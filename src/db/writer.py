"""DatabaseWriter — psycopg2-based upserts for Phase 3 extraction output."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg2  # type: ignore[import]
import psycopg2.extras  # type: ignore[import]
from loguru import logger

from src.db.schema import ALL_STATEMENTS, CREATE_VECTOR_EXTENSION
from src.models.country import Country
from src.models.extraction import SectionExtractionResult
from src.models.retrieval import RetrievedDocument


class DatabaseWriter:
    """Writes Phase 3 extraction results to PostgreSQL using idempotent upserts."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn: Optional[Any] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_conn(self) -> Any:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self._dsn)
            self._conn.autocommit = False
        return self._conn

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    # Fixed key for the session-level advisory lock that serialises schema setup.
    # The idempotent DDL (CREATE/ALTER/CREATE INDEX) takes AccessExclusiveLocks;
    # two processes running ensure_schema() concurrently deadlock without this.
    _SCHEMA_LOCK_KEY = 823476101

    def ensure_schema(self) -> None:
        """Run all CREATE TABLE IF NOT EXISTS statements.

        Serialised across processes via a Postgres advisory lock so parallel
        pipeline runs queue for the DDL instead of deadlocking on table locks.
        The lock is session-level; it auto-releases if the connection drops.
        """
        conn = self._get_conn()

        # Fast path: if the schema is already fully migrated, skip the DDL
        # entirely. ALTER TABLE ... IF NOT EXISTS is a no-op when the column
        # already exists, but it still takes an AccessExclusiveLock to check —
        # which can deadlock against unrelated long-running DML from other
        # concurrent processes (e.g. a batch embedding UPDATE on a child table
        # via FK to `documents`), not just against another ensure_schema() call.
        # Checking these two columns (the last two ALTERs applied, in order) is
        # enough: statements run sequentially, so if the last one landed, every
        # earlier one already did too.
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    EXISTS (SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'criterion_scores'
                              AND column_name = 'retrieval_limited') AS a,
                    EXISTS (SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'country_index_scores'
                              AND column_name = 'partial_coverage') AS b
                """
            )
            already_migrated = all(cur.fetchone())
        if already_migrated:
            logger.debug("Database schema already up to date — skipping DDL")
            return

        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(%s)", (self._SCHEMA_LOCK_KEY,))
        try:
            with conn.cursor() as cur:
                # pgvector extension must exist before vector columns can be created.
                # Run it first so a missing extension surfaces with actionable instructions.
                try:
                    cur.execute(CREATE_VECTOR_EXTENSION)
                    conn.commit()
                except psycopg2.Error as e:
                    conn.rollback()
                    raise RuntimeError(
                        "pgvector extension is not installed in this PostgreSQL instance.\n"
                        "Fix: change the Docker image in docker-compose.yml from\n"
                        "  'postgres:16-alpine'  →  'pgvector/pgvector:pg16'\n"
                        "then run:  docker compose up -d --force-recreate"
                    ) from e
                for stmt in ALL_STATEMENTS:
                    cur.execute(stmt)
            conn.commit()
        finally:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock(%s)", (self._SCHEMA_LOCK_KEY,))
            conn.commit()
        logger.debug("Database schema ensured")

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_country_id_by_name(self, country_name: str) -> Optional[int]:
        """Return the DB id for a country by name (or alias), or None if not found."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM countries
                WHERE name = %s
                   OR %s = ANY(aliases)
                LIMIT 1
                """,
                (country_name, country_name),
            )
            row = cur.fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------------
    # Upserts
    # ------------------------------------------------------------------

    def upsert_country(self, country: Country) -> int:
        """Upsert country row; returns its id."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO countries (name, iso_code, region, languages, aliases)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (iso_code) DO UPDATE
                    SET name      = EXCLUDED.name,
                        region    = EXCLUDED.region,
                        languages = EXCLUDED.languages
                RETURNING id
                """,
                (
                    country.name,
                    country.iso_code,
                    country.region,
                    country.official_languages,
                    country.aliases,
                ),
            )
            row = cur.fetchone()
        conn.commit()
        return row[0]

    def find_country(self, name: str) -> Optional[Dict[str, Any]]:
        """Look up a country by name (case-insensitive) or alias. Returns metadata dict or None."""
        conn = self._get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT name, iso_code, region, languages, aliases
                FROM   countries
                WHERE  name ILIKE %s
                   OR  %s = ANY(aliases)
                LIMIT 1
                """,
                (name, name),
            )
            row = cur.fetchone()
        return _db_row_to_metadata(dict(row)) if row is not None else None

    def find_country_by_iso(self, iso_code: str) -> Optional[Dict[str, Any]]:
        """Look up a country by ISO 3166-1 alpha-2 code. Returns metadata dict or None."""
        conn = self._get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT name, iso_code, region, languages, aliases
                FROM   countries
                WHERE  iso_code = %s
                """,
                (iso_code.upper(),),
            )
            row = cur.fetchone()
        return _db_row_to_metadata(dict(row)) if row is not None else None

    def add_alias(self, iso_code: str, alias: str) -> None:
        """Append alias to the country's aliases array if not already present."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE countries
                SET    aliases = array_append(aliases, %s)
                WHERE  iso_code = %s
                  AND  NOT (%s = ANY(aliases))
                """,
                (alias, iso_code.upper(), alias),
            )
        conn.commit()

    def upsert_document(
        self,
        country_id: int,
        doc: RetrievedDocument,
        detected_language: str,
        information_opacity: bool = False,
    ) -> int:
        """Upsert document row; returns its id."""
        content = doc.content
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents
                    (country_id, document_type, official_name, source_url,
                     content_type, char_count, detected_language, criteria_ids,
                     retrieved_at, information_opacity)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (country_id, official_name) DO UPDATE
                    SET source_url          = EXCLUDED.source_url,
                        content_type        = EXCLUDED.content_type,
                        char_count          = EXCLUDED.char_count,
                        detected_language   = EXCLUDED.detected_language,
                        criteria_ids        = EXCLUDED.criteria_ids,
                        retrieved_at        = EXCLUDED.retrieved_at,
                        information_opacity = EXCLUDED.information_opacity
                RETURNING id
                """,
                (
                    country_id,
                    doc.document.document_type,
                    doc.document.official_name,
                    doc.successful_url,
                    content.content_type if content else None,
                    content.char_count if content else None,
                    detected_language,
                    doc.document.criteria_ids,
                    datetime.now(),
                    information_opacity,
                ),
            )
            row = cur.fetchone()
        conn.commit()
        return row[0]

    def upsert_section_extraction(
        self,
        document_id: int,
        result: SectionExtractionResult,
        llm_provider: str,
        llm_model: str,
        extraction_dimension: Optional[str] = None,
    ) -> int:
        """Upsert a single section extraction row; returns its id."""
        conn = self._get_conn()
        fields_json = (
            psycopg2.extras.Json(result.extracted_fields)
            if result.extracted_fields is not None
            else None
        )
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO section_extractions
                    (document_id, section_index, section_header, section_text_original,
                     split_tier_used, extracted_fields, all_null, llm_provider, llm_model,
                     processing_time_seconds, error_message, extraction_dimension)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id, section_index) DO UPDATE
                    SET section_header          = EXCLUDED.section_header,
                        section_text_original   = EXCLUDED.section_text_original,
                        split_tier_used         = EXCLUDED.split_tier_used,
                        extracted_fields        = EXCLUDED.extracted_fields,
                        all_null                = EXCLUDED.all_null,
                        llm_provider            = EXCLUDED.llm_provider,
                        llm_model               = EXCLUDED.llm_model,
                        processing_time_seconds = EXCLUDED.processing_time_seconds,
                        error_message           = EXCLUDED.error_message,
                        extraction_dimension    = EXCLUDED.extraction_dimension,
                        extracted_at            = NOW()
                RETURNING id
                """,
                (
                    document_id,
                    result.section_index,
                    result.section_header,
                    result.section_text_original,
                    result.split_tier_used,
                    fields_json,
                    result.all_null,
                    llm_provider,
                    llm_model,
                    result.processing_time_seconds,
                    result.error_message,
                    extraction_dimension,
                ),
            )
            row = cur.fetchone()
        conn.commit()
        return row[0]

    def upsert_document_extraction(
        self,
        document_id: int,
        aggregated: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> int:
        """Upsert aggregated document extraction; returns its id."""
        conn = self._get_conn()
        enforcement = aggregated.get("enforcement_body")
        constitutional_privacy_right = aggregated.get("constitutional_privacy_right")
        dpa_exists = aggregated.get("dpa_exists")
        dpa_independence = aggregated.get("dpa_independence")
        # Problem 2C fix: the column existed but was never written. Wire the mapping
        # so a legitimate flag from the extractor persists. Expected null/false today
        # (the extractor rarely emits it — see diagnosis Problem 2B, intentionally
        # left untouched), which is the honest state.
        information_opacity_flag = aggregated.get("information_opacity_flag")
        extraction_dimension = metadata.get("extraction_dimension")
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_extractions
                    (document_id, extracted_fields, enforcement_authority,
                     total_sections, sections_with_signal, split_tier_used,
                     detected_language, status, extracted_at, error_message,
                     extraction_dimension, constitutional_privacy_right,
                     dpa_exists, dpa_independence, information_opacity_flag)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id) DO UPDATE
                    SET extracted_fields             = EXCLUDED.extracted_fields,
                        enforcement_authority        = EXCLUDED.enforcement_authority,
                        total_sections               = EXCLUDED.total_sections,
                        sections_with_signal         = EXCLUDED.sections_with_signal,
                        split_tier_used              = EXCLUDED.split_tier_used,
                        detected_language            = EXCLUDED.detected_language,
                        status                       = EXCLUDED.status,
                        extracted_at                 = NOW(),
                        error_message                = EXCLUDED.error_message,
                        extraction_dimension         = EXCLUDED.extraction_dimension,
                        constitutional_privacy_right = EXCLUDED.constitutional_privacy_right,
                        dpa_exists                   = EXCLUDED.dpa_exists,
                        dpa_independence             = EXCLUDED.dpa_independence,
                        information_opacity_flag     = EXCLUDED.information_opacity_flag
                RETURNING id
                """,
                (
                    document_id,
                    psycopg2.extras.Json(aggregated),
                    enforcement,
                    metadata.get("total_sections", 0),
                    metadata.get("sections_with_signal", 0),
                    metadata.get("split_tier_used", "unknown"),
                    metadata.get("detected_language", "unknown"),
                    metadata.get("status", "success"),
                    metadata.get("error_message"),
                    extraction_dimension,
                    constitutional_privacy_right,
                    dpa_exists,
                    dpa_independence,
                    information_opacity_flag,
                ),
            )
            row = cur.fetchone()
        conn.commit()
        return row[0]


def _db_row_to_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a DB countries row into the metadata dict shape expected by _build_country."""
    return {
        "name":               row["name"],
        "iso_code":           row["iso_code"].strip(),
        "official_languages": row.get("languages") or [],
        "government_domains": [],
        "region":             row.get("region") or "",
        "known_documents":    {},
        "search_hints":       [],
        "aliases":            row.get("aliases") or [],
    }
