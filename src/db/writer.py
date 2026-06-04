"""DatabaseWriter — psycopg2-based upserts for Phase 3 extraction output."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg2  # type: ignore[import]
import psycopg2.extras  # type: ignore[import]
from loguru import logger

from src.db.schema import ALL_STATEMENTS
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

    def ensure_schema(self) -> None:
        """Run all CREATE TABLE IF NOT EXISTS statements."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            for stmt in ALL_STATEMENTS:
                cur.execute(stmt)
        conn.commit()
        logger.debug("Database schema ensured")

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
    ) -> int:
        """Upsert document row; returns its id."""
        content = doc.content
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents
                    (country_id, document_type, official_name, source_url,
                     content_type, char_count, detected_language, criteria_ids, retrieved_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (country_id, official_name) DO UPDATE
                    SET source_url        = EXCLUDED.source_url,
                        content_type      = EXCLUDED.content_type,
                        char_count        = EXCLUDED.char_count,
                        detected_language = EXCLUDED.detected_language,
                        criteria_ids      = EXCLUDED.criteria_ids,
                        retrieved_at      = EXCLUDED.retrieved_at
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
        extraction_dimension = metadata.get("extraction_dimension")
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_extractions
                    (document_id, extracted_fields, enforcement_authority,
                     total_sections, sections_with_signal, split_tier_used,
                     detected_language, status, extracted_at, error_message,
                     extraction_dimension, constitutional_privacy_right,
                     dpa_exists, dpa_independence)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s)
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
                        dpa_independence             = EXCLUDED.dpa_independence
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
