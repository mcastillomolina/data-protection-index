"""Phase 4 — CriterionScorer: scores all 14 PI criteria for a country using LLM + pgvector."""

import json
from datetime import datetime
from typing import Any

import psycopg2
import psycopg2.extras
from loguru import logger

from src.clients.embedding_client import EmbeddingClient
from src.clients.llm_client import LLMClient
from src.config.criteria import CRITERIA, CRITERION_QUERY_SENTENCES, CRITERION_RUBRICS
from src.models.scoring import CriterionScore
from src.prompts.criterion_scoring import (
    CRITERION_SCORER_SYSTEM,
    CRITERION_SCORER_USER,
    MIXED_INSTRUCTIONS,
    OPACITY_BLOCK,
)

_MIXED_CRITERIA = {4, 5, 7, 9, 14}


def _vec_str(embedding: list[float]) -> str:
    """Format a float list as a pgvector literal: '[x,y,...]'."""
    return "[" + ",".join(str(v) for v in embedding) + "]"


class CriterionScorer:
    """
    Scores all 14 Privacy International criteria for a country.

    One instance per country run — loops all 14 criteria internally.
    Query embeddings are computed once per criterion and cached in the instance.
    Results are written to criterion_scores table after each criterion.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        embedding_client: EmbeddingClient,
        dsn: str,
        model_name: str,
        cosine_threshold: float = 0.35,
        max_sections: int = 20,
    ) -> None:
        self._llm = llm_client
        self._emb = embedding_client
        self._dsn = dsn
        self._model_name = model_name
        self._cosine_threshold = cosine_threshold
        self._max_sections = max_sections
        self._embedding_cache: dict[int, list[float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_all_criteria(
        self,
        country_id: int,
        country_name: str,
        country_code: str,
        reference_year: int,
        information_environment: str = "open",
        skip_if_scored: bool = False,
    ) -> list[CriterionScore]:
        """
        Score all 14 criteria for a country.

        Each score is written to criterion_scores before moving to the next.
        Returns the list of all successfully scored criteria.
        If skip_if_scored=True, criteria already in criterion_scores are returned
        from the DB without calling the LLM.
        """
        scores: list[CriterionScore] = []
        conn = psycopg2.connect(self._dsn)
        try:
            for criterion_number in range(1, 15):
                logger.info(
                    f"Scoring criterion {criterion_number}/14 "
                    f"({CRITERIA[criterion_number]['name']}) for {country_name}"
                )
                score = self._score_one(
                    conn,
                    country_id=country_id,
                    country_name=country_name,
                    country_code=country_code,
                    criterion_number=criterion_number,
                    reference_year=reference_year,
                    information_environment=information_environment,
                    skip_if_scored=skip_if_scored,
                )
                if score is not None:
                    self._write_score(conn, score, country_id)
                    conn.commit()
                    scores.append(score)
                    logger.info(
                        f"Criterion {criterion_number} scored: "
                        f"{score.criterion_score:.2f} ({score.confidence})"
                    )
                else:
                    logger.warning(
                        f"Criterion {criterion_number} returned no score — skipping"
                    )
        finally:
            conn.close()

        logger.info(
            f"Scoring complete for {country_name}: "
            f"{len(scores)}/14 criteria scored"
        )
        return scores

    # ------------------------------------------------------------------
    # Internal: score one criterion
    # ------------------------------------------------------------------

    def _score_one(
        self,
        conn: psycopg2.extensions.connection,
        country_id: int,
        country_name: str,
        country_code: str,
        criterion_number: int,
        reference_year: int,
        information_environment: str,
        skip_if_scored: bool = False,
    ) -> CriterionScore | None:
        if skip_if_scored:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT criterion_number, criterion_name, dimension,
                           legal_subscore, enforcement_subscore, criterion_score,
                           confidence, evidence_count, information_opacity,
                           rationale, evidence_gaps, key_sources,
                           model_used, reference_year
                    FROM criterion_scores
                    WHERE country_id = %s AND criterion_number = %s
                      AND reference_year = %s AND model_used = %s
                    """,
                    (country_id, criterion_number, reference_year, self._model_name),
                )
                row = cur.fetchone()
            if row is not None:
                logger.info(
                    f"[CACHE HIT] Criterion {criterion_number} already scored "
                    f"({row['criterion_score']:.2f}) — skipping LLM"
                )
                return CriterionScore(
                    country_code=country_code,
                    criterion_number=row["criterion_number"],
                    criterion_name=row["criterion_name"],
                    dimension=row["dimension"],
                    legal_subscore=row["legal_subscore"],
                    enforcement_subscore=row["enforcement_subscore"],
                    criterion_score=row["criterion_score"],
                    confidence=row["confidence"],
                    evidence_count=row["evidence_count"],
                    information_opacity=row["information_opacity"],
                    rationale=row["rationale"],
                    evidence_gaps=row["evidence_gaps"],
                    key_sources=row["key_sources"] or [],
                    model_used=row["model_used"],
                    reference_year=row["reference_year"],
                    created_at=datetime.now(),
                )

        query_embedding = self._get_query_embedding(criterion_number)
        evidence = self._assemble_evidence(
            conn, country_id, criterion_number, query_embedding
        )

        user_prompt = self._build_user_prompt(
            country_name=country_name,
            criterion_number=criterion_number,
            reference_year=reference_year,
            information_environment=information_environment,
            evidence=evidence,
        )

        try:
            raw = self._llm.complete_json(
                prompt=user_prompt,
                system_prompt=CRITERION_SCORER_SYSTEM,
                temperature=0.2,
                max_tokens=1500,
            )
        except Exception as exc:
            logger.error(
                f"LLM call failed for criterion {criterion_number}: {exc}"
            )
            return None

        # Retry with a shorter context window when the LLM returns an empty response.
        if not raw or "criterion_score" not in raw:
            reduced = max(5, self._max_sections // 2)
            logger.warning(
                f"Criterion {criterion_number}: empty LLM response — "
                f"retrying with max_sections={reduced} (was {self._max_sections})"
            )
            evidence = self._assemble_evidence(
                conn, country_id, criterion_number, query_embedding,
                max_sections=reduced,
            )
            user_prompt = self._build_user_prompt(
                country_name=country_name,
                criterion_number=criterion_number,
                reference_year=reference_year,
                information_environment=information_environment,
                evidence=evidence,
            )
            try:
                raw = self._llm.complete_json(
                    prompt=user_prompt,
                    system_prompt=CRITERION_SCORER_SYSTEM,
                    temperature=0.2,
                    max_tokens=1500,
                )
            except Exception as exc:
                logger.error(
                    f"LLM retry failed for criterion {criterion_number}: {exc}"
                )
                return None

        return self._parse_score(
            raw=raw,
            criterion_number=criterion_number,
            country_code=country_code,
            reference_year=reference_year,
            evidence_count=len(evidence),
        )

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------

    def _get_query_embedding(self, criterion_number: int) -> list[float]:
        """Embed the criterion query sentence once; cache the result."""
        if criterion_number not in self._embedding_cache:
            sentence = CRITERION_QUERY_SENTENCES[criterion_number]
            vectors = self._emb.embed([sentence])
            self._embedding_cache[criterion_number] = vectors[0]
        return self._embedding_cache[criterion_number]

    # ------------------------------------------------------------------
    # Evidence assembly
    # ------------------------------------------------------------------

    def _assemble_evidence(
        self,
        conn: psycopg2.extensions.connection,
        country_id: int,
        criterion_number: int,
        query_embedding: list[float],
        max_sections: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Pull evidence from two sources:
        1. section_extractions — top-k via cosine similarity (Gate 2)
        2. enforcement_records — by criterion number
        """
        evidence: list[dict[str, Any]] = []
        vec = _vec_str(query_embedding)

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # --- Pull 1: sections via pgvector ---
            cur.execute(
                """
                SELECT
                    se.section_text_original,
                    se.extracted_fields,
                    d.document_type,
                    d.official_name,
                    d.source_url,
                    1 - (se.embedding <=> %s::vector) AS similarity
                FROM section_extractions se
                JOIN documents d ON d.id = se.document_id
                WHERE
                    d.country_id = %s
                    AND %s = ANY(d.criteria_ids)
                    AND se.all_null = false
                    AND se.embedding IS NOT NULL
                    AND 1 - (se.embedding <=> %s::vector) >= %s
                ORDER BY se.embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    vec,
                    country_id,
                    criterion_number,
                    vec,
                    self._cosine_threshold,
                    vec,
                    max_sections if max_sections is not None else self._max_sections,
                ),
            )
            for row in cur.fetchall():
                evidence.append(
                    {
                        "source_type": "section",
                        "document_type": row["document_type"],
                        "official_name": row["official_name"],
                        "source_url": row["source_url"],
                        "similarity": round(row["similarity"], 3),
                        "text": row["section_text_original"],
                        "extracted_fields": row["extracted_fields"],
                    }
                )

            # --- Pull 2: enforcement records ---
            cur.execute(
                """
                SELECT
                    source_type, source_url, source_domain, enforcing_body,
                    subject_entity, sanction_type, sanction_amount,
                    sanction_currency, sanction_date, summary, reliability_score
                FROM enforcement_records
                WHERE country_id = %s AND pi_criterion_number = %s
                ORDER BY reliability_score DESC, sanction_date DESC
                LIMIT 10
                """,
                (country_id, criterion_number),
            )
            for row in cur.fetchall():
                evidence.append(
                    {
                        "source_type": "enforcement_record",
                        "source_url": row["source_url"],
                        "source_domain": row["source_domain"],
                        "enforcing_body": row["enforcing_body"],
                        "subject_entity": row["subject_entity"],
                        "sanction_type": row["sanction_type"],
                        "sanction_amount": row["sanction_amount"],
                        "sanction_currency": row["sanction_currency"],
                        "sanction_date": (
                            str(row["sanction_date"]) if row["sanction_date"] else None
                        ),
                        "summary": row["summary"],
                        "reliability_score": row["reliability_score"],
                    }
                )

            # --- Pull 3: external indicators ---
            cur.execute(
                """
                SELECT source_name, indicator_name, indicator_value,
                       indicator_normalised, source_year, notes
                FROM external_indicators
                WHERE country_id = %s AND pi_criterion_number = %s
                ORDER BY source_year DESC NULLS LAST
                """,
                (country_id, criterion_number),
            )
            for row in cur.fetchall():
                evidence.append(
                    {
                        "source_type": "external_indicator",
                        "source_name": row["source_name"],
                        "indicator_name": row["indicator_name"],
                        "indicator_value": row["indicator_value"],
                        "indicator_normalised": row["indicator_normalised"],
                        "source_year": row["source_year"],
                        "notes": row["notes"],
                    }
                )

        logger.debug(
            f"Criterion {criterion_number}: assembled {len(evidence)} evidence items"
        )
        return evidence

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_user_prompt(
        self,
        country_name: str,
        criterion_number: int,
        reference_year: int,
        information_environment: str,
        evidence: list[dict[str, Any]],
    ) -> str:
        criterion_info = CRITERIA[criterion_number]
        is_mixed = criterion_number in _MIXED_CRITERIA
        is_opacity = information_environment == "restricted"

        return CRITERION_SCORER_USER.format(
            country_name=country_name,
            criterion_number=criterion_number,
            criterion_name=criterion_info["name"],
            dimension=criterion_info["dimension"],
            reference_year=reference_year,
            information_environment=information_environment,
            evidence_count=len(evidence),
            formatted_evidence=self._format_evidence(evidence),
            opacity_block=OPACITY_BLOCK if is_opacity else "",
            criterion_rubric=CRITERION_RUBRICS[criterion_number],
            mixed_instructions=MIXED_INSTRUCTIONS if is_mixed else "",
        )

    def _format_evidence(self, evidence: list[dict[str, Any]]) -> str:
        """Format evidence items as numbered blocks for the LLM prompt."""
        if not evidence:
            return "(no evidence found)"
        parts: list[str] = []
        for i, item in enumerate(evidence, 1):
            if item["source_type"] == "section":
                header = (
                    f"[{i}] {item['document_type']} — {item['official_name']} "
                    f"(similarity: {item['similarity']})"
                )
                body = item["text"][:2000]  # cap section length in prompt
            elif item["source_type"] == "external_indicator":
                header = (
                    f"[{i}] external_indicator — {item.get('source_name', 'unknown')} "
                    f"({item.get('indicator_name', '')}, year={item.get('source_year', '?')})"
                )
                body = (
                    f"Raw value: {item.get('indicator_value')}  "
                    f"Normalised (1–5): {item.get('indicator_normalised')}"
                )
                if item.get("notes"):
                    body += f"\n{item['notes']}"
            else:
                header = (
                    f"[{i}] enforcement_record — {item.get('enforcing_body', 'unknown')} "
                    f"(reliability: {item.get('reliability_score', '?')})"
                )
                body = item.get("summary") or ""
                if item.get("sanction_amount"):
                    body += (
                        f"\nSanction: {item['sanction_amount']} "
                        f"{item.get('sanction_currency', '')} on {item.get('sanction_date', '?')}"
                    )
            parts.append(f"{header}\n{body}")
        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Parse LLM response
    # ------------------------------------------------------------------

    def _parse_score(
        self,
        raw: dict[str, Any],
        criterion_number: int,
        country_code: str,
        reference_year: int,
        evidence_count: int,
    ) -> CriterionScore | None:
        criterion_info = CRITERIA[criterion_number]
        is_mixed = criterion_number in _MIXED_CRITERIA

        try:
            criterion_score = float(raw["criterion_score"])
            legal_subscore = (
                float(raw["legal_subscore"]) if raw.get("legal_subscore") is not None else None
            )
            enforcement_subscore = (
                float(raw["enforcement_subscore"])
                if raw.get("enforcement_subscore") is not None
                else None
            )

            if is_mixed and (legal_subscore is None or enforcement_subscore is None):
                logger.warning(
                    f"Criterion {criterion_number} is mixed but LLM returned "
                    f"legal_subscore={legal_subscore}, enforcement_subscore={enforcement_subscore}. "
                    "Using criterion_score for missing sub-score."
                )
                if legal_subscore is None:
                    legal_subscore = criterion_score
                if enforcement_subscore is None:
                    enforcement_subscore = criterion_score

            return CriterionScore(
                country_code=country_code,
                criterion_number=criterion_number,
                criterion_name=criterion_info["name"],
                dimension=criterion_info["dimension"],
                legal_subscore=legal_subscore,
                enforcement_subscore=enforcement_subscore,
                criterion_score=criterion_score,
                confidence=raw.get("confidence", "low"),
                evidence_count=evidence_count,
                information_opacity=bool(raw.get("information_opacity", False)),
                rationale=raw.get("rationale", ""),
                evidence_gaps=raw.get("evidence_gaps", ""),
                key_sources=raw.get("key_sources", []),
                model_used=self._model_name,
                reference_year=reference_year,
                created_at=datetime.now(),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.error(
                f"Failed to parse LLM score for criterion {criterion_number}: {exc}. "
                f"Raw response: {raw}"
            )
            return None

    # ------------------------------------------------------------------
    # DB write
    # ------------------------------------------------------------------

    def _write_score(
        self,
        conn: psycopg2.extensions.connection,
        score: CriterionScore,
        country_id: int,
    ) -> None:
        """Upsert a CriterionScore into criterion_scores (idempotent)."""
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO criterion_scores (
                    country_id, criterion_number, criterion_name, dimension,
                    legal_subscore, enforcement_subscore, criterion_score,
                    confidence, evidence_count, information_opacity,
                    rationale, evidence_gaps, key_sources,
                    model_used, reference_year
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (country_id, criterion_number, reference_year, model_used)
                DO UPDATE SET
                    criterion_score      = EXCLUDED.criterion_score,
                    legal_subscore       = EXCLUDED.legal_subscore,
                    enforcement_subscore = EXCLUDED.enforcement_subscore,
                    confidence           = EXCLUDED.confidence,
                    evidence_count       = EXCLUDED.evidence_count,
                    information_opacity  = EXCLUDED.information_opacity,
                    rationale            = EXCLUDED.rationale,
                    evidence_gaps        = EXCLUDED.evidence_gaps,
                    key_sources          = EXCLUDED.key_sources,
                    created_at           = NOW()
                """,
                (
                    country_id,
                    score.criterion_number,
                    score.criterion_name,
                    score.dimension,
                    score.legal_subscore,
                    score.enforcement_subscore,
                    score.criterion_score,
                    score.confidence,
                    score.evidence_count,
                    score.information_opacity,
                    score.rationale,
                    score.evidence_gaps,
                    json.dumps(score.key_sources),
                    score.model_used,
                    score.reference_year,
                ),
            )
