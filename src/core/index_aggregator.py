"""Phase 4 Step G — IndexAggregator: computes dual-framework country index scores."""

import csv
import json
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras
from loguru import logger

from src.config.criteria import CONFIDENCE_WEIGHTS, CRITERIA
from src.models.scoring import CountryIndexScore

PI_CATEGORIES = [
    (4.1, "Consistently upholds human rights standards"),
    (3.6, "Significant protections and safeguards"),
    (3.1, "Adequate safeguards against abuse"),
    (2.6, "Some safeguards but weakened protections"),
    (2.1, "Systemic failure to uphold safeguards"),
    (1.6, "Extensive surveillance societies"),
    (1.1, "Endemic surveillance societies"),
]

_LEGAL_DIMENSIONS = {"legal", "mixed"}
_ENFORCEMENT_DIMS = {"enforcement", "mixed"}
_MIXED_CRITERIA   = {4, 5, 7, 9, 14}

_EXPORT_FIELDS = [
    "country_name", "iso_code", "final_score", "legal_score",
    "enforcement_score", "pi_category", "rank", "criteria_count", "opacity_affected",
]


class IndexAggregator:
    """
    Aggregates criterion_scores rows into country_index_scores.

    Reads legal_weight, enforcement_weight, missing_strategy, and
    confidence_weighting from the ScoringConfig passed at construction —
    not hardcoded.
    """

    def __init__(self, dsn: str, scoring_config: Any) -> None:
        self._dsn               = dsn
        self._legal_weight      = scoring_config.legal_weight
        self._enforce_weight    = scoring_config.enforcement_weight
        self._missing_strategy  = scoring_config.missing_strategy
        self._conf_weighting    = scoring_config.confidence_weighting
        self._min_criteria      = getattr(scoring_config, "min_criteria_for_ranking", 12)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_country_score(
        self,
        country_id: int,
        reference_year: int,
        model_used: str | None = None,
    ) -> CountryIndexScore:
        """Aggregate criterion scores for one country and write to country_index_scores."""
        scores = self._fetch_criterion_scores(country_id, reference_year, model_used)
        scored_nums = {s["criterion_number"] for s in scores}
        missing = [i for i in range(1, 15) if i not in scored_nums]

        if missing:
            logger.warning(
                f"country_id={country_id}: {len(missing)} criteria missing: {missing}"
            )

        if self._missing_strategy == "penalise_as_1":
            for n in missing:
                dim = CRITERIA[n]["dimension"]
                scores.append({
                    "criterion_number": n,
                    "dimension": dim,
                    "criterion_score": 1.0,
                    "legal_subscore": 1.0 if dim in _LEGAL_DIMENSIONS else None,
                    "enforcement_subscore": 1.0 if dim in _ENFORCEMENT_DIMS else None,
                    "confidence": "low",
                    "information_opacity": False,
                })

        legal_pairs      = self._extract_pairs(scores, "legal")
        enforcement_pairs = self._extract_pairs(scores, "enforcement")

        legal_mean      = self._weighted_mean(legal_pairs)
        enforcement_mean = self._weighted_mean(enforcement_pairs)

        if legal_mean is not None and enforcement_mean is not None:
            final_score = (
                legal_mean * self._legal_weight
                + enforcement_mean * self._enforce_weight
            )
        elif legal_mean is not None:
            final_score = legal_mean
        elif enforcement_mean is not None:
            final_score = enforcement_mean
        else:
            final_score = 1.0

        opacity_criteria = [
            s["criterion_number"] for s in scores if s.get("information_opacity")
        ]

        criteria_count = len({s["criterion_number"] for s in scores})

        result = CountryIndexScore(
            country_id=country_id,
            reference_year=reference_year,
            legal_score=round(legal_mean, 4) if legal_mean is not None else None,
            enforcement_score=round(enforcement_mean, 4) if enforcement_mean is not None else None,
            final_score=round(final_score, 4),
            pi_category=self._map_category(final_score),
            criteria_count=criteria_count,
            missing_criteria=missing,
            opacity_affected_criteria=opacity_criteria,
            partial_coverage=criteria_count < self._min_criteria,
            model_used=model_used or "default",
            confidence_weighting=self._conf_weighting,
            missing_strategy=self._missing_strategy,
        )
        self._write_index_score(result)
        return result

    def rank_countries(
        self,
        reference_year: int,
        model_used: str | None = None,
    ) -> list[CountryIndexScore]:
        """Assign ranks by final_score DESC and write back to DB.

        Only countries with partial_coverage=false are ranked. Partial-coverage
        countries are computed and persisted but kept out of the ranking; their
        rank is reset to NULL so a stale rank from an earlier run never leaks
        into the export.
        """
        rows = [
            r for r in self._fetch_all_index_scores(reference_year, model_used)
            if not r.partial_coverage
        ]
        rows.sort(key=lambda r: r.final_score, reverse=True)

        conn = psycopg2.connect(self._dsn)
        try:
            with conn.cursor() as cur:
                # Clear ranks on partial-coverage rows for this year/model.
                cur.execute(
                    """
                    UPDATE country_index_scores
                    SET rank = NULL
                    WHERE reference_year = %s
                      AND COALESCE(model_used, 'default') = COALESCE(%s, 'default')
                      AND partial_coverage = true
                    """,
                    (reference_year, model_used or "default"),
                )
                for rank, row in enumerate(rows, 1):
                    row.rank = rank
                    cur.execute(
                        """
                        UPDATE country_index_scores
                        SET rank = %s
                        WHERE country_id = %s
                          AND reference_year = %s
                          AND COALESCE(model_used, 'default') = COALESCE(%s, 'default')
                        """,
                        (rank, row.country_id, reference_year, model_used or "default"),
                    )
            conn.commit()
        finally:
            conn.close()

        return rows

    def export_index(
        self,
        reference_year: int,
        model_used: str | None = None,
        fmt: str = "csv",
    ) -> Path:
        """Write ranked index to data/outputs/index/ as CSV or JSON."""
        rows = self._fetch_index_with_country_names(reference_year, model_used)

        out_dir = Path("data/outputs/index")
        out_dir.mkdir(parents=True, exist_ok=True)
        slug = (model_used or "default").replace("/", "_").replace(":", "_")
        out_path = out_dir / f"index_{reference_year}_{slug}.{fmt}"

        if fmt == "json":
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False, default=str)
        else:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_EXPORT_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

        logger.info(f"Index exported to {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Internal: math
    # ------------------------------------------------------------------

    def _extract_pairs(
        self, scores: list[dict], dimension_group: str
    ) -> list[tuple[float, str]]:
        """
        Return (score_value, confidence) pairs for the requested dimension group.

        For mixed criteria, use the dimension-specific subscore when available;
        fall back to criterion_score.
        """
        pairs: list[tuple[float, str]] = []
        for s in scores:
            dim = s.get("dimension", "")
            if dimension_group == "legal" and dim not in _LEGAL_DIMENSIONS:
                continue
            if dimension_group == "enforcement" and dim not in _ENFORCEMENT_DIMS:
                continue

            if dim == "mixed" and s["criterion_number"] in _MIXED_CRITERIA:
                if dimension_group == "legal":
                    val = s.get("legal_subscore") or s["criterion_score"]
                else:
                    val = s.get("enforcement_subscore") or s["criterion_score"]
            else:
                val = s["criterion_score"]

            pairs.append((float(val), s.get("confidence", "low")))
        return pairs

    def _weighted_mean(self, pairs: list[tuple[float, str]]) -> float | None:
        if not pairs:
            return None
        if not self._conf_weighting:
            return sum(v for v, _ in pairs) / len(pairs)
        weights = [CONFIDENCE_WEIGHTS.get(c, 0.4) for _, c in pairs]
        values  = [v for v, _ in pairs]
        total_w = sum(weights)
        if total_w == 0:
            return sum(values) / len(values)
        return sum(v * w for v, w in zip(values, weights)) / total_w

    def _map_category(self, score: float) -> str:
        for threshold, label in PI_CATEGORIES:
            if score >= threshold:
                return label
        return "Endemic surveillance societies"

    # ------------------------------------------------------------------
    # Internal: DB reads
    # ------------------------------------------------------------------

    def _fetch_criterion_scores(
        self, country_id: int, reference_year: int, model_used: str | None
    ) -> list[dict]:
        conn = psycopg2.connect(self._dsn)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if model_used:
                    cur.execute(
                        """
                        SELECT criterion_number, dimension, criterion_score,
                               legal_subscore, enforcement_subscore,
                               confidence, information_opacity
                        FROM criterion_scores
                        WHERE country_id = %s
                          AND reference_year = %s
                          AND model_used = %s
                        """,
                        (country_id, reference_year, model_used),
                    )
                else:
                    cur.execute(
                        """
                        SELECT criterion_number, dimension, criterion_score,
                               legal_subscore, enforcement_subscore,
                               confidence, information_opacity
                        FROM criterion_scores
                        WHERE country_id = %s AND reference_year = %s
                        """,
                        (country_id, reference_year),
                    )
                return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

    def _fetch_all_index_scores(
        self, reference_year: int, model_used: str | None
    ) -> list[CountryIndexScore]:
        conn = psycopg2.connect(self._dsn)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if model_used:
                    cur.execute(
                        """
                        SELECT country_id, reference_year, legal_score, enforcement_score,
                               final_score, pi_category, rank, criteria_count,
                               missing_criteria, opacity_affected, model_used,
                               confidence_weighting, missing_strategy, partial_coverage
                        FROM country_index_scores
                        WHERE reference_year = %s AND model_used = %s
                        """,
                        (reference_year, model_used),
                    )
                else:
                    cur.execute(
                        """
                        SELECT country_id, reference_year, legal_score, enforcement_score,
                               final_score, pi_category, rank, criteria_count,
                               missing_criteria, opacity_affected, model_used,
                               confidence_weighting, missing_strategy, partial_coverage
                        FROM country_index_scores
                        WHERE reference_year = %s
                        """,
                        (reference_year,),
                    )
                rows = cur.fetchall()
        finally:
            conn.close()

        results = []
        for r in rows:
            results.append(CountryIndexScore(
                country_id=r["country_id"],
                reference_year=r["reference_year"],
                legal_score=r["legal_score"],
                enforcement_score=r["enforcement_score"],
                final_score=r["final_score"],
                pi_category=r["pi_category"] or "",
                rank=r["rank"],
                criteria_count=r["criteria_count"] or 0,
                missing_criteria=r["missing_criteria"] or [],
                opacity_affected_criteria=[],  # not stored per-criterion in index table
                model_used=r["model_used"],
                confidence_weighting=bool(r["confidence_weighting"]),
                missing_strategy=r["missing_strategy"] or "exclude",
                partial_coverage=bool(r["partial_coverage"]),
            ))
        return results

    def _fetch_index_with_country_names(
        self, reference_year: int, model_used: str | None
    ) -> list[dict]:
        conn = psycopg2.connect(self._dsn)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if model_used:
                    cur.execute(
                        """
                        SELECT c.name AS country_name, c.iso_code,
                               cis.final_score, cis.legal_score, cis.enforcement_score,
                               cis.pi_category, cis.rank, cis.criteria_count,
                               cis.opacity_affected
                        FROM country_index_scores cis
                        JOIN countries c ON c.id = cis.country_id
                        WHERE cis.reference_year = %s AND cis.model_used = %s
                          AND cis.partial_coverage = false
                        ORDER BY cis.rank NULLS LAST
                        """,
                        (reference_year, model_used),
                    )
                else:
                    cur.execute(
                        """
                        SELECT c.name AS country_name, c.iso_code,
                               cis.final_score, cis.legal_score, cis.enforcement_score,
                               cis.pi_category, cis.rank, cis.criteria_count,
                               cis.opacity_affected
                        FROM country_index_scores cis
                        JOIN countries c ON c.id = cis.country_id
                        WHERE cis.reference_year = %s
                          AND cis.partial_coverage = false
                        ORDER BY cis.rank NULLS LAST
                        """,
                        (reference_year,),
                    )
                return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Internal: DB write
    # ------------------------------------------------------------------

    def _write_index_score(self, score: CountryIndexScore) -> None:
        model_used = score.model_used or "default"
        opacity_count = len(score.opacity_affected_criteria)
        conn = psycopg2.connect(self._dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO country_index_scores (
                        country_id, reference_year,
                        legal_score, enforcement_score, final_score,
                        pi_category, criteria_count, missing_criteria, opacity_affected,
                        partial_coverage,
                        model_used, confidence_weighting, missing_strategy
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (country_id, reference_year, model_used)
                    DO UPDATE SET
                        legal_score          = EXCLUDED.legal_score,
                        enforcement_score    = EXCLUDED.enforcement_score,
                        final_score          = EXCLUDED.final_score,
                        pi_category          = EXCLUDED.pi_category,
                        criteria_count       = EXCLUDED.criteria_count,
                        missing_criteria     = EXCLUDED.missing_criteria,
                        opacity_affected     = EXCLUDED.opacity_affected,
                        partial_coverage     = EXCLUDED.partial_coverage,
                        confidence_weighting = EXCLUDED.confidence_weighting,
                        missing_strategy     = EXCLUDED.missing_strategy,
                        created_at           = NOW()
                    """,
                    (
                        score.country_id,
                        score.reference_year,
                        score.legal_score,
                        score.enforcement_score,
                        score.final_score,
                        score.pi_category,
                        score.criteria_count,
                        json.dumps(score.missing_criteria),
                        opacity_count,
                        score.partial_coverage,
                        model_used,
                        score.confidence_weighting,
                        score.missing_strategy,
                    ),
                )
            conn.commit()
        finally:
            conn.close()
