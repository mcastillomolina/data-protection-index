"""Unit tests for CriterionScorer._parse_score — parses mock LLM JSON, no real LLM calls."""

from unittest.mock import MagicMock

import pytest

from src.core.criterion_scorer import CriterionScorer
from src.models.scoring import CriterionScore


def _make_scorer() -> CriterionScorer:
    return CriterionScorer(
        llm_client=MagicMock(),
        embedding_client=MagicMock(),
        dsn="postgresql://fake:fake@localhost/fake",
        model_name="deepseek-chat",
    )


def _full_response(**overrides) -> dict:
    base = {
        "criterion_score": 4.2,
        "legal_subscore": None,
        "enforcement_subscore": 4.2,
        "confidence": "high",
        "rationale": "BfDI imposed several GDPR fines...",
        "evidence_gaps": "No data on investigation close rates.",
        "key_sources": ["gdprhub.eu", "bfdi.bund.de"],
    }
    base.update(overrides)
    return base


class TestParseScorePureCriterion:
    def test_parses_full_response(self):
        scorer = _make_scorer()
        raw = _full_response()

        result = scorer._parse_score(
            raw=raw,
            criterion_number=3,  # pure enforcement
            country_code="DE",
            reference_year=2024,
            evidence_count=12,
        )

        assert isinstance(result, CriterionScore)
        assert result.criterion_number == 3
        assert result.criterion_name == "Privacy Enforcement"
        assert result.dimension == "enforcement"
        assert result.criterion_score == 4.2
        assert result.enforcement_subscore == 4.2
        assert result.legal_subscore is None
        assert result.confidence == "high"
        assert result.evidence_count == 12
        assert result.rationale == "BfDI imposed several GDPR fines..."
        assert result.evidence_gaps == "No data on investigation close rates."
        assert result.key_sources == ["gdprhub.eu", "bfdi.bund.de"]
        assert result.model_used == "deepseek-chat"
        assert result.reference_year == 2024
        assert result.country_code == "DE"

    def test_information_opacity_always_false_regardless_of_raw(self):
        """information_opacity is never trusted from the LLM (self-reported prior, not evidence)."""
        scorer = _make_scorer()
        raw = _full_response(information_opacity=True)

        result = scorer._parse_score(
            raw=raw, criterion_number=3, country_code="DE", reference_year=2024, evidence_count=0
        )

        assert result.information_opacity is False


class TestParseScoreMixedCriterion:
    def test_mixed_with_both_subscores_present(self):
        scorer = _make_scorer()
        raw = _full_response(criterion_score=3.4, legal_subscore=4.0, enforcement_subscore=3.0)

        result = scorer._parse_score(
            raw=raw, criterion_number=4, country_code="DE", reference_year=2024, evidence_count=5
        )

        assert result.dimension == "mixed"
        assert result.legal_subscore == 4.0
        assert result.enforcement_subscore == 3.0
        assert result.criterion_score == 3.4

    def test_mixed_missing_legal_subscore_backfills_from_criterion_score(self):
        scorer = _make_scorer()
        raw = _full_response(criterion_score=3.0, legal_subscore=None, enforcement_subscore=2.5)

        result = scorer._parse_score(
            raw=raw, criterion_number=5, country_code="DE", reference_year=2024, evidence_count=5
        )

        assert result.legal_subscore == 3.0  # backfilled with criterion_score
        assert result.enforcement_subscore == 2.5

    def test_mixed_missing_both_subscores_backfills_both(self):
        scorer = _make_scorer()
        raw = _full_response(criterion_score=3.0, legal_subscore=None, enforcement_subscore=None)

        result = scorer._parse_score(
            raw=raw, criterion_number=7, country_code="DE", reference_year=2024, evidence_count=5
        )

        assert result.legal_subscore == 3.0
        assert result.enforcement_subscore == 3.0


class TestParseScoreMalformedResponses:
    def test_missing_criterion_score_key_returns_none(self):
        scorer = _make_scorer()
        raw = {"confidence": "high"}

        result = scorer._parse_score(
            raw=raw, criterion_number=1, country_code="DE", reference_year=2024, evidence_count=0
        )

        assert result is None

    def test_non_numeric_criterion_score_returns_none(self):
        scorer = _make_scorer()
        raw = _full_response(criterion_score="not-a-number")

        result = scorer._parse_score(
            raw=raw, criterion_number=1, country_code="DE", reference_year=2024, evidence_count=0
        )

        assert result is None

    def test_non_numeric_subscore_returns_none(self):
        scorer = _make_scorer()
        raw = _full_response(enforcement_subscore="bad")

        result = scorer._parse_score(
            raw=raw, criterion_number=3, country_code="DE", reference_year=2024, evidence_count=0
        )

        assert result is None

    def test_empty_dict_returns_none(self):
        scorer = _make_scorer()

        result = scorer._parse_score(
            raw={}, criterion_number=1, country_code="DE", reference_year=2024, evidence_count=0
        )

        assert result is None

    def test_missing_confidence_defaults_to_low(self):
        scorer = _make_scorer()
        raw = _full_response()
        del raw["confidence"]

        result = scorer._parse_score(
            raw=raw, criterion_number=1, country_code="DE", reference_year=2024, evidence_count=0
        )

        assert result.confidence == "low"

    def test_missing_rationale_and_evidence_gaps_default_to_empty_string(self):
        scorer = _make_scorer()
        raw = _full_response()
        del raw["rationale"]
        del raw["evidence_gaps"]

        result = scorer._parse_score(
            raw=raw, criterion_number=1, country_code="DE", reference_year=2024, evidence_count=0
        )

        assert result.rationale == ""
        assert result.evidence_gaps == ""

    def test_missing_key_sources_defaults_to_empty_list(self):
        scorer = _make_scorer()
        raw = _full_response()
        del raw["key_sources"]

        result = scorer._parse_score(
            raw=raw, criterion_number=1, country_code="DE", reference_year=2024, evidence_count=0
        )

        assert result.key_sources == []
