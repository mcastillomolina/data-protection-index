"""Unit tests for CriterionExtractor — mocks LLMClient and yaml loading."""

from unittest.mock import MagicMock, patch

import src.core.criterion_extractor as _module
from src.core.criterion_extractor import CriterionExtractor
from src.core.section_splitter import DocumentSection
from src.models.retrieval import RetrievedDocument, DocumentContent
from src.models.document import DocumentMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_section(index: int = 0, text: str = "Article text.", tier: str = "tier1") -> DocumentSection:
    return DocumentSection(index=index, header=f"Art. {index + 1}", text=text, tier_used=tier)


def _make_retrieved_doc(document_type: str, criteria_ids: list[int]) -> RetrievedDocument:
    doc_meta = DocumentMetadata(
        document_type=document_type,
        official_name="Test Document",
        description="Test",
        priority_score=8,
        expected_language="en",
        criteria_ids=criteria_ids,
    )
    content = DocumentContent(
        url="https://example.com/doc.pdf",
        content_type="pdf",
        extracted_text="Article text " * 100,
        char_count=1300,
        extraction_success=True,
    )
    return RetrievedDocument(
        document=doc_meta,
        content=content,
        successful_url="https://example.com/doc.pdf",
        attempted_urls=["https://example.com/doc.pdf"],
        status="success",
    )


# Controlled dimension map injected via patch so tests never touch the filesystem
_FAKE_DIMENSIONS = {
    "constitution":        "legal",
    "data_protection_law": "legal",
    "enforcement_report":  "enforcement",
    "dpa_annual_report":   "enforcement",
    "surveillance_law":    "mixed",
    "biometrics_id_law":   "mixed",
    "legislative_history": None,
}

# Long enough to pass the pre-filter and min-chars check
_SIGNAL_TEXT = "personal data protection rights consent enforcement authority. " * 20


# ---------------------------------------------------------------------------
# Test dispatch by dimension
# ---------------------------------------------------------------------------

class TestCriterionExtractorDispatch:

    def setup_method(self):
        # Reset the dimension cache so each test gets a fresh load
        _module._DIMENSION_CACHE = _FAKE_DIMENSIONS.copy()

    # ------------------------------------------------------------------
    # Test 1: legal document → LegalExtractor → constitutional_privacy_right
    # ------------------------------------------------------------------

    def test_legal_document_uses_legal_extractor(self):
        """constitution (dimension=legal) → one LLM call with LEGAL_EXTRACTOR_SYSTEM."""
        from src.prompts.criterion_extraction import LEGAL_EXTRACTOR_SYSTEM

        llm_result = {
            "constitutional_privacy_right": True,
            "constitutional_articles": ["Article 19"],
            "right_scope": "Home and communications",
            "limitations_clause": None,
            "jurisprudence_mentioned": [],
            "notes": None,
        }
        llm = MagicMock()
        llm.complete_json.return_value = llm_result

        extractor = CriterionExtractor(llm, country_name="Germany")
        section = _make_section(text=_SIGNAL_TEXT)
        retrieved = _make_retrieved_doc("constitution", criteria_ids=[1])

        results, aggregated = extractor.extract_document(retrieved, [section])

        assert len(results) == 1
        result = results[0]
        assert result.all_null is False
        assert result.error_message is None
        assert result.extracted_fields is not None
        assert result.extracted_fields["constitutional_privacy_right"] is True
        assert "Article 19" in result.extracted_fields["constitutional_articles"]

        # Aggregated document-level fields
        assert aggregated["constitutional_privacy_right"] is True
        assert "Article 19" in aggregated["constitutional_articles"]

        # Exactly one LLM call; system prompt must be the legal one
        llm.complete_json.assert_called_once()
        call_kwargs = llm.complete_json.call_args[1]
        assert call_kwargs["system_prompt"] == LEGAL_EXTRACTOR_SYSTEM

    # ------------------------------------------------------------------
    # Test 2: enforcement document → EnforcementExtractor → actual_sanctions
    # ------------------------------------------------------------------

    def test_enforcement_document_uses_enforcement_extractor(self):
        """dpa_annual_report (dimension=enforcement) → one LLM call with ENFORCEMENT_EXTRACTOR_SYSTEM."""
        from src.prompts.criterion_extraction import ENFORCEMENT_EXTRACTOR_SYSTEM

        llm_result = {
            "dpa_exists": True,
            "dpa_name": "Federal Data Protection Authority",
            "dpa_independence": "fully_independent",
            "dpa_staff_count": 42,
            "dpa_budget_mentioned": "€5M",
            "actual_sanctions": [
                {"entity": "Acme Corp", "amount": "€100,000", "date": "2023-05-01", "summary": "Unlawful processing"}
            ],
            "investigations_count": 12,
            "proactive_enforcement": True,
            "enforcement_blocked": None,
            "notes": None,
        }
        llm = MagicMock()
        llm.complete_json.return_value = llm_result

        extractor = CriterionExtractor(llm, country_name="Germany")
        section = _make_section(text=_SIGNAL_TEXT)
        retrieved = _make_retrieved_doc("dpa_annual_report", criteria_ids=[3])

        results, aggregated = extractor.extract_document(retrieved, [section])

        assert len(results) == 1
        result = results[0]
        assert result.all_null is False
        assert result.extracted_fields is not None
        assert "actual_sanctions" in result.extracted_fields
        assert result.extracted_fields["actual_sanctions"][0]["entity"] == "Acme Corp"

        # Aggregated — actual_sanctions is a _DICT_LIST_FIELDS entry (concatenated)
        assert aggregated["dpa_exists"] is True
        assert len(aggregated["actual_sanctions"]) == 1

        # System prompt must be enforcement one
        call_kwargs = llm.complete_json.call_args[1]
        assert call_kwargs["system_prompt"] == ENFORCEMENT_EXTRACTOR_SYSTEM

    # ------------------------------------------------------------------
    # Test 3: mixed document → both extractors → merged fields
    # ------------------------------------------------------------------

    def test_mixed_document_runs_both_extractors_and_merges(self):
        """surveillance_law (dimension=mixed) → two LLM calls; legal AND enforcement fields present."""
        from src.prompts.criterion_extraction import (
            LEGAL_EXTRACTOR_SYSTEM, ENFORCEMENT_EXTRACTOR_SYSTEM,
        )

        legal_result = {
            "interception_legal_standard": "judicial_warrant",
            "authorising_body": "Regional Court",
            "crime_threshold": "4+ years imprisonment",
            "duration_limit": "3 months",
            "oversight_mechanism": "Parliamentary committee",
            "service_provider_obligations": ["Lawful intercept capability"],
            "notes": "Ambiguous emergency clause",
        }
        enforcement_result = {
            "interception_volume": "~5,000 per year",
            "illegal_wiretapping_cases": [],
            "oversight_effectiveness": "Largely effective",
            "service_provider_compliance": "High",
            "notes": "No documented abuses",
        }
        llm = MagicMock()
        llm.complete_json.side_effect = [legal_result, enforcement_result]

        extractor = CriterionExtractor(llm, country_name="Germany")
        section = _make_section(text=_SIGNAL_TEXT)
        # surveillance_law covers criteria 6, 7, 11; criterion 7 has both legal and enforcement schemas
        retrieved = _make_retrieved_doc("surveillance_law", criteria_ids=[6, 7, 11])

        results, aggregated = extractor.extract_document(retrieved, [section])

        assert len(results) == 1
        result = results[0]
        assert result.all_null is False
        fields = result.extracted_fields

        # Legal fields present
        assert fields["interception_legal_standard"] == "judicial_warrant"
        assert fields["authorising_body"] == "Regional Court"

        # Enforcement fields present
        assert fields["interception_volume"] == "~5,000 per year"
        assert fields["oversight_effectiveness"] == "Largely effective"

        # Notes from both extractors merged
        assert "Ambiguous emergency clause" in fields["notes"]
        assert "No documented abuses" in fields["notes"]

        # Two LLM calls — one legal, one enforcement
        assert llm.complete_json.call_count == 2
        system_prompts = [call[1]["system_prompt"] for call in llm.complete_json.call_args_list]
        assert LEGAL_EXTRACTOR_SYSTEM in system_prompts
        assert ENFORCEMENT_EXTRACTOR_SYSTEM in system_prompts

    # ------------------------------------------------------------------
    # Test 4: no-dimension document → generic fallback
    # ------------------------------------------------------------------

    def test_no_dimension_falls_back_to_generic_extraction(self):
        """legislative_history (no dimension) → generic prompt → key_provisions in extracted_fields."""
        from src.prompts.information_extraction import SYSTEM_PROMPT as GENERIC_SYSTEM

        llm_result = {
            "key_provisions": ["Parliamentary debate note"],
            "data_subject_rights": None,
            "enforcement_body": None,
            "statutory_penalties": None,
            "actual_sanctions": None,
            "lawful_basis": None,
            "constitutional_privacy_right": None,
            "constitutional_articles": None,
            "data_retention_period": None,
            "interception_legal_standard": None,
            "biometric_legal_basis": None,
            "treaties_signed": None,
            "dpa_exists": None,
            "dpa_independence": None,
            "dpa_staff_count": None,
            "sanctions_count": None,
            "sanctions_total_amount": None,
            "cctv_regulatory_status": None,
            "border_biometrics_deployed": None,
            "information_opacity_flag": None,
            "notes": None,
            "enforcement_body": None,
        }
        llm = MagicMock()
        llm.complete_json.return_value = llm_result

        extractor = CriterionExtractor(llm, country_name="Germany")
        section = _make_section(text=_SIGNAL_TEXT)
        retrieved = _make_retrieved_doc("legislative_history", criteria_ids=[])

        results, aggregated = extractor.extract_document(retrieved, [section])

        assert len(results) == 1
        result = results[0]
        assert result.extracted_fields is not None
        assert "key_provisions" in result.extracted_fields
        assert "Parliamentary debate note" in result.extracted_fields["key_provisions"]

        # Generic system prompt used — not legal or enforcement
        call_kwargs = llm.complete_json.call_args[1]
        assert call_kwargs["system_prompt"] == GENERIC_SYSTEM

        # Aggregated
        assert "Parliamentary debate note" in aggregated["key_provisions"]
