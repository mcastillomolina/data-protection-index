"""Unit tests for InformationExtractor — mocks LLMClient."""

from unittest.mock import MagicMock, patch

import pytest

from src.core.information_extractor import InformationExtractor
from src.core.section_splitter import DocumentSection
from src.models.retrieval import RetrievedDocument, DocumentContent
from src.models.document import DocumentMetadata


def _make_section(index: int = 0, text: str = "Article text.", tier: str = "tier1") -> DocumentSection:
    return DocumentSection(index=index, header=f"Art. {index + 1}", text=text, tier_used=tier)


def _make_retrieved_doc() -> RetrievedDocument:
    doc_meta = DocumentMetadata(
        document_type="data_protection_law",
        official_name="Ley 19.628",
        description="Ley sobre Protección de la Vida Privada",
        priority_score=10,
        expected_language="es",
    )
    content = DocumentContent(
        url="https://example.com/ley.pdf",
        content_type="pdf",
        extracted_text="Article text " * 100,
        char_count=1200,
        extraction_success=True,
    )
    return RetrievedDocument(
        document=doc_meta,
        content=content,
        successful_url="https://example.com/ley.pdf",
        attempted_urls=["https://example.com/ley.pdf"],
        status="success",
    )


class TestInformationExtractor:
    def _make_llm(self, return_value: dict) -> MagicMock:
        llm = MagicMock()
        llm.complete_json.return_value = return_value
        return llm

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    def test_extracts_section_fields(self):
        llm = self._make_llm({
            "key_provisions": ["Right to access", "Right to erasure"],
            "data_subject_rights": ["Access", "Deletion"],
            "enforcement_body": "Data Protection Authority",
            "penalties": ["Up to €20M"],
            "lawful_basis": ["Consent", "Legitimate interest"],
            "notes": None,
        })
        extractor = InformationExtractor(llm)
        section = _make_section(text="Article 1. Data subjects have the right to access their personal data. "
                                     "Controllers must ensure lawful processing under applicable law. " * 2)
        retrieved = _make_retrieved_doc()

        results, aggregated = extractor.extract_document(retrieved, [section])

        assert len(results) == 1
        result = results[0]
        assert result.section_index == 0
        assert result.all_null is False
        assert result.error_message is None
        assert result.extracted_fields is not None
        assert "key_provisions" in result.extracted_fields

    def test_aggregates_list_fields_across_sections(self):
        llm = MagicMock()
        llm.complete_json.side_effect = [
            {
                "key_provisions": ["Right to access"],
                "data_subject_rights": ["Access"],
                "enforcement_body": "DPA",
                "penalties": None,
                "lawful_basis": ["Consent"],
                "notes": None,
            },
            {
                "key_provisions": ["Right to erasure"],
                "data_subject_rights": ["Deletion"],
                "enforcement_body": None,
                "penalties": ["€10M fine"],
                "lawful_basis": ["Consent"],  # duplicate — should appear once
                "notes": "Ambiguous scope",
            },
        ]
        extractor = InformationExtractor(llm)
        sections = [_make_section(0, "A" * 200), _make_section(1, "B" * 200)]
        retrieved = _make_retrieved_doc()

        _, aggregated = extractor.extract_document(retrieved, sections)

        # List fields merged and deduplicated
        assert "Right to access" in aggregated["key_provisions"]
        assert "Right to erasure" in aggregated["key_provisions"]
        assert aggregated["key_provisions"].count("Consent") == 0  # Consent is in lawful_basis
        assert aggregated["lawful_basis"].count("Consent") == 1    # deduplicated

        # Scalar: first non-null wins
        assert aggregated["enforcement_body"] == "DPA"

        # Notes concatenated
        assert aggregated["notes"] is not None
        assert "Ambiguous scope" in aggregated["notes"]

    def test_all_null_detection(self):
        llm = self._make_llm({
            "key_provisions": None,
            "data_subject_rights": None,
            "enforcement_body": None,
            "penalties": None,
            "lawful_basis": None,
            "notes": None,
        })
        extractor = InformationExtractor(llm)
        section = _make_section(text="Transitional provision. " * 20)
        retrieved = _make_retrieved_doc()

        results, _ = extractor.extract_document(retrieved, [section])
        assert results[0].all_null is True

    def test_llm_error_sets_error_message_not_raised(self):
        llm = MagicMock()
        llm.complete_json.side_effect = RuntimeError("LLM unavailable")
        extractor = InformationExtractor(llm)
        section = _make_section(text="Some text. " * 20)
        retrieved = _make_retrieved_doc()

        results, aggregated = extractor.extract_document(retrieved, [section])

        result = results[0]
        assert result.error_message == "LLM unavailable"
        assert result.extracted_fields is None
        assert result.all_null is True
        # Should not raise — pipeline continues

    # ------------------------------------------------------------------
    # min_section_chars filtering
    # ------------------------------------------------------------------

    def test_short_sections_skipped(self):
        llm = MagicMock()
        extractor = InformationExtractor(llm, min_section_chars=500)
        short_section = _make_section(text="Short.")
        retrieved = _make_retrieved_doc()

        results, _ = extractor.extract_document(retrieved, [short_section])

        assert results == []
        llm.complete_json.assert_not_called()

    def test_sections_above_min_chars_are_processed(self):
        llm = self._make_llm({
            "key_provisions": ["Principle"],
            "data_subject_rights": None,
            "enforcement_body": None,
            "penalties": None,
            "lawful_basis": None,
            "notes": None,
        })
        extractor = InformationExtractor(llm, min_section_chars=50)
        section = _make_section(text="X" * 100)
        retrieved = _make_retrieved_doc()

        results, _ = extractor.extract_document(retrieved, [section])
        assert len(results) == 1
        llm.complete_json.assert_called_once()

    # ------------------------------------------------------------------
    # Processing time is recorded
    # ------------------------------------------------------------------

    def test_processing_time_is_positive(self):
        llm = self._make_llm({
            "key_provisions": [], "data_subject_rights": [], "enforcement_body": None,
            "penalties": [], "lawful_basis": [], "notes": None,
        })
        extractor = InformationExtractor(llm)
        section = _make_section(text="Content. " * 30)
        retrieved = _make_retrieved_doc()

        results, _ = extractor.extract_document(retrieved, [section])
        assert results[0].processing_time_seconds >= 0.0
