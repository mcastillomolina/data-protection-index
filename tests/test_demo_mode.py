"""Unit tests for --demo mode pipeline behavior."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.main import DEMO_CRITERIA_NUMBERS, DEMO_CRITERIA_NAMES
from src.core.document_identifier import DocumentIdentifier
from src.models.country import Country
from src.models.document import DocumentMetadata
from src.models.retrieval import RetrievedDocument, DocumentContent, RetrievalOutput
from src.core.section_splitter import DocumentSection


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_country(**kw) -> Country:
    return Country(
        name=kw.get("name", "TestLand"),
        iso_code=kw.get("iso_code", "TL"),
        official_languages=["en"],
        government_domains=["gov.tl"],
        region="Test Region",
        metadata={},
    )


def _make_doc_meta(name: str = "Test Privacy Act", priority: int = 10) -> DocumentMetadata:
    return DocumentMetadata(
        document_type="data_protection_law",
        official_name=name,
        description="Test statute",
        expected_language="en",
        priority_score=priority,
        criteria_ids=[2, 3],
    )


def _make_section(i: int) -> DocumentSection:
    return DocumentSection(
        index=i,
        header=f"S{i}",
        text="personal data protection rights consent enforcement " * 10,
        tier_used="tier1",
    )


def _make_retrieved_doc(text: str = "body text for extraction " * 200) -> RetrievedDocument:
    return RetrievedDocument(
        document=_make_doc_meta(),
        content=DocumentContent(
            url="https://example.com/law.pdf",
            content_type="pdf",
            extracted_text=text,
            char_count=len(text),
            extraction_success=True,
        ),
        successful_url="https://example.com/law.pdf",
        attempted_urls=["https://example.com/law.pdf"],
        status="success",
    )


def _make_config(**overrides) -> MagicMock:
    cfg = MagicMock()
    cfg.pipeline.enable_caching = False
    cfg.pipeline.enable_deduplication = True
    cfg.pipeline.min_relevance_score = 6.0
    cfg.llm.temperature = 0.3
    cfg.llm.max_tokens = 4000
    cfg.llm.model = "test-model"
    cfg.search.max_results_per_query = 10
    cfg.extraction.min_section_chars = 10
    cfg.extraction.llm_provider = "test"
    cfg.extraction.llm_model = "test-model"
    cfg.scoring.cosine_similarity_threshold = 0.7
    cfg.scoring.max_sections_per_criterion = 10
    cfg.output.directory = "/tmp/dpi_test_output"
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Test 1: DocumentIdentifier injects demo constraint into LLM prompt
# ---------------------------------------------------------------------------

def test_demo_p1_prompt_contains_constraint():
    """demo_mode=True appends 'at most 3 documents' to the LLM prompt."""
    mock_llm = MagicMock()
    mock_llm.complete_json.return_value = {
        "documents": [],
        "metadata": {"country": "TestLand", "total_documents": 0, "notes": ""},
    }

    identifier = DocumentIdentifier(llm_client=mock_llm, demo_mode=True)
    identifier.identify_documents(_make_country())

    prompt = mock_llm.complete_json.call_args.kwargs["prompt"]
    assert "at most 3 documents" in prompt


# ---------------------------------------------------------------------------
# Test 2: discover_documents uses queries_per_doc=2 and max_results=2 in demo
# ---------------------------------------------------------------------------

def test_demo_p1_query_and_search_params():
    """discover_documents_for_country with demo_mode=True initialises components with demo limits."""
    with (
        patch("src.main.resolve_country", return_value=_make_country()),
        patch("src.main.DocumentIdentifier") as mock_di_cls,
        patch("src.main.QueryGenerator") as mock_qg_cls,
        patch("src.main.SearchExecutor") as mock_se_cls,
        patch("src.main.RelevanceFilter"),
    ):
        mock_di_cls.return_value.identify_documents.return_value = [_make_doc_meta()]
        mock_qg_cls.return_value.generate_queries_for_multiple.return_value = {
            "Test Privacy Act": []
        }
        mock_se_cls.return_value.execute_searches_by_document.return_value = {
            "Test Privacy Act": []
        }

        from src.main import discover_documents_for_country
        discover_documents_for_country(
            country_name="TestLand",
            config=_make_config(),
            queries_per_document=2,
            demo_mode=True,
        )

        qg_kwargs = mock_qg_cls.call_args.kwargs
        assert qg_kwargs.get("queries_per_document") == 2

        se_kwargs = mock_se_cls.call_args.kwargs
        assert se_kwargs.get("max_results_per_query") == 2


# ---------------------------------------------------------------------------
# Test 3: RelevanceFilter is never called in demo mode
# ---------------------------------------------------------------------------

def test_demo_relevance_filter_skipped():
    """In demo mode, RelevanceFilter.filter_results_batch is never called."""
    with (
        patch("src.main.resolve_country", return_value=_make_country()),
        patch("src.main.DocumentIdentifier") as mock_di_cls,
        patch("src.main.QueryGenerator") as mock_qg_cls,
        patch("src.main.SearchExecutor") as mock_se_cls,
        patch("src.main.RelevanceFilter") as mock_rf_cls,
    ):
        mock_di_cls.return_value.identify_documents.return_value = [_make_doc_meta()]
        mock_qg_cls.return_value.generate_queries_for_multiple.return_value = {
            "Test Privacy Act": []
        }
        mock_se_cls.return_value.execute_searches_by_document.return_value = {
            "Test Privacy Act": []
        }
        mock_rf_cls.return_value.filter_results_batch.side_effect = AssertionError(
            "RelevanceFilter must not be called in demo mode"
        )

        from src.main import discover_documents_for_country
        discover_documents_for_country(
            country_name="TestLand",
            config=_make_config(),
            demo_mode=True,
        )

        mock_rf_cls.return_value.filter_results_batch.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: P3 caps sections at 20 in demo mode
# ---------------------------------------------------------------------------

def test_demo_section_cap():
    """demo_mode=True caps sections at 20 before passing to CriterionExtractor."""
    retrieval_output = RetrievalOutput(
        country=_make_country(),
        documents=[_make_retrieved_doc()],
        total_documents=1,
        successful_retrievals=1,
        failed_retrievals=0,
        metadata={"phase": "2", "version": "1.0", "processing_time_seconds": 0.1},
    )

    fifty_sections = [_make_section(i) for i in range(50)]
    captured: list[list[DocumentSection]] = []

    cfg = _make_config()
    cfg.get_extraction_llm_client.return_value.total_usage.prompt_tokens = 0
    cfg.get_extraction_llm_client.return_value.total_usage.completion_tokens = 0
    cfg.get_extraction_llm_client.return_value.total_usage.total_tokens = 0

    with (
        patch("src.main.LanguageDetector") as mock_ld,
        patch("src.main.SectionSplitter") as mock_ss,
        patch("src.main.CriterionExtractor") as mock_ce,
    ):
        mock_ld.return_value.detect_with_fallback.return_value = "en"
        mock_ss.return_value.split.return_value = fifty_sections

        def capture_extract(doc, sections):
            captured.append(list(sections))
            return [], {}

        mock_ce.return_value.extract_document.side_effect = capture_extract

        from src.main import extract_information_from_retrieval
        extract_information_from_retrieval(
            retrieval_output=retrieval_output,
            config=cfg,
            db_writer=None,
            demo_mode=True,
        )

    assert len(captured) == 1, "extract_document should be called once"
    assert len(captured[0]) == 20, f"Expected 20 sections, got {len(captured[0])}"


# ---------------------------------------------------------------------------
# Test 5: criteria_filter limits which criteria are scored
# ---------------------------------------------------------------------------

def test_demo_criteria_filter():
    """score_all_criteria with criteria_filter only scores those criterion numbers."""
    from src.core.criterion_scorer import CriterionScorer

    with patch("src.core.criterion_scorer.psycopg2") as mock_pg:
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)

        scorer = CriterionScorer(
            llm_client=MagicMock(),
            embedding_client=MagicMock(),
            dsn="postgresql://fake",
            model_name="test-model",
            cosine_threshold=0.7,
            max_sections=10,
        )

        with patch.object(scorer, "_score_one", return_value=None) as mock_score:
            scorer.score_all_criteria(
                country_id=1,
                country_name="TestLand",
                country_code="TL",
                reference_year=2024,
                criteria_filter=DEMO_CRITERIA_NUMBERS,
            )

        called = [c.kwargs["criterion_number"] for c in mock_score.call_args_list]
        assert sorted(called) == sorted(DEMO_CRITERIA_NUMBERS)
        assert len(called) == len(DEMO_CRITERIA_NUMBERS)
        # No criteria outside the filter should be called
        excluded = set(range(1, 15)) - set(DEMO_CRITERIA_NUMBERS)
        for num in excluded:
            assert num not in called, f"Criterion {num} should not have been scored"


# ---------------------------------------------------------------------------
# Test 6: P3 still writes to DB in demo mode
# ---------------------------------------------------------------------------

def test_demo_writes_to_db():
    """extract_information_from_retrieval in demo mode still calls db_writer."""
    retrieval_output = RetrievalOutput(
        country=_make_country(),
        documents=[_make_retrieved_doc()],
        total_documents=1,
        successful_retrievals=1,
        failed_retrievals=0,
        metadata={"phase": "2", "version": "1.0", "processing_time_seconds": 0.1},
    )

    mock_db = MagicMock()
    mock_db.upsert_country.return_value = 42
    mock_db.upsert_document.return_value = 99

    cfg = _make_config()
    cfg.get_extraction_llm_client.return_value.total_usage.prompt_tokens = 0
    cfg.get_extraction_llm_client.return_value.total_usage.completion_tokens = 0
    cfg.get_extraction_llm_client.return_value.total_usage.total_tokens = 0

    with (
        patch("src.main.LanguageDetector") as mock_ld,
        patch("src.main.SectionSplitter") as mock_ss,
        patch("src.main.CriterionExtractor") as mock_ce,
    ):
        mock_ld.return_value.detect_with_fallback.return_value = "en"
        mock_ss.return_value.split.return_value = [_make_section(i) for i in range(5)]
        mock_ce.return_value.extract_document.return_value = ([], {"key_provisions": []})

        from src.main import extract_information_from_retrieval
        extract_information_from_retrieval(
            retrieval_output=retrieval_output,
            config=cfg,
            db_writer=mock_db,
            demo_mode=True,
        )

    mock_db.upsert_country.assert_called_once()
    mock_db.upsert_document.assert_called_once()
