"""Tests for async concurrent section extraction in CriterionExtractor."""

import threading
import time
from unittest.mock import MagicMock, patch

import src.core.criterion_extractor as _module
from src.core.criterion_extractor import CriterionExtractor
from src.core.section_splitter import DocumentSection
from src.models.extraction import SectionExtractionResult
from src.models.retrieval import RetrievedDocument, DocumentContent
from src.models.document import DocumentMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIGNAL_TEXT = "personal data protection rights consent enforcement authority. " * 20

_FAKE_DIMENSIONS = {"constitution": "legal"}


def _make_section(index: int, text: str = _SIGNAL_TEXT) -> DocumentSection:
    return DocumentSection(index=index, header=f"Section {index}", text=text, tier_used="tier1")


def _make_retrieved_doc() -> RetrievedDocument:
    doc_meta = DocumentMetadata(
        document_type="constitution",
        official_name="Test Document",
        description="Test",
        priority_score=8,
        expected_language="en",
        criteria_ids=[1],
    )
    content = DocumentContent(
        url="https://example.com/doc.pdf",
        content_type="pdf",
        extracted_text=_SIGNAL_TEXT,
        char_count=len(_SIGNAL_TEXT),
        extraction_success=True,
    )
    return RetrievedDocument(
        document=doc_meta,
        content=content,
        successful_url="https://example.com/doc.pdf",
        attempted_urls=["https://example.com/doc.pdf"],
        status="success",
    )


def _ok_result(section: DocumentSection) -> SectionExtractionResult:
    return SectionExtractionResult(
        section_index=section.index,
        section_header=section.header,
        section_text_original=section.text,
        split_tier_used=section.tier_used,
        extracted_fields={"key_provisions": [f"provision-{section.index}"]},
        all_null=False,
        processing_time_seconds=0.1,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAsyncExtraction:

    def setup_method(self):
        _module._DIMENSION_CACHE = _FAKE_DIMENSIONS.copy()
        _module.FORCE_SERIAL = False
        _module.MAX_CONCURRENT_EXTRACTIONS = 10

    def teardown_method(self):
        _module.FORCE_SERIAL = False
        _module.MAX_CONCURRENT_EXTRACTIONS = 10

    # ------------------------------------------------------------------

    def test_concurrent_faster_than_serial(self):
        """10 sections each sleeping 0.1s should finish in < 0.6s, not the 1.0s serial would take."""
        sections = [_make_section(i) for i in range(10)]
        retrieved = _make_retrieved_doc()

        def slow_extract(section, doc, dimension, criteria_ids):
            time.sleep(0.1)
            return _ok_result(section)

        extractor = CriterionExtractor(MagicMock(), country_name="Test")
        with patch.object(extractor, "_extract_section", side_effect=slow_extract):
            start = time.monotonic()
            results, _ = extractor.extract_document(retrieved, sections)
            elapsed = time.monotonic() - start

        assert len(results) == 10
        assert elapsed < 0.6, (
            f"Concurrent extraction took {elapsed:.2f}s — expected < 0.6s "
            f"(serial would be ~1.0s)"
        )

    # ------------------------------------------------------------------

    def test_one_section_failure_does_not_abort(self):
        """An exception from one section must not abort the document; 9/10 succeed."""
        sections = [_make_section(i) for i in range(10)]
        retrieved = _make_retrieved_doc()

        def flaky_extract(section, doc, dimension, criteria_ids):
            if section.index == 5:
                raise RuntimeError("simulated LLM timeout")
            return _ok_result(section)

        extractor = CriterionExtractor(MagicMock(), country_name="Test")
        with patch.object(extractor, "_extract_section", side_effect=flaky_extract):
            results, _ = extractor.extract_document(retrieved, sections)

        assert len(results) == 10

        failed = [r for r in results if r.error_message and "simulated LLM timeout" in r.error_message]
        ok = [r for r in results if r.error_message is None]

        assert len(failed) == 1, f"Expected 1 failed section, got {len(failed)}"
        assert failed[0].section_index == 5
        assert len(ok) == 9

    # ------------------------------------------------------------------

    def test_result_order_preserved(self):
        """Results must appear in the original section-index order regardless of completion order."""
        sections = [_make_section(i) for i in range(20)]
        retrieved = _make_retrieved_doc()

        def staggered_extract(section, doc, dimension, criteria_ids):
            # High-index sections finish faster, so gather would reorder if order weren't preserved
            time.sleep(0.02 * (20 - section.index) / 20)
            return _ok_result(section)

        extractor = CriterionExtractor(MagicMock(), country_name="Test")
        with patch.object(extractor, "_extract_section", side_effect=staggered_extract):
            results, _ = extractor.extract_document(retrieved, sections)

        assert [r.section_index for r in results] == list(range(20))

    # ------------------------------------------------------------------

    def test_semaphore_limits_concurrency(self):
        """Peak in-flight extractions must never exceed MAX_CONCURRENT_EXTRACTIONS."""
        _module.MAX_CONCURRENT_EXTRACTIONS = 3

        sections = [_make_section(i) for i in range(20)]
        retrieved = _make_retrieved_doc()

        lock = threading.Lock()
        active = [0]
        peak = [0]

        def counted_extract(section, doc, dimension, criteria_ids):
            with lock:
                active[0] += 1
                if active[0] > peak[0]:
                    peak[0] = active[0]
            time.sleep(0.05)
            with lock:
                active[0] -= 1
            return _ok_result(section)

        extractor = CriterionExtractor(MagicMock(), country_name="Test")
        with patch.object(extractor, "_extract_section", side_effect=counted_extract):
            results, _ = extractor.extract_document(retrieved, sections)

        assert len(results) == 20
        assert peak[0] <= 3, f"Peak concurrency was {peak[0]}, expected ≤ 3"
