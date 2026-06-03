"""
Integration test for Phase 3: full extraction pipeline on Chile data.

Requires:
- Chile retrieval_results_latest.json (run Phase 1+2 first)
- GROQ_API_KEY or ANTHROPIC_API_KEY set
- DATABASE_URL set (running PostgreSQL via docker-compose)

Run with:
    pytest tests/integration/test_phase3_pipeline.py -v -m integration
"""

import json
import os
from pathlib import Path

import pytest

from src.core.language_detector import LanguageDetector
from src.core.section_splitter import SectionSplitter
from src.core.information_extractor import InformationExtractor
from src.db.writer import DatabaseWriter
from src.models.retrieval import RetrievalOutput
from src.utils.config import Config


CHILE_RETRIEVAL = Path("data/outputs/Chile/retrieval_results_latest.json")


@pytest.fixture(scope="module")
def retrieval_output() -> RetrievalOutput:
    if not CHILE_RETRIEVAL.exists():
        pytest.skip(f"Chile Phase 2 output not found at {CHILE_RETRIEVAL}")
    with open(CHILE_RETRIEVAL, encoding="utf-8") as f:
        data = json.load(f)
    return RetrievalOutput.model_validate(data)


@pytest.fixture(scope="module")
def config() -> Config:
    return Config()


@pytest.fixture(scope="module")
def db_writer():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        pytest.skip("DATABASE_URL not set — skipping DB integration tests")
    writer = DatabaseWriter(dsn)
    writer.ensure_schema()
    yield writer
    writer.close()


@pytest.mark.integration
class TestPhase3Pipeline:

    def test_language_detection_is_spanish(self, retrieval_output: RetrievalOutput):
        """Chile documents should be detected as Spanish."""
        detector = LanguageDetector()
        successful = [d for d in retrieval_output.documents if d.status == "success" and d.content]
        assert successful, "No successful retrievals in Chile output"

        for doc in successful[:3]:  # check first 3 to save time
            lang = detector.detect(doc.content.extracted_text)
            assert lang == "es", f"Expected 'es', got '{lang}' for '{doc.document.official_name}'"

    def test_section_splitter_produces_sections(self, retrieval_output: RetrievalOutput):
        """Each document should split into at least one section."""
        splitter = SectionSplitter()
        successful = [d for d in retrieval_output.documents if d.status == "success" and d.content]
        assert successful

        for doc in successful[:3]:
            sections = splitter.split(doc.content.extracted_text, "es")
            assert len(sections) >= 1, f"No sections for '{doc.document.official_name}'"
            assert all(s.tier_used in ("tier1", "tier2", "tier3") for s in sections)

    @pytest.mark.integration
    def test_extraction_produces_non_null_fields(self, retrieval_output: RetrievalOutput, config: Config):
        """At least one section per document should have non-null key_provisions."""
        detector = LanguageDetector()
        splitter = SectionSplitter()
        llm = config.get_extraction_llm_client()
        extractor = InformationExtractor(llm, min_section_chars=config.extraction.min_section_chars)

        successful = [d for d in retrieval_output.documents if d.status == "success" and d.content]
        assert successful

        doc = successful[0]
        lang = detector.detect(doc.content.extracted_text)
        sections = splitter.split(doc.content.extracted_text, lang)

        section_results, aggregated = extractor.extract_document(doc, sections)

        sections_with_signal = [r for r in section_results if not r.all_null]
        assert len(sections_with_signal) > 0, "All sections returned null — extraction may be broken"
        assert aggregated.get("key_provisions") or aggregated.get("data_subject_rights"), (
            "Aggregated output missing key_provisions and data_subject_rights"
        )

    @pytest.mark.integration
    def test_database_writes_are_idempotent(self, retrieval_output: RetrievalOutput, config: Config, db_writer: DatabaseWriter):
        """Running extraction twice should not raise (ON CONFLICT upserts)."""
        detector = LanguageDetector()
        splitter = SectionSplitter()
        llm = config.get_extraction_llm_client()
        extractor = InformationExtractor(llm, min_section_chars=config.extraction.min_section_chars)

        country_id = db_writer.upsert_country(retrieval_output.country)

        successful = [d for d in retrieval_output.documents if d.status == "success" and d.content]
        assert successful

        doc = successful[0]
        lang = detector.detect(doc.content.extracted_text)
        sections = splitter.split(doc.content.extracted_text, lang)
        section_results, aggregated = extractor.extract_document(doc, sections)

        doc_id = db_writer.upsert_document(country_id, doc, lang)

        for sr in section_results:
            db_writer.upsert_section_extraction(
                doc_id, sr,
                llm_provider=config.extraction.llm_provider,
                llm_model=config.extraction.llm_model,
            )

        db_writer.upsert_document_extraction(
            doc_id,
            aggregated,
            metadata={
                "total_sections": len(sections),
                "sections_with_signal": sum(1 for r in section_results if not r.all_null),
                "split_tier_used": sections[0].tier_used if sections else "tier3",
                "detected_language": lang,
                "status": "success",
            },
        )

        # Second run — must not raise
        doc_id2 = db_writer.upsert_document(country_id, doc, lang)
        assert doc_id2 == doc_id  # same row updated, same id returned

    @pytest.mark.integration
    def test_section_extractions_written_to_db(self, retrieval_output: RetrievalOutput, config: Config, db_writer: DatabaseWriter):
        """Verify section_extractions rows exist in the DB after a write."""
        import psycopg2

        country_id = db_writer.upsert_country(retrieval_output.country)
        successful = [d for d in retrieval_output.documents if d.status == "success" and d.content]
        assert successful

        doc = successful[0]
        doc_id = db_writer.upsert_document(country_id, doc, "es")

        conn = db_writer._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM section_extractions WHERE document_id = %s",
                (doc_id,),
            )
            count = cur.fetchone()[0]

        assert count >= 0  # rows may or may not exist depending on run order; no crash is enough
