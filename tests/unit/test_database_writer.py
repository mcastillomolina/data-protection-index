"""Unit tests for DatabaseWriter — mocks psycopg2."""

from unittest.mock import MagicMock, patch, call
import pytest

from src.db.writer import DatabaseWriter
from src.db.schema import ALL_STATEMENTS
from src.models.country import Country
from src.models.extraction import SectionExtractionResult
from src.models.retrieval import RetrievedDocument, DocumentContent
from src.models.document import DocumentMetadata


def _make_country() -> Country:
    return Country(
        name="Chile",
        iso_code="CL",
        official_languages=["es"],
        government_domains=[".cl"],
        region="South America",
        metadata={},
    )


def _make_retrieved_doc(status: str = "success") -> RetrievedDocument:
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
        extracted_text="Text content",
        char_count=12,
        extraction_success=True,
    ) if status == "success" else None

    return RetrievedDocument(
        document=doc_meta,
        content=content,
        successful_url="https://example.com/ley.pdf" if status == "success" else None,
        attempted_urls=["https://example.com/ley.pdf"],
        status=status,
    )


def _make_section_result(index: int = 0, all_null: bool = False) -> SectionExtractionResult:
    return SectionExtractionResult(
        section_index=index,
        section_header=f"Art. {index + 1}",
        section_text_original="Section text " * 10,
        split_tier_used="tier1",
        extracted_fields={"key_provisions": ["Right to access"], "enforcement_body": "DPA",
                          "data_subject_rights": None, "penalties": None,
                          "lawful_basis": None, "notes": None} if not all_null else None,
        all_null=all_null,
        processing_time_seconds=0.5,
    )


@pytest.fixture
def mock_conn():
    """A mock psycopg2 connection with a working cursor context manager."""
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    cursor.fetchone.return_value = (42,)

    conn = MagicMock()
    conn.closed = False
    conn.cursor.return_value = cursor
    return conn, cursor


class TestDatabaseWriter:

    @patch("psycopg2.connect")
    def test_ensure_schema_executes_all_statements(self, mock_connect, mock_conn):
        conn, cursor = mock_conn
        mock_connect.return_value = conn

        writer = DatabaseWriter("postgresql://dpi:dpi@localhost:5433/dpi")
        writer.ensure_schema()

        # +1 for CREATE_VECTOR_EXTENSION, which runs before the ALL_STATEMENTS loop
        assert cursor.execute.call_count == len(ALL_STATEMENTS) + 1
        assert conn.commit.call_count == 2  # once after extension, once after all DDL

    @patch("psycopg2.connect")
    def test_upsert_country_returns_id(self, mock_connect, mock_conn):
        conn, cursor = mock_conn
        mock_connect.return_value = conn

        writer = DatabaseWriter("postgresql://dpi:dpi@localhost:5433/dpi")
        country_id = writer.upsert_country(_make_country())

        assert country_id == 42
        cursor.execute.assert_called_once()
        # SQL should contain ON CONFLICT
        sql = cursor.execute.call_args[0][0]
        assert "ON CONFLICT" in sql
        conn.commit.assert_called_once()

    @patch("psycopg2.connect")
    def test_upsert_document_returns_id(self, mock_connect, mock_conn):
        conn, cursor = mock_conn
        mock_connect.return_value = conn

        writer = DatabaseWriter("postgresql://dpi:dpi@localhost:5433/dpi")
        doc_id = writer.upsert_document(1, _make_retrieved_doc(), "es")

        assert doc_id == 42
        cursor.execute.assert_called_once()
        sql = cursor.execute.call_args[0][0]
        assert "ON CONFLICT" in sql

    @patch("psycopg2.connect")
    def test_upsert_section_extraction_returns_id(self, mock_connect, mock_conn):
        conn, cursor = mock_conn
        mock_connect.return_value = conn

        writer = DatabaseWriter("postgresql://dpi:dpi@localhost:5433/dpi")
        row_id = writer.upsert_section_extraction(
            document_id=1,
            result=_make_section_result(),
            llm_provider="groq",
            llm_model="llama-3.3-70b-versatile",
        )

        assert row_id == 42
        sql = cursor.execute.call_args[0][0]
        assert "section_extractions" in sql
        assert "ON CONFLICT" in sql

    @patch("psycopg2.connect")
    def test_upsert_document_extraction_returns_id(self, mock_connect, mock_conn):
        conn, cursor = mock_conn
        mock_connect.return_value = conn

        writer = DatabaseWriter("postgresql://dpi:dpi@localhost:5433/dpi")
        row_id = writer.upsert_document_extraction(
            document_id=1,
            aggregated={"key_provisions": ["Right to access"], "enforcement_body": "DPA",
                        "data_subject_rights": [], "penalties": [], "lawful_basis": [], "notes": None},
            metadata={
                "total_sections": 5,
                "sections_with_signal": 3,
                "split_tier_used": "tier1",
                "detected_language": "es",
                "status": "success",
            },
        )

        assert row_id == 42
        sql = cursor.execute.call_args[0][0]
        assert "document_extractions" in sql
        assert "ON CONFLICT" in sql

    @patch("psycopg2.connect")
    def test_close_closes_connection(self, mock_connect, mock_conn):
        conn, _ = mock_conn
        mock_connect.return_value = conn

        writer = DatabaseWriter("postgresql://dpi:dpi@localhost:5433/dpi")
        writer._conn = conn  # inject directly
        writer.close()

        conn.close.assert_called_once()

    @patch("psycopg2.connect")
    def test_upsert_section_with_null_fields(self, mock_connect, mock_conn):
        conn, cursor = mock_conn
        mock_connect.return_value = conn

        writer = DatabaseWriter("postgresql://dpi:dpi@localhost:5433/dpi")
        result = _make_section_result(all_null=True)
        row_id = writer.upsert_section_extraction(
            document_id=1, result=result, llm_provider="groq", llm_model="llama"
        )
        assert row_id == 42
        # extracted_fields param should be None
        params = cursor.execute.call_args[0][1]
        assert params[5] is None  # extracted_fields is 6th param (0-based index 5)
