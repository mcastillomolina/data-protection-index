"""Unit tests for DocumentRetriever."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.core.document_retriever import DocumentRetriever


class TestDocumentRetrieverInit:
    def test_defaults(self):
        with patch("src.core.document_retriever.httpx.Client"):
            r = DocumentRetriever()
        assert r.timeout == 30
        assert r.max_retries == 3
        assert r.retry_delay == 2.0

    def test_custom_params(self):
        with patch("src.core.document_retriever.httpx.Client"):
            r = DocumentRetriever(timeout=10, max_retries=1, retry_delay=0.5)
        assert r.timeout == 10
        assert r.max_retries == 1


class TestDocumentRetrieverRetrieve:
    def _make_response(self, content: bytes, content_type: str, status_code: int = 200):
        resp = MagicMock(spec=httpx.Response)
        resp.content = content
        resp.headers = httpx.Headers({"content-type": content_type})
        resp.status_code = status_code
        resp.raise_for_status = MagicMock()
        return resp

    def test_successful_pdf_download(self):
        pdf_bytes = b"%PDF-1.4 fake pdf content"
        with patch("src.core.document_retriever.httpx.Client") as mock_client_cls:
            mock_session = MagicMock()
            mock_client_cls.return_value = mock_session
            mock_session.get.return_value = self._make_response(pdf_bytes, "application/pdf")

            retriever = DocumentRetriever(max_retries=1, retry_delay=0)
            result = retriever.retrieve("https://example.com/doc.pdf")

        assert result is not None
        raw, content_type = result
        assert raw == pdf_bytes
        assert content_type == "pdf"

    def test_successful_html_download(self):
        html_bytes = b"<html><body>Hello</body></html>"
        with patch("src.core.document_retriever.httpx.Client") as mock_client_cls:
            mock_session = MagicMock()
            mock_client_cls.return_value = mock_session
            mock_session.get.return_value = self._make_response(html_bytes, "text/html; charset=utf-8")

            retriever = DocumentRetriever(max_retries=1, retry_delay=0)
            result = retriever.retrieve("https://example.com/page.html")

        assert result is not None
        _, content_type = result
        assert content_type == "html"

    def test_returns_none_after_max_retries(self):
        with patch("src.core.document_retriever.httpx.Client") as mock_client_cls:
            mock_session = MagicMock()
            mock_client_cls.return_value = mock_session
            mock_session.get.side_effect = httpx.ConnectError("connection refused")

            with patch("src.core.document_retriever.time.sleep"):
                retriever = DocumentRetriever(max_retries=2, retry_delay=0.1)
                result = retriever.retrieve("https://example.com/fail")

        assert result is None
        assert mock_session.get.call_count == 2

    def test_no_retry_on_404(self):
        resp = self._make_response(b"not found", "text/html", status_code=404)
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=resp
        )
        with patch("src.core.document_retriever.httpx.Client") as mock_client_cls:
            mock_session = MagicMock()
            mock_client_cls.return_value = mock_session
            mock_session.get.return_value = resp

            retriever = DocumentRetriever(max_retries=3, retry_delay=0)
            result = retriever.retrieve("https://example.com/missing")

        assert result is None
        # Should not retry on 4xx
        assert mock_session.get.call_count == 1


class TestContentTypeDetection:
    def _retriever(self):
        with patch("src.core.document_retriever.httpx.Client"):
            return DocumentRetriever()

    def test_pdf_from_content_type_header(self):
        r = self._retriever()
        ct = r._detect_content_type("https://x.com/f", httpx.Headers({"content-type": "application/pdf"}), b"")
        assert ct == "pdf"

    def test_html_from_content_type_header(self):
        r = self._retriever()
        ct = r._detect_content_type("https://x.com/f", httpx.Headers({"content-type": "text/html"}), b"")
        assert ct == "html"

    def test_pdf_from_url_extension(self):
        r = self._retriever()
        ct = r._detect_content_type("https://x.com/doc.pdf", httpx.Headers({}), b"")
        assert ct == "pdf"

    def test_pdf_from_magic_bytes(self):
        r = self._retriever()
        ct = r._detect_content_type("https://x.com/file", httpx.Headers({}), b"%PDF-1.4")
        assert ct == "pdf"

    def test_default_to_html(self):
        r = self._retriever()
        ct = r._detect_content_type("https://x.com/file", httpx.Headers({}), b"<html>")
        assert ct == "html"
