"""Unit tests for TextExtractor."""

from unittest.mock import MagicMock, patch

import pytest

from src.core.text_extractor import TextExtractor


SAMPLE_HTML = b"""
<html>
<head><title>Test</title></head>
<body>
  <nav>Navigation content</nav>
  <script>var x = 1;</script>
  <style>.foo { color: red; }</style>
  <main>
    <h1>Data Protection Law</h1>
    <p>This law governs the processing of personal data by public and private entities.
       It establishes the rights of data subjects and the obligations of controllers.</p>
  </main>
  <footer>Footer content</footer>
</body>
</html>
"""

NOISY_HTML = b"<html><body><p>   lots   of   whitespace   </p></body></html>"


class TestTextExtractorInit:
    def test_default_min_length(self):
        e = TextExtractor()
        assert e.min_text_length == 200

    def test_custom_min_length(self):
        e = TextExtractor(min_text_length=50)
        assert e.min_text_length == 50


class TestHTMLExtraction:
    def test_extracts_main_text(self):
        e = TextExtractor(min_text_length=10)
        text = e.extract(SAMPLE_HTML, "html")
        assert text is not None
        assert "Data Protection Law" in text
        assert "data subjects" in text

    def test_removes_script_and_style(self):
        e = TextExtractor(min_text_length=10)
        text = e.extract(SAMPLE_HTML, "html")
        assert text is not None
        assert "var x = 1" not in text
        assert ".foo" not in text

    def test_removes_nav_and_footer(self):
        e = TextExtractor(min_text_length=10)
        text = e.extract(SAMPLE_HTML, "html")
        assert text is not None
        assert "Navigation content" not in text
        assert "Footer content" not in text

    def test_collapses_whitespace(self):
        e = TextExtractor(min_text_length=1)
        text = e.extract(NOISY_HTML, "html")
        assert text is not None
        assert "  " not in text

    def test_returns_none_when_too_short(self):
        e = TextExtractor(min_text_length=10_000)
        result = e.extract(SAMPLE_HTML, "html")
        assert result is None

    def test_unknown_content_type_treated_as_html(self):
        e = TextExtractor(min_text_length=10)
        text = e.extract(SAMPLE_HTML, "unknown")
        assert text is not None

    def test_returns_none_on_parse_error(self):
        e = TextExtractor(min_text_length=10)
        with patch("src.core.text_extractor.BeautifulSoup", side_effect=Exception("parse error")):
            result = e.extract(b"<html>", "html")
        assert result is None




class TestPDFExtraction:
    def _make_pdf_mock(self, pages_text: list[str]):
        mock_pdf = MagicMock()
        mock_pages = []
        for text in pages_text:
            page = MagicMock()
            page.extract_text.return_value = text
            mock_pages.append(page)
        mock_pdf.pages = mock_pages
        mock_pdf.__enter__ = lambda s: s
        mock_pdf.__exit__ = MagicMock(return_value=False)
        return mock_pdf

    def test_extracts_text_from_pages(self):
        long_text = "Data protection law text. " * 20
        mock_pdf = self._make_pdf_mock([long_text, "Page two content."])

        with patch("pdfplumber.open", return_value=mock_pdf):
            e = TextExtractor(min_text_length=10)
            text = e.extract(b"%PDF-fake", "pdf")

        assert text is not None
        assert "Data protection law text" in text

    def test_skips_none_pages(self):
        long_text = "Valid page content with enough characters. " * 10
        mock_pdf = self._make_pdf_mock([None, long_text])

        with patch("pdfplumber.open", return_value=mock_pdf):
            e = TextExtractor(min_text_length=10)
            text = e.extract(b"%PDF-fake", "pdf")

        assert text is not None
        assert "Valid page content" in text

    def test_returns_none_on_extraction_error(self):
        with patch("pdfplumber.open", side_effect=Exception("corrupt pdf")):
            e = TextExtractor(min_text_length=10)
            result = e.extract(b"%PDF-bad", "pdf")
        assert result is None

    def test_returns_none_when_too_short(self):
        mock_pdf = self._make_pdf_mock(["short"])

        with patch("pdfplumber.open", return_value=mock_pdf):
            e = TextExtractor(min_text_length=10_000)
            result = e.extract(b"%PDF-fake", "pdf")

        assert result is None


class TestTextCleaning:
    def test_removes_null_bytes(self):
        text = TextExtractor._clean("hello\x00world")
        assert "\x00" not in text
        assert "helloworld" in text

    def test_collapses_multiple_spaces(self):
        text = TextExtractor._clean("a   b\t\tc\n\nd")
        assert "  " not in text
