"""Phase 2: Text extraction from PDF, HTML, and Word document content."""

import io
import re
import subprocess
import tempfile
from typing import Optional

import pdfplumber
from bs4 import BeautifulSoup
from loguru import logger


class TextExtractor:
    """Extracts clean plain text from PDF bytes, HTML bytes, or Word documents."""

    def __init__(self, min_text_length: int = 200):
        self.min_text_length = min_text_length
        logger.info("Initialized TextExtractor")

    def extract(self, data: bytes, content_type: str) -> Optional[str]:
        """
        Extract text from raw bytes.

        Returns clean text, or None if extraction fails or the result is too short.
        content_type must be 'pdf', 'html', 'doc', 'docx', or 'unknown'.
        """
        try:
            if content_type == "pdf":
                text = self._extract_pdf(data)
            elif content_type == "docx":
                text = self._extract_docx(data)
            elif content_type == "doc":
                text = self._extract_doc(data)
            else:
                text = self._extract_html(data)
        except Exception as e:
            logger.warning(f"Text extraction failed ({content_type}): {e}")
            return None

        if not text or len(text) < self.min_text_length:
            logger.debug(
                f"Extracted text too short ({len(text) if text else 0} chars, "
                f"min {self.min_text_length})"
            )
            return None

        return text

    def _extract_pdf(self, data: bytes) -> str:
        pages: list[str] = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)

        raw = "\n".join(pages)
        return self._clean(raw)

    def _extract_docx(self, data: bytes) -> str:
        import docx  # type: ignore[import]

        doc = docx.Document(io.BytesIO(data))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return self._clean("\n".join(paragraphs))

    def _extract_doc(self, data: bytes) -> str:
        """Extract text from legacy .doc (OLE) using macOS textutil, falling back to docx."""
        # Try macOS textutil first (handles OLE .doc natively)
        try:
            with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            result = subprocess.run(
                ["textutil", "-convert", "txt", "-stdout", tmp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return self._clean(result.stdout)
            logger.debug(f"textutil failed (rc={result.returncode}): {result.stderr.strip()}")
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.debug(f"textutil not available: {e}")

        # Fallback: try python-docx (works if the file is actually OOXML)
        try:
            return self._extract_docx(data)
        except Exception as e:
            logger.debug(f"python-docx fallback failed: {e}")

        raise RuntimeError("Could not extract text from .doc file — no suitable parser available")

    def _extract_html(self, data: bytes) -> str:
        soup = BeautifulSoup(data, "lxml")

        # Remove noise tags
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        raw = soup.get_text(separator=" ", strip=True)
        return self._clean(raw)

    @staticmethod
    def _clean(text: str) -> str:
        """Collapse whitespace and remove null bytes."""
        text = text.replace("\x00", "")
        text = re.sub(r"\s+", " ", text)
        return text.strip()
