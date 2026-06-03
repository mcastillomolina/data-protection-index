"""Three-tier deterministic section splitter for Phase 3."""

import re
from dataclasses import dataclass
from typing import Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Tier 1 — Universal numeric patterns (language-agnostic)
# ---------------------------------------------------------------------------
_TIER1_PATTERN = re.compile(
    r"(?m)^(?:"
    r"\d+\.\s|"               # 1.  or 2.
    r"\d+\.\d+\s|"            # 1.1  or 2.3
    r"§\s*\d+|"               # § 17  (Germanic laws)
    r"Art(?:icle|\.)\s*\d+|"  # Article 5, Art. 5
    r"第\d+条|"               # Japanese/Chinese Article X
    r"제\d+조|"               # Korean Article X
    r"المادة\s*\d+|"          # Arabic Article X
    r"Статья\s*\d+"           # Russian Article X
    r")"
)

# ---------------------------------------------------------------------------
# Tier 2 — Language-specific article labels
# ---------------------------------------------------------------------------
_TIER2_PATTERNS: dict[str, re.Pattern[str]] = {
    "es": re.compile(r"(?m)^Art[íi]culo\s+\d+"),
    "pt": re.compile(r"(?m)^Artigo\s+\d+"),
    "de": re.compile(r"(?m)^(?:Artikel|§)\s*\d+"),
    "fr": re.compile(r"(?m)^Article\s+\d+"),
    "it": re.compile(r"(?m)^Articolo\s+\d+"),
    "nl": re.compile(r"(?m)^Artikel\s+\d+"),
}

_MIN_SECTIONS = 3
_PARAGRAPH_SIZE_CHARS = 2000


@dataclass
class DocumentSection:
    index: int
    header: Optional[str]
    text: str
    tier_used: str  # "tier1" | "tier2" | "tier3"


class SectionSplitter:
    """Splits document text into sections using a three-tier regex strategy."""

    def split(self, text: str, language: str) -> list[DocumentSection]:
        """
        Split *text* into sections.

        Tries Tier 1 → Tier 2 → Tier 3 (paragraph fallback).
        Returns a list of DocumentSection with tier_used set uniformly.
        """
        # Tier 1
        raw_sections = self._try_tier1(text)
        if len(raw_sections) >= _MIN_SECTIONS:
            logger.info(f"Section split: tier1 ({len(raw_sections)} sections)")
            return self._build_sections(raw_sections, "tier1")

        # Tier 2 (language-specific)
        raw_sections = self._try_tier2(text, language)
        if len(raw_sections) >= _MIN_SECTIONS:
            logger.info(f"Section split: tier2/{language} ({len(raw_sections)} sections)")
            return self._build_sections(raw_sections, "tier2")

        # Tier 3 — paragraph fallback
        raw_sections = self._tier3_fallback(text)
        logger.info(f"Section split: tier3/paragraph ({len(raw_sections)} sections)")
        return self._build_sections(raw_sections, "tier3")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _try_tier1(self, text: str) -> list[str]:
        return self._split_on_pattern(text, _TIER1_PATTERN)

    def _try_tier2(self, text: str, language: str) -> list[str]:
        pattern = _TIER2_PATTERNS.get(language)
        if pattern is None:
            return []
        return self._split_on_pattern(text, pattern)

    def _tier3_fallback(self, text: str) -> list[str]:
        """Fixed-size paragraph chunks (~2000 chars, split at newlines)."""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + _PARAGRAPH_SIZE_CHARS
            if end >= len(text):
                chunk = text[start:]
                if chunk.strip():
                    chunks.append(chunk)
                break
            # Try to break at a newline boundary
            boundary = text.rfind("\n", start, end)
            if boundary <= start:
                boundary = end
            chunk = text[start:boundary]
            if chunk.strip():
                chunks.append(chunk)
            start = boundary
        return chunks

    @staticmethod
    def _split_on_pattern(text: str, pattern: re.Pattern[str]) -> list[str]:
        """Split text on all matches of *pattern*; return non-empty parts."""
        parts = pattern.split(text)
        return [p for p in parts if p and p.strip()]

    @staticmethod
    def _build_sections(raw: list[str], tier: str) -> list[DocumentSection]:
        sections: list[DocumentSection] = []
        for i, text in enumerate(raw):
            # The first line often contains the matched header
            first_line = text.split("\n", 1)[0].strip()
            header = first_line if first_line else None
            sections.append(DocumentSection(index=i, header=header, text=text.strip(), tier_used=tier))
        return sections
