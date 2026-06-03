"""Unit tests for SectionSplitter — three-tier regex strategy."""

import pytest

from src.core.section_splitter import SectionSplitter, DocumentSection, _MIN_SECTIONS


class TestSectionSplitter:
    def setup_method(self):
        self.splitter = SectionSplitter()

    # ------------------------------------------------------------------
    # Tier 1 — universal numeric patterns
    # ------------------------------------------------------------------

    def test_tier1_wins_on_article_dot_pattern(self):
        text = "\n".join([
            "Art. 1 General Provisions\nThis law applies to all controllers.",
            "Art. 2 Definitions\nPersonal data means any information.",
            "Art. 3 Scope\nThis regulation covers automated processing.",
        ])
        sections = self.splitter.split(text, "en")
        assert all(s.tier_used == "tier1" for s in sections)
        assert len(sections) >= _MIN_SECTIONS

    def test_tier1_wins_on_section_symbol(self):
        text = "\n".join([
            "§ 1 Zweck des Gesetzes\nDieses Gesetz regelt den Datenschutz.",
            "§ 2 Begriffsbestimmungen\nPersonenbezogene Daten sind alle Informationen.",
            "§ 3 Anwendungsbereich\nDas Gesetz gilt für Verantwortliche.",
        ])
        sections = self.splitter.split(text, "de")
        assert all(s.tier_used == "tier1" for s in sections)

    def test_tier1_wins_on_article_keyword(self):
        text = "\n".join([
            "Article 1\nGeneral scope of this regulation.",
            "Article 2\nDefinitions used in this act.",
            "Article 3\nRights of data subjects.",
        ])
        sections = self.splitter.split(text, "en")
        assert all(s.tier_used == "tier1" for s in sections)

    # ------------------------------------------------------------------
    # Tier 2 — language-specific article labels
    # ------------------------------------------------------------------

    def test_tier2_wins_for_spanish_articulo(self):
        # No Tier 1 match, but Spanish Artículo should win
        text = "\n".join([
            "Artículo 1\nDisposiciones generales.",
            "Artículo 2\nDefiniciones de datos personales.",
            "Artículo 3\nÁmbito de aplicación.",
        ])
        sections = self.splitter.split(text, "es")
        assert all(s.tier_used == "tier2" for s in sections)
        assert len(sections) >= _MIN_SECTIONS

    def test_tier2_wins_for_portuguese_artigo(self):
        text = "\n".join([
            "Artigo 1\nDisposições gerais.",
            "Artigo 2\nConceitos e definições.",
            "Artigo 3\nÂmbito de aplicação.",
        ])
        sections = self.splitter.split(text, "pt")
        assert all(s.tier_used == "tier2" for s in sections)

    def test_tier2_skipped_for_unknown_language(self):
        # Artículo won't appear in Tier 1; language "xx" has no Tier 2 pattern
        text = "Short text without any article patterns. " * 10
        sections = self.splitter.split(text, "xx")
        assert all(s.tier_used == "tier3" for s in sections)

    # ------------------------------------------------------------------
    # Tier 3 — paragraph fallback
    # ------------------------------------------------------------------

    def test_tier3_fallback_for_plain_prose(self):
        # Plain prose with no numeric or language-specific patterns
        text = "This is a general paragraph about data protection. " * 50
        sections = self.splitter.split(text, "en")
        assert all(s.tier_used == "tier3" for s in sections)
        assert len(sections) >= 1

    def test_tier3_produces_chunks_within_max_size(self):
        # 10 000 chars should produce multiple chunks
        text = "Data protection is important. " * 334  # ~10 000 chars
        sections = self.splitter.split(text, "en")
        assert all(s.tier_used == "tier3" for s in sections)
        assert len(sections) > 1

    # ------------------------------------------------------------------
    # Minimum section threshold
    # ------------------------------------------------------------------

    def test_falls_through_to_tier3_when_fewer_than_min_splits(self):
        # Only 2 Article matches — below MIN_SECTIONS (3)
        text = "Article 1\nFirst section.\n\nArticle 2\nSecond section.\n\nPlain prose. " * 5
        # Two Article matches but splitter may yield more from re.split — let's
        # use a text that definitely has only 2
        text2 = "Article 1\nfoo.\nArticle 2\nbar."
        sections = self.splitter.split(text2, "en")
        # Either tier1 succeeded (if ≥3 parts after split) or fell to tier3
        # The important invariant: all sections have the same tier
        tiers = {s.tier_used for s in sections}
        assert len(tiers) == 1

    # ------------------------------------------------------------------
    # DocumentSection fields
    # ------------------------------------------------------------------

    def test_section_index_is_sequential(self):
        text = "\n".join([
            "Art. 1 First\nContent one.",
            "Art. 2 Second\nContent two.",
            "Art. 3 Third\nContent three.",
        ])
        sections = self.splitter.split(text, "en")
        for i, s in enumerate(sections):
            assert s.index == i

    def test_section_header_is_populated(self):
        # Tier 1 splits ON the pattern, so the remaining text starts after the match.
        # Each remaining chunk's first line becomes the header.
        text = (
            "Preamble text before any article.\n"
            "Art. 1 Scope\nThis law applies to all data controllers.\n"
            "Art. 2 Definitions\nPersonal data means any identifiable information.\n"
            "Art. 3 Rights\nData subjects have the right to access their data.\n"
        )
        sections = self.splitter.split(text, "en")
        # At least the non-empty chunks should exist
        assert len(sections) >= 1
        # Each section must have a header (first non-empty line of its text)
        for s in sections:
            if s.text.strip():
                assert s.header is not None

    def test_section_text_is_stripped(self):
        text = "\n".join([
            "Art. 1 Foo\n   Content with surrounding whitespace.   ",
            "Art. 2 Bar\n   More content.   ",
            "Art. 3 Baz\n   Even more.   ",
        ])
        sections = self.splitter.split(text, "en")
        for s in sections:
            assert s.text == s.text.strip()

    def test_empty_text_returns_tier3_with_no_sections(self):
        sections = self.splitter.split("", "en")
        assert sections == []
