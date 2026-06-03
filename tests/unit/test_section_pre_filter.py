"""Unit tests for SectionPreFilter, is_structural_noise, and has_signal_terms."""

import pytest

from src.core.section_pre_filter import (
    SectionPreFilter,
    is_structural_noise,
    has_signal_terms,
)
from src.core.section_splitter import DocumentSection


def _make_section(index: int, text: str) -> DocumentSection:
    return DocumentSection(index=index, header=None, text=text, tier_used="tier1")


# ---------------------------------------------------------------------------
# Gate 1: is_structural_noise
# ---------------------------------------------------------------------------


class TestIsStructuralNoise:

    # --- blocked by regex ---

    def test_blocks_archival_banner(self):
        # Bilingual Government of Canada archival notice; "ARCHIVÉE -" triggers the pattern
        assert is_structural_noise(
            "ARCHIVED - Archiving Content ARCHIVÉE - Content archivée"
        )

    def test_blocks_dot_leader(self):
        assert is_structural_noise("Personal Data Rights ........ 45")

    def test_blocks_dots_only(self):
        assert is_structural_noise(".....")

    def test_blocks_table_of_contents_exact(self):
        assert is_structural_noise("TABLE OF CONTENTS")

    def test_blocks_table_of_contents_lowercase(self):
        assert is_structural_noise("table of contents")

    def test_blocks_lone_page_number(self):
        assert is_structural_noise("1")

    def test_blocks_multidigit_page_number(self):
        assert is_structural_noise("42")

    def test_blocks_visual_separator(self):
        assert is_structural_noise("-----")

    def test_blocks_equals_separator(self):
        assert is_structural_noise("=====")

    def test_blocks_all_rights_reserved(self):
        assert is_structural_noise("© 2023 The Organization. All rights reserved.")

    def test_blocks_copyright_year(self):
        assert is_structural_noise("© 2021")

    def test_blocks_page_n_of_m(self):
        assert is_structural_noise("Page 3 of 12")

    # --- blocked by short-line ratio ---

    def test_blocks_short_line_ratio(self):
        # 8 of 10 lines are < 60 chars → 80 % > 65 % threshold → ToC / index page
        short = "Short line under sixty chars here"                          # 32 chars
        long = "This line is deliberately long enough to exceed sixty characters total."  # 71 chars
        lines = [short] * 8 + [long, long]
        assert is_structural_noise("\n".join(lines))

    # --- false-positive protection ---

    def test_not_blocked_three_line_section(self):
        # Below the 4-line minimum — ratio check is never applied
        text = "Short line.\nAlso short.\nYet another short line."
        assert not is_structural_noise(text)

    def test_not_blocked_copyright_in_quote(self):
        # © not followed by a year — copyright pattern must not fire
        text = 'The law states "© data belongs to the subject"'
        assert not is_structural_noise(text)

    def test_not_blocked_normal_article(self):
        assert not is_structural_noise(
            "Article 6. Processing shall be lawful only if the data subject "
            "has given consent for one or more specific purposes."
        )


# ---------------------------------------------------------------------------
# Gate 2: has_signal_terms
# ---------------------------------------------------------------------------


class TestHasSignalTerms:

    # --- passes ---

    def test_passes_personal_data(self):
        assert has_signal_terms("This section covers personal data processing.")

    def test_passes_consent(self):
        assert has_signal_terms("Consent of the data subject is required.")

    def test_passes_supervisory_authority_fine(self):
        assert has_signal_terms("The supervisory authority may impose a fine.")

    def test_passes_data_breach(self):
        assert has_signal_terms("A data breach must be reported within 72 hours.")

    def test_passes_surveillance(self):
        assert has_signal_terms("Communications surveillance powers are regulated.")

    def test_passes_biometric_retention(self):
        assert has_signal_terms(
            "Biometric identifiers must not be retained beyond the period "
            "strictly necessary for the stated purpose."
        )

    def test_passes_anonymisation(self):
        assert has_signal_terms("Data must be anonymised before transfer.")

    def test_passes_anonymization(self):
        assert has_signal_terms("All records must be anonymized prior to sharing.")

    def test_passes_pseudonymisation(self):
        assert has_signal_terms("Pseudonymisation reduces the risk of identification.")

    def test_passes_lawful_basis(self):
        assert has_signal_terms("The controller must identify a lawful basis for processing.")

    def test_passes_case_insensitive(self):
        assert has_signal_terms("PERSONAL DATA PROCESSING RIGHTS")

    # --- blocked ---

    def test_blocks_tax_law_section(self):
        assert not has_signal_terms(
            "Article 5. The taxpayer shall file a return within thirty days "
            "of the assessment. Late filing attracts additional charges from "
            "the Revenue Authority."
        )

    def test_blocks_bare_definitions_boilerplate(self):
        assert not has_signal_terms("Article 1. The following definitions apply.")

    def test_blocks_preamble(self):
        assert not has_signal_terms("Whereas, having regard to the provisions herein.")

    def test_no_false_positive_metadata(self):
        # "metadata" must not match as "data" — we only recognise multi-word phrases
        assert not has_signal_terms("Metadata standards shall apply to all filings.")

    def test_no_false_positive_mandate(self):
        # "mandate" must not match as "data"
        assert not has_signal_terms(
            "The regulatory mandate covers commercial transactions only."
        )

    def test_no_false_positive_penalties_plural(self):
        # "penalties" must not trigger \bpenalty\b
        assert not has_signal_terms(
            "Penalties for late filing shall be assessed by the Revenue Authority."
        )

    # --- multilingual ---

    def test_passes_spanish_datos_personales(self):
        assert has_signal_terms(
            "El responsable del tratamiento debe obtener el consentimiento "
            "del titular de los datos personales."
        )

    def test_passes_french_donnees_personnelles(self):
        assert has_signal_terms(
            "La CNIL a prononcé une sanction contre la société pour traitement "
            "illicite de données personnelles."
        )

    def test_passes_german_datenschutz(self):
        assert has_signal_terms(
            "Nach der Datenschutz-Grundverordnung muss die Einwilligung der "
            "betroffenen Person eingeholt werden."
        )

    def test_passes_chinese_geren_shuju(self):
        assert has_signal_terms(
            "根据数据保护法律，个人数据的处理需要数据主体的同意。"
        )

    def test_blocks_english_tax_law_still_fails(self):
        assert not has_signal_terms(
            "Article 5. The taxpayer shall file a return within thirty days "
            "of the assessment. Late filing attracts additional charges from "
            "the Revenue Authority."
        )


# ---------------------------------------------------------------------------
# Integration: SectionPreFilter.passes() and filter()
# ---------------------------------------------------------------------------


class TestSectionPreFilter:
    def setup_method(self):
        self.pf = SectionPreFilter()

    # --- both gates pass ---

    def test_passes_gdpr_consent_article(self):
        assert self.pf.passes(
            "Article 6. Processing shall be lawful only if the data subject "
            "has given consent for one or more specific purposes."
        )

    def test_passes_dpa_enforcement_fines(self):
        assert self.pf.passes(
            "The supervisory authority found violations and imposed a fine "
            "of ten million euros for unlawful data processing."
        )

    def test_passes_biometric_retention(self):
        assert self.pf.passes(
            "Biometric identifiers must not be retained beyond the period "
            "strictly necessary for the stated purpose."
        )

    def test_passes_right_to_access(self):
        assert self.pf.passes(
            "Every data subject has the right to access their personal data "
            "held by the controller."
        )

    def test_passes_surveillance_warrant(self):
        assert self.pf.passes(
            "Interception of communications requires a judicial warrant issued "
            "by a competent authority."
        )

    # --- blocked by Gate 1 (structural noise) ---

    def test_blocks_archival_notice(self):
        assert not self.pf.passes(
            "ARCHIVED - Archiving Content ARCHIVÉE - Content archivée"
        )

    def test_blocks_dot_leader(self):
        assert not self.pf.passes("Personal Data Rights ........ 45")

    def test_blocks_table_of_contents(self):
        assert not self.pf.passes("TABLE OF CONTENTS")

    def test_blocks_lone_page_number(self):
        assert not self.pf.passes("1")

    def test_blocks_short_line_ratio_toc_page(self):
        short = "Short line under sixty chars here"
        long = "This line is deliberately long enough to exceed sixty characters total."
        assert not self.pf.passes("\n".join([short] * 8 + [long, long]))

    # --- blocked by Gate 2 (no signal terms) ---

    def test_blocks_tax_law_no_privacy_terms(self):
        assert not self.pf.passes(
            "Article 5. The taxpayer shall file a return within thirty days "
            "of the assessment. Late filing attracts additional charges from "
            "the Revenue Authority."
        )

    def test_blocks_bare_definitions_header(self):
        assert not self.pf.passes("Article 1. The following definitions apply.")

    def test_blocks_empty_text(self):
        assert not self.pf.passes("")

    def test_blocks_pure_preamble(self):
        assert not self.pf.passes("Whereas, having regard to the provisions herein.")

    def test_blocks_administrative_boilerplate(self):
        assert not self.pf.passes(
            "This act shall enter into force on the date of its publication. "
            "Repealing provisions contrary to the present law."
        )

    def test_case_insensitive_passes(self):
        assert self.pf.passes("PERSONAL DATA PROCESSING RIGHTS")

    def test_case_insensitive_blocks(self):
        assert not self.pf.passes("PREAMBLE AND TRANSITIONAL CLAUSE")

    # --- filter() partition logic ---

    def test_filter_all_passing(self):
        sections = [
            _make_section(0, "The controller must protect personal data."),
            _make_section(1, "Consent must be obtained before processing."),
        ]
        passing, blocked = self.pf.filter(sections)
        assert len(passing) == 2
        assert len(blocked) == 0

    def test_filter_all_blocked(self):
        sections = [
            _make_section(0, "Preamble and general introduction."),
            _make_section(1, "Transitional and final provisions."),
        ]
        passing, blocked = self.pf.filter(sections)
        assert len(passing) == 0
        assert len(blocked) == 2

    def test_filter_mixed(self):
        sections = [
            _make_section(0, "The controller must protect personal data."),
            _make_section(1, "Preamble only."),
            _make_section(2, "The supervisory authority may impose enforcement sanctions."),
        ]
        passing, blocked = self.pf.filter(sections)
        assert len(passing) == 2
        assert len(blocked) == 1
        assert blocked[0].index == 1

    def test_filter_preserves_order(self):
        sections = [
            _make_section(i, f"consent and personal data rights section {i}")
            for i in range(5)
        ]
        passing, _ = self.pf.filter(sections)
        assert [s.index for s in passing] == list(range(5))

    def test_filter_empty_input(self):
        passing, blocked = self.pf.filter([])
        assert passing == []
        assert blocked == []
