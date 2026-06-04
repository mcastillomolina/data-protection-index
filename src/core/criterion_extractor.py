"""Phase 3 — criterion-aware LLM extractor replacing InformationExtractor."""

import time
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from src.clients.llm_client import LLMClient
from src.core.section_splitter import DocumentSection
from src.core.section_pre_filter import SectionPreFilter
from src.models.extraction import SectionExtractionResult
from src.models.retrieval import RetrievedDocument
from src.prompts.criterion_extraction import (
    LEGAL_EXTRACTOR_SYSTEM,
    ENFORCEMENT_EXTRACTOR_SYSTEM,
    LEGAL_SCHEMAS,
    ENFORCEMENT_SCHEMAS,
    build_legal_prompt,
    build_enforcement_prompt,
    merge_legal_json_schema,
    merge_enforcement_json_schema,
)
from src.prompts.information_extraction import (
    SYSTEM_PROMPT as _GENERIC_SYSTEM_PROMPT,
    EXTRACTION_SCHEMA as _GENERIC_SCHEMA,
    build_extraction_prompt as _build_generic_prompt,
)

# ---------------------------------------------------------------------------
# Document-dimension cache (loaded once from document_types.yaml)
# ---------------------------------------------------------------------------

_DIMENSION_CACHE: dict[str, str | None] | None = None


def _load_document_dimension(document_type: str) -> str | None:
    """Return dimension ('legal'|'enforcement'|'mixed') or None if not set."""
    global _DIMENSION_CACHE
    if _DIMENSION_CACHE is None:
        yaml_path = Path(__file__).parents[2] / "config" / "document_types.yaml"
        with yaml_path.open() as fh:
            data = yaml.safe_load(fh)
        _DIMENSION_CACHE = {
            entry["id"]: entry.get("dimension")
            for entry in data.get("document_types", [])
        }
    return _DIMENSION_CACHE.get(document_type)


# ---------------------------------------------------------------------------
# Field classification for aggregation
# ---------------------------------------------------------------------------

# list[str] fields — deduplicate by exact string across sections
_STRING_LIST_FIELDS = (
    "key_provisions",
    "data_subject_rights",
    "lawful_basis",
    "lawful_bases",
    "constitutional_articles",
    "jurisprudence_mentioned",
    "sectoral_laws_mentioned",
    "data_sharing_restrictions",
    "exceptions_to_restrictions",
    "service_provider_obligations",
    "agencies_with_access",
    "data_types_accessible",
    "data_types_covered",
    "entities_obligated",
    "exceptions",
    "treaties_mentioned",
    "international_data_sharing",
    "biometrics_collected",
    "privacy_safeguards_stated",
    "enforcement_actions",
    "regulator_rulings",
    "financial_surveillance_programs",
    "location_tracking_programs",
    "sensitive_data_enforcement",
    "biometric_types_at_border",
    "passenger_data_shared_with",
    "data_sharing_agreements",
    "parliament_oversight_actions",
    "executive_overreach_documented",
    "press_freedom_incidents",
    "abuse_documented",
    "data_sharing_programs_active",
    "companies_compelled_to_share",
    "illegal_wiretapping_cases",
    "warrantless_access_documented",
    "legal_challenges",
    "public_spaces_covered",
    # backward-compatible string-list fields from InformationExtractor
    "statutory_penalties",
    "treaties_signed",
)

# list[dict] fields — concatenate across sections (dicts cannot be string-deduped)
_DICT_LIST_FIELDS = (
    "actual_sanctions",       # list[{entity, amount, date, summary}] in enforcement schema
    "treaty_status",
    "employer_surveillance_cases",
    "medical_data_breaches",
    "court_rulings_limiting_surveillance",
    "secondary_use_violations",
)

# scalar fields — first non-null wins
_SCALAR_FIELDS = (
    "constitutional_privacy_right",
    "right_scope",
    "limitations_clause",
    "law_name",
    "enactment_year",
    "scope",
    "secondary_use_prohibition",
    "consent_requirement",
    "id_card_law_exists",
    "id_card_mandatory",
    "central_database",
    "legal_basis_for_biometrics",
    "interception_legal_standard",
    "authorising_body",
    "crime_threshold",
    "duration_limit",
    "oversight_mechanism",
    "warrant_requirement",
    "access_process",
    "emergency_provisions",
    "retention_law_exists",
    "retention_period",
    "oversight",
    "leadership_stance",
    "dpa_exists",
    "dpa_name",
    "dpa_independence",
    "dpa_staff_count",
    "dpa_budget_mentioned",
    "investigations_count",
    "proactive_enforcement",
    "enforcement_blocked",
    "cctv_scale",
    "cctv_regulated",
    "regulatory_body",
    "facial_recognition_deployed",
    "employee_protections_enforced",
    "guidelines_issued",
    "rfid_tracking_deployed",
    "border_biometrics_deployed",
    "privacy_impact_assessed",
    "democratic_backsliding",
    "biometric_system_operational",
    "deployment_scale",
    "opt_out_possible",
    "interception_volume",
    "oversight_effectiveness",
    "service_provider_compliance",
    "access_requests_volume",
    # backward-compatible scalar fields from InformationExtractor
    "enforcement_body",
    "data_retention_period",
    "biometric_legal_basis",
    "sanctions_count",
    "sanctions_total_amount",
    "cctv_regulatory_status",
    "information_opacity_flag",
)

_NOTE_FIELD = "notes"


# ---------------------------------------------------------------------------
# LegalExtractor
# ---------------------------------------------------------------------------

class LegalExtractor:
    """Extracts legal-dimension fields for the relevant PI criteria."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def extract_section(
        self,
        section: DocumentSection,
        document_type: str,
        official_name: str,
        country_name: str,
        criteria_ids: list[int],
    ) -> dict[str, Any]:
        """One complete_json() call for the legal dimension. Returns {} if no relevant criteria."""
        relevant = [c for c in criteria_ids if c in LEGAL_SCHEMAS]
        if not relevant:
            return {}
        prompt = build_legal_prompt(document_type, official_name, country_name, relevant, section.text)
        schema = merge_legal_json_schema(relevant)
        return self._llm.complete_json(
            prompt=prompt,
            system_prompt=LEGAL_EXTRACTOR_SYSTEM,
            schema=schema,
            temperature=0,
            max_tokens=4000,
        )


# ---------------------------------------------------------------------------
# EnforcementExtractor
# ---------------------------------------------------------------------------

class EnforcementExtractor:
    """Extracts enforcement-dimension fields for the relevant PI criteria."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def extract_section(
        self,
        section: DocumentSection,
        document_type: str,
        official_name: str,
        country_name: str,
        criteria_ids: list[int],
    ) -> dict[str, Any]:
        """One complete_json() call for the enforcement dimension. Returns {} if no relevant criteria."""
        relevant = [c for c in criteria_ids if c in ENFORCEMENT_SCHEMAS]
        if not relevant:
            return {}
        prompt = build_enforcement_prompt(document_type, official_name, country_name, relevant, section.text)
        schema = merge_enforcement_json_schema(relevant)
        return self._llm.complete_json(
            prompt=prompt,
            system_prompt=ENFORCEMENT_EXTRACTOR_SYSTEM,
            schema=schema,
            temperature=0,
            max_tokens=4000,
        )


# ---------------------------------------------------------------------------
# CriterionExtractor — evolution of InformationExtractor
# ---------------------------------------------------------------------------

class CriterionExtractor:
    """
    Criterion-aware Phase 3 extractor.

    Dispatches to LegalExtractor, EnforcementExtractor, or both based on the
    document dimension from document_types.yaml. Documents without a dimension
    field fall back to the generic InformationExtractor prompt.

    Public interface is identical to InformationExtractor so main.py requires
    only a two-line change.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        min_section_chars: int = 100,
        country_name: str = "",
    ) -> None:
        self._llm = llm_client
        self._min_section_chars = min_section_chars
        self._country_name = country_name
        self._pre_filter = SectionPreFilter()
        self._legal = LegalExtractor(llm_client)
        self._enforcement = EnforcementExtractor(llm_client)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_document(
        self,
        retrieved_doc: RetrievedDocument,
        sections: list[DocumentSection],
    ) -> tuple[list[SectionExtractionResult], dict[str, Any]]:
        """
        Extract information from all sections of a document.

        Returns (section_results, aggregated_fields).
        Signature is identical to InformationExtractor.extract_document().
        """
        doc = retrieved_doc.document
        dimension = _load_document_dimension(doc.document_type)
        criteria_ids = doc.criteria_ids or []

        results: list[SectionExtractionResult] = []

        for section in sections:
            if len(section.text) < self._min_section_chars:
                logger.debug(
                    f"Skipping section {section.index} — "
                    f"too short ({len(section.text)} chars)"
                )
                continue

            if not self._pre_filter.passes(section.text):
                logger.debug(f"Section {section.index}: pre-filter blocked (no signal terms)")
                results.append(SectionExtractionResult(
                    section_index=section.index,
                    section_header=section.header,
                    section_text_original=section.text,
                    split_tier_used=section.tier_used,
                    extracted_fields=None,
                    all_null=True,
                    processing_time_seconds=0.0,
                    error_message="pre-filter:no-signal",
                ))
                continue

            result = self._extract_section(section, doc, dimension, criteria_ids)
            results.append(result)

        aggregated = self._aggregate(results)
        return results, aggregated

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_section(
        self,
        section: DocumentSection,
        doc: Any,
        dimension: str | None,
        criteria_ids: list[int],
    ) -> SectionExtractionResult:
        start = time.time()
        try:
            raw = self._dispatch(section, doc, dimension, criteria_ids)
            elapsed = time.time() - start
            all_null = _is_all_null(raw)
            if all_null:
                logger.debug(f"Section {section.index}: all fields null")
            return SectionExtractionResult(
                section_index=section.index,
                section_header=section.header,
                section_text_original=section.text,
                split_tier_used=section.tier_used,
                extracted_fields=raw if raw else None,
                all_null=all_null,
                processing_time_seconds=elapsed,
            )
        except Exception as exc:
            elapsed = time.time() - start
            logger.warning(f"LLM extraction failed for section {section.index}: {exc}")
            return SectionExtractionResult(
                section_index=section.index,
                section_header=section.header,
                section_text_original=section.text,
                split_tier_used=section.tier_used,
                extracted_fields=None,
                all_null=True,
                processing_time_seconds=elapsed,
                error_message=str(exc),
            )

    def _dispatch(
        self,
        section: DocumentSection,
        doc: Any,
        dimension: str | None,
        criteria_ids: list[int],
    ) -> dict[str, Any]:
        """Route to the appropriate extractor(s) based on document dimension."""
        doc_type = doc.document_type
        official_name = doc.official_name
        country = self._country_name

        if dimension is None:
            # No dimension → generic fallback (uses original InformationExtractor prompt)
            return self._llm.complete_json(
                prompt=_build_generic_prompt(section.text),
                system_prompt=_GENERIC_SYSTEM_PROMPT,
                schema=_GENERIC_SCHEMA,
                temperature=0,
                max_tokens=4000,
            )

        if dimension == "legal":
            return self._legal.extract_section(
                section, doc_type, official_name, country, criteria_ids
            )

        if dimension == "enforcement":
            return self._enforcement.extract_section(
                section, doc_type, official_name, country, criteria_ids
            )

        if dimension == "mixed":
            legal_raw = self._legal.extract_section(
                section, doc_type, official_name, country, criteria_ids
            )
            enf_raw = self._enforcement.extract_section(
                section, doc_type, official_name, country, criteria_ids
            )
            # Merge; enforcement fields don't overlap with legal fields
            merged: dict[str, Any] = {}
            merged.update(legal_raw)
            merged.update(enf_raw)
            # Prefer non-null notes; concatenate if both present
            legal_note = legal_raw.get(_NOTE_FIELD)
            enf_note = enf_raw.get(_NOTE_FIELD)
            if legal_note and enf_note:
                merged[_NOTE_FIELD] = f"{legal_note} | {enf_note}"
            else:
                merged[_NOTE_FIELD] = legal_note or enf_note
            return merged

        # Unknown dimension value — fall back to generic
        logger.warning(f"Unknown dimension '{dimension}' for {doc_type}; using generic extraction")
        return self._llm.complete_json(
            prompt=_build_generic_prompt(section.text),
            system_prompt=_GENERIC_SYSTEM_PROMPT,
            schema=_GENERIC_SCHEMA,
            temperature=0,
            max_tokens=4000,
        )

    @staticmethod
    def _aggregate(results: list[SectionExtractionResult]) -> dict[str, Any]:
        """
        Merge section-level extractions into a single document-level dict.

        String-list fields: extend and deduplicate by exact string.
        Dict-list fields: extend without deduplication.
        Scalar fields: first non-null value wins.
        Notes: concatenate non-null values with section-index prefix.
        """
        agg: dict[str, Any] = {}
        for f in _STRING_LIST_FIELDS:
            agg[f] = []
        for f in _DICT_LIST_FIELDS:
            agg[f] = []
        for f in _SCALAR_FIELDS:
            agg[f] = None
        agg[_NOTE_FIELD] = None

        seen_strings: dict[str, set] = {f: set() for f in _STRING_LIST_FIELDS}
        note_parts: list[str] = []

        for result in results:
            fields = result.extracted_fields
            if not fields:
                continue

            for f in _STRING_LIST_FIELDS:
                items = fields.get(f) or []
                if not isinstance(items, list):
                    continue
                for item in items:
                    if isinstance(item, str) and item not in seen_strings[f]:
                        agg[f].append(item)
                        seen_strings[f].add(item)

            for f in _DICT_LIST_FIELDS:
                items = fields.get(f) or []
                if isinstance(items, list):
                    agg[f].extend(items)

            for f in _SCALAR_FIELDS:
                if agg[f] is None:
                    val = fields.get(f)
                    if val is not None:
                        agg[f] = val

            note = fields.get(_NOTE_FIELD)
            if note:
                note_parts.append(f"[§{result.section_index}] {note}")

        if note_parts:
            agg[_NOTE_FIELD] = " | ".join(note_parts)

        return agg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_all_null(fields: dict[str, Any]) -> bool:
    return all(v in (None, [], "") for v in fields.values())
