"""Phase 3 — LLM-based information extractor, one call per section."""

import time
from typing import Any, Dict, Optional

from loguru import logger

from src.clients.llm_client import LLMClient
from src.core.section_splitter import DocumentSection
from src.core.section_pre_filter import SectionPreFilter
from src.models.extraction import SectionExtractionResult, DocumentExtractionResult
from src.models.document import DocumentMetadata
from src.models.retrieval import RetrievedDocument
from src.prompts.information_extraction import (
    SYSTEM_PROMPT,
    EXTRACTION_SCHEMA,
    build_extraction_prompt,
)

_LIST_FIELDS = ("key_provisions", "data_subject_rights", "penalties", "lawful_basis")
_SCALAR_FIELDS = ("enforcement_body",)
_NOTE_FIELD = "notes"


class InformationExtractor:
    """Extracts structured information from document sections using an LLM."""

    def __init__(self, llm_client: LLMClient, min_section_chars: int = 100) -> None:
        self._llm = llm_client
        self._min_section_chars = min_section_chars
        self._pre_filter = SectionPreFilter()

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
        """
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

            result = self._extract_section(section)
            results.append(result)

        aggregated = self._aggregate(results)
        return results, aggregated

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_section(self, section: DocumentSection) -> SectionExtractionResult:
        start = time.time()
        prompt = build_extraction_prompt(section.text)

        try:
            raw: dict = self._llm.complete_json(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                schema=EXTRACTION_SCHEMA,
                temperature=0,
                max_tokens=4000,
            )
            elapsed = time.time() - start
            all_null = all(
                raw.get(field) in (None, [], "") for field in EXTRACTION_SCHEMA["properties"]
            )
            if all_null:
                logger.debug(f"Section {section.index}: all fields null (administrative provision)")

            return SectionExtractionResult(
                section_index=section.index,
                section_header=section.header,
                section_text_original=section.text,
                split_tier_used=section.tier_used,
                extracted_fields=raw,
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

    @staticmethod
    def _aggregate(results: list[SectionExtractionResult]) -> dict[str, Any]:
        """
        Merge section-level extractions into a single document-level dict.

        List fields: extend and deduplicate by exact string match.
        Scalar fields: first non-null value wins.
        Notes: concatenate non-null values.
        """
        agg: dict[str, Any] = {f: [] for f in _LIST_FIELDS}
        agg[_NOTE_FIELD] = None
        for f in _SCALAR_FIELDS:
            agg[f] = None

        seen: dict[str, set] = {f: set() for f in _LIST_FIELDS}
        note_parts: list[str] = []

        for result in results:
            fields = result.extracted_fields
            if not fields:
                continue

            for f in _LIST_FIELDS:
                items = fields.get(f) or []
                for item in items:
                    if isinstance(item, str) and item not in seen[f]:
                        agg[f].append(item)
                        seen[f].add(item)

            for f in _SCALAR_FIELDS:
                if agg[f] is None:
                    val = fields.get(f)
                    if val:
                        agg[f] = val

            note = fields.get(_NOTE_FIELD)
            if note:
                header = f"[§{result.section_index}]"
                note_parts.append(f"{header} {note}")

        if note_parts:
            agg[_NOTE_FIELD] = " | ".join(note_parts)

        return agg
