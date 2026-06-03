"""Phase 3 extraction data models."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.models.country import Country
from src.models.document import DocumentMetadata


class SectionExtractionResult(BaseModel):
    """Extraction result for a single document section."""

    section_index: int = Field(..., description="0-based position within document")
    section_header: Optional[str] = Field(None, description="Matched pattern or first line")
    section_text_original: str = Field(..., description="Source-language section text")
    split_tier_used: str = Field(..., description="tier1 | tier2 | tier3")
    extracted_fields: Optional[Dict[str, Any]] = Field(
        None, description="English extraction JSON; None on LLM failure"
    )
    all_null: bool = Field(False, description="True if every extracted field was null")
    processing_time_seconds: float = Field(0.0)
    error_message: Optional[str] = Field(None)


class DocumentExtractionResult(BaseModel):
    """Aggregated extraction result for a full document."""

    document: DocumentMetadata
    detected_language: str = Field(..., description="ISO 639-1 code or 'unknown'")
    split_tier_used: str = Field(..., description="Tier that won for this document")
    total_sections: int
    sections_with_signal: int = Field(..., description="Sections with at least one non-null field")
    sections: List[SectionExtractionResult]
    aggregated_fields: Dict[str, Any] = Field(
        default_factory=dict, description="Merged English extraction across all sections"
    )
    enforcement_authority: Optional[str] = Field(
        None, description="Promoted scalar for fast SQL filtering"
    )
    status: str = Field("pending", description="success | partial | failed")
    error_message: Optional[str] = Field(None)
    processing_time_seconds: float = Field(0.0)
    llm_provider: str = Field("")
    llm_model: str = Field("")


class ExtractionOutput(BaseModel):
    """Complete Phase 3 output for a country."""

    country: Country
    documents: List[DocumentExtractionResult]
    timestamp: datetime = Field(default_factory=datetime.now)
    total_documents: int
    successful_extractions: int
    failed_extractions: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
