"""Phase 2 retrieval data models."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.models.country import Country
from src.models.document import DocumentMetadata


class DocumentContent(BaseModel):
    """Extracted text content from a single URL."""

    url: str = Field(..., description="URL the content was downloaded from")
    content_type: str = Field(..., description="Detected content type: pdf, html, or unknown")
    extracted_text: str = Field(..., description="Clean extracted text")
    char_count: int = Field(..., description="Character count of extracted text")
    extraction_success: bool = Field(..., description="Whether extraction produced usable text")
    error_message: Optional[str] = Field(None, description="Error detail if extraction failed")


class RetrievedDocument(BaseModel):
    """A legal document with its retrieved text content."""

    document: DocumentMetadata = Field(..., description="Document metadata from Phase 1")
    content: Optional[DocumentContent] = Field(None, description="Extracted content, or None if all URLs failed")
    successful_url: Optional[str] = Field(None, description="The URL that yielded usable text")
    attempted_urls: List[str] = Field(..., description="All URLs attempted in order")
    status: str = Field(..., description="success | failed | no_results")


class RetrievalOutput(BaseModel):
    """Complete output from Phase 2 document retrieval."""

    country: Country = Field(..., description="Country being processed")
    documents: List[RetrievedDocument] = Field(..., description="Retrieved documents")
    timestamp: datetime = Field(default_factory=datetime.now, description="When retrieval ran")
    total_documents: int = Field(..., description="Total documents attempted")
    successful_retrievals: int = Field(..., description="Documents with successfully extracted text")
    failed_retrievals: int = Field(..., description="Documents where all URLs failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Phase/run metadata")
