"""Document-related data models."""

from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata about a document to discover."""

    document_type: str = Field(..., description="Type of document (constitution, law, etc.)")
    official_name: str = Field(..., description="Official name of the document")
    description: str = Field(..., description="Brief description")
    expected_language: str = Field(..., description="Expected language (ISO code)")
    priority_score: int = Field(..., ge=1, le=10, description="Priority for discovery (1-10)")
    alternate_names: List[str] = Field(default_factory=list, description="Alternative names")
    expected_file_types: List[str] = Field(
        default_factory=list, description="Expected file types (pdf, html, etc.)"
    )

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "document_type": "data_protection_law",
                "official_name": "Ley 19.628",
                "description": "Ley sobre Protección de la Vida Privada",
                "expected_language": "es",
                "priority_score": 10,
                "alternate_names": ["Ley 19628", "Ley de Protección de Datos"],
                "expected_file_types": ["pdf", "html"],
            }
        }


class SearchQuery(BaseModel):
    """Represents a search query for document discovery."""

    query_string: str = Field(..., description="The search query text")
    document_id: str = Field(..., description="ID linking to DocumentMetadata")
    site_restrictions: List[str] = Field(
        default_factory=list, description="Site restrictions (e.g., site:.gob.cl)"
    )
    file_type_hint: str = Field(default="", description="File type hint (pdf, html, etc.)")
    priority: int = Field(default=5, ge=1, le=10, description="Query priority")

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "query_string": "Chile Ley 19.628 texto completo",
                "document_id": "ley_19628",
                "site_restrictions": ["site:.gob.cl"],
                "file_type_hint": "pdf",
                "priority": 10,
            }
        }


class SearchResult(BaseModel):
    """Raw search result from search engine."""

    url: str = Field(..., description="URL of the result")
    title: str = Field(..., description="Title of the page/document")
    snippet: str = Field(..., description="Snippet/description from search engine")
    source_domain: str = Field(..., description="Domain name of the source")
    query_used: str = Field(..., description="The query that found this result")
    search_engine: str = Field(default="serpapi", description="Search engine used")
    timestamp: datetime = Field(default_factory=datetime.now, description="When result was found")

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "url": "https://www.bcn.cl/leychile/navegar?idNorma=141599",
                "title": "Ley 19628 - Protección de la Vida Privada",
                "snippet": "Texto completo de la Ley 19.628...",
                "source_domain": "bcn.cl",
                "query_used": "Chile Ley 19.628 texto completo site:.gob.cl",
                "search_engine": "serpapi",
            }
        }


class ScoredResult(BaseModel):
    """Search result with relevance score from LLM."""

    search_result: SearchResult = Field(..., description="The search result")
    relevance_score: float = Field(..., ge=0, le=10, description="Relevance score (0-10)")
    reasoning: str = Field(..., description="Why this score was given")
    is_likely_official: bool = Field(..., description="Whether this appears to be official")
    confidence: str = Field(..., description="Confidence level (high, medium, low)")

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "search_result": {
                    "url": "https://www.bcn.cl/leychile/navegar?idNorma=141599",
                    "title": "Ley 19628",
                    "snippet": "...",
                    "source_domain": "bcn.cl",
                    "query_used": "...",
                },
                "relevance_score": 9.5,
                "reasoning": "Official government source, exact law text",
                "is_likely_official": True,
                "confidence": "high",
            }
        }


class DocumentWithResults(BaseModel):
    """Document metadata along with discovered URLs."""

    document: DocumentMetadata = Field(..., description="The document metadata")
    top_results: List[ScoredResult] = Field(..., description="Top scored results")
    search_queries_used: List[SearchQuery] = Field(..., description="Queries used to find results")

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "document": {"document_type": "data_protection_law", "official_name": "Ley 19.628"},
                "top_results": [],
                "search_queries_used": [],
            }
        }


class DiscoveryOutput(BaseModel):
    """Complete output from document discovery pipeline."""

    country_name: str = Field(..., description="Name of the country")
    country_metadata: Dict[str, Any] = Field(..., description="Country metadata")
    documents: List[DocumentWithResults] = Field(..., description="All discovered documents")
    timestamp: datetime = Field(default_factory=datetime.now, description="When discovery ran")
    total_documents_identified: int = Field(..., description="Number of documents identified")
    total_urls_found: int = Field(..., description="Total URLs discovered")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "country_name": "Chile",
                "country_metadata": {"iso_code": "CL", "region": "Latin America"},
                "documents": [],
                "timestamp": "2024-02-11T10:00:00",
                "total_documents_identified": 8,
                "total_urls_found": 34,
                "metadata": {"phase": "1", "version": "1.0"},
            }
        }
