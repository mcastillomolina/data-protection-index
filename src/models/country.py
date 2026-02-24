"""Country data model."""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class Country(BaseModel):
    """Represents a country for analysis."""

    name: str = Field(..., description="Country name")
    iso_code: str = Field(..., min_length=2, max_length=2, description="ISO 3166-1 alpha-2 code")
    official_languages: List[str] = Field(..., description="List of official languages")
    government_domains: List[str] = Field(
        ..., description="Common government domain extensions"
    )
    region: str = Field(..., description="Geographic region")
    known_documents: Dict[str, str] = Field(
        default_factory=dict, description="Known document names by type"
    )
    search_hints: List[str] = Field(
        default_factory=list, description="Helpful domains/sites for search"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "name": "Chile",
                "iso_code": "CL",
                "official_languages": ["es"],
                "government_domains": [".gob.cl", ".cl"],
                "region": "Latin America",
                "known_documents": {
                    "constitution": "Constitución Política de Chile",
                    "data_protection_law": "Ley 19.628",
                },
                "search_hints": ["bcn.cl", "consejotransparencia.cl"],
                "metadata": {},
            }
        }
