"""Scoring output models for Phase 4 criterion scoring and index aggregation."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class CriterionScore(BaseModel):
    """LLM-produced score for one criterion, one country, one run."""

    country_code: str
    criterion_number: int = Field(..., ge=1, le=14)
    criterion_name: str
    dimension: str  # 'legal', 'enforcement', 'mixed'

    legal_subscore: Optional[float] = Field(None, ge=1.0, le=5.0)
    enforcement_subscore: Optional[float] = Field(None, ge=1.0, le=5.0)
    criterion_score: float = Field(..., ge=1.0, le=5.0)

    confidence: str  # 'high', 'medium', 'low'
    evidence_count: int
    information_opacity: bool = False

    rationale: str
    evidence_gaps: str
    key_sources: List[str] = Field(default_factory=list)

    model_used: str
    reference_year: int
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "country_code": "DE",
                "criterion_number": 3,
                "criterion_name": "Privacy Enforcement",
                "dimension": "enforcement",
                "legal_subscore": None,
                "enforcement_subscore": 4.2,
                "criterion_score": 4.2,
                "confidence": "high",
                "evidence_count": 12,
                "information_opacity": False,
                "rationale": "BfDI imposed several GDPR fines...",
                "evidence_gaps": "No data on investigation close rates.",
                "key_sources": ["gdprhub.eu", "bfdi.bund.de"],
                "model_used": "llama-3.3-70b-versatile",
                "reference_year": 2024,
            }
        }


class CountryIndexScore(BaseModel):
    """Aggregated dual-framework score for one country."""

    country_id: int
    reference_year: int

    legal_score: Optional[float] = None
    enforcement_score: Optional[float] = None
    final_score: float

    pi_category: str
    rank: Optional[int] = None

    criteria_count: int
    missing_criteria: List[int] = Field(default_factory=list)
    opacity_affected_criteria: List[int] = Field(default_factory=list)
    partial_coverage: bool = False

    model_used: Optional[str] = None
    confidence_weighting: bool = True
    missing_strategy: str = "exclude"

    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "country_id": 1,
                "reference_year": 2024,
                "legal_score": 3.8,
                "enforcement_score": 3.2,
                "final_score": 3.44,
                "pi_category": "Adequate safeguards against abuse",
                "rank": 12,
                "criteria_count": 14,
                "missing_criteria": [],
                "opacity_affected_criteria": [],
                "model_used": "llama-3.3-70b-versatile",
            }
        }
