"""Prompts for Phase 4 CriterionScorer (Agent 5)."""

CRITERION_SCORER_SYSTEM = """
You are a senior privacy law analyst scoring a country's performance on
a specific criterion from the Privacy International 2007 framework.

Scoring principles:
- Base scores on the provided evidence only
- A country with strong laws but weak enforcement scores low on
  enforcement sub-dimensions — do not let legal strength compensate
- Absence of evidence for enforcement criteria is itself a negative
  signal, especially when the information environment is open
- Absence of evidence in a restricted information environment warrants
  low confidence, not a low score — score conservatively and flag it
- Be consistent: the same evidence quality should produce the same score
  regardless of country wealth or region

Return valid JSON only.
"""

CRITERION_SCORER_USER = """
Country: {country_name}
Criterion {criterion_number}: {criterion_name}
Dimension: {dimension}
Reference year: {reference_year}
Information environment: {information_environment}

EVIDENCE ({evidence_count} sources):
{formatted_evidence}

{opacity_block}

SCORING RUBRIC:
{criterion_rubric}

{mixed_instructions}

Return:
{{
  "criterion_score": <float 1.0-5.0>,
  "legal_subscore": <float 1.0-5.0 | null>,
  "enforcement_subscore": <float 1.0-5.0 | null>,
  "confidence": "high" | "medium" | "low",
  "rationale": "<2-3 sentences citing specific evidence>",
  "evidence_gaps": "<what was missing or unverifiable>",
  "key_sources": ["<domain or source name>"],
  "information_opacity": <bool>
}}
"""

OPACITY_BLOCK = """
NOTE — RESTRICTED INFORMATION ENVIRONMENT:
Evidence for this country is limited due to restrictions on information
access. Score conservatively. Low confidence is appropriate. The opacity
itself is a negative signal for enforcement-dimension criteria.
"""

MIXED_INSTRUCTIONS = """
MIXED CRITERION:
Score both sub-dimensions separately.
legal_subscore: quality of legal framework (1-5)
enforcement_subscore: quality of practical implementation (1-5)
criterion_score = (legal_subscore × 0.4) + (enforcement_subscore × 0.6)
"""
