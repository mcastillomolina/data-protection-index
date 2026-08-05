"""Prompts for Phase 4 CriterionScorer (Agent 5)."""

CRITERION_SCORER_SYSTEM = """
You are a senior privacy law analyst scoring a country's performance on
a specific criterion from the Privacy International 2007 framework.

Scoring principles:
- Base scores on the provided evidence only
- A country with strong laws but weak enforcement scores low on
  enforcement sub-dimensions — do not let legal strength compensate
- Absence of evidence for enforcement criteria is itself a negative
  signal — do not assume evidence exists but was simply not surfaced
- Be consistent: the same evidence quality should produce the same score
  regardless of country wealth or region

Return valid JSON only.
"""

# NOTE: information_environment / opacity handling was removed from the scorer prompt
# (both the {information_environment}/{opacity_block} placeholders and, as of this
# revision, the "restricted information environment" language in the system prompt and
# the "information_opacity" field in the expected JSON below). Confirmed live: with the
# country-level opacity_block gone, the model was still self-reporting
# information_opacity=true from its own pretrained country-level priors — flat across
# legal/enforcement/mixed dimensions and uncorrelated with evidence_count for the same
# call (e.g. China: 14/14 criteria marked opaque including ones with evidence_count=20,
# the max). That is exactly the a priori geopolitical assumption Decision B removed from
# document_identification.py, leaking back through this channel. information_opacity
# stays in the DB schema as an intentional, unpopulated column for future positive
# evidence — see diagnosis Decision A — but the model is no longer asked to guess it.

CRITERION_SCORER_USER = """
Country: {country_name}
Criterion {criterion_number}: {criterion_name}
Dimension: {dimension}
Reference year: {reference_year}

EVIDENCE ({evidence_count} sources):
{formatted_evidence}

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
  "key_sources": ["<domain or source name>"]
}}
"""

MIXED_INSTRUCTIONS = """
MIXED CRITERION:
Score both sub-dimensions separately.
legal_subscore: quality of legal framework (1-5)
enforcement_subscore: quality of practical implementation (1-5)
criterion_score = (legal_subscore × 0.4) + (enforcement_subscore × 0.6)
"""
