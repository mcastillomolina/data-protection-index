"""Prompt and JSON schema for Phase 3 information extraction."""

SYSTEM_PROMPT = """You are a legal analyst extracting structured information from data \
protection and privacy legislation. The document section may be in any language. You must:
1. Extract only information present in the section — return null for absent fields
2. Respond ONLY in English regardless of the document language
3. Respond with valid JSON only — no commentary, no markdown fences"""

EXTRACTION_PROMPT_TEMPLATE = """Extract data protection and privacy information from this document section.
Return null for any field not present in this specific section.

Fields to extract:
- key_provisions: list of core data protection principles or rules stated (strings)
- data_subject_rights: list of rights granted to individuals (strings)
- enforcement_body: name of supervisory/enforcement authority (string or null)
- statutory_penalties: list of maximum fines or sanctions prescribed by law — what the law says CAN happen (strings)
- actual_sanctions: list of fines or sanctions actually issued by a regulator — include entity, amount, date if present (strings)
- lawful_basis: list of legal grounds for data processing (strings)
- constitutional_privacy_right: true if the section explicitly establishes a constitutional right to privacy or data protection, false if it explicitly denies it, null if not addressed (bool or null)
- constitutional_articles: list of specific constitutional articles cited (e.g. "Article 18") (strings)
- data_retention_period: legally mandated data retention duration if stated (e.g. "12 months", "2 years") (string or null)
- interception_legal_standard: legal standard required for communications interception if stated (e.g. "judicial warrant", "ministerial authorisation") (string or null)
- biometric_legal_basis: legal basis for biometric collection or storage if stated (string or null)
- treaties_signed: list of international treaties relevant to privacy/surveillance referenced (strings)
- dpa_exists: true if the section confirms an operational Data Protection Authority exists, null if not addressed (bool or null)
- dpa_independence: level of DPA independence if stated — one of "fully_independent", "ministerial", "within_government", "none" (string or null)
- dpa_staff_count: number of staff at DPA if reported (integer or null)
- sanctions_count: number of sanctions or fines issued in the reference period if reported (integer or null)
- sanctions_total_amount: total value of sanctions in reference period if stated (e.g. "€2.4M") (string or null)
- cctv_regulatory_status: CCTV regulatory status if stated — one of "regulated_and_enforced", "regulated_not_enforced", "unregulated", "unknown" (string or null)
- border_biometrics_deployed: true if biometric systems are confirmed deployed at borders (bool or null)
- information_opacity_flag: true if evidence is limited due to restricted information environment (bool or null)
- notes: any ambiguity or caveats worth flagging (string or null)

Document section:
{section_text}
"""

EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "key_provisions":             {"type": ["array", "null"], "items": {"type": "string"}},
        "data_subject_rights":        {"type": ["array", "null"], "items": {"type": "string"}},
        "enforcement_body":           {"type": ["string", "null"]},
        "statutory_penalties":        {"type": ["array", "null"], "items": {"type": "string"}},
        "actual_sanctions":           {"type": ["array", "null"], "items": {"type": "string"}},
        "lawful_basis":               {"type": ["array", "null"], "items": {"type": "string"}},
        "constitutional_privacy_right": {"type": ["boolean", "null"]},
        "constitutional_articles":    {"type": ["array", "null"], "items": {"type": "string"}},
        "data_retention_period":      {"type": ["string", "null"]},
        "interception_legal_standard": {"type": ["string", "null"]},
        "biometric_legal_basis":      {"type": ["string", "null"]},
        "treaties_signed":            {"type": ["array", "null"], "items": {"type": "string"}},
        "dpa_exists":                 {"type": ["boolean", "null"]},
        "dpa_independence":           {"type": ["string", "null"]},
        "dpa_staff_count":            {"type": ["integer", "null"]},
        "sanctions_count":            {"type": ["integer", "null"]},
        "sanctions_total_amount":     {"type": ["string", "null"]},
        "cctv_regulatory_status":     {"type": ["string", "null"]},
        "border_biometrics_deployed": {"type": ["boolean", "null"]},
        "information_opacity_flag":   {"type": ["boolean", "null"]},
        "notes":                      {"type": ["string", "null"]},
    },
    "required": [
        "key_provisions",
        "data_subject_rights",
        "enforcement_body",
        "statutory_penalties",
        "actual_sanctions",
        "lawful_basis",
        "constitutional_privacy_right",
        "constitutional_articles",
        "data_retention_period",
        "interception_legal_standard",
        "biometric_legal_basis",
        "treaties_signed",
        "dpa_exists",
        "dpa_independence",
        "dpa_staff_count",
        "sanctions_count",
        "sanctions_total_amount",
        "cctv_regulatory_status",
        "border_biometrics_deployed",
        "information_opacity_flag",
        "notes",
    ],
}


def build_extraction_prompt(section_text: str) -> str:
    return EXTRACTION_PROMPT_TEMPLATE.format(section_text=section_text)
