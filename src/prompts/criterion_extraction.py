"""Prompts and JSON schemas for Phase 3 criterion-aware extraction (CriterionExtractor)."""

from src.config.criteria import CRITERIA

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

LEGAL_EXTRACTOR_SYSTEM = """\
You are a legal analyst extracting structured information from legal texts.

Rules:
- Extract only what is explicitly stated in the provided text
- Do not infer, assume, or extrapolate beyond the text
- Extract in English regardless of source language
- Return null for any field not addressed in this section
- Be precise with legal article numbers and statute names

Return valid JSON only."""

ENFORCEMENT_EXTRACTOR_SYSTEM = """\
You are a legal analyst extracting enforcement evidence from regulatory \
documents, court decisions, NGO reports, and news sources.

Rules:
- Focus on what actually happened, not what the law says can happen
- Distinguish between statutory penalties (what law allows) and actual
  sanctions (what was actually imposed)
- Extract specific entities, amounts, and dates where present
- Extract in English regardless of source language
- Return null for any field not present in the text

Return valid JSON only."""

# ---------------------------------------------------------------------------
# User prompt templates
# ---------------------------------------------------------------------------

LEGAL_EXTRACTOR_USER = """\
Document: {document_type} — {official_name}
Country: {country_name}
Criteria: {criteria_block}

Section text:
---
{section_text}
---

Extract fields relevant to these criteria from this section.
Return null for any field not present in the text.

Schema:
{schema_block}
"""

ENFORCEMENT_EXTRACTOR_USER = """\
Document: {document_type} — {official_name}
Country: {country_name}
Criteria: {criteria_block}

Section text:
---
{section_text}
---

Extract enforcement evidence relevant to these criteria.
Focus on observable facts: what happened, who acted, when, with what result.

Schema:
{schema_block}
"""

# ---------------------------------------------------------------------------
# Per-criterion schema description strings (injected into user prompts)
# Legal dimension schemas — keys: 1, 2, 4, 5, 7, 9, 10, 13
# ---------------------------------------------------------------------------

LEGAL_SCHEMAS: dict[int, str] = {

    1: """{
  "constitutional_privacy_right": "bool — explicit privacy/data protection right",
  "constitutional_articles": "list[str] — article numbers (e.g. ['Article 18', 'Section 10'])",
  "right_scope": "str — what the right covers: home, communications, personal data, etc.",
  "limitations_clause": "str — how and when the right can be limited",
  "jurisprudence_mentioned": "list[str] — court cases or rulings cited by name",
  "notes": "str | null — any ambiguity or caveats worth flagging"
}""",

    2: """{
  "law_name": "str — official name of the law",
  "enactment_year": "int | null",
  "scope": "str — who and what is covered",
  "data_subject_rights": "list[str] — rights granted: access, erasure, portability, etc.",
  "lawful_bases": "list[str] — legal bases for processing",
  "sectoral_laws_mentioned": "list[str] — sector-specific privacy laws referenced",
  "key_provisions": "list[str] — most important obligations on data controllers",
  "notes": "str | null — any ambiguity or caveats worth flagging"
}""",

    4: """{
  "id_card_law_exists": "bool",
  "id_card_mandatory": "bool | null",
  "biometrics_collected": "list[str] — e.g. ['fingerprint', 'facial', 'iris']",
  "central_database": "bool | null — is biometric data stored centrally?",
  "legal_basis_for_biometrics": "str | null",
  "privacy_safeguards_stated": "list[str]",
  "notes": "str | null — any ambiguity or caveats worth flagging"
}""",

    5: """{
  "secondary_use_prohibition": "bool — does law prohibit using data for purposes beyond original collection?",
  "data_sharing_restrictions": "list[str] — specific restrictions on inter-agency sharing",
  "exceptions_to_restrictions": "list[str]",
  "consent_requirement": "str — when is consent required for sharing?",
  "notes": "str | null — any ambiguity or caveats worth flagging"
}""",

    7: """{
  "interception_legal_standard": "str — 'judicial_warrant' | 'ministerial_authorisation' | 'none' | 'mixed'",
  "authorising_body": "str — who authorises interception",
  "crime_threshold": "str — minimum offence severity required (e.g. '4+ years imprisonment')",
  "duration_limit": "str — maximum duration per warrant",
  "oversight_mechanism": "str — who reviews/audits interceptions",
  "service_provider_obligations": "list[str] — what providers must implement",
  "notes": "str | null — any ambiguity or caveats worth flagging"
}""",

    9: """{
  "warrant_requirement": "str — 'judicial' | 'administrative' | 'none' | 'mixed'",
  "agencies_with_access": "list[str] — law enforcement agencies named",
  "data_types_accessible": "list[str]",
  "access_process": "str — how access is obtained legally",
  "emergency_provisions": "str | null — warrantless access provisions",
  "notes": "str | null — any ambiguity or caveats worth flagging"
}""",

    10: """{
  "retention_law_exists": "bool",
  "retention_period": "str — e.g. '12 months', '2 years', 'not specified'",
  "data_types_covered": "list[str] — traffic data, content, location, etc.",
  "entities_obligated": "list[str] — ISPs, telcos, etc.",
  "oversight": "str | null",
  "exceptions": "list[str]",
  "notes": "str | null — any ambiguity or caveats worth flagging"
}""",

    13: """{
  "treaties_mentioned": "list[str] — treaty names referenced in the text",
  "treaty_status": "list[{name: str, status: 'signed'|'ratified'|'not_party'}]",
  "international_data_sharing": "list[str] — named bilateral or multilateral sharing agreements",
  "leadership_stance": "str | null — country's stated position on privacy at international level",
  "notes": "str | null — any ambiguity or caveats worth flagging"
}""",

}

# ---------------------------------------------------------------------------
# Per-criterion schema description strings — Enforcement dimension
# Keys: 3, 4, 5, 6, 7, 8, 9, 11, 12, 14
# ---------------------------------------------------------------------------

ENFORCEMENT_SCHEMAS: dict[int, str] = {

    3: """{
  "dpa_exists": "bool",
  "dpa_name": "str | null",
  "dpa_independence": "'fully_independent' | 'ministerial' | 'within_government' | 'none' | null",
  "dpa_staff_count": "int | null",
  "dpa_budget_mentioned": "str | null",
  "actual_sanctions": "list[{entity: str, amount: str, date: str, summary: str}]",
  "investigations_count": "int | null — number of investigations opened",
  "proactive_enforcement": "bool | null — did DPA act without complaint?",
  "enforcement_blocked": "bool | null — was enforcement prevented by government?",
  "notes": "str | null — any ambiguity or caveats worth flagging"
}""",

    4: """{
  "biometric_system_operational": "bool | null",
  "deployment_scale": "str | null — how widely deployed in practice",
  "abuse_documented": "list[str] — documented misuse of ID/biometric systems",
  "opt_out_possible": "bool | null",
  "notes": "str | null — any ambiguity or caveats worth flagging"
}""",

    5: """{
  "data_sharing_programs_active": "list[str] — named inter-agency programs",
  "secondary_use_violations": "list[{summary: str, date: str}]",
  "companies_compelled_to_share": "list[str] — cases where private sector compelled",
  "notes": "str | null — any ambiguity or caveats worth flagging"
}""",

    6: """{
  "cctv_scale": "'extensive' | 'moderate' | 'limited' | 'unknown'",
  "cctv_regulated": "bool | null",
  "regulatory_body": "str | null — who oversees CCTV",
  "enforcement_actions": "list[str] — any sanctions for illegal CCTV use",
  "facial_recognition_deployed": "bool | null",
  "public_spaces_covered": "list[str] — types of public space with CCTV",
  "notes": "str | null — any ambiguity or caveats worth flagging"
}""",

    7: """{
  "interception_volume": "str | null — reported number of intercepts per year",
  "illegal_wiretapping_cases": "list[str] — documented illegal intercepts",
  "oversight_effectiveness": "str | null — evidence of whether oversight works",
  "service_provider_compliance": "str | null",
  "notes": "str | null — any ambiguity or caveats worth flagging"
}""",

    8: """{
  "employer_surveillance_cases": "list[{summary: str, outcome: str, date: str}]",
  "employee_protections_enforced": "bool | null",
  "guidelines_issued": "bool | null",
  "regulator_rulings": "list[str] — DPA or labour court rulings on workplace monitoring",
  "notes": "str | null — any ambiguity or caveats worth flagging"
}""",

    9: """{
  "warrantless_access_documented": "list[str] — cases of access without proper authorisation",
  "access_requests_volume": "str | null — reported number of requests to providers",
  "legal_challenges": "list[str] — court challenges to government data access",
  "notes": "str | null — any ambiguity or caveats worth flagging"
}""",

    11: """{
  "medical_data_breaches": "list[{summary: str, date: str}]",
  "financial_surveillance_programs": "list[str] — named programs",
  "location_tracking_programs": "list[str] — named programs",
  "sensitive_data_enforcement": "list[str] — enforcement actions on medical/financial/movement data",
  "rfid_tracking_deployed": "bool | null",
  "notes": "str | null — any ambiguity or caveats worth flagging"
}""",

    12: """{
  "border_biometrics_deployed": "bool | null",
  "biometric_types_at_border": "list[str] — fingerprint, facial, iris, etc.",
  "passenger_data_shared_with": "list[str] — countries or organisations receiving passenger data",
  "data_sharing_agreements": "list[str] — named PNR or similar agreements",
  "privacy_impact_assessed": "bool | null",
  "notes": "str | null — any ambiguity or caveats worth flagging"
}""",

    14: """{
  "court_rulings_limiting_surveillance": "list[{summary: str, court: str, date: str}]",
  "parliament_oversight_actions": "list[str] — committees, hearings, legislation blocking executive",
  "executive_overreach_documented": "list[str] — documented cases of unchecked surveillance",
  "press_freedom_incidents": "list[str] — journalists or lawyers targeted by surveillance",
  "democratic_backsliding": "bool | null — evidence of systematic erosion of safeguards",
  "notes": "str | null — any ambiguity or caveats worth flagging"
}""",

}

# ---------------------------------------------------------------------------
# Per-criterion JSON Schema dicts for complete_json() validation
# ---------------------------------------------------------------------------

LEGAL_JSON_SCHEMAS: dict[int, dict] = {

    1: {
        "type": "object",
        "properties": {
            "constitutional_privacy_right":  {"type": ["boolean", "null"]},
            "constitutional_articles":       {"type": ["array", "null"], "items": {"type": "string"}},
            "right_scope":                   {"type": ["string", "null"]},
            "limitations_clause":            {"type": ["string", "null"]},
            "jurisprudence_mentioned":       {"type": ["array", "null"], "items": {"type": "string"}},
            "notes":                         {"type": ["string", "null"]},
        },
        "required": ["constitutional_privacy_right", "constitutional_articles",
                     "right_scope", "limitations_clause", "jurisprudence_mentioned", "notes"],
    },

    2: {
        "type": "object",
        "properties": {
            "law_name":                 {"type": ["string", "null"]},
            "enactment_year":           {"type": ["integer", "null"]},
            "scope":                    {"type": ["string", "null"]},
            "data_subject_rights":      {"type": ["array", "null"], "items": {"type": "string"}},
            "lawful_bases":             {"type": ["array", "null"], "items": {"type": "string"}},
            "sectoral_laws_mentioned":  {"type": ["array", "null"], "items": {"type": "string"}},
            "key_provisions":           {"type": ["array", "null"], "items": {"type": "string"}},
            "notes":                    {"type": ["string", "null"]},
        },
        "required": ["law_name", "enactment_year", "scope", "data_subject_rights",
                     "lawful_bases", "sectoral_laws_mentioned", "key_provisions", "notes"],
    },

    4: {
        "type": "object",
        "properties": {
            "id_card_law_exists":          {"type": ["boolean", "null"]},
            "id_card_mandatory":           {"type": ["boolean", "null"]},
            "biometrics_collected":        {"type": ["array", "null"], "items": {"type": "string"}},
            "central_database":            {"type": ["boolean", "null"]},
            "legal_basis_for_biometrics":  {"type": ["string", "null"]},
            "privacy_safeguards_stated":   {"type": ["array", "null"], "items": {"type": "string"}},
            "notes":                       {"type": ["string", "null"]},
        },
        "required": ["id_card_law_exists", "id_card_mandatory", "biometrics_collected",
                     "central_database", "legal_basis_for_biometrics", "privacy_safeguards_stated", "notes"],
    },

    5: {
        "type": "object",
        "properties": {
            "secondary_use_prohibition":    {"type": ["boolean", "null"]},
            "data_sharing_restrictions":    {"type": ["array", "null"], "items": {"type": "string"}},
            "exceptions_to_restrictions":   {"type": ["array", "null"], "items": {"type": "string"}},
            "consent_requirement":          {"type": ["string", "null"]},
            "notes":                        {"type": ["string", "null"]},
        },
        "required": ["secondary_use_prohibition", "data_sharing_restrictions",
                     "exceptions_to_restrictions", "consent_requirement", "notes"],
    },

    7: {
        "type": "object",
        "properties": {
            "interception_legal_standard":      {"type": ["string", "null"]},
            "authorising_body":                 {"type": ["string", "null"]},
            "crime_threshold":                  {"type": ["string", "null"]},
            "duration_limit":                   {"type": ["string", "null"]},
            "oversight_mechanism":              {"type": ["string", "null"]},
            "service_provider_obligations":     {"type": ["array", "null"], "items": {"type": "string"}},
            "notes":                            {"type": ["string", "null"]},
        },
        "required": ["interception_legal_standard", "authorising_body", "crime_threshold",
                     "duration_limit", "oversight_mechanism", "service_provider_obligations", "notes"],
    },

    9: {
        "type": "object",
        "properties": {
            "warrant_requirement":    {"type": ["string", "null"]},
            "agencies_with_access":   {"type": ["array", "null"], "items": {"type": "string"}},
            "data_types_accessible":  {"type": ["array", "null"], "items": {"type": "string"}},
            "access_process":         {"type": ["string", "null"]},
            "emergency_provisions":   {"type": ["string", "null"]},
            "notes":                  {"type": ["string", "null"]},
        },
        "required": ["warrant_requirement", "agencies_with_access", "data_types_accessible",
                     "access_process", "emergency_provisions", "notes"],
    },

    10: {
        "type": "object",
        "properties": {
            "retention_law_exists":  {"type": ["boolean", "null"]},
            "retention_period":      {"type": ["string", "null"]},
            "data_types_covered":    {"type": ["array", "null"], "items": {"type": "string"}},
            "entities_obligated":    {"type": ["array", "null"], "items": {"type": "string"}},
            "oversight":             {"type": ["string", "null"]},
            "exceptions":            {"type": ["array", "null"], "items": {"type": "string"}},
            "notes":                 {"type": ["string", "null"]},
        },
        "required": ["retention_law_exists", "retention_period", "data_types_covered",
                     "entities_obligated", "oversight", "exceptions", "notes"],
    },

    13: {
        "type": "object",
        "properties": {
            "treaties_mentioned":          {"type": ["array", "null"], "items": {"type": "string"}},
            "treaty_status": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "properties": {
                        "name":   {"type": "string"},
                        "status": {"type": "string"},
                    },
                },
            },
            "international_data_sharing":  {"type": ["array", "null"], "items": {"type": "string"}},
            "leadership_stance":           {"type": ["string", "null"]},
            "notes":                       {"type": ["string", "null"]},
        },
        "required": ["treaties_mentioned", "treaty_status", "international_data_sharing",
                     "leadership_stance", "notes"],
    },

}

ENFORCEMENT_JSON_SCHEMAS: dict[int, dict] = {

    3: {
        "type": "object",
        "properties": {
            "dpa_exists":          {"type": ["boolean", "null"]},
            "dpa_name":            {"type": ["string", "null"]},
            "dpa_independence":    {"type": ["string", "null"]},
            "dpa_staff_count":     {"type": ["integer", "null"]},
            "dpa_budget_mentioned": {"type": ["string", "null"]},
            "actual_sanctions": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "properties": {
                        "entity":  {"type": "string"},
                        "amount":  {"type": "string"},
                        "date":    {"type": "string"},
                        "summary": {"type": "string"},
                    },
                },
            },
            "investigations_count":   {"type": ["integer", "null"]},
            "proactive_enforcement":  {"type": ["boolean", "null"]},
            "enforcement_blocked":    {"type": ["boolean", "null"]},
            "notes":                  {"type": ["string", "null"]},
        },
        "required": ["dpa_exists", "dpa_name", "dpa_independence", "dpa_staff_count",
                     "dpa_budget_mentioned", "actual_sanctions", "investigations_count",
                     "proactive_enforcement", "enforcement_blocked", "notes"],
    },

    4: {
        "type": "object",
        "properties": {
            "biometric_system_operational": {"type": ["boolean", "null"]},
            "deployment_scale":             {"type": ["string", "null"]},
            "abuse_documented":             {"type": ["array", "null"], "items": {"type": "string"}},
            "opt_out_possible":             {"type": ["boolean", "null"]},
            "notes":                        {"type": ["string", "null"]},
        },
        "required": ["biometric_system_operational", "deployment_scale",
                     "abuse_documented", "opt_out_possible", "notes"],
    },

    5: {
        "type": "object",
        "properties": {
            "data_sharing_programs_active":  {"type": ["array", "null"], "items": {"type": "string"}},
            "secondary_use_violations": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "date":    {"type": "string"},
                    },
                },
            },
            "companies_compelled_to_share":  {"type": ["array", "null"], "items": {"type": "string"}},
            "notes":                         {"type": ["string", "null"]},
        },
        "required": ["data_sharing_programs_active", "secondary_use_violations",
                     "companies_compelled_to_share", "notes"],
    },

    6: {
        "type": "object",
        "properties": {
            "cctv_scale":                    {"type": ["string", "null"]},
            "cctv_regulated":                {"type": ["boolean", "null"]},
            "regulatory_body":               {"type": ["string", "null"]},
            "enforcement_actions":           {"type": ["array", "null"], "items": {"type": "string"}},
            "facial_recognition_deployed":   {"type": ["boolean", "null"]},
            "public_spaces_covered":         {"type": ["array", "null"], "items": {"type": "string"}},
            "notes":                         {"type": ["string", "null"]},
        },
        "required": ["cctv_scale", "cctv_regulated", "regulatory_body", "enforcement_actions",
                     "facial_recognition_deployed", "public_spaces_covered", "notes"],
    },

    7: {
        "type": "object",
        "properties": {
            "interception_volume":         {"type": ["string", "null"]},
            "illegal_wiretapping_cases":   {"type": ["array", "null"], "items": {"type": "string"}},
            "oversight_effectiveness":     {"type": ["string", "null"]},
            "service_provider_compliance": {"type": ["string", "null"]},
            "notes":                       {"type": ["string", "null"]},
        },
        "required": ["interception_volume", "illegal_wiretapping_cases",
                     "oversight_effectiveness", "service_provider_compliance", "notes"],
    },

    8: {
        "type": "object",
        "properties": {
            "employer_surveillance_cases": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "outcome": {"type": "string"},
                        "date":    {"type": "string"},
                    },
                },
            },
            "employee_protections_enforced": {"type": ["boolean", "null"]},
            "guidelines_issued":             {"type": ["boolean", "null"]},
            "regulator_rulings":             {"type": ["array", "null"], "items": {"type": "string"}},
            "notes":                         {"type": ["string", "null"]},
        },
        "required": ["employer_surveillance_cases", "employee_protections_enforced",
                     "guidelines_issued", "regulator_rulings", "notes"],
    },

    9: {
        "type": "object",
        "properties": {
            "warrantless_access_documented": {"type": ["array", "null"], "items": {"type": "string"}},
            "access_requests_volume":        {"type": ["string", "null"]},
            "legal_challenges":              {"type": ["array", "null"], "items": {"type": "string"}},
            "notes":                         {"type": ["string", "null"]},
        },
        "required": ["warrantless_access_documented", "access_requests_volume",
                     "legal_challenges", "notes"],
    },

    11: {
        "type": "object",
        "properties": {
            "medical_data_breaches": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "date":    {"type": "string"},
                    },
                },
            },
            "financial_surveillance_programs": {"type": ["array", "null"], "items": {"type": "string"}},
            "location_tracking_programs":      {"type": ["array", "null"], "items": {"type": "string"}},
            "sensitive_data_enforcement":      {"type": ["array", "null"], "items": {"type": "string"}},
            "rfid_tracking_deployed":          {"type": ["boolean", "null"]},
            "notes":                           {"type": ["string", "null"]},
        },
        "required": ["medical_data_breaches", "financial_surveillance_programs",
                     "location_tracking_programs", "sensitive_data_enforcement",
                     "rfid_tracking_deployed", "notes"],
    },

    12: {
        "type": "object",
        "properties": {
            "border_biometrics_deployed":    {"type": ["boolean", "null"]},
            "biometric_types_at_border":     {"type": ["array", "null"], "items": {"type": "string"}},
            "passenger_data_shared_with":    {"type": ["array", "null"], "items": {"type": "string"}},
            "data_sharing_agreements":       {"type": ["array", "null"], "items": {"type": "string"}},
            "privacy_impact_assessed":       {"type": ["boolean", "null"]},
            "notes":                         {"type": ["string", "null"]},
        },
        "required": ["border_biometrics_deployed", "biometric_types_at_border",
                     "passenger_data_shared_with", "data_sharing_agreements",
                     "privacy_impact_assessed", "notes"],
    },

    14: {
        "type": "object",
        "properties": {
            "court_rulings_limiting_surveillance": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "court":   {"type": "string"},
                        "date":    {"type": "string"},
                    },
                },
            },
            "parliament_oversight_actions":   {"type": ["array", "null"], "items": {"type": "string"}},
            "executive_overreach_documented": {"type": ["array", "null"], "items": {"type": "string"}},
            "press_freedom_incidents":        {"type": ["array", "null"], "items": {"type": "string"}},
            "democratic_backsliding":         {"type": ["boolean", "null"]},
            "notes":                          {"type": ["string", "null"]},
        },
        "required": ["court_rulings_limiting_surveillance", "parliament_oversight_actions",
                     "executive_overreach_documented", "press_freedom_incidents",
                     "democratic_backsliding", "notes"],
    },

}

# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------


def _criteria_block(criteria_ids: list[int]) -> str:
    parts = [f"Criterion {cid}: {CRITERIA[cid]['name']}" for cid in criteria_ids if cid in CRITERIA]
    return "\n".join(parts) if parts else "Unknown criteria"


def build_legal_prompt(
    document_type: str,
    official_name: str,
    country_name: str,
    criteria_ids: list[int],
    section_text: str,
) -> str:
    """Build the user prompt for LegalExtractor. Only uses criteria present in LEGAL_SCHEMAS."""
    relevant = [c for c in criteria_ids if c in LEGAL_SCHEMAS]
    schema_block = "\n\n".join(
        f"Criterion {cid} — {CRITERIA[cid]['name']}:\n{LEGAL_SCHEMAS[cid]}"
        for cid in relevant
    )
    return LEGAL_EXTRACTOR_USER.format(
        document_type=document_type,
        official_name=official_name,
        country_name=country_name,
        criteria_block=_criteria_block(relevant),
        section_text=section_text,
        schema_block=schema_block,
    )


def build_enforcement_prompt(
    document_type: str,
    official_name: str,
    country_name: str,
    criteria_ids: list[int],
    section_text: str,
) -> str:
    """Build the user prompt for EnforcementExtractor. Only uses criteria present in ENFORCEMENT_SCHEMAS."""
    relevant = [c for c in criteria_ids if c in ENFORCEMENT_SCHEMAS]
    schema_block = "\n\n".join(
        f"Criterion {cid} — {CRITERIA[cid]['name']}:\n{ENFORCEMENT_SCHEMAS[cid]}"
        for cid in relevant
    )
    return ENFORCEMENT_EXTRACTOR_USER.format(
        document_type=document_type,
        official_name=official_name,
        country_name=country_name,
        criteria_block=_criteria_block(relevant),
        section_text=section_text,
        schema_block=schema_block,
    )


def _base_json_schema() -> dict:
    return {
        "type": "object",
        "properties": {"notes": {"type": ["string", "null"]}},
        "required": ["notes"],
    }


def _merge_schemas(criteria_ids: list[int], source: dict[int, dict]) -> dict:
    merged_props: dict = {}
    merged_required: list = []
    for cid in criteria_ids:
        schema = source.get(cid)
        if schema is None:
            continue
        merged_props.update(schema.get("properties", {}))
        merged_required.extend(
            r for r in schema.get("required", []) if r not in merged_required
        )
    if not merged_props:
        return _base_json_schema()
    # notes is always present
    if "notes" not in merged_props:
        merged_props["notes"] = {"type": ["string", "null"]}
    if "notes" not in merged_required:
        merged_required.append("notes")
    return {"type": "object", "properties": merged_props, "required": merged_required}


def merge_legal_json_schema(criteria_ids: list[int]) -> dict:
    """Merge JSON Schemas for the given criteria (legal dimension)."""
    relevant = [c for c in criteria_ids if c in LEGAL_JSON_SCHEMAS]
    return _merge_schemas(relevant, LEGAL_JSON_SCHEMAS)


def merge_enforcement_json_schema(criteria_ids: list[int]) -> dict:
    """Merge JSON Schemas for the given criteria (enforcement dimension)."""
    relevant = [c for c in criteria_ids if c in ENFORCEMENT_JSON_SCHEMAS]
    return _merge_schemas(relevant, ENFORCEMENT_JSON_SCHEMAS)
