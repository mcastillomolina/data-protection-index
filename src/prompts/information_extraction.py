"""Prompt and JSON schema for Phase 3 information extraction."""

SYSTEM_PROMPT = """You are a legal analyst extracting structured information from data \
protection legislation. The document section may be in any language. You must:
1. Extract only information present in the section — return null for absent fields
2. Respond ONLY in English regardless of the document language
3. Respond with valid JSON only — no commentary, no markdown fences"""

EXTRACTION_PROMPT_TEMPLATE = """Extract data protection information from this document section.
Return null for any field not present in this specific section.

Fields to extract:
- key_provisions: list of core data protection principles or rules stated (strings)
- data_subject_rights: list of rights granted to individuals (strings)
- enforcement_body: name of supervisory/enforcement authority (string or null)
- penalties: list of fines or sanctions described, include amounts if present (strings)
- lawful_basis: list of legal grounds for data processing (strings)
- notes: any ambiguity or caveats worth flagging (string or null)

Document section:
{section_text}
"""

EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "key_provisions": {"type": ["array", "null"], "items": {"type": "string"}},
        "data_subject_rights": {"type": ["array", "null"], "items": {"type": "string"}},
        "enforcement_body": {"type": ["string", "null"]},
        "penalties": {"type": ["array", "null"], "items": {"type": "string"}},
        "lawful_basis": {"type": ["array", "null"], "items": {"type": "string"}},
        "notes": {"type": ["string", "null"]},
    },
    "required": [
        "key_provisions",
        "data_subject_rights",
        "enforcement_body",
        "penalties",
        "lawful_basis",
        "notes",
    ],
}


def build_extraction_prompt(section_text: str) -> str:
    return EXTRACTION_PROMPT_TEMPLATE.format(section_text=section_text)
