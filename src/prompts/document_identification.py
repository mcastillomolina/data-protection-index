"""
Prompts for document identification.

This module contains system and user prompts for identifying relevant
legal documents for a country using LLM.
"""

from pathlib import Path
from typing import Dict, Any
import yaml


_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


def _build_document_types_block(allowed_ids: set = None) -> str:
    """Load document_types.yaml and format non-legacy types for prompt injection.

    Args:
        allowed_ids: If provided, only include types whose id is in this set.
                     Legacy types are always excluded regardless.
    """
    yaml_path = _CONFIG_DIR / "document_types.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    lines = []
    for dt in data.get("document_types", []):
        if dt.get("legacy"):
            continue
        if allowed_ids is not None and dt["id"] not in allowed_ids:
            continue
        lines.append(f"- id: {dt['id']}")
        lines.append(f"  name: {dt['name']}")
        lines.append(f"  description: {dt['description']}")
        if dt.get("pi_criteria_covered"):
            lines.append(f"  pi_criteria_covered: {dt['pi_criteria_covered']}")
        if dt.get("common_keywords"):
            lines.append(f"  keywords: {dt['common_keywords']}")
        if dt.get("known_sources"):
            lines.append(f"  known_sources: [{', '.join(dt['known_sources'])}]")
    return "\n".join(lines)


# JSON schema for document identification response
DOCUMENT_IDENTIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "documents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "document_type": {
                        "type": "string",
                        "description": "Type of document — must be one of the allowed ids listed in the prompt"
                    },
                    "official_name": {
                        "type": "string",
                        "description": "Official name of the document in original language"
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief description of the document and its relevance"
                    },
                    "expected_language": {
                        "type": "string",
                        "description": "Expected language code (e.g., 'es', 'en', 'de')"
                    },
                    "priority_score": {
                        "type": "integer",
                        "description": "Priority score from 1-10, where 10 is most important",
                        "minimum": 1,
                        "maximum": 10
                    },
                    "alternate_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Alternative names or abbreviations for the document"
                    },
                    "expected_file_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Expected file formats (e.g., ['pdf', 'html'])"
                    },
                    "information_opacity": {
                        "type": "boolean",
                        "description": "True if enforcement evidence is likely behind national firewalls. Omit or false otherwise."
                    }
                },
                "required": [
                    "document_type",
                    "official_name",
                    "description",
                    "expected_language",
                    "priority_score"
                ]
            }
        },
        "metadata": {
            "type": "object",
            "properties": {
                "country": {"type": "string"},
                "total_documents": {"type": "integer"},
                "notes": {"type": "string"}
            }
        }
    },
    "required": ["documents", "metadata"]
}


SYSTEM_PROMPT = """You are an expert legal researcher specialising in data protection, \
privacy law, surveillance law, and civil liberties. Your task is to identify relevant \
legal documents for a given country across ALL dimensions of privacy protection — not \
just comprehensive data protection laws.

For each document provide:
- document_type: must be one of the allowed type ids listed in the user message
- official_name: exact official name in the original language
- description: brief description of the document and its relevance
- expected_language: ISO 639-1 language code
- priority_score: 1-10 (10 = most critical for understanding the country's privacy posture)
- alternate_names: common abbreviations or alternative names (if any)
- expected_file_types: likely formats (pdf, html, etc.)

Use the most specific matching type. If a document fits multiple types, prefer the one \
with the narrowest scope.

Focus on:
- Official government sources
- Currently in force legislation (not repealed laws)
- Documents that actually exist (not theoretical)
- Authoritative sources

Your goal is to find documentary evidence for all 14 Privacy International criteria:
1.  Constitutional protection        — constitution, court_decision
2.  Statutory protection             — data_protection_law
3.  Privacy enforcement              — enforcement_report, dpa_annual_report
4.  Identity cards & biometrics      — biometrics_id_law
5.  Data sharing                     — data_protection_law, legislative_history
6.  Visual surveillance              — surveillance_law
7.  Communication interception       — surveillance_law
8.  Workplace monitoring             — workplace_privacy_law
9.  Government access to data        — surveillance_law, data_protection_law
10. Communications data retention    — data_retention_law
11. Surveillance of medical/financial/movement — surveillance_law, sectoral laws
12. Border & trans-border issues     — border_surveillance_law
13. Leadership                       — international_treaty
14. Democratic safeguards            — court_decision, parliamentary_report

For countries with restricted information environments (China, Russia, Belarus, \
Iran, North Korea), set "information_opacity": true on documents where enforcement \
evidence is likely inaccessible outside national firewalls. This is a signal for \
the scoring engine, not a reason to skip the document.

Try to find at least one document per criterion. A country without a surveillance_law \
identified means criteria 6, 7, 9, 11 will have no evidence — identify it even if \
it is not a dedicated privacy law.

Respond with valid JSON only."""


def create_identification_prompt(
    country_name: str,
    iso_code: str,
    official_languages: list,
    government_domains: list,
    region: str,
    known_documents: Dict[str, str] = None,
    document_types: list = None
) -> str:
    """
    Create a prompt for identifying documents for a specific country.

    Args:
        country_name: Name of the country
        iso_code: ISO 3166-1 alpha-2 country code
        official_languages: List of official language codes
        government_domains: List of government domain extensions
        region: Geographic region (e.g., "Latin America", "Europe")
        known_documents: Optional dict of known document names by type
        document_types: Optional list of specific document type ids to focus on
            (if None, all non-legacy types from document_types.yaml are used)

    Returns:
        Formatted user prompt string
    """
    prompt = f"""Identify all relevant data protection and privacy legal documents for {country_name}.

Country Information:
- Name: {country_name}
- ISO Code: {iso_code}
- Official Languages: {', '.join(official_languages)}
- Government Domains: {', '.join(government_domains) if government_domains else 'unknown'}
- Region: {region}
"""

    if known_documents:
        prompt += "\nKnown Documents (for reference):\n"
        for doc_type, doc_name in known_documents.items():
            prompt += f"- {doc_type}: {doc_name}\n"

    allowed_ids = set(document_types) if document_types else None
    types_block = _build_document_types_block(allowed_ids)

    prompt += f"""
Allowed document_type values (use the id field exactly):
{types_block}

Identify documents that exist for {country_name}. Prioritise types that cover \
pi_criteria_covered values — these are essential for the privacy index. \
Try to cover all 14 criteria. Do not invent documents — only include documents \
you are reasonably certain exist for this country.

Respond with a JSON object containing an array of documents and metadata."""

    return prompt


def create_simple_identification_prompt(country_name: str) -> str:
    """
    Create a simplified prompt for quick document identification.

    Args:
        country_name: Name of the country

    Returns:
        Simplified user prompt string
    """
    types_block = _build_document_types_block()
    return f"""Identify the key data protection and privacy legal documents for {country_name}.

Allowed document_type values:
{types_block}

For each document provide the official name, description, priority (1-10), and alternate names.

Respond with JSON containing an array of documents."""


# Example of expected response format
EXAMPLE_RESPONSE = {
    "documents": [
        {
            "document_type": "constitution",
            "official_name": "Constitución Política de Chile",
            "description": "National constitution with Article 19(4) guaranteeing privacy rights",
            "expected_language": "es",
            "priority_score": 9,
            "alternate_names": ["CPR", "Constitución"],
            "expected_file_types": ["pdf", "html"]
        },
        {
            "document_type": "data_protection_law",
            "official_name": "Ley 19.628 sobre Protección de la Vida Privada",
            "description": "Primary data protection legislation from 1999",
            "expected_language": "es",
            "priority_score": 10,
            "alternate_names": ["Ley 19.628", "Ley de Protección de Datos"],
            "expected_file_types": ["pdf", "html"]
        },
        {
            "document_type": "enforcement_report",
            "official_name": "Informe Anual del Consejo para la Transparencia",
            "description": "Annual enforcement report from Chile's DPA with sanctions and cases",
            "expected_language": "es",
            "priority_score": 8,
            "alternate_names": ["CPLT Annual Report"],
            "expected_file_types": ["pdf"]
        }
    ],
    "metadata": {
        "country": "Chile",
        "total_documents": 3,
        "notes": "Chile is in the process of modernizing its data protection framework"
    }
}
