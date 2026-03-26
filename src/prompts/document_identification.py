"""
Prompts for document identification.

This module contains system and user prompts for identifying relevant
legal documents for a country using LLM.
"""

from typing import Dict, Any


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
                        "description": "Type of document (e.g., 'constitution', 'data_protection_law', 'dpa_reports')"
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


SYSTEM_PROMPT = """You are an expert legal researcher specializing in data protection and privacy law. Your task is to identify relevant legal documents for a given country that relate to data protection, privacy rights, and related regulations.

You should identify documents in the following categories:
1. Constitution - National constitution with privacy/data protection provisions
2. Data Protection Law - Primary legislation governing data protection
3. Data Protection Authority Reports - Annual reports, enforcement decisions
4. Enforcement Records - Published fines, sanctions, case decisions
5. Regulatory Guidance - Official guidance documents from the DPA
6. Legislative History - Parliamentary debates, amendments, explanatory notes

For each document, provide:
- document_type: The category from above (use snake_case)
- official_name: The exact official name in the original language
- description: A brief description of the document and its relevance
- expected_language: ISO 639-1 language code
- priority_score: 1-10, where 10 is most critical for understanding data protection
- alternate_names: Common abbreviations or alternative names (if any)
- expected_file_types: Likely formats this document exists in (pdf, html, etc.)

Focus on:
- Official government sources
- Currently in force legislation (not repealed laws)
- Documents that actually exist (not theoretical)
- Authoritative sources

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
        document_types: Optional list of specific document types to focus on

    Returns:
        Formatted user prompt string
    """
    prompt = f"""Identify all relevant data protection and privacy legal documents for {country_name}.

Country Information:
- Name: {country_name}
- ISO Code: {iso_code}
- Official Languages: {', '.join(official_languages)}
- Government Domains: {', '.join(government_domains)}
- Region: {region}
"""

    if known_documents:
        prompt += f"\nKnown Documents (for reference):\n"
        for doc_type, doc_name in known_documents.items():
            prompt += f"- {doc_type}: {doc_name}\n"

    if document_types:
        prompt += f"\nFocus on these document types: {', '.join(document_types)}\n"

    prompt += """
Please identify:
1. The national constitution (if it contains privacy/data protection provisions)
2. The primary data protection law(s)
3. Any sector-specific data protection regulations
4. The Data Protection Authority (DPA) and its key publications
5. Recent enforcement decisions or fines (if publicly available)
6. Official regulatory guidance documents

For each document, use your knowledge of this country's legal system to provide accurate official names and relevant details.

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
    return f"""Identify the key data protection and privacy legal documents for {country_name}.

Include:
1. Constitution (if it has privacy provisions)
2. Main data protection law(s)
3. Data Protection Authority reports
4. Any other critical privacy/data protection legislation

For each document, provide the official name, description, priority (1-10), and any alternate names.

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
        }
    ],
    "metadata": {
        "country": "Chile",
        "total_documents": 2,
        "notes": "Chile is in the process of modernizing its data protection framework"
    }
}
