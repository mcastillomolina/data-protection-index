"""
Prompts for search query generation.

This module contains system and user prompts for generating targeted
search queries to find specific legal documents.
"""

from typing import List, Dict, Any


CRITERION_QUERY_HINTS: Dict[str, Dict] = {
    "constitution": {
        "hint": "Look for the official constitutional text and any amendments related to privacy, data protection, or communications secrecy.",
        "trusted_domains": ["constituteproject.org", "venice.coe.int"],
        "example_queries": [
            '"{country}" constitution "right to privacy" OR "data protection" article',
            'site:constituteproject.org "{country}"',
        ],
    },
    "enforcement_report": {
        "hint": "Look for DPA enforcement decisions, fines issued, annual reports, and case outcomes. Prefer GDPRhub and official DPA sites.",
        "trusted_domains": ["gdprhub.eu", "edpb.europa.eu"],
        "example_queries": [
            'site:gdprhub.eu "{country}"',
            '"{country}" data protection authority enforcement fine sanction {year}',
            '"{country}" DPA annual report {year}',
        ],
    },
    "surveillance_law": {
        "hint": "Look for laws on CCTV regulation, wiretapping, lawful interception, and state surveillance powers.",
        "trusted_domains": ["privacyinternational.org", "eff.org", "accessnow.org"],
        "example_queries": [
            '"{country}" surveillance law interception wiretapping legal framework',
            '"{country}" CCTV regulation privacy law',
            'site:privacyinternational.org "{country}" surveillance',
        ],
    },
    "international_treaty": {
        "hint": "Look for treaties signed: Budapest Convention on Cybercrime, Treaty of Prum, bilateral data-sharing agreements.",
        "trusted_domains": ["treaty.un.org", "coe.int"],
        "example_queries": [
            '"{country}" "Budapest Convention" cybercrime ratified',
            '"{country}" data sharing agreement law enforcement treaty',
            'site:coe.int "{country}" convention privacy',
        ],
    },
    "dpa_annual_report": {
        "hint": "Look for official DPA annual reports with staffing numbers, budget, complaints received, investigations opened, fines issued.",
        "trusted_domains": [],
        "example_queries": [
            '"{country}" data protection authority annual report {year}',
            '"{country}" DPA budget staff independence report',
        ],
    },
    "biometrics_id_law": {
        "hint": "Look for national ID card law, biometric passport regulation, and any central biometric database legislation.",
        "trusted_domains": [],
        "example_queries": [
            '"{country}" national identity card biometrics law',
            '"{country}" biometric passport fingerprint database regulation',
        ],
    },
    "workplace_privacy_law": {
        "hint": "Look for labor laws, workplace monitoring regulations, employee data protection guidelines.",
        "trusted_domains": [],
        "example_queries": [
            '"{country}" workplace monitoring privacy law employee',
            '"{country}" employee surveillance regulation data protection',
        ],
    },
}


# JSON schema for query generation response
QUERY_GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "query_string": {
                        "type": "string",
                        "description": "The search query to execute"
                    },
                    "priority": {
                        "type": "integer",
                        "description": "Priority score 1-10, where 10 is most likely to find the document",
                        "minimum": 1,
                        "maximum": 10
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why this query is likely to be effective"
                    },
                    "site_restrictions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Recommended site restrictions (e.g., ['site:gob.cl'])"
                    },
                    "expected_result_type": {
                        "type": "string",
                        "description": "Expected type of result (e.g., 'official_text', 'pdf_document', 'summary_page')"
                    }
                },
                "required": ["query_string", "priority", "reasoning"]
            }
        },
        "metadata": {
            "type": "object",
            "properties": {
                "document_name": {"type": "string"},
                "total_queries": {"type": "integer"},
                "search_strategy": {"type": "string"}
            }
        }
    },
    "required": ["queries", "metadata"]
}


SYSTEM_PROMPT = """You are an expert at crafting effective web search queries to find legal documents. Your task is to generate multiple targeted search queries that will help locate a specific document.

Guidelines for creating effective queries:
1. Use the official document name in the original language
2. Include country-specific terms and identifiers
3. Vary the queries to capture different sources (official sites, legal databases, summaries)
4. Use site restrictions to focus on authoritative sources (e.g., site:.gob.cl for Chile)
5. Consider both full official names and common abbreviations
6. Include file type hints when appropriate (PDF for official texts)
7. Use quotes for exact phrases when needed
8. Consider both direct document searches and contextual searches

Generate 3-5 queries per document, ordered by priority (most likely to succeed first).

For each query provide:
- query_string: The actual search query
- priority: 1-10 score (10 = most likely to find the document)
- reasoning: Brief explanation of the search strategy
- site_restrictions: Recommended domain filters (if any)
- expected_result_type: What kind of page you expect to find

Respond with valid JSON only."""


def create_query_generation_prompt(
    document_name: str,
    document_type: str,
    country_name: str,
    government_domains: List[str],
    language: str,
    alternate_names: List[str] = None,
    known_sources: List[str] = None,
    criterion_number: int = None,
    criterion_core_question: str = None,
    trusted_domains: List[str] = None,
) -> str:
    """
    Create a prompt for generating search queries for a specific document.

    Args:
        document_name: Official name of the document
        document_type: Type of document (e.g., 'data_protection_law')
        country_name: Name of the country
        government_domains: List of government domain extensions
        language: Expected language of the document
        alternate_names: Alternative names for the document
        known_sources: Known authoritative sources/domains

    Returns:
        Formatted user prompt string
    """
    prompt = f"""Generate effective search queries to find this document:

Document Information:
- Official Name: {document_name}
- Document Type: {document_type}
- Country: {country_name}
- Language: {language}
- Government Domains: {', '.join(government_domains)}
"""

    if alternate_names:
        prompt += f"- Alternate Names: {', '.join(alternate_names)}\n"

    # Merge hint trusted_domains into known_sources
    hints = CRITERION_QUERY_HINTS.get(document_type)
    merged_sources = list(known_sources) if known_sources else []
    if hints:
        for domain in hints["trusted_domains"]:
            if domain not in merged_sources:
                merged_sources.append(domain)

    if merged_sources:
        prompt += f"- Known Authoritative Sources: {', '.join(merged_sources)}\n"

    if hints:
        prompt += f"\nDocument-type search guidance:\n"
        prompt += f"  Hint: {hints['hint']}\n"
        if hints["example_queries"]:
            prompt += "  Example query patterns (adapt to the actual document and country):\n"
            for eq in hints["example_queries"]:
                prompt += f"    - {eq}\n"

    if criterion_number is not None:
        prompt += f"""
This document is sought as evidence for:
Criterion {criterion_number}: {criterion_core_question}
Queries should surface evidence that answers this criterion's question,
not just locate the document in general.
"""

    if trusted_domains:
        prompt += (
            "Prioritise these trusted domains "
            "(use site: operator for Query 1 if applicable):\n"
            f"{trusted_domains}\n"
        )

    prompt += f"""
Generate 3-5 search queries with different strategies:

Strategy Mix:
1. Direct official name search with site restriction
2. Official name + "texto completo" or "full text" or "official text"
3. Official name + country + "PDF"
4. Abbreviated name + contextual terms
5. Regulatory context search (e.g., "data protection law {country_name}")

Prioritize queries that:
- Target official government websites
- Look for authoritative legal databases
- Search for PDF versions of official texts
- Use exact official terminology

Respond with a JSON object containing an array of queries and metadata."""

    return prompt


def create_simple_query_generation_prompt(
    document_name: str,
    country_name: str,
    government_domains: List[str]
) -> str:
    """
    Create a simplified prompt for quick query generation.

    Args:
        document_name: Official name of the document
        country_name: Name of the country
        government_domains: List of government domain extensions

    Returns:
        Simplified user prompt string
    """
    site_restriction = government_domains[0] if government_domains else ""

    return f"""Generate 3-5 web search queries to find this document: "{document_name}" from {country_name}.

Include:
1. A query with site restriction ({site_restriction})
2. A query searching for the official text or PDF
3. A query using contextual terms about data protection law
4. Alternative name or abbreviation queries (if applicable)

For each query, provide the query string, priority score (1-10), and brief reasoning.

Respond with JSON containing an array of queries."""


def create_multilingual_query_prompt(
    document_name: str,
    country_name: str,
    primary_language: str,
    secondary_languages: List[str] = None
) -> str:
    """
    Create queries in multiple languages for international documents.

    Args:
        document_name: Official name of the document
        country_name: Name of the country
        primary_language: Primary language code
        secondary_languages: Optional list of secondary language codes

    Returns:
        Multilingual query prompt string
    """
    prompt = f"""Generate multilingual search queries to find: "{document_name}" from {country_name}.

Primary language: {primary_language}
"""

    if secondary_languages:
        prompt += f"Also create queries in: {', '.join(secondary_languages)}\n"

    prompt += """
Generate queries that:
1. Use the official name in the primary language
2. Include English translations (e.g., "data protection law")
3. Search for both local and international legal databases
4. Consider how international organizations might reference this law

Respond with JSON containing queries in multiple languages."""

    return prompt


# Example of expected response format
EXAMPLE_RESPONSE = {
    "queries": [
        {
            "query_string": "site:bcn.cl \"Ley 19.628\" texto completo",
            "priority": 10,
            "reasoning": "Direct search on official legislative website for full text",
            "site_restrictions": ["site:bcn.cl"],
            "expected_result_type": "official_text"
        },
        {
            "query_string": "\"Ley 19.628 sobre Protección de la Vida Privada\" PDF site:.gob.cl",
            "priority": 9,
            "reasoning": "Full official name with PDF filter on government domains",
            "site_restrictions": ["site:.gob.cl"],
            "expected_result_type": "pdf_document"
        },
        {
            "query_string": "Chile data protection law \"Ley 19.628\" official text",
            "priority": 8,
            "reasoning": "Mixed language search for English and Spanish sources",
            "site_restrictions": [],
            "expected_result_type": "summary_page"
        },
        {
            "query_string": "Ley 19628 protección datos personales Chile",
            "priority": 7,
            "reasoning": "Abbreviated name with contextual keywords",
            "site_restrictions": [],
            "expected_result_type": "official_text"
        }
    ],
    "metadata": {
        "document_name": "Ley 19.628",
        "total_queries": 4,
        "search_strategy": "Multi-pronged approach targeting official sources, PDFs, and contextual searches"
    }
}
