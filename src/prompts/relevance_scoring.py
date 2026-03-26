"""
Prompts for relevance scoring of search results.

This module contains system and user prompts for scoring search results
by their relevance to a specific document using LLM.
"""

from typing import List, Dict, Any


# JSON schema for relevance scoring response
RELEVANCE_SCORING_SCHEMA = {
    "type": "object",
    "properties": {
        "scored_results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the search result"
                    },
                    "relevance_score": {
                        "type": "number",
                        "description": "Relevance score from 0-10",
                        "minimum": 0,
                        "maximum": 10
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Explanation for the score"
                    },
                    "is_likely_official": {
                        "type": "boolean",
                        "description": "Whether this appears to be from an official/authoritative source"
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Confidence in the relevance assessment"
                    },
                    "document_type": {
                        "type": "string",
                        "description": "Apparent document type (e.g., 'full_text', 'summary', 'news_article', 'analysis')"
                    }
                },
                "required": ["url", "relevance_score", "reasoning", "is_likely_official", "confidence"]
            }
        },
        "summary": {
            "type": "object",
            "properties": {
                "total_results": {"type": "integer"},
                "highly_relevant_count": {"type": "integer"},
                "official_source_count": {"type": "integer"},
                "recommended_top_result": {"type": "string"}
            }
        }
    },
    "required": ["scored_results", "summary"]
}


SYSTEM_PROMPT = """You are an expert legal researcher specializing in evaluating the relevance and authority of web search results for legal documents.

Your task is to score search results based on their relevance to finding a specific legal document.

Scoring criteria (0-10 scale):

10 - Perfect Match:
- Official government source with the exact document
- Full official text from authoritative database
- Direct link to the document in original format

8-9 - Highly Relevant:
- Official government page about the document
- Authoritative legal database with the document
- Official PDF or HTML version
- Verified translation or summary from official source

6-7 - Relevant:
- Government page that references the document
- Academic or legal analysis from credible source
- News from official sources about the law
- Secondary sources with links to official documents

4-5 - Somewhat Relevant:
- General information about the topic
- Non-official summaries or analyses
- News articles or blog posts about the law
- Related but not the specific document

1-3 - Marginally Relevant:
- Tangentially related content
- Different country's similar law
- General privacy/data protection information
- Commercial sites discussing the topic

0 - Not Relevant:
- Unrelated content
- Spam or advertising
- Different topic entirely

Authority indicators:
- Official government domains (.gov, .gob, .gouv, etc.)
- Parliamentary/legislative databases
- National legal information systems
- Data Protection Authority websites
- International organization databases (OECD, Council of Europe, etc.)
- Established legal research platforms

For each result, provide:
- relevance_score: 0-10 score
- reasoning: Clear explanation of the score
- is_likely_official: Boolean indicating if it's an official source
- confidence: high/medium/low confidence in the assessment
- document_type: What kind of page/document this appears to be

Respond with valid JSON only."""


def create_relevance_scoring_prompt(
    document_name: str,
    document_type: str,
    country_name: str,
    search_results: List[Dict[str, str]],
    min_score: float = 6.0
) -> str:
    """
    Create a prompt for scoring search results relevance.

    Args:
        document_name: Official name of the document
        document_type: Type of document being searched for
        country_name: Name of the country
        search_results: List of search results with url, title, snippet
        min_score: Minimum score to consider relevant (default 6.0)

    Returns:
        Formatted user prompt string
    """
    prompt = f"""Score the relevance of these search results for finding this document:

Target Document:
- Name: {document_name}
- Type: {document_type}
- Country: {country_name}

Search Results to Score:
"""

    for i, result in enumerate(search_results, 1):
        url = result.get('url', 'N/A')
        title = result.get('title', 'N/A')
        snippet = result.get('snippet', 'N/A')
        domain = result.get('domain', 'N/A')

        prompt += f"""
Result #{i}:
- URL: {url}
- Domain: {domain}
- Title: {title}
- Snippet: {snippet}
"""

    prompt += f"""
For each result, provide:
1. A relevance score (0-10)
2. Clear reasoning for the score
3. Whether it appears to be an official/authoritative source
4. Your confidence level (high/medium/low)
5. The apparent document type

Prioritize:
- Official government sources
- Full official texts over summaries
- Authoritative legal databases
- Direct document links over informational pages

Focus on results scoring {min_score} or higher as these are most likely to contain the actual document.

Respond with a JSON object containing scored results and a summary."""

    return prompt


def create_simple_relevance_prompt(
    document_name: str,
    search_results: List[Dict[str, str]]
) -> str:
    """
    Create a simplified prompt for quick relevance scoring.

    Args:
        document_name: Official name of the document
        search_results: List of search results with url, title, snippet

    Returns:
        Simplified user prompt string
    """
    prompt = f"""Score how relevant each search result is for finding: "{document_name}"

Results:
"""

    for i, result in enumerate(search_results, 1):
        prompt += f"\n{i}. {result.get('title', 'N/A')}\n"
        prompt += f"   URL: {result.get('url', 'N/A')}\n"
        prompt += f"   Snippet: {result.get('snippet', 'N/A')[:200]}...\n"

    prompt += """
For each result, provide a score (0-10) and brief reasoning.
10 = exact official document, 0 = completely irrelevant.

Respond with JSON containing scored results."""

    return prompt


def create_comparative_scoring_prompt(
    document_name: str,
    search_results: List[Dict[str, str]],
    top_n: int = 5
) -> str:
    """
    Create a prompt for comparative scoring to find top N results.

    Args:
        document_name: Official name of the document
        search_results: List of search results
        top_n: Number of top results to identify

    Returns:
        Comparative scoring prompt string
    """
    prompt = f"""Compare and rank these search results for finding: "{document_name}"

Identify the top {top_n} most relevant results.

Results to compare:
"""

    for i, result in enumerate(search_results, 1):
        prompt += f"\n{i}. {result.get('title', 'N/A')}\n"
        prompt += f"   {result.get('url', 'N/A')}\n"
        prompt += f"   {result.get('snippet', 'N/A')[:150]}...\n"

    prompt += f"""
Score each result (0-10) and identify the top {top_n}.

For the top {top_n}, provide detailed reasoning about:
- Why it's likely to contain the document
- Whether it's an official source
- What type of document/page it appears to be

Respond with JSON containing all scored results and highlighting the top {top_n}."""

    return prompt


def create_batch_scoring_prompt(
    documents: List[Dict[str, str]],
    search_results: List[Dict[str, str]]
) -> str:
    """
    Create a prompt for scoring results against multiple documents.

    Useful when you have results that might match different documents.

    Args:
        documents: List of documents with name and type
        search_results: List of search results

    Returns:
        Batch scoring prompt string
    """
    prompt = "Score these search results for relevance to multiple documents:\n\nDocuments:\n"

    for i, doc in enumerate(documents, 1):
        prompt += f"{i}. {doc.get('name', 'N/A')} ({doc.get('type', 'unknown')})\n"

    prompt += "\nResults:\n"

    for i, result in enumerate(search_results, 1):
        prompt += f"\n{i}. {result.get('title', 'N/A')}\n"
        prompt += f"   {result.get('url', 'N/A')}\n"

    prompt += """
For each result, identify which document(s) it's most relevant to and provide a score.

Respond with JSON mapping each result to its most relevant document with scores."""

    return prompt


# Example of expected response format
EXAMPLE_RESPONSE = {
    "scored_results": [
        {
            "url": "https://www.bcn.cl/leychile/navegar?idNorma=141599",
            "relevance_score": 10.0,
            "reasoning": "This is the official Chilean legislative database (BCN) with the exact law number. Highest authority source for the full official text.",
            "is_likely_official": True,
            "confidence": "high",
            "document_type": "full_text"
        },
        {
            "url": "https://www.consejotransparencia.cl/ley-19-628/",
            "relevance_score": 8.5,
            "reasoning": "Official Chilean transparency council website discussing the law. Authoritative but may be a summary rather than full text.",
            "is_likely_official": True,
            "confidence": "high",
            "document_type": "summary"
        },
        {
            "url": "https://www.example.com/chile-privacy-law",
            "relevance_score": 5.0,
            "reasoning": "General information page about Chilean privacy law. Not official source, likely secondary analysis.",
            "is_likely_official": False,
            "confidence": "medium",
            "document_type": "analysis"
        }
    ],
    "summary": {
        "total_results": 3,
        "highly_relevant_count": 2,
        "official_source_count": 2,
        "recommended_top_result": "https://www.bcn.cl/leychile/navegar?idNorma=141599"
    }
}
