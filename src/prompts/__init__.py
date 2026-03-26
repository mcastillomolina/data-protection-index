"""
Prompt templates for LLM interactions.

This package contains system and user prompts for document identification,
query generation, and relevance scoring.
"""

from . import document_identification
from . import query_generation
from . import relevance_scoring

# Re-export commonly used items
from .document_identification import (
    SYSTEM_PROMPT as DOC_ID_SYSTEM_PROMPT,
    DOCUMENT_IDENTIFICATION_SCHEMA,
    create_identification_prompt,
    create_simple_identification_prompt,
)

from .query_generation import (
    SYSTEM_PROMPT as QUERY_GEN_SYSTEM_PROMPT,
    QUERY_GENERATION_SCHEMA,
    create_query_generation_prompt,
    create_simple_query_generation_prompt,
    create_multilingual_query_prompt,
)

from .relevance_scoring import (
    SYSTEM_PROMPT as RELEVANCE_SYSTEM_PROMPT,
    RELEVANCE_SCORING_SCHEMA,
    create_relevance_scoring_prompt,
    create_simple_relevance_prompt,
    create_comparative_scoring_prompt,
    create_batch_scoring_prompt,
)

__all__ = [
    # Modules
    "document_identification",
    "query_generation",
    "relevance_scoring",
    # Document Identification
    "DOC_ID_SYSTEM_PROMPT",
    "DOCUMENT_IDENTIFICATION_SCHEMA",
    "create_identification_prompt",
    "create_simple_identification_prompt",
    # Query Generation
    "QUERY_GEN_SYSTEM_PROMPT",
    "QUERY_GENERATION_SCHEMA",
    "create_query_generation_prompt",
    "create_simple_query_generation_prompt",
    "create_multilingual_query_prompt",
    # Relevance Scoring
    "RELEVANCE_SYSTEM_PROMPT",
    "RELEVANCE_SCORING_SCHEMA",
    "create_relevance_scoring_prompt",
    "create_simple_relevance_prompt",
    "create_comparative_scoring_prompt",
    "create_batch_scoring_prompt",
]
