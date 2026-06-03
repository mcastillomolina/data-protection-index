"""
Core pipeline components.

Phase 1 — Document Discovery:
- DocumentIdentifier: Identifies relevant documents using LLM
- QueryGenerator: Generates search queries using LLM
- SearchExecutor: Executes searches and collects results
- RelevanceFilter: Scores and filters results using LLM

Phase 2 — Document Retrieval:
- DocumentRetriever: Downloads content from URLs via HTTP
- TextExtractor: Extracts clean text from PDF and HTML content

Phase 3 — Information Extraction:
- LanguageDetector: Detects document language (no LLM)
- SectionSplitter: Splits document into sections (three-tier regex, no LLM)
- InformationExtractor: Extracts structured fields via LLM (one call per section)
"""

from .document_identifier import DocumentIdentifier
from .query_generator import QueryGenerator
from .search_executor import SearchExecutor
from .relevance_filter import RelevanceFilter
from .document_retriever import DocumentRetriever
from .text_extractor import TextExtractor
from .language_detector import LanguageDetector
from .section_splitter import SectionSplitter
from .information_extractor import InformationExtractor

__all__ = [
    "DocumentIdentifier",
    "QueryGenerator",
    "SearchExecutor",
    "RelevanceFilter",
    "DocumentRetriever",
    "TextExtractor",
    "LanguageDetector",
    "SectionSplitter",
    "InformationExtractor",
]
