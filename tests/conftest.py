"""
Pytest fixtures for integration tests in test_core_pipeline.py.

These are session-scoped so each fixture runs once per pytest session,
chaining the pipeline steps the same way the standalone script does.
Fixtures return None when API keys are absent; the tests handle that gracefully.
"""

import os
import pytest

from src.core import DocumentIdentifier, QueryGenerator, SearchExecutor
from src.models.country import Country
from src.utils.config import Config


def _has_llm_key() -> bool:
    return bool(
        os.getenv("OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("GROQ_API_KEY")
    )


@pytest.fixture(scope="session")
def documents():
    if not _has_llm_key():
        return None
    config = Config()
    llm_client = config.get_llm_client()
    country = Country(
        name="Chile",
        iso_code="CL",
        official_languages=["es"],
        government_domains=[".gob.cl", ".cl"],
        region="Latin America",
        metadata={"known_documents": {"data_protection_law": "Ley 19.628"}},
    )
    identifier = DocumentIdentifier(llm_client, temperature=0.3)
    return identifier.identify_documents(country)


@pytest.fixture(scope="session")
def queries(documents):
    if not documents or not _has_llm_key():
        return None
    config = Config()
    llm_client = config.get_llm_client()
    country = Country(
        name="Chile",
        iso_code="CL",
        official_languages=["es"],
        government_domains=[".gob.cl", ".cl"],
        region="Latin America",
        metadata={"known_documents": {"data_protection_law": "Ley 19.628"}},
    )
    generator = QueryGenerator(llm_client, temperature=0.5, queries_per_document=3)
    return generator.generate_queries(
        document=documents[0],
        country=country,
        known_sources=["bcn.cl", "consejotransparencia.cl"],
    )


@pytest.fixture(scope="session")
def results(queries):
    if not queries or not os.getenv("SERPAPI_KEY"):
        return None
    config = Config()
    search_client = config.get_search_client()
    executor = SearchExecutor(
        search_client,
        max_results_per_query=5,
        enable_deduplication=True,
        show_progress=False,
    )
    return executor.execute_searches(queries[:2], country_code="cl", language="es")
