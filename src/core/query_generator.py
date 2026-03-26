"""
Query generator using LLM.

This module uses an LLM to generate targeted search queries for finding
specific legal documents.
"""

from typing import List, Optional
from loguru import logger

from src.clients.llm_client import LLMClient
from src.models.country import Country
from src.models.document import DocumentMetadata, SearchQuery
from src.prompts.query_generation import (
    SYSTEM_PROMPT,
    QUERY_GENERATION_SCHEMA,
    create_query_generation_prompt,
)


class QueryGenerator:
    """
    Uses LLM to generate search queries for finding documents.

    This class leverages a language model to create optimized search queries
    that are likely to find specific legal documents.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        temperature: float = 0.5,
        max_tokens: int = 1500,
        queries_per_document: int = 5
    ):
        """
        Initialize query generator.

        Args:
            llm_client: LLM client instance
            temperature: Sampling temperature (0.5 recommended for query variety)
            max_tokens: Maximum tokens for LLM response
            queries_per_document: Target number of queries to generate per document
        """
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.queries_per_document = queries_per_document
        logger.info(f"Initialized QueryGenerator (queries_per_doc={queries_per_document})")

    def generate_queries(
        self,
        document: DocumentMetadata,
        country: Country,
        known_sources: Optional[List[str]] = None
    ) -> List[SearchQuery]:
        """
        Generate search queries for a specific document.

        Args:
            document: Document to generate queries for
            country: Country context
            known_sources: Optional list of known authoritative sources

        Returns:
            List of SearchQuery objects

        Raises:
            ValueError: If LLM response is invalid
            Exception: If LLM call fails
        """
        logger.info(
            f"Generating queries for '{document.official_name}' in {country.name}"
        )

        # Create prompt
        prompt = create_query_generation_prompt(
            document_name=document.official_name,
            document_type=document.document_type,
            country_name=country.name,
            government_domains=country.government_domains,
            language=document.expected_language,
            alternate_names=document.alternate_names,
            known_sources=known_sources
        )

        try:
            # Call LLM
            logger.debug(f"Calling LLM with temperature={self.temperature}")
            response = self.llm_client.complete_json(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                schema=QUERY_GENERATION_SCHEMA,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Validate response structure
            if "queries" not in response:
                raise ValueError("LLM response missing 'queries' field")

            queries_data = response["queries"]
            logger.info(f"LLM generated {len(queries_data)} queries")

            # Convert to SearchQuery objects
            queries = []
            for i, query_data in enumerate(queries_data):
                try:
                    # Extract site restrictions
                    site_restrictions = query_data.get("site_restrictions", [])

                    # Determine file type hint from query or restrictions
                    file_type_hint = None
                    query_string = query_data["query_string"]
                    if "PDF" in query_string.upper() or "filetype:pdf" in query_string.lower():
                        file_type_hint = "pdf"
                    elif "HTML" in query_string.upper() or "filetype:html" in query_string.lower():
                        file_type_hint = "html"

                    query = SearchQuery(
                        query_string=query_string,
                        document_id=document.official_name,  # Use official name as ID
                        site_restrictions=site_restrictions,
                        file_type_hint=file_type_hint,
                        priority=query_data.get("priority", 5)
                    )
                    queries.append(query)

                    logger.debug(
                        f"Query {i+1}: '{query.query_string[:60]}...' "
                        f"(priority: {query.priority})"
                    )

                except Exception as e:
                    logger.warning(f"Failed to create SearchQuery: {e}")
                    logger.debug(f"Query data: {query_data}")
                    continue

            if not queries:
                logger.warning("No valid queries created from LLM response")

            # Sort by priority (highest first)
            queries.sort(key=lambda q: q.priority, reverse=True)

            # Limit to target number
            if len(queries) > self.queries_per_document:
                logger.debug(
                    f"Limiting from {len(queries)} to {self.queries_per_document} queries"
                )
                queries = queries[:self.queries_per_document]

            return queries

        except ValueError as e:
            logger.error(f"Invalid LLM response: {e}")
            raise

        except Exception as e:
            logger.error(f"Error generating queries: {e}")
            raise

    def generate_queries_for_multiple(
        self,
        documents: List[DocumentMetadata],
        country: Country,
        known_sources: Optional[List[str]] = None
    ) -> dict[str, List[SearchQuery]]:
        """
        Generate queries for multiple documents.

        Args:
            documents: List of documents to generate queries for
            country: Country context
            known_sources: Optional list of known authoritative sources

        Returns:
            Dictionary mapping document ID -> list of SearchQuery objects
        """
        logger.info(f"Generating queries for {len(documents)} documents")

        all_queries = {}
        for i, document in enumerate(documents, 1):
            logger.info(f"Processing document {i}/{len(documents)}: {document.official_name}")

            try:
                queries = self.generate_queries(document, country, known_sources)
                all_queries[document.official_name] = queries

            except Exception as e:
                logger.error(f"Failed to generate queries for '{document.official_name}': {e}")
                all_queries[document.official_name] = []

        total_queries = sum(len(q) for q in all_queries.values())
        logger.info(f"Generated {total_queries} total queries for {len(documents)} documents")

        return all_queries

    def get_top_queries(
        self,
        document: DocumentMetadata,
        country: Country,
        top_n: int = 3
    ) -> List[SearchQuery]:
        """
        Get only the top N highest-priority queries for a document.

        Args:
            document: Document to generate queries for
            country: Country context
            top_n: Number of top queries to return

        Returns:
            List of top N SearchQuery objects
        """
        all_queries = self.generate_queries(document, country)

        # Already sorted by priority in generate_queries
        top_queries = all_queries[:top_n]

        logger.info(
            f"Returning top {len(top_queries)} queries (from {len(all_queries)} total)"
        )

        return top_queries

    def generate_multilingual_queries(
        self,
        document: DocumentMetadata,
        country: Country,
        additional_languages: List[str]
    ) -> List[SearchQuery]:
        """
        Generate queries in multiple languages.

        This creates queries in both the document's language and additional
        languages (typically English for international databases).

        Args:
            document: Document to generate queries for
            country: Country context
            additional_languages: Additional language codes (e.g., ["en"])

        Returns:
            List of SearchQuery objects in multiple languages
        """
        from src.prompts.query_generation import create_multilingual_query_prompt

        logger.info(
            f"Generating multilingual queries for '{document.official_name}' "
            f"(primary: {document.expected_language}, additional: {additional_languages})"
        )

        prompt = create_multilingual_query_prompt(
            document_name=document.official_name,
            country_name=country.name,
            primary_language=document.expected_language,
            secondary_languages=additional_languages
        )

        try:
            response = self.llm_client.complete_json(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            queries_data = response.get("queries", [])
            queries = []

            for query_data in queries_data:
                query = SearchQuery(
                    query_string=query_data["query_string"],
                    document_id=document.official_name,
                    site_restrictions=query_data.get("site_restrictions", []),
                    file_type_hint=None,
                    priority=query_data.get("priority", 5)
                )
                queries.append(query)

            logger.info(f"Generated {len(queries)} multilingual queries")
            return queries

        except Exception as e:
            logger.error(f"Error generating multilingual queries: {e}")
            raise
