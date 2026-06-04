"""
Query generator using LLM.

This module uses an LLM to generate targeted search queries for finding
specific legal documents.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from loguru import logger

from src.clients.llm_client import LLMClient
from src.config.criteria import CRITERION_CORE_QUESTIONS, TRUSTED_DOMAINS_BY_CRITERION
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
        queries_per_document: int = 5,
        cache_dir: Optional[Path] = None,
    ):
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.queries_per_document = queries_per_document
        self._cache_dir: Optional[Path] = None

        if cache_dir is not None:
            self._cache_dir = Path(cache_dir) / "queries"
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Query cache enabled — dir: {self._cache_dir}")

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

        if self._cache_dir is not None:
            key = self._query_cache_key(document.official_name, country.iso_code)
            cached = self._load_query_cache(key)
            if cached is not None:
                logger.info(f"[CACHE HIT] Queries for '{document.official_name}' ({len(cached)} queries)")
                return cached

        # Look up criterion context from the document's first criterion id
        criterion_number = None
        criterion_core_question = None
        criterion_trusted_domains = None
        if document.criteria_ids:
            cid = document.criteria_ids[0]
            criterion_number = cid
            criterion_core_question = CRITERION_CORE_QUESTIONS.get(cid)
            domains = TRUSTED_DOMAINS_BY_CRITERION.get(cid) or []
            criterion_trusted_domains = domains if domains else None

        # Create prompt
        prompt = create_query_generation_prompt(
            document_name=document.official_name,
            document_type=document.document_type,
            country_name=country.name,
            government_domains=country.government_domains,
            language=document.expected_language,
            alternate_names=document.alternate_names,
            known_sources=known_sources,
            criterion_number=criterion_number,
            criterion_core_question=criterion_core_question,
            trusted_domains=criterion_trusted_domains,
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
                    file_type_hint = ""
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

            if self._cache_dir is not None and queries:
                self._save_query_cache(key, document.official_name, country.iso_code, queries)

            return queries

        except ValueError as e:
            logger.error(f"Invalid LLM response: {e}")
            raise

        except Exception as e:
            logger.error(f"Error generating queries: {e}")
            raise

    def _query_cache_key(self, doc_name: str, country_iso: str) -> str:
        raw = f"{doc_name}|{country_iso}|{self.queries_per_document}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def _load_query_cache(self, key: str) -> Optional[List[SearchQuery]]:
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return [SearchQuery(**q) for q in data["queries"]]
        except Exception as e:
            logger.warning(f"Failed to read query cache {path}: {e}")
            return None

    def _save_query_cache(
        self, key: str, doc_name: str, country_iso: str, queries: List[SearchQuery]
    ) -> None:
        path = self._cache_dir / f"{key}.json"
        payload = {
            "document_name": doc_name,
            "country_iso": country_iso,
            "queries_per_document": self.queries_per_document,
            "cached_at": datetime.now().isoformat(),
            "queries": [q.model_dump() for q in queries],
        }
        try:
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.debug(f"Cached {len(queries)} queries → {path.name}")
        except Exception as e:
            logger.warning(f"Failed to write query cache {path}: {e}")

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
                    file_type_hint="",
                    priority=query_data.get("priority", 5)
                )
                queries.append(query)

            logger.info(f"Generated {len(queries)} multilingual queries")
            return queries

        except Exception as e:
            logger.error(f"Error generating multilingual queries: {e}")
            raise
