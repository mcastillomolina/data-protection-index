"""
Search executor for executing web searches.

This module executes search queries using a SearchClient and aggregates results.
"""

from typing import List, Dict
from loguru import logger
from tqdm import tqdm

from src.clients.search_client import SearchClient
from src.models.document import SearchQuery, SearchResult


class SearchExecutor:
    """
    Executes web searches and aggregates results.

    This class takes a list of search queries and executes them using
    a SearchClient, handling deduplication and progress tracking.
    """

    def __init__(
        self,
        search_client: SearchClient,
        max_results_per_query: int = 10,
        enable_deduplication: bool = True,
        show_progress: bool = True
    ):
        """
        Initialize search executor.

        Args:
            search_client: SearchClient instance
            max_results_per_query: Maximum results to retrieve per query
            enable_deduplication: Whether to deduplicate URLs across queries
            show_progress: Whether to show progress bar
        """
        self.search_client = search_client
        self.max_results_per_query = max_results_per_query
        self.enable_deduplication = enable_deduplication
        self.show_progress = show_progress
        logger.info(
            f"Initialized SearchExecutor "
            f"(max_results={max_results_per_query}, dedup={enable_deduplication})"
        )

    def execute_searches(
        self,
        queries: List[SearchQuery],
        country_code: str = None,
        language: str = None
    ) -> List[SearchResult]:
        """
        Execute all search queries and aggregate results.

        Args:
            queries: List of SearchQuery objects to execute
            country_code: Optional country code for localized search
            language: Optional language code for search results

        Returns:
            List of SearchResult objects (deduplicated if enabled)
        """
        logger.info(f"Executing {len(queries)} search queries")

        all_results = []

        # Use tqdm for progress if enabled
        iterator = tqdm(queries, desc="Searching") if self.show_progress else queries

        for query in iterator:
            try:
                # Build search query string
                search_string = self._build_search_string(query)

                logger.debug(f"Executing search: {search_string[:100]}...")

                # Execute search
                raw_results = self.search_client.search(
                    query=search_string,
                    num_results=self.max_results_per_query,
                    country=country_code,
                    language=language
                )

                # Convert to SearchResult objects
                for raw_result in raw_results:
                    result = SearchResult(
                        url=raw_result["url"],
                        title=raw_result["title"],
                        snippet=raw_result["snippet"],
                        source_domain=raw_result["domain"],
                        query_used=search_string,
                        search_engine="serpapi",  # From search_client
                    )
                    all_results.append(result)

                logger.debug(f"Found {len(raw_results)} results for query")

            except Exception as e:
                logger.error(f"Search failed for query '{query.query_string}': {e}")
                continue

        # Deduplicate if enabled
        if self.enable_deduplication:
            all_results = self._deduplicate_results(all_results)

        logger.info(f"Total results collected: {len(all_results)}")

        return all_results

    def execute_searches_by_document(
        self,
        queries_by_doc: Dict[str, List[SearchQuery]],
        country_code: str = None,
        language: str = None
    ) -> Dict[str, List[SearchResult]]:
        """
        Execute searches grouped by document.

        Args:
            queries_by_doc: Dictionary mapping document_id -> list of queries
            country_code: Optional country code for localized search
            language: Optional language code for search results

        Returns:
            Dictionary mapping document_id -> list of SearchResult objects
        """
        logger.info(f"Executing searches for {len(queries_by_doc)} documents")

        results_by_doc = {}

        for doc_id, queries in queries_by_doc.items():
            logger.info(f"Searching for document: {doc_id} ({len(queries)} queries)")

            # Reset deduplication for each document if enabled
            if self.enable_deduplication:
                self.search_client.reset_deduplication()

            results = self.execute_searches(queries, country_code, language)
            results_by_doc[doc_id] = results

            logger.info(f"Found {len(results)} results for '{doc_id}'")

        return results_by_doc

    def _build_search_string(self, query: SearchQuery) -> str:
        """
        Build the final search string from a SearchQuery.

        Combines the base query with site restrictions and file type hints.

        Args:
            query: SearchQuery object

        Returns:
            Complete search query string
        """
        search_parts = [query.query_string]

        # Add site restrictions
        for site in query.site_restrictions:
            if not site.startswith("site:"):
                site = f"site:{site}"
            search_parts.append(site)

        # Add file type hint if present
        if query.file_type_hint:
            search_parts.append(f"filetype:{query.file_type_hint}")

        search_string = " ".join(search_parts)

        return search_string

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Remove duplicate results based on URL.

        Args:
            results: List of SearchResult objects

        Returns:
            Deduplicated list of SearchResult objects
        """
        seen_urls = set()
        unique_results = []

        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        removed = len(results) - len(unique_results)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate URLs")

        return unique_results

    def get_results_summary(self, results: List[SearchResult]) -> Dict:
        """
        Get a summary of search results.

        Args:
            results: List of SearchResult objects

        Returns:
            Dictionary with summary statistics
        """
        domains = {}
        for result in results:
            domain = result.source_domain
            domains[domain] = domains.get(domain, 0) + 1

        summary = {
            "total_results": len(results),
            "unique_domains": len(domains),
            "top_domains": sorted(
                domains.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "domains_breakdown": domains
        }

        return summary

    def filter_by_domain(
        self,
        results: List[SearchResult],
        domains: List[str]
    ) -> List[SearchResult]:
        """
        Filter results to only specific domains.

        Args:
            results: List of SearchResult objects
            domains: List of domain names to include

        Returns:
            Filtered list of SearchResult objects
        """
        filtered = [
            result for result in results
            if any(domain in result.source_domain for domain in domains)
        ]

        logger.info(
            f"Filtered to {len(filtered)}/{len(results)} results "
            f"from domains: {domains}"
        )

        return filtered

    def get_official_results(
        self,
        results: List[SearchResult],
        government_domains: List[str]
    ) -> List[SearchResult]:
        """
        Filter results to only official government sources.

        Args:
            results: List of SearchResult objects
            government_domains: List of government domain extensions (e.g., [".gob.cl"])

        Returns:
            List of SearchResult objects from official sources
        """
        official = [
            result for result in results
            if any(
                gov_domain in result.source_domain
                for gov_domain in government_domains
            )
        ]

        logger.info(
            f"Found {len(official)}/{len(results)} results "
            f"from official government sources"
        )

        return official
