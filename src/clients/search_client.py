"""
Search client for web search APIs (SerpAPI, etc.).

This module provides a unified interface for executing web searches through
various search providers, with rate limiting, error handling, and deduplication.
"""

import time
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse

from loguru import logger

try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False
    logger.warning("serpapi package not installed. Install with: pip install google-search-results")


class SearchClient:
    """
    Web search client supporting multiple providers.

    Currently supports:
    - SerpAPI (Google Search API wrapper)

    Features:
    - Rate limiting with configurable delays
    - Automatic retry on failures
    - Result deduplication by URL
    - Error handling and logging
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        rate_limit_delay: float = 1.0,
        max_retries: int = 3,
        timeout: int = 15
    ):
        """
        Initialize search client.

        Args:
            provider: Search provider ("serpapi")
            api_key: API key for the search provider
            rate_limit_delay: Delay between requests in seconds
            max_retries: Maximum retry attempts for failed searches
            timeout: Request timeout in seconds

        Raises:
            ValueError: If provider is unsupported or API key is missing
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.last_request_time = 0
        self._seen_urls: Set[str] = set()

        if not api_key:
            raise ValueError(f"API key required for {provider}")

        if self.provider == "serpapi":
            if not SERPAPI_AVAILABLE:
                raise ImportError(
                    "SerpAPI package not installed. Install with: pip install google-search-results"
                )
        else:
            raise ValueError(f"Unsupported search provider: {provider}")

        logger.info(f"Initialized SearchClient with provider: {provider}")

    def search(
        self,
        query: str,
        num_results: int = 10,
        country: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs
    ) -> List[Dict]:
        """
        Execute a web search query.

        Args:
            query: Search query string
            num_results: Number of results to return (max)
            country: Country code (e.g., "us", "cl") for localized results
            language: Language code (e.g., "en", "es") for results
            **kwargs: Additional provider-specific parameters

        Returns:
            List of search result dictionaries with keys:
                - url: Result URL
                - title: Page title
                - snippet: Text snippet
                - position: Result position (1-indexed)
                - domain: Domain name extracted from URL

        Raises:
            Exception: If search fails after all retries
        """
        # Apply rate limiting
        self._apply_rate_limit()

        for attempt in range(self.max_retries):
            try:
                if self.provider == "serpapi":
                    results = self._search_serpapi(
                        query, num_results, country, language, **kwargs
                    )
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")

                results = self._deduplicate_results(results)
                logger.info(f"Search completed: '{query[:50]}...' returned {len(results)} results")
                return results

            except Exception as e:
                logger.error(f"Search error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    delay = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Search failed after {self.max_retries} attempts")
                    raise

    def _search_serpapi(
        self,
        query: str,
        num_results: int,
        country: Optional[str],
        language: Optional[str],
        **kwargs
    ) -> List[Dict]:
        """
        Execute search using SerpAPI.

        Args:
            query: Search query
            num_results: Number of results
            country: Country code
            language: Language code
            **kwargs: Additional SerpAPI parameters

        Returns:
            List of normalized search results
        """
        # Build search parameters
        params = {
            "q": query,
            "api_key": self.api_key,
            "num": min(num_results, 100),  # SerpAPI max is 100
            "engine": "google",
            **kwargs
        }

        # Add optional parameters
        if country:
            params["gl"] = country  # Geographic location
        if language:
            params["hl"] = language  # Interface language

        # Execute search
        search = GoogleSearch(params)
        data = search.get_dict()

        # Extract and normalize results
        results = []
        organic_results = data.get("organic_results", [])

        for idx, result in enumerate(organic_results[:num_results]):
            url = result.get("link")
            if not url:
                continue

            # Extract domain
            try:
                domain = urlparse(url).netloc
            except Exception:
                domain = ""

            normalized = {
                "url": url,
                "title": result.get("title", ""),
                "snippet": result.get("snippet", ""),
                "position": idx + 1,
                "domain": domain,
            }

            results.append(normalized)

        return results

    def search_multiple(
        self,
        queries: List[str],
        num_results: int = 10,
        **kwargs
    ) -> Dict[str, List[Dict]]:
        """
        Execute multiple search queries.

        Args:
            queries: List of search query strings
            num_results: Number of results per query
            **kwargs: Additional search parameters

        Returns:
            Dictionary mapping query -> list of results
        """
        all_results = {}

        for i, query in enumerate(queries):
            logger.info(f"Executing search {i + 1}/{len(queries)}: {query[:50]}...")

            try:
                results = self.search(query, num_results, **kwargs)
                all_results[query] = results

            except Exception as e:
                logger.error(f"Failed to search '{query}': {e}")
                all_results[query] = []

        return all_results

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """
        Remove duplicate URLs from results.

        Uses internal state to track seen URLs across all searches
        for this client instance.

        Args:
            results: List of search results

        Returns:
            Filtered list with duplicates removed
        """
        unique_results = []

        for result in results:
            url = result.get("url")
            if url and url not in self._seen_urls:
                self._seen_urls.add(url)
                unique_results.append(result)

        if len(results) != len(unique_results):
            logger.debug(
                f"Removed {len(results) - len(unique_results)} duplicate URLs"
            )

        return unique_results

    def _apply_rate_limit(self) -> None:
        """
        Apply rate limiting delay between requests.

        Ensures minimum time between consecutive API calls.
        """
        if self.rate_limit_delay > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)

        self.last_request_time = time.time()

    def reset_deduplication(self) -> None:
        """
        Reset the URL deduplication cache.

        Call this when starting a new batch of searches where
        you want to allow previously seen URLs.
        """
        self._seen_urls.clear()
        logger.debug("Reset URL deduplication cache")

    def get_seen_urls_count(self) -> int:
        """
        Get the number of unique URLs seen.

        Returns:
            Count of unique URLs in deduplication cache
        """
        return len(self._seen_urls)
