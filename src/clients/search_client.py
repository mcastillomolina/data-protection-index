"""
Search client for web search APIs (SerpAPI, etc.).

This module provides a unified interface for executing web searches through
various search providers, with rate limiting, error handling, and deduplication.
"""

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
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
        timeout: int = 15,
        enable_caching: bool = False,
        cache_dir: Optional[str] = None,
        cache_ttl_seconds: int = 2592000,
    ):
        self.provider = provider.lower()
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.last_request_time = 0
        self._seen_urls: Set[str] = set()
        self.enable_caching = enable_caching
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache_search_dir: Optional[Path] = None

        if not api_key:
            raise ValueError(f"API key required for {provider}")

        if self.provider == "serpapi":
            if not SERPAPI_AVAILABLE:
                raise ImportError(
                    "SerpAPI package not installed. Install with: pip install google-search-results"
                )
        else:
            raise ValueError(f"Unsupported search provider: {provider}")

        if enable_caching and cache_dir:
            self._cache_search_dir = Path(cache_dir) / "search"
            self._cache_search_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Search cache enabled — dir: {self._cache_search_dir}, TTL: {cache_ttl_seconds}s")

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
        # Check cache before hitting SerpAPI
        if self.enable_caching and self._cache_search_dir is not None:
            key = self._cache_key(query, country, language, num_results)
            cached = self._load_from_cache(key)
            if cached is not None:
                logger.info(f"[CACHE HIT] '{query[:60]}' — {len(cached)} results")
                return self._deduplicate_results(cached)

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

                if self.enable_caching and self._cache_search_dir is not None:
                    self._save_to_cache(key, query, country, language, num_results, results)

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

    def _cache_key(self, query: str, country: Optional[str], language: Optional[str], num_results: int) -> str:
        raw = f"{query}|{country or ''}|{language or ''}|{num_results}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def _cache_path(self, key: str) -> Path:
        return self._cache_search_dir / f"{key}.json"

    def _load_from_cache(self, key: str) -> Optional[List[Dict]]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cached_at = datetime.fromisoformat(data["cached_at"])
            if (datetime.now() - cached_at).total_seconds() > self.cache_ttl_seconds:
                logger.debug(f"Cache expired for key {key}")
                return None
            return data["results"]
        except Exception as e:
            logger.warning(f"Failed to read cache file {path}: {e}")
            return None

    def _save_to_cache(
        self, key: str, query: str, country: Optional[str], language: Optional[str],
        num_results: int, results: List[Dict]
    ) -> None:
        path = self._cache_path(key)
        payload = {
            "query": query,
            "country": country,
            "language": language,
            "num_results": num_results,
            "cached_at": datetime.now().isoformat(),
            "results": results,
        }
        try:
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.debug(f"Cached {len(results)} results → {path.name}")
        except Exception as e:
            logger.warning(f"Failed to write cache file {path}: {e}")

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
        # The google-search-results library defaults SerpApiClient.timeout to 60000
        # and passes it straight to requests.get(timeout=...), which treats it as
        # SECONDS — a ~16.7 hour effective timeout. Our configured self.timeout was
        # never reaching the actual HTTP call because GoogleSearch's constructor
        # doesn't accept it; override the attribute directly after construction.
        # Confirmed live: a single search hung 16m15s waiting on this default.
        search.timeout = self.timeout
        data = search.get_dict()

        # SerpAPI can return HTTP 200 with a soft error (e.g. an invalid or
        # blocked `gl` value) instead of raising. Surface it as an exception
        # so the retry logic sees it, instead of silently caching it as
        # zero organic results indistinguishable from a genuine empty search.
        api_error = data.get("error")
        status = data.get("search_metadata", {}).get("status")
        if api_error or status == "Error":
            raise RuntimeError(f"SerpAPI returned an error (status={status}): {api_error}")

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
