"""
Unit tests for SearchClient.

Tests the search client functionality with mocked API calls.
"""

import time
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.clients.search_client import SearchClient


class TestSearchClientInitialization:
    """Test SearchClient initialization."""

    def test_init_with_valid_provider(self):
        """Test initialization with valid provider."""
        client = SearchClient(
            provider="serpapi",
            api_key="test_key",
            rate_limit_delay=1.0,
            max_retries=3
        )
        assert client.provider == "serpapi"
        assert client.api_key == "test_key"
        assert client.rate_limit_delay == 1.0
        assert client.max_retries == 3

    def test_init_without_api_key(self):
        """Test initialization fails without API key."""
        with pytest.raises(ValueError, match="API key required"):
            SearchClient(provider="serpapi", api_key="")

    def test_init_with_invalid_provider(self):
        """Test initialization fails with invalid provider."""
        with pytest.raises(ValueError, match="Unsupported search provider"):
            SearchClient(provider="invalid_provider", api_key="test_key")


class TestSearchClientSearch:
    """Test search functionality."""

    @patch('src.clients.search_client.GoogleSearch')
    def test_search_basic(self, mock_google_search):
        """Test basic search with mocked SerpAPI."""
        # Mock SerpAPI response
        mock_instance = MagicMock()
        mock_instance.get_dict.return_value = {
            "organic_results": [
                {
                    "link": "https://example.com/1",
                    "title": "Test Result 1",
                    "snippet": "This is test result 1"
                },
                {
                    "link": "https://example.com/2",
                    "title": "Test Result 2",
                    "snippet": "This is test result 2"
                }
            ]
        }
        mock_google_search.return_value = mock_instance

        client = SearchClient(
            provider="serpapi",
            api_key="test_key",
            rate_limit_delay=0  # No delay for testing
        )

        results = client.search("test query", num_results=10)

        assert len(results) == 2
        assert results[0]["url"] == "https://example.com/1"
        assert results[0]["title"] == "Test Result 1"
        assert results[0]["snippet"] == "This is test result 1"
        assert results[0]["position"] == 1
        assert results[0]["domain"] == "example.com"

    @patch('src.clients.search_client.GoogleSearch')
    def test_search_with_country_and_language(self, mock_google_search):
        """Test search with country and language parameters."""
        mock_instance = MagicMock()
        mock_instance.get_dict.return_value = {"organic_results": []}
        mock_google_search.return_value = mock_instance

        client = SearchClient(
            provider="serpapi",
            api_key="test_key",
            rate_limit_delay=0
        )

        client.search(
            "test query",
            num_results=10,
            country="cl",
            language="es"
        )

        # Check that GoogleSearch was called with correct params
        call_args = mock_google_search.call_args
        params = call_args[0][0]
        assert params["gl"] == "cl"
        assert params["hl"] == "es"

    @patch('src.clients.search_client.GoogleSearch')
    def test_search_limits_results(self, mock_google_search):
        """Test that search respects num_results parameter."""
        mock_instance = MagicMock()
        mock_instance.get_dict.return_value = {
            "organic_results": [
                {"link": f"https://example.com/{i}", "title": f"Result {i}", "snippet": f"Snippet {i}"}
                for i in range(20)
            ]
        }
        mock_google_search.return_value = mock_instance

        client = SearchClient(
            provider="serpapi",
            api_key="test_key",
            rate_limit_delay=0
        )

        results = client.search("test query", num_results=5)

        assert len(results) == 5

    @patch('src.clients.search_client.GoogleSearch')
    def test_search_handles_missing_fields(self, mock_google_search):
        """Test search handles results with missing fields gracefully."""
        mock_instance = MagicMock()
        mock_instance.get_dict.return_value = {
            "organic_results": [
                {
                    "link": "https://example.com/1",
                    # Missing title and snippet
                },
                {
                    # Missing link - should be skipped
                    "title": "Result without link",
                    "snippet": "This has no link"
                }
            ]
        }
        mock_google_search.return_value = mock_instance

        client = SearchClient(
            provider="serpapi",
            api_key="test_key",
            rate_limit_delay=0
        )

        results = client.search("test query")

        assert len(results) == 1
        assert results[0]["url"] == "https://example.com/1"
        assert results[0]["title"] == ""
        assert results[0]["snippet"] == ""

    @patch('src.clients.search_client.GoogleSearch')
    def test_search_retry_on_failure(self, mock_google_search):
        """Test that search retries on failure."""
        mock_instance = MagicMock()
        # First two calls fail, third succeeds
        mock_instance.get_dict.side_effect = [
            Exception("API Error 1"),
            Exception("API Error 2"),
            {"organic_results": [{"link": "https://example.com", "title": "Success", "snippet": "Test"}]}
        ]
        mock_google_search.return_value = mock_instance

        client = SearchClient(
            provider="serpapi",
            api_key="test_key",
            rate_limit_delay=0,
            max_retries=3
        )

        # Should succeed on third try
        results = client.search("test query")
        assert len(results) == 1
        assert results[0]["title"] == "Success"


class TestSearchClientRateLimiting:
    """Test rate limiting functionality."""

    @patch('src.clients.search_client.GoogleSearch')
    def test_rate_limiting_delay(self, mock_google_search):
        """Test that rate limiting adds delay between requests."""
        mock_instance = MagicMock()
        mock_instance.get_dict.return_value = {"organic_results": []}
        mock_google_search.return_value = mock_instance

        client = SearchClient(
            provider="serpapi",
            api_key="test_key",
            rate_limit_delay=0.1  # 100ms delay
        )

        start_time = time.time()
        client.search("query 1")
        client.search("query 2")
        elapsed = time.time() - start_time

        # Should take at least 100ms due to rate limiting
        assert elapsed >= 0.1


class TestSearchClientDeduplication:
    """Test URL deduplication functionality."""

    @patch('src.clients.search_client.GoogleSearch')
    def test_deduplication_removes_duplicates(self, mock_google_search):
        """Test that deduplication removes duplicate URLs."""
        mock_instance = MagicMock()
        mock_google_search.return_value = mock_instance

        client = SearchClient(
            provider="serpapi",
            api_key="test_key",
            rate_limit_delay=0
        )

        # First search
        mock_instance.get_dict.return_value = {
            "organic_results": [
                {"link": "https://example.com/1", "title": "Result 1", "snippet": "Test 1"},
                {"link": "https://example.com/2", "title": "Result 2", "snippet": "Test 2"}
            ]
        }
        results1 = client.search("query 1")
        assert len(results1) == 2

        # Second search with duplicate URL
        mock_instance.get_dict.return_value = {
            "organic_results": [
                {"link": "https://example.com/1", "title": "Result 1 Again", "snippet": "Test 1 Again"},
                {"link": "https://example.com/3", "title": "Result 3", "snippet": "Test 3"}
            ]
        }
        results2 = client._deduplicate_results(
            [
                {"url": "https://example.com/1", "title": "Result 1 Again", "snippet": "Test 1 Again"},
                {"url": "https://example.com/3", "title": "Result 3", "snippet": "Test 3"}
            ]
        )

        # Should only return the new URL
        assert len(results2) == 1
        assert results2[0]["url"] == "https://example.com/3"

    def test_reset_deduplication(self):
        """Test resetting deduplication cache."""
        client = SearchClient(
            provider="serpapi",
            api_key="test_key",
            rate_limit_delay=0
        )

        # Add some URLs to the cache
        client._seen_urls.add("https://example.com/1")
        client._seen_urls.add("https://example.com/2")
        assert client.get_seen_urls_count() == 2

        # Reset
        client.reset_deduplication()
        assert client.get_seen_urls_count() == 0

    @patch('src.clients.search_client.GoogleSearch')
    def test_search_multiple_with_deduplication(self, mock_google_search):
        """Test search_multiple with deduplication enabled."""
        mock_instance = MagicMock()
        mock_google_search.return_value = mock_instance

        # First query returns results
        results_q1 = {
            "organic_results": [
                {"link": "https://example.com/1", "title": "Result 1", "snippet": "Test 1"},
                {"link": "https://example.com/2", "title": "Result 2", "snippet": "Test 2"}
            ]
        }

        # Second query has duplicate
        results_q2 = {
            "organic_results": [
                {"link": "https://example.com/2", "title": "Result 2 Dup", "snippet": "Test 2 Dup"},
                {"link": "https://example.com/3", "title": "Result 3", "snippet": "Test 3"}
            ]
        }

        mock_instance.get_dict.side_effect = [results_q1, results_q2]

        client = SearchClient(
            provider="serpapi",
            api_key="test_key",
            rate_limit_delay=0
        )

        results = client.search_multiple(
            queries=["query 1", "query 2"],
            num_results=10,
            deduplicate=True
        )

        # First query should have 2 results
        assert len(results["query 1"]) == 2

        # Second query should have only 1 result (duplicate removed)
        assert len(results["query 2"]) == 1
        assert results["query 2"][0]["url"] == "https://example.com/3"


class TestSearchClientMultipleQueries:
    """Test multiple query search functionality."""

    @patch('src.clients.search_client.GoogleSearch')
    def test_search_multiple_queries(self, mock_google_search):
        """Test searching multiple queries."""
        mock_instance = MagicMock()
        mock_instance.get_dict.return_value = {
            "organic_results": [
                {"link": "https://example.com/1", "title": "Result", "snippet": "Test"}
            ]
        }
        mock_google_search.return_value = mock_instance

        client = SearchClient(
            provider="serpapi",
            api_key="test_key",
            rate_limit_delay=0
        )

        queries = ["query 1", "query 2", "query 3"]
        results = client.search_multiple(queries, num_results=5, deduplicate=False)

        assert len(results) == 3
        assert "query 1" in results
        assert "query 2" in results
        assert "query 3" in results

    @patch('src.clients.search_client.GoogleSearch')
    def test_search_multiple_handles_failures(self, mock_google_search):
        """Test search_multiple handles individual query failures."""
        mock_instance = MagicMock()
        # First query succeeds, second fails
        mock_instance.get_dict.side_effect = [
            {"organic_results": [{"link": "https://example.com/1", "title": "Result 1", "snippet": "Test 1"}]},
            Exception("API Error")
        ]
        mock_google_search.return_value = mock_instance

        client = SearchClient(
            provider="serpapi",
            api_key="test_key",
            rate_limit_delay=0,
            max_retries=1  # Fail quickly
        )

        results = client.search_multiple(
            queries=["query 1", "query 2"],
            num_results=5,
            deduplicate=False
        )

        # First query should have results
        assert len(results["query 1"]) == 1

        # Second query should have empty list
        assert len(results["query 2"]) == 0
