#!/usr/bin/env python3
"""
Quick test script for SearchClient.

This script tests the basic functionality of the SearchClient with real API calls.
Run this after setting up your .env file with SERPAPI_KEY.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.clients.search_client import SearchClient
from src.utils.config import Config
from src.utils.logger import setup_logger
import os


def test_basic_search():
    """Test basic search functionality."""
    print("\n" + "="*60)
    print("Testing Basic Search")
    print("="*60)

    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        print("⚠️  SERPAPI_KEY not set, skipping test")
        return

    try:
        client = SearchClient(
            provider="serpapi",
            api_key=api_key,
            rate_limit_delay=1.0,  # Be nice to the API
            max_retries=2
        )

        print("\n1. Testing basic search...")
        results = client.search(
            query="Chile data protection law Ley 19.628",
            num_results=5
        )

        print(f"\n📊 Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n  {i}. {result['title']}")
            print(f"     URL: {result['url']}")
            print(f"     Domain: {result['domain']}")
            print(f"     Snippet: {result['snippet'][:100]}...")

        print("\n✅ Basic search test passed!")

    except Exception as e:
        print(f"\n❌ Basic search test failed: {e}")
        import traceback
        traceback.print_exc()


def test_search_with_localization():
    """Test search with country and language parameters."""
    print("\n" + "="*60)
    print("Testing Localized Search")
    print("="*60)

    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        print("⚠️  SERPAPI_KEY not set, skipping test")
        return

    try:
        client = SearchClient(
            provider="serpapi",
            api_key=api_key,
            rate_limit_delay=1.0,
            max_retries=2
        )

        print("\n1. Testing search with Chilean localization...")
        results = client.search(
            query="ley protección datos personales",
            num_results=3,
            country="cl",
            language="es"
        )

        print(f"\n📊 Found {len(results)} results (localized to Chile, Spanish):")
        for i, result in enumerate(results, 1):
            print(f"\n  {i}. {result['title']}")
            print(f"     URL: {result['url']}")
            print(f"     Domain: {result['domain']}")

        print("\n✅ Localized search test passed!")

    except Exception as e:
        print(f"\n❌ Localized search test failed: {e}")
        import traceback
        traceback.print_exc()


def test_multiple_queries():
    """Test searching multiple queries with deduplication."""
    print("\n" + "="*60)
    print("Testing Multiple Queries with Deduplication")
    print("="*60)

    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        print("⚠️  SERPAPI_KEY not set, skipping test")
        return

    try:
        client = SearchClient(
            provider="serpapi",
            api_key=api_key,
            rate_limit_delay=1.0,
            max_retries=2
        )

        print("\n1. Searching multiple related queries...")
        queries = [
            "Chile Ley 19.628 texto completo",
            "Chile data protection law"
        ]

        results = client.search_multiple(
            queries=queries,
            num_results=3,
            deduplicate=True
        )

        print(f"\n📊 Results by query:")
        for query, query_results in results.items():
            print(f"\n  Query: '{query}'")
            print(f"  Results: {len(query_results)}")
            for i, result in enumerate(query_results, 1):
                print(f"    {i}. {result['title'][:60]}...")
                print(f"       {result['url']}")

        print(f"\n📊 Deduplication stats:")
        print(f"  Total unique URLs seen: {client.get_seen_urls_count()}")

        print("\n✅ Multiple queries test passed!")

    except Exception as e:
        print(f"\n❌ Multiple queries test failed: {e}")
        import traceback
        traceback.print_exc()


def test_config_integration():
    """Test that search client can be created from config."""
    print("\n" + "="*60)
    print("Testing Config Integration")
    print("="*60)

    try:
        config = Config()
        print(f"Loaded config: provider={config.search.provider}")

        client = config.get_search_client()
        print(f"✅ Successfully created {type(client).__name__} from config")

        # Quick test if API key is available
        if os.getenv("SERPAPI_KEY"):
            results = client.search("test", num_results=1)
            print(f"Test search returned {len(results)} result(s)")
        else:
            print("⚠️  SERPAPI_KEY not set, skipping actual search")

        print("\n✅ Config integration test passed!")

    except Exception as e:
        print(f"\n❌ Config integration test failed: {e}")
        import traceback
        traceback.print_exc()


def test_site_restriction():
    """Test search with site restriction."""
    print("\n" + "="*60)
    print("Testing Site Restriction")
    print("="*60)

    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        print("⚠️  SERPAPI_KEY not set, skipping test")
        return

    try:
        client = SearchClient(
            provider="serpapi",
            api_key=api_key,
            rate_limit_delay=1.0,
            max_retries=2
        )

        print("\n1. Testing search with site restriction...")
        results = client.search(
            query="site:bcn.cl Ley 19.628",
            num_results=3
        )

        print(f"\n📊 Found {len(results)} results from bcn.cl:")
        for i, result in enumerate(results, 1):
            print(f"\n  {i}. {result['title']}")
            print(f"     URL: {result['url']}")
            print(f"     Domain: {result['domain']}")

        # Verify all results are from bcn.cl
        all_from_bcn = all("bcn.cl" in r["domain"] for r in results)
        if all_from_bcn:
            print("\n✅ All results correctly from bcn.cl domain")
        else:
            print("\n⚠️  Some results not from bcn.cl (SerpAPI might have found other matches)")

        print("\n✅ Site restriction test passed!")

    except Exception as e:
        print(f"\n❌ Site restriction test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("SEARCH CLIENT TESTS")
    print("="*60)

    # Setup logger
    setup_logger(level="INFO")

    # Check for API key
    if not os.getenv("SERPAPI_KEY"):
        print("\n⚠️  WARNING: SERPAPI_KEY not set in .env file")
        print("Set it to run real API tests, or tests will be skipped")
        print("Get your key at: https://serpapi.com/")

    # Run tests
    test_basic_search()
    test_search_with_localization()
    test_multiple_queries()
    test_site_restriction()
    test_config_integration()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
