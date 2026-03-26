#!/usr/bin/env python3
"""
Test script for core pipeline components.

This script tests the complete document discovery pipeline with real API calls.
Requires API keys in .env file.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core import DocumentIdentifier, QueryGenerator, SearchExecutor, RelevanceFilter
from src.models.country import Country
from src.utils.config import Config
from src.utils.logger import setup_logger
import os


def create_test_country() -> Country:
    """Create a test Country object for Chile."""
    return Country(
        name="Chile",
        iso_code="CL",
        official_languages=["es"],
        government_domains=[".gob.cl", ".cl"],
        region="Latin America",
        metadata={
            "known_documents": {
                "data_protection_law": "Ley 19.628"
            }
        }
    )


def test_document_identifier():
    """Test DocumentIdentifier component."""
    print("\n" + "="*60)
    print("Testing DocumentIdentifier")
    print("="*60)

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("GROQ_API_KEY"):
        print("⚠️  No LLM API keys found, skipping test")
        return None

    try:
        config = Config()
        llm_client = config.get_llm_client()

        identifier = DocumentIdentifier(llm_client, temperature=0.3)
        country = create_test_country()

        print(f"\n1. Identifying documents for {country.name}...")
        documents = identifier.identify_documents(country)

        print(f"\n📄 Found {len(documents)} documents:")
        for i, doc in enumerate(documents, 1):
            print(f"\n  {i}. {doc.official_name}")
            print(f"     Type: {doc.document_type}")
            print(f"     Priority: {doc.priority_score}/10")
            print(f"     Language: {doc.expected_language}")
            if doc.alternate_names:
                print(f"     Alternate names: {', '.join(doc.alternate_names)}")

        # Test high-priority filter
        high_priority = identifier.get_high_priority_documents(country, min_priority=8)
        print(f"\n📌 High priority documents (>=8): {len(high_priority)}")

        print("\n✅ DocumentIdentifier test passed!")
        return documents

    except Exception as e:
        print(f"\n❌ DocumentIdentifier test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_query_generator(documents):
    """Test QueryGenerator component."""
    print("\n" + "="*60)
    print("Testing QueryGenerator")
    print("="*60)

    if not documents:
        print("⚠️  No documents from previous test, skipping")
        return None

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("GROQ_API_KEY"):
        print("⚠️  No LLM API keys found, skipping test")
        return None

    try:
        config = Config()
        llm_client = config.get_llm_client()

        generator = QueryGenerator(llm_client, temperature=0.5, queries_per_document=3)
        country = create_test_country()

        # Test with first document
        doc = documents[0]
        print(f"\n1. Generating queries for: {doc.official_name}")

        queries = generator.generate_queries(
            document=doc,
            country=country,
            known_sources=["bcn.cl", "consejotransparencia.cl"]
        )

        print(f"\n🔍 Generated {len(queries)} queries:")
        for i, query in enumerate(queries, 1):
            print(f"\n  {i}. {query.query_string}")
            print(f"     Priority: {query.priority}/10")
            if query.site_restrictions:
                print(f"     Site restrictions: {', '.join(query.site_restrictions)}")
            if query.file_type_hint:
                print(f"     File type: {query.file_type_hint}")

        print("\n✅ QueryGenerator test passed!")
        return queries

    except Exception as e:
        print(f"\n❌ QueryGenerator test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_search_executor(queries):
    """Test SearchExecutor component."""
    print("\n" + "="*60)
    print("Testing SearchExecutor")
    print("="*60)

    if not queries:
        print("⚠️  No queries from previous test, skipping")
        return None

    if not os.getenv("SERPAPI_KEY"):
        print("⚠️  No SERPAPI_KEY found, skipping test")
        return None

    try:
        config = Config()
        search_client = config.get_search_client()

        executor = SearchExecutor(
            search_client,
            max_results_per_query=5,
            enable_deduplication=True,
            show_progress=True
        )

        print(f"\n1. Executing {len(queries)} search queries...")
        results = executor.execute_searches(
            queries=queries[:2],  # Limit to 2 queries for testing
            country_code="cl",
            language="es"
        )

        print(f"\n🌐 Found {len(results)} search results:")
        for i, result in enumerate(results[:5], 1):  # Show first 5
            print(f"\n  {i}. {result.title}")
            print(f"     URL: {result.url}")
            print(f"     Domain: {result.source_domain}")
            print(f"     Query: {result.query_used[:60]}...")

        # Test summary
        summary = executor.get_results_summary(results)
        print(f"\n📊 Summary:")
        print(f"   Total results: {summary['total_results']}")
        print(f"   Unique domains: {summary['unique_domains']}")
        print(f"   Top domains: {summary['top_domains'][:3]}")

        print("\n✅ SearchExecutor test passed!")
        return results

    except Exception as e:
        print(f"\n❌ SearchExecutor test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_relevance_filter(documents, results):
    """Test RelevanceFilter component."""
    print("\n" + "="*60)
    print("Testing RelevanceFilter")
    print("="*60)

    if not documents or not results:
        print("⚠️  Missing documents or results from previous tests, skipping")
        return None

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("GROQ_API_KEY"):
        print("⚠️  No LLM API keys found, skipping test")
        return None

    try:
        config = Config()
        llm_client = config.get_llm_client()

        filter_obj = RelevanceFilter(
            llm_client,
            temperature=0.2,
            min_relevance_score=6.0
        )

        doc = documents[0]
        country = create_test_country()

        print(f"\n1. Scoring {len(results)} results for: {doc.official_name}")

        scored_results = filter_obj.filter_results(
            document=doc,
            results=results,
            country_name=country.name,
            top_n=5
        )

        print(f"\n⭐ Top {len(scored_results)} relevant results:")
        for i, scored in enumerate(scored_results, 1):
            result = scored.search_result
            print(f"\n  {i}. {result.title[:60]}...")
            print(f"     Score: {scored.relevance_score:.1f}/10")
            print(f"     Official: {scored.is_likely_official}")
            print(f"     Confidence: {scored.confidence}")
            print(f"     URL: {result.url}")
            print(f"     Reasoning: {scored.reasoning[:100]}...")

        # Test summary
        summary = filter_obj.get_scoring_summary(scored_results)
        print(f"\n📊 Scoring Summary:")
        print(f"   Average score: {summary['avg_score']:.2f}")
        print(f"   Official sources: {summary['official_count']}")
        print(f"   High confidence: {summary['high_confidence_count']}")

        print("\n✅ RelevanceFilter test passed!")
        return scored_results

    except Exception as e:
        print(f"\n❌ RelevanceFilter test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_full_pipeline():
    """Test the complete pipeline end-to-end."""
    print("\n" + "="*60)
    print("FULL PIPELINE TEST")
    print("="*60)

    # Check all required API keys
    has_llm = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GROQ_API_KEY")
    has_search = os.getenv("SERPAPI_KEY")

    if not has_llm:
        print("\n⚠️  No LLM API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return

    if not has_search:
        print("\n⚠️  No SERPAPI_KEY found. Search tests will be skipped.")

    try:
        config = Config()
        country = create_test_country()

        print(f"\n🌍 Running full pipeline for: {country.name}\n")

        # Step 1: Identify documents
        print("Step 1/4: Identifying documents...")
        llm_client = config.get_llm_client()
        identifier = DocumentIdentifier(llm_client)
        documents = identifier.identify_documents(country)
        print(f"✓ Found {len(documents)} documents")

        if not documents:
            print("No documents found, stopping pipeline")
            return

        # Use first document for testing
        doc = documents[0]
        print(f"\nFocusing on: {doc.official_name}")

        # Step 2: Generate queries
        print("\nStep 2/4: Generating search queries...")
        generator = QueryGenerator(llm_client, queries_per_document=3)
        queries = generator.generate_queries(doc, country)
        print(f"✓ Generated {len(queries)} queries")

        if not has_search:
            print("\n⚠️  Stopping here (no SERPAPI_KEY)")
            return

        # Step 3: Execute searches
        print("\nStep 3/4: Executing searches...")
        search_client = config.get_search_client()
        executor = SearchExecutor(search_client, max_results_per_query=5)
        results = executor.execute_searches(queries[:2], country_code="cl")
        print(f"✓ Found {len(results)} results")

        if not results:
            print("No search results found, stopping pipeline")
            return

        # Step 4: Filter by relevance
        print("\nStep 4/4: Filtering by relevance...")
        filter_obj = RelevanceFilter(llm_client, min_relevance_score=6.0)
        scored_results = filter_obj.filter_results(
            doc, results, country.name, top_n=3
        )
        print(f"✓ Top {len(scored_results)} relevant results")

        # Final summary
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"\n📋 Summary:")
        print(f"   Documents identified: {len(documents)}")
        print(f"   Queries generated: {len(queries)}")
        print(f"   Search results: {len(results)}")
        print(f"   Highly relevant results: {len(scored_results)}")

        if scored_results:
            best = scored_results[0]
            print(f"\n🏆 Best result:")
            print(f"   Title: {best.search_result.title}")
            print(f"   URL: {best.search_result.url}")
            print(f"   Score: {best.relevance_score:.1f}/10")
            print(f"   Official: {best.is_likely_official}")

        print("\n✅ Full pipeline test passed!")

    except Exception as e:
        print(f"\n❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("CORE PIPELINE TESTS")
    print("="*60)

    # Setup logger
    setup_logger(level="INFO")

    # Run individual component tests
    documents = test_document_identifier()
    queries = test_query_generator(documents)
    results = test_search_executor(queries)
    scored_results = test_relevance_filter(documents, results)

    # Run full pipeline test
    test_full_pipeline()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
