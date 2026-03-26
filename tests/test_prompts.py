#!/usr/bin/env python3
"""
Test script for prompt templates.

This script demonstrates how to use the prompt templates and optionally
tests them with real LLM calls if API keys are available.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.prompts import (
    DOC_ID_SYSTEM_PROMPT,
    QUERY_GEN_SYSTEM_PROMPT,
    RELEVANCE_SYSTEM_PROMPT,
    create_identification_prompt,
    create_query_generation_prompt,
    create_relevance_scoring_prompt,
)
from src.utils.logger import setup_logger


def test_document_identification_prompt():
    """Test document identification prompt generation."""
    print("\n" + "="*60)
    print("Document Identification Prompt")
    print("="*60)

    # Create prompt for Chile
    prompt = create_identification_prompt(
        country_name="Chile",
        iso_code="CL",
        official_languages=["es"],
        government_domains=[".gob.cl", ".cl"],
        region="Latin America",
        known_documents={
            "data_protection_law": "Ley 19.628"
        }
    )

    print("\nSystem Prompt (first 300 chars):")
    print("-" * 60)
    print(DOC_ID_SYSTEM_PROMPT[:300] + "...")

    print("\nUser Prompt:")
    print("-" * 60)
    print(prompt)

    print("\n✅ Document identification prompt created successfully")


def test_query_generation_prompt():
    """Test query generation prompt."""
    print("\n" + "="*60)
    print("Query Generation Prompt")
    print("="*60)

    prompt = create_query_generation_prompt(
        document_name="Ley 19.628 sobre Protección de la Vida Privada",
        document_type="data_protection_law",
        country_name="Chile",
        government_domains=[".gob.cl", ".cl"],
        language="es",
        alternate_names=["Ley 19.628", "Ley de Protección de Datos"],
        known_sources=["bcn.cl", "consejotransparencia.cl"]
    )

    print("\nSystem Prompt (first 300 chars):")
    print("-" * 60)
    print(QUERY_GEN_SYSTEM_PROMPT[:300] + "...")

    print("\nUser Prompt:")
    print("-" * 60)
    print(prompt)

    print("\n✅ Query generation prompt created successfully")


def test_relevance_scoring_prompt():
    """Test relevance scoring prompt."""
    print("\n" + "="*60)
    print("Relevance Scoring Prompt")
    print("="*60)

    # Sample search results
    search_results = [
        {
            "url": "https://www.bcn.cl/leychile/navegar?idNorma=141599",
            "title": "Ley 19628 - Protección de la Vida Privada",
            "snippet": "Texto completo de la Ley 19.628 sobre Protección de la Vida Privada...",
            "domain": "bcn.cl"
        },
        {
            "url": "https://www.consejotransparencia.cl/ley-19-628/",
            "title": "Ley 19.628 | Consejo para la Transparencia",
            "snippet": "Información sobre la Ley 19.628 de protección de datos personales en Chile...",
            "domain": "consejotransparencia.cl"
        },
        {
            "url": "https://www.example.com/chile-privacy-laws",
            "title": "Understanding Chile's Privacy Laws",
            "snippet": "A guide to privacy legislation in Chile including the Ley 19.628...",
            "domain": "example.com"
        }
    ]

    prompt = create_relevance_scoring_prompt(
        document_name="Ley 19.628",
        document_type="data_protection_law",
        country_name="Chile",
        search_results=search_results
    )

    print("\nSystem Prompt (first 300 chars):")
    print("-" * 60)
    print(RELEVANCE_SYSTEM_PROMPT[:300] + "...")

    print("\nUser Prompt:")
    print("-" * 60)
    print(prompt)

    print("\n✅ Relevance scoring prompt created successfully")


def test_with_real_llm():
    """Test prompts with real LLM calls (if API keys available)."""
    print("\n" + "="*60)
    print("Testing with Real LLM (Optional)")
    print("="*60)

    import os
    from src.utils.config import Config

    # Check if API keys are available
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  No API keys found. Skipping real LLM tests.")
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env to test with real LLM.")
        return

    try:
        config = Config()
        client = config.get_llm_client()

        print(f"\n1. Testing document identification with {type(client).__name__}...")

        prompt = create_identification_prompt(
            country_name="Chile",
            iso_code="CL",
            official_languages=["es"],
            government_domains=[".gob.cl"],
            region="Latin America"
        )

        response = client.complete_json(
            prompt=prompt,
            system_prompt=DOC_ID_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=2000
        )

        print("\n📄 Response:")
        print(json.dumps(response, indent=2, ensure_ascii=False))

        # Validate response structure
        if "documents" in response and "metadata" in response:
            print(f"\n✅ Valid response structure")
            print(f"   Found {len(response['documents'])} documents")
            print(f"   Metadata: {response['metadata']}")
        else:
            print("\n⚠️  Response missing expected fields")

        # Check usage
        usage = client.get_total_usage()
        print(f"\n💰 Cost: ${usage.estimated_cost_usd:.4f}")

        print("\n✅ Real LLM test completed successfully!")

    except Exception as e:
        print(f"\n❌ Real LLM test failed: {e}")
        import traceback
        traceback.print_exc()


def test_all_prompt_functions():
    """Test all prompt generation functions."""
    print("\n" + "="*60)
    print("Testing All Prompt Functions")
    print("="*60)

    from src.prompts import (
        create_simple_identification_prompt,
        create_simple_query_generation_prompt,
        create_multilingual_query_prompt,
        create_simple_relevance_prompt,
        create_comparative_scoring_prompt
    )

    # Test simple identification
    prompt1 = create_simple_identification_prompt("Germany")
    print("\n✓ Simple identification prompt created")

    # Test simple query generation
    prompt2 = create_simple_query_generation_prompt(
        "Bundesdatenschutzgesetz",
        "Germany",
        [".de", ".bund.de"]
    )
    print("✓ Simple query generation prompt created")

    # Test multilingual query
    prompt3 = create_multilingual_query_prompt(
        "GDPR",
        "European Union",
        "en",
        ["fr", "de", "es"]
    )
    print("✓ Multilingual query prompt created")

    # Test simple relevance
    results = [
        {"url": "https://example.com", "title": "Test", "snippet": "Test snippet"}
    ]
    prompt4 = create_simple_relevance_prompt("Test Document", results)
    print("✓ Simple relevance prompt created")

    # Test comparative scoring
    prompt5 = create_comparative_scoring_prompt("Test Document", results, top_n=3)
    print("✓ Comparative scoring prompt created")

    print("\n✅ All prompt functions working correctly!")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("PROMPT TEMPLATE TESTS")
    print("="*60)

    # Setup logger
    setup_logger(level="INFO")

    # Run tests
    test_document_identification_prompt()
    test_query_generation_prompt()
    test_relevance_scoring_prompt()
    test_all_prompt_functions()
    test_with_real_llm()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
