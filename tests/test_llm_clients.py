#!/usr/bin/env python3
"""
Quick test script for LLM clients.

This script tests the basic functionality of the OpenAI and Anthropic clients.
Run this after setting up your .env file with API keys.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import Config
from src.utils.logger import setup_logger


def test_openai_client():
    """Test OpenAI client."""
    print("\n" + "="*60)
    print("Testing OpenAI Client")
    print("="*60)

    try:
        from src.clients.openai_client import OpenAIClient
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  OPENAI_API_KEY not set, skipping OpenAI test")
            return

        client = OpenAIClient(
            api_key=api_key,
            model="gpt-4o-mini",  # Using cheaper model for testing
            timeout=30,
            max_retries=2
        )

        # Test text completion
        print("\n1. Testing text completion...")
        response = client.complete(
            prompt="Say 'Hello from OpenAI!' and nothing else.",
            temperature=0.3,
            max_tokens=50
        )
        print(f"Response: {response}")

        # Test JSON completion
        print("\n2. Testing JSON completion...")
        json_response = client.complete_json(
            prompt="Return a JSON object with two fields: 'message' (set to 'test') and 'number' (set to 42).",
            temperature=0.1,
            max_tokens=100
        )
        print(f"JSON Response: {json_response}")

        # Print usage stats
        usage = client.get_total_usage()
        print(f"\n📊 Total Usage:")
        print(f"   Tokens: {usage.total_tokens} (prompt: {usage.prompt_tokens}, completion: {usage.completion_tokens})")
        print(f"   Estimated cost: ${usage.estimated_cost_usd:.4f}")

        print("\n✅ OpenAI client test passed!")

    except Exception as e:
        print(f"\n❌ OpenAI client test failed: {e}")
        import traceback
        traceback.print_exc()


def test_anthropic_client():
    """Test Anthropic client."""
    print("\n" + "="*60)
    print("Testing Anthropic Client")
    print("="*60)

    try:
        from src.clients.anthropic_client import AnthropicClient
        import os

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("⚠️  ANTHROPIC_API_KEY not set, skipping Anthropic test")
            return

        client = AnthropicClient(
            api_key=api_key,
            model="claude-3-haiku-20240307",  # Using cheaper model for testing
            timeout=30,
            max_retries=2
        )

        # Test text completion
        print("\n1. Testing text completion...")
        response = client.complete(
            prompt="Say 'Hello from Claude!' and nothing else.",
            temperature=0.3,
            max_tokens=50
        )
        print(f"Response: {response}")

        # Test JSON completion
        print("\n2. Testing JSON completion...")
        json_response = client.complete_json(
            prompt="Return a JSON object with two fields: 'message' (set to 'test') and 'number' (set to 42).",
            temperature=0.1,
            max_tokens=100
        )
        print(f"JSON Response: {json_response}")

        # Print usage stats
        usage = client.get_total_usage()
        print(f"\n📊 Total Usage:")
        print(f"   Tokens: {usage.total_tokens} (prompt: {usage.prompt_tokens}, completion: {usage.completion_tokens})")
        print(f"   Estimated cost: ${usage.estimated_cost_usd:.6f}")

        print("\n✅ Anthropic client test passed!")

    except Exception as e:
        print(f"\n❌ Anthropic client test failed: {e}")
        import traceback
        traceback.print_exc()


def test_config_integration():
    """Test that clients can be created from config."""
    print("\n" + "="*60)
    print("Testing Config Integration")
    print("="*60)

    try:
        config = Config()
        print(f"Loaded config: provider={config.llm.provider}, model={config.llm.model}")

        client = config.get_llm_client()
        print(f"✅ Successfully created {type(client).__name__} from config")

        # Quick test
        response = client.complete(
            prompt="Say 'Config integration works!' and nothing else.",
            temperature=0.3,
            max_tokens=50
        )
        print(f"Response: {response}")

        usage = client.get_total_usage()
        print(f"Cost: ${usage.estimated_cost_usd:.6f}")

        print("\n✅ Config integration test passed!")

    except Exception as e:
        print(f"\n❌ Config integration test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("LLM CLIENT TESTS")
    print("="*60)

    # Setup logger
    setup_logger(level="INFO")

    # Run tests
    test_openai_client()
    test_anthropic_client()
    test_config_integration()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
