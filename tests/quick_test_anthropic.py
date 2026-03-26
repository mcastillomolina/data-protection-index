#!/usr/bin/env python
"""Quick test to verify Anthropic integration is working."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import Config
from src.clients import AnthropicClient

print("=" * 60)
print("Testing Anthropic Integration")
print("=" * 60)

# Check API key
api_key_present = "ANTHROPIC_API_KEY" in os.environ and os.environ["ANTHROPIC_API_KEY"] != ""
print(f"✓ ANTHROPIC_API_KEY present: {api_key_present}")

if not api_key_present:
    print("❌ ANTHROPIC_API_KEY not found in environment")
    print("Make sure .env file exists and contains ANTHROPIC_API_KEY")
    sys.exit(1)

# Load config
try:
    config = Config(Path("config/config.yaml"))
    print(f"✓ Config loaded")
    print(f"  Provider: {config.llm.provider}")
    print(f"  Model: {config.llm.model}")
except Exception as e:
    print(f"❌ Config load failed: {e}")
    sys.exit(1)

# Get client
try:
    client = config.get_llm_client()
    print(f"✓ LLM client created: {type(client).__name__}")
except Exception as e:
    print(f"❌ Client creation failed: {e}")
    sys.exit(1)

# Test simple completion
try:
    print("\nTesting simple completion...")
    response = client.complete(
        'Say "Hello from Claude!" and nothing else.',
        temperature=0.3,
        max_tokens=50
    )
    print(f"✓ Response received: {response}")
except Exception as e:
    print(f"❌ Completion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test JSON completion
try:
    print("\nTesting JSON completion...")
    response = client.complete_json(
        'Return a JSON object with fields: message="Hello from JSON", status="ok"',
        temperature=0.3,
        max_tokens=100
    )
    print(f"✓ JSON response received: {response}")
except Exception as e:
    print(f"❌ JSON completion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All tests passed! Anthropic integration is working.")
print("=" * 60)
