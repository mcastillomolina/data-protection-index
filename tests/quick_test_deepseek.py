#!/usr/bin/env python
"""Quick test to verify DeepSeek integration is working."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clients import DeepSeekClient

print("=" * 60)
print("Testing DeepSeek Integration")
print("=" * 60)

# Check API key
api_key = os.environ.get("DEEPSEEK_API_KEY", "")
print(f"✓ DEEPSEEK_API_KEY present: {bool(api_key)}")

if not api_key:
    print("❌ DEEPSEEK_API_KEY not found in environment")
    print("Add DEEPSEEK_API_KEY=your_key to your .env file")
    sys.exit(1)

# Create client directly (no config change needed for this test)
try:
    client = DeepSeekClient(api_key=api_key, model="deepseek-v4-flash")
    print(f"✓ DeepSeekClient created: model={client.model}")
except Exception as e:
    print(f"❌ Client creation failed: {e}")
    sys.exit(1)

# Test simple completion
try:
    print("\nTesting simple completion...")
    response = client.complete(
        'Say "Hello from DeepSeek!" and nothing else.',
        temperature=0.3,
        max_tokens=50
    )
    print(f"✓ Response: {response}")
except Exception as e:
    print(f"❌ Completion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test JSON completion
try:
    print("\nTesting JSON completion...")
    response = client.complete_json(
        'Return a JSON object with fields: message="Hello from DeepSeek", status="ok"',
        temperature=0.3,
        max_tokens=100
    )
    print(f"✓ JSON response: {response}")
except Exception as e:
    print(f"❌ JSON completion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Show usage
usage = client.get_total_usage()
print(f"\n✓ Total tokens used: {usage.total_tokens}")
print(f"✓ Estimated cost: ${usage.estimated_cost_usd:.6f}")

print("\n" + "=" * 60)
print("✅ All tests passed! DeepSeek integration is working.")
print("=" * 60)
