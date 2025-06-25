"""
Test LLM Connectivity and Basic Functionality
Tests if LLM providers are working correctly.
"""

import sys
import os
import io

# Force UTF-8 encoding for Windows
if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from llm_provider.unified_client import llm_client
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_llm_connectivity():
    """Test LLM connectivity and basic functionality."""
    print("🤖 Testing LLM Connectivity...")
    print("=" * 50)
    
    try:
        # Test 1: Provider availability
        print("\n1. Testing provider availability...")
        test_results = llm_client.test_providers()
        
        working_providers = [p for p, r in test_results.items() if r["status"] == "success"]
        if working_providers:
            print(f"✅ SUCCESS: Working providers: {', '.join(working_providers)}")
        else:
            print("❌ FAILED: No working LLM providers")
            return False
        
        # Test 2: Basic text generation
        print("\n2. Testing basic text generation...")
        simple_prompt = [{"role": "user", "content": "Say 'Hello World' in a single line"}]
        
        response = llm_client.generate(simple_prompt)
        
        if response and response.content:
            print(f"✅ SUCCESS: Generated response from {response.provider}")
            print(f"   Response: {response.content.strip()}")
        else:
            print("❌ FAILED: No response generated")
            return False
        
        # Test 3: Code-related generation
        print("\n3. Testing code-related generation...")
        code_prompt = [{"role": "user", "content": "Write a simple Python function that adds two numbers. Return only the function code."}]
        
        code_response = llm_client.generate(code_prompt)
        
        if code_response and code_response.content and "def" in code_response.content:
            print(f"✅ SUCCESS: Generated code response")
            first_line = code_response.content.strip().split('\n')[0]
            print(f"   First line: {first_line}")
        else:
            print("❌ FAILED: No valid code response")
            return False
        
        # Test 4: Different providers (if available)
        print("\n4. Testing provider fallback...")
        if len(working_providers) > 1:
            # Try with specific provider
            for provider in working_providers[:2]:  # Test first 2 providers
                provider_response = llm_client.generate(
                    simple_prompt,
                    model=f"{provider}/default"
                )
                if provider_response:
                    print(f"✅ SUCCESS: {provider} provider working")
                else:
                    print(f"⚠️  WARNING: {provider} provider not responding")
        
        print("\n🎉 All LLM tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: LLM test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_llm_connectivity()
    if success:
        print("\n✅ LLM is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ LLM has issues!")
        sys.exit(1)
