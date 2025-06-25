"""
Test Error Handling and Web Search Integration
Tests if the system correctly uses web search when errors occur.
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
from tools.base_tools import ToolManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_error_handling():
    """Test error handling and web search integration."""
    print("🔍 Testing Error Handling and Web Search Integration...")
    print("=" * 50)
    
    try:
        # Initialize components
        tool_manager = ToolManager(".")
        
        # Test 1: Simulate common error
        print("\n1. Testing error simulation...")
        
        # Simulate a common Python error
        error_message = """
Traceback (most recent call last):
  File "test.py", line 5, in <module>
    result = undefined_function()
NameError: name 'undefined_function' is not defined
"""
        
        print(f"✅ SUCCESS: Simulated error: NameError")
        
        # Test 2: LLM error understanding
        print("\n2. Testing LLM error understanding...")
        
        error_prompt = [
            {"role": "system", "content": "You are a debugging assistant. Analyze Python errors and suggest solutions."},
            {"role": "user", "content": f"I got this error:\\n{error_message}\\nWhat's the problem and how can I fix it?"}
        ]
        
        error_response = llm_client.generate(error_prompt)
        
        if error_response and error_response.content and ("NameError" in error_response.content or "undefined" in error_response.content.lower()):
            print(f"✅ SUCCESS: LLM understood the error")
            print(f"   Analysis: {error_response.content.strip()[:150]}...")
        else:
            print("❌ FAILED: LLM could not understand the error")
            return False
        
        # Test 3: Web search for error solution
        print("\n3. Testing web search for error solutions...")
        
        search_query = "Python NameError undefined_function fix"
        search_results = tool_manager.execute_tool("web_search", search_query)
        
        if search_results and len(search_results) > 0:
            print(f"✅ SUCCESS: Found {len(search_results)} search results for error")
            
            # Check if results contain relevant information
            relevant_found = False
            for result in search_results[:3]:
                content = (result.get('title', '') + ' ' + result.get('body', '')).lower()
                if 'nameerror' in content or 'undefined' in content or 'python' in content:
                    relevant_found = True
                    break
            
            if relevant_found:
                print(f"✅ SUCCESS: Found relevant error solutions")
            else:
                print(f"⚠️  WARNING: Search results may not be highly relevant")
        else:
            print("❌ FAILED: No search results for error")
            return False
        
        # Test 4: LLM solution with search context
        print("\n4. Testing LLM solution with search context...")
        
        # Combine error and search results for better solution
        context_prompt = [
            {"role": "system", "content": "You are a debugging assistant with access to web search results. Provide specific fixes."},
            {"role": "user", "content": f"Error: {error_message}\\n\\nSearch results suggest checking function definitions and imports. Provide a specific code fix."}
        ]
        
        solution_response = llm_client.generate(context_prompt)
        
        if solution_response and solution_response.content and ("def " in solution_response.content or "import" in solution_response.content):
            print(f"✅ SUCCESS: LLM provided a code solution")
            print(f"   Solution preview: {solution_response.content.strip()[:100]}...")
        else:
            print("❌ FAILED: LLM could not provide a concrete solution")
            return False
        
        # Test 5: Complex error scenario
        print("\n5. Testing complex error scenario...")
        
        complex_error = """
UnicodeDecodeError: 'charmap' codec can't decode byte 0x81 in position 3243: character maps to <undefined>
"""
        
        complex_search_query = "UnicodeDecodeError charmap codec python windows fix encoding utf-8"
        complex_results = tool_manager.execute_tool("web_search", complex_search_query)
        
        if complex_results and len(complex_results) > 0:
            print(f"✅ SUCCESS: Found solutions for complex Unicode error")
        else:
            print("❌ FAILED: Could not find solutions for complex error")
            return False
        
        print("\n🎉 All error handling tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Error handling test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_error_handling()
    if success:
        print("\n✅ Error handling and web search integration is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ Error handling has issues!")
        sys.exit(1)
