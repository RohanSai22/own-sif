"""
Test Web Search Functionality
Tests if the web search component is working correctly.
"""

import sys
import os
import io

# Force UTF-8 encoding for Windows
if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from tools.base_tools import ToolManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_web_search():
    """Test web search functionality."""
    print("🔍 Testing Web Search Functionality...")
    print("=" * 50)
    
    try:
        # Initialize tool manager
        tool_manager = ToolManager(".")
        
        # Test 1: Basic web search
        print("\n1. Testing basic web search...")
        search_query = "Python programming tutorial"
        results = tool_manager.execute_tool("web_search", search_query)
        
        if results and len(results) > 0:
            print(f"✅ SUCCESS: Found {len(results)} results for '{search_query}'")
            for i, result in enumerate(results[:3]):  # Show first 3 results
                print(f"   {i+1}. {result.get('title', 'No title')}")
                print(f"      URL: {result.get('href', 'No URL')}")
                print(f"      Preview: {result.get('body', 'No preview')[:100]}...")
        else:
            print("❌ FAILED: No results returned")
            return False
        
        # Test 2: Technical search
        print("\n2. Testing technical search...")
        tech_query = "UnicodeDecodeError Python subprocess fix"
        tech_results = tool_manager.execute_tool("web_search", tech_query)
        
        if tech_results and len(tech_results) > 0:
            print(f"✅ SUCCESS: Found {len(tech_results)} technical results")
        else:
            print("❌ FAILED: No technical results returned")
            return False
        
        # Test 3: Error handling
        print("\n3. Testing error handling...")
        try:
            empty_results = tool_manager.execute_tool("web_search", "")
            print("⚠️  WARNING: Empty query didn't raise error")
        except Exception as e:
            print(f"✅ SUCCESS: Empty query handled properly: {e}")
        
        print("\n🎉 All web search tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Web search test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_web_search()
    if success:
        print("\n✅ Web search is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ Web search has issues!")
        sys.exit(1)
