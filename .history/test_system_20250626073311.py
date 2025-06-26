#!/usr/bin/env python3
"""
Quick test script to verify all components are working
"""

import os
import sys
from pathlib import Path

# Set up path
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)
sys.path.insert(0, str(script_dir))

def test_llm_connectivity():
    """Test LLM connectivity with qwen3 model."""
    print("🧪 Testing LLM connectivity...")
    
    try:
        from llm_provider.unified_client import llm_client
        
        # Test simple generation
        test_messages = [{"role": "user", "content": "Say 'Hello, Prometheus 2.0!' and explain what you are in one sentence."}]
        
        print(f"📡 Using model: {llm_client.default_model}")
        print("🤖 Generating response...")
        
        response = llm_client.generate(test_messages, temperature=0.7)
        
        print(f"✅ LLM Response from {response.provider}:")
        print(f"   Content: {response.content}")
        print(f"   Model: {response.model}")
        print(f"   Tokens: {response.tokens_used}")
        print(f"   Time: {response.response_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return False

def test_components():
    """Test core components."""
    print("\n🔧 Testing core components...")
    
    try:
        # Test TUI
        from framework.tui import tui
        print("✅ TUI imported successfully")
        
        # Test agent
        from agent.agent_core import PrometheusAgent
        print("✅ Agent imported successfully")
        
        # Test evaluator
        from evaluation.swe_bench_harness import SWEBenchHarness
        evaluator = SWEBenchHarness()
        print("✅ Evaluator imported successfully")
        
        # Test orchestrator
        from main import PrometheusOrchestrator
        print("✅ Orchestrator imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def test_agent_creation():
    """Test agent creation and basic functionality."""
    print("\n🤖 Testing agent creation...")
    
    try:
        from agent.agent_core import PrometheusAgent
        from config import config
        
        agent = PrometheusAgent(project_root=config.project_root)
        print(f"✅ Created agent: {agent.agent_id}")
        
        # Test basic methods
        if hasattr(agent, 'self_reflect_and_improve'):
            print("✅ Agent has self-reflection capability")
        
        if hasattr(agent, '_analyze_action_patterns'):
            print("✅ Agent has pattern analysis capability")
        
        if hasattr(agent, '_design_and_implement_tool'):
            print("✅ Agent has tool creation capability")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent creation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔥 Prometheus 2.0 - System Test")
    print("=" * 50)
    
    tests = [
        ("LLM Connectivity", test_llm_connectivity),
        ("Core Components", test_components),
        ("Agent Creation", test_agent_creation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        print("\nTo run the system:")
        print("  GUI: python run_gui.py")
        print("  TUI: python run_tui.py")
        return 0
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
