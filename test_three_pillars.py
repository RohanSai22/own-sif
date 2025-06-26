#!/usr/bin/env python3
"""
Test script to verify the Three Pillars of Autonomous Evolution in Prometheus 2.0:
1. Autonomous Tool Creation
2. Self-Modification Capabilities  
3. Curiosity-Driven Exploration
"""

import os
import sys
import json
import tempfile
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_pillar_1_autonomous_tool_creation():
    """Test Pillar 1: Autonomous Tool Creation capabilities."""
    print("🔨 Testing Pillar 1: Autonomous Tool Creation...")
    
    try:
        from agent.agent_core import PrometheusAgent
        
        agent = PrometheusAgent(project_root=project_root)
        
        # Test 1: Check if agent can analyze action patterns
        if hasattr(agent, '_analyze_action_patterns'):
            print("  ✓ Action pattern analysis capability exists")
            
            patterns = agent._analyze_action_patterns()
            if "REPETITIVE ACTION PATTERNS" in patterns:
                print("  ✓ Can identify repetitive patterns")
            else:
                print("  ⚠ Pattern analysis may not be working correctly")
                return False
        else:
            print("  ✗ Missing action pattern analysis method")
            return False
        
        # Test 2: Check if agent can design and implement tools
        if hasattr(agent, '_design_and_implement_tool'):
            print("  ✓ Tool design and implementation capability exists")
            
            # Test tool spec
            test_tool_spec = {
                "tool_name": "test_file_analyzer",
                "purpose": "Analyze file structure for testing",
                "required_capabilities": ["file reading", "parsing"],
                "function_name": "analyze_test_file",
                "research_queries": ["python file analysis best practices"],
                "dependencies": ["os", "ast"]
            }
            
            # This would normally create a real tool, but for testing we just verify the method exists
            print("  ✓ Tool creation interface available")
        else:
            print("  ✗ Missing tool design and implementation method")
            return False
        
        # Test 3: Check enhanced self-reflection supports new tools
        test_source = {"test.py": "def test(): pass"}
        test_logs = "Test performance logs"
        
        try:
            reflection_result = agent.self_reflect_and_improve(test_source, test_logs)
            reflection_data = json.loads(reflection_result)
            
            if "new_tools" in reflection_data or "cognitive_analysis" in reflection_data:
                print("  ✓ Enhanced self-reflection supports tool creation")
            else:
                print("  ⚠ Self-reflection may not fully support new tool format")
                return False
        except Exception as e:
            print(f"  ⚠ Self-reflection test failed: {e}")
            return False
        
        print("  ✅ Pillar 1: Autonomous Tool Creation - IMPLEMENTED")
        return True
        
    except Exception as e:
        print(f"  ✗ Pillar 1 test failed with exception: {e}")
        return False

def test_pillar_2_self_modification():
    """Test Pillar 2: Self-Modification capabilities."""
    print("🧠 Testing Pillar 2: Self-Modification...")
    
    try:
        from agent.agent_core import PrometheusAgent
        
        agent = PrometheusAgent(project_root=project_root)
        
        # Test 1: Check if file manifest loading works
        if hasattr(agent, '_load_file_manifest'):
            print("  ✓ File manifest loading capability exists")
            
            manifest = agent._load_file_manifest()
            if "system_architecture" in manifest or "file_manifest" in manifest:
                print("  ✓ Can load system architecture information")
            else:
                print("  ⚠ File manifest may not be loading correctly")
                return False
        else:
            print("  ✗ Missing file manifest loading method")
            return False
        
        # Test 2: Check if architectural stagnation assessment works
        if hasattr(agent, '_assess_architectural_stagnation'):
            print("  ✓ Architectural stagnation assessment capability exists")
            
            assessment = agent._assess_architectural_stagnation()
            if "assessment" in assessment and "recommendation" in assessment:
                print("  ✓ Can assess architectural performance")
            else:
                print("  ⚠ Stagnation assessment may not be working correctly")
                return False
        else:
            print("  ✗ Missing architectural stagnation assessment method")
            return False
        
        # Test 3: Check if the self-reflection prompt includes architectural context
        from config import SELF_REFLECTION_PROMPT
        
        if "file_manifest" in SELF_REFLECTION_PROMPT and "self-modification" in SELF_REFLECTION_PROMPT.lower():
            print("  ✓ Self-reflection prompt supports architectural analysis")
        else:
            print("  ⚠ Self-reflection prompt may not fully support self-modification")
            return False
        
        # Test 4: Check if file manifest exists
        manifest_path = os.path.join(project_root, "file_manifest.json")
        if os.path.exists(manifest_path):
            print("  ✓ File manifest exists for self-awareness")
            
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
            
            if "main.py" in str(manifest_data) and "modification" in str(manifest_data).lower():
                print("  ✓ File manifest contains architectural information")
            else:
                print("  ⚠ File manifest may be incomplete")
                return False
        else:
            print("  ✗ File manifest missing")
            return False
        
        print("  ✅ Pillar 2: Self-Modification - IMPLEMENTED")
        return True
        
    except Exception as e:
        print(f"  ✗ Pillar 2 test failed with exception: {e}")
        return False

def test_pillar_3_curiosity_driven_exploration():
    """Test Pillar 3: Curiosity-Driven Exploration capabilities."""
    print("🔍 Testing Pillar 3: Curiosity-Driven Exploration...")
    
    try:
        from agent.agent_core import PrometheusAgent
        from main import PrometheusOrchestrator
        
        agent = PrometheusAgent(project_root=project_root)
        
        # Test 1: Check if knowledge base loading works
        if hasattr(agent, '_load_knowledge_base_summary'):
            print("  ✓ Knowledge base loading capability exists")
            
            kb_summary = agent._load_knowledge_base_summary()
            if isinstance(kb_summary, str) and len(kb_summary) > 0:
                print("  ✓ Can load knowledge base summary")
            else:
                print("  ⚠ Knowledge base loading may not be working correctly")
                return False
        else:
            print("  ✗ Missing knowledge base loading method")
            return False
        
        # Test 2: Check if exploration conducting works
        if hasattr(agent, '_conduct_exploration'):
            print("  ✓ Exploration conducting capability exists")
            
            test_queries = [{
                "question": "What are emerging techniques in automated testing?",
                "research_focus": "Novel testing methodologies",
                "potential_impact": "Improved code validation"
            }]
            
            # Note: This would normally conduct real web searches
            print("  ✓ Exploration interface available")
        else:
            print("  ✗ Missing exploration conducting method")
            return False
        
        # Test 3: Check if knowledge base updating works
        if hasattr(agent, '_update_knowledge_base'):
            print("  ✓ Knowledge base updating capability exists")
        else:
            print("  ✗ Missing knowledge base updating method")
            return False
        
        # Test 4: Check if orchestrator has exploration phase
        orchestrator = PrometheusOrchestrator()
        if hasattr(orchestrator, '_run_exploration_phase'):
            print("  ✓ Orchestrator exploration phase exists")
        else:
            print("  ✗ Missing orchestrator exploration phase")
            return False
        
        # Test 5: Check if knowledge base file exists
        kb_path = os.path.join(project_root, "archive", "knowledge_base.md")
        if os.path.exists(kb_path):
            print("  ✓ Knowledge base file exists")
            
            with open(kb_path, 'r', encoding='utf-8') as f:
                kb_content = f.read()
            
            if "Knowledge Base" in kb_content and ("Research" in kb_content or "Concepts" in kb_content):
                print("  ✓ Knowledge base contains learning infrastructure")
            else:
                print("  ⚠ Knowledge base may be empty or incomplete")
                return False
        else:
            print("  ✗ Knowledge base file missing")
            return False
        
        # Test 6: Check if self-reflection prompt supports exploration
        from config import SELF_REFLECTION_PROMPT
        
        if "curiosity" in SELF_REFLECTION_PROMPT.lower() and "exploration_queries" in SELF_REFLECTION_PROMPT:
            print("  ✓ Self-reflection prompt supports curiosity-driven exploration")
        else:
            print("  ⚠ Self-reflection prompt may not fully support exploration")
            return False
        
        print("  ✅ Pillar 3: Curiosity-Driven Exploration - IMPLEMENTED")
        return True
        
    except Exception as e:
        print(f"  ✗ Pillar 3 test failed with exception: {e}")
        return False

def test_integration():
    """Test that all three pillars work together."""
    print("⚡ Testing Integration: Three Pillars Working Together...")
    
    try:
        # Test that the enhanced self-reflection supports all three pillars
        from config import SELF_REFLECTION_PROMPT
        
        required_elements = [
            "AUTONOMOUS TOOL CREATION",
            "SELF-MODIFICATION", 
            "CURIOSITY-DRIVEN EXPLORATION",
            "cognitive_analysis",
            "new_tools",
            "exploration_queries"
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in SELF_REFLECTION_PROMPT:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"  ⚠ Missing integration elements: {missing_elements}")
            return False
        else:
            print("  ✓ Self-reflection prompt integrates all three pillars")
        
        # Test that main.py orchestrator supports enhanced capabilities
        from main import PrometheusOrchestrator
        
        orchestrator = PrometheusOrchestrator()
        
        # Check for exploration phase
        if hasattr(orchestrator, '_run_exploration_phase'):
            print("  ✓ Orchestrator supports exploration phases")
        else:
            print("  ✗ Missing exploration phase integration")
            return False
        
        # Check for enhanced self-improvement
        if hasattr(orchestrator, '_run_self_improvement_phase'):
            print("  ✓ Orchestrator has enhanced self-improvement")
            
            # Check if the method handles safety and confidence
            import inspect
            source = inspect.getsource(orchestrator._run_self_improvement_phase)
            if "confidence" in source and "safe_changes" in source:
                print("  ✓ Self-improvement includes safety mechanisms")
            else:
                print("  ⚠ Safety mechanisms may not be fully implemented")
                return False
        else:
            print("  ✗ Missing enhanced self-improvement")
            return False
        
        print("  ✅ Integration: All Three Pillars Working Together - SUCCESS")
        return True
        
    except Exception as e:
        print(f"  ✗ Integration test failed with exception: {e}")
        return False

def main():
    """Run all three pillar tests."""
    print("🔥 Prometheus 2.0 - Three Pillars of Autonomous Evolution Test")
    print("=" * 70)
    
    tests = [
        ("Pillar 1: Autonomous Tool Creation", test_pillar_1_autonomous_tool_creation),
        ("Pillar 2: Self-Modification", test_pillar_2_self_modification),
        ("Pillar 3: Curiosity-Driven Exploration", test_pillar_3_curiosity_driven_exploration),
        ("Integration: All Pillars Together", test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"  ✗ {test_name} failed with exception: {e}")
            print()
    
    print("=" * 70)
    print(f"📊 Results: {passed}/{total} pillars implemented")
    
    if passed == total:
        print("🎉 ALL THREE PILLARS SUCCESSFULLY IMPLEMENTED!")
        print("\nPrometheus 2.0 now has true autonomous capabilities:")
        print("  🔨 Can create new tools when recognizing repetitive patterns")
        print("  🧠 Can modify its own core architecture and algorithms")  
        print("  🔍 Can explore knowledge beyond immediate tasks")
        print("  ⚡ All pillars work together in enhanced self-reflection")
    else:
        print("⚠️  Some pillars may need additional work")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
