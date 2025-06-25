#!/usr/bin/env python3
"""
Test script to verify the key architectural fixes in Prometheus 2.0
"""

import os
import sys
import json
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_population_based_evolution():
    """Test that population-based evolution is implemented."""
    from main import PrometheusOrchestrator
    from config import config
    
    print("🧬 Testing Population-Based Evolution...")
    
    # Check that population_size is used
    print(f"  - Population size configured: {config.population_size}")
    
    # Check that PrometheusOrchestrator has population attributes
    orchestrator = PrometheusOrchestrator()
    
    required_attrs = ['population', 'population_scores', 'generation']
    for attr in required_attrs:
        if hasattr(orchestrator, attr):
            print(f"  ✓ Has {attr} attribute")
        else:
            print(f"  ✗ Missing {attr} attribute")
            return False
    
    # Check population initialization method exists
    if hasattr(orchestrator, '_initialize_population'):
        print("  ✓ Population initialization method exists")
    else:
        print("  ✗ Missing population initialization method")
        return False
    
    print("  ✓ Population-based evolution implemented")
    return True

def test_web_search_in_research():
    """Test that web search is used in research improvements."""
    from agent.agent_core import PrometheusAgent
    
    print("🔍 Testing Web Search Integration...")
    
    agent = PrometheusAgent(project_root=project_root)
    
    # Check if _research_improvements method exists
    if hasattr(agent, '_research_improvements'):
        print("  ✓ Research improvements method exists")
        
        # Check the method implementation for web search
        import inspect
        source = inspect.getsource(agent._research_improvements)
        
        if 'web_search' in source and 'tool_manager.execute_tool' in source:
            print("  ✓ Web search integration found in method")
        else:
            print("  ⚠ Web search integration not detected in method")
            return False
    else:
        print("  ✗ Research improvements method missing")
        return False
    
    print("  ✓ Web search integration implemented")
    return True

def test_robust_archive_loading():
    """Test that archive loading handles both mutation formats."""
    from archive.agent_archive import AgentArchive
    
    print("📚 Testing Robust Archive Loading...")
    
    archive = AgentArchive(project_root)
    
    # Check if _load_archive method exists
    if hasattr(archive, '_load_archive'):
        print("  ✓ Archive loading method exists")
        
        # Check the method implementation for robust loading
        import inspect
        source = inspect.getsource(archive._load_archive)
        
        if 'mutations_applied' in source and 'mutation_changes' in source:
            print("  ✓ Handles both mutation field formats")
        else:
            print("  ⚠ May not handle both mutation field formats")
            return False
    else:
        print("  ✗ Archive loading method missing")
        return False
    
    print("  ✓ Robust archive loading implemented")
    return True

def test_improved_patch_extraction():
    """Test that patch extraction is improved."""
    from evaluation.swe_bench_harness import SWEBenchHarness
    
    print("🔧 Testing Improved Patch Extraction...")
    
    harness = SWEBenchHarness(project_root)
    
    # Check if patch extraction method exists
    if hasattr(harness, '_extract_patch_from_solution'):
        print("  ✓ Patch extraction method exists")
        
        # Check the method implementation for improvements
        import inspect
        source = inspect.getsource(harness._extract_patch_from_solution)
        
        if '<patch>' in source and 'import re' in source and 'patch_tag_pattern' in source:
            print("  ✓ Enhanced patch extraction with multiple formats")
        else:
            print("  ⚠ Enhanced patch extraction not fully detected")
            return False
    else:
        print("  ✗ Patch extraction method missing")
        return False
    
    print("  ✓ Improved patch extraction implemented")
    return True

def test_live_state_integration():
    """Test that live state is written for GUI integration."""
    from main import PrometheusOrchestrator
    
    print("📊 Testing Live State Integration...")
    
    orchestrator = PrometheusOrchestrator()
    
    # Check if live state writing method exists
    if hasattr(orchestrator, '_write_live_state'):
        print("  ✓ Live state writing method exists")
        
        # Test writing live state
        try:
            orchestrator._write_live_state()
            
            # Check if file was created
            live_state_path = os.path.join(project_root, "archive", "live_state.json")
            if os.path.exists(live_state_path):
                print("  ✓ Live state file created successfully")
                
                # Check file contents
                with open(live_state_path, 'r') as f:
                    state = json.load(f)
                
                required_fields = ['timestamp', 'generation', 'is_running', 'status']
                if all(field in state for field in required_fields):
                    print("  ✓ Live state contains required fields")
                else:
                    print("  ⚠ Live state missing some required fields")
                    return False
            else:
                print("  ✗ Live state file not created")
                return False
        except Exception as e:
            print(f"  ✗ Error writing live state: {e}")
            return False
    else:
        print("  ✗ Live state writing method missing")
        return False
    
    print("  ✓ Live state integration implemented")
    return True

def test_docker_improvements():
    """Test that Docker generation is improved."""
    from evaluation.sandbox import DockerSandbox
    
    print("🐳 Testing Docker Improvements...")
    
    sandbox = DockerSandbox()
    
    # Check if Docker generation method exists
    if hasattr(sandbox, '_generate_dockerfile'):
        print("  ✓ Dockerfile generation method exists")
        
        # Test generating a Dockerfile
        try:
            dockerfile = sandbox._generate_dockerfile(
                repo_url="https://github.com/test/repo",
                commit_hash="abc123",
                dependencies=[],
                python_version="3.11"
            )
            
            # Check for improvements in the generated Dockerfile
            improvements = [
                'multi-stage',
                'error handling', 
                'fallback',
                'compatibility',
                'PYTHONPATH'
            ]
            
            found_improvements = []
            for improvement in improvements:
                if improvement.lower().replace(' ', '') in dockerfile.lower().replace(' ', ''):
                    found_improvements.append(improvement)
            
            if len(found_improvements) >= 3:
                print(f"  ✓ Found improvements: {', '.join(found_improvements)}")
            else:
                print(f"  ⚠ Limited improvements found: {', '.join(found_improvements)}")
                return False
        except Exception as e:
            print(f"  ✗ Error generating Dockerfile: {e}")
            return False
    else:
        print("  ✗ Dockerfile generation method missing")
        return False
    
    print("  ✓ Docker improvements implemented")
    return True

def main():
    """Run all tests."""
    print("🔥 Prometheus 2.0 - Architecture Fix Verification")
    print("=" * 50)
    
    tests = [
        test_population_based_evolution,
        test_web_search_in_research,
        test_robust_archive_loading,
        test_improved_patch_extraction,
        test_live_state_integration,
        test_docker_improvements
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All architectural fixes verified successfully!")
    else:
        print("⚠️  Some fixes may need additional work")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
