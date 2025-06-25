"""
Test SWE-bench Problem Solving
Tests if LLM can understand and attempt to solve SWE-bench problems.
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
from evaluation.swe_bench_harness import SWEBenchHarness
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_swe_problem_solving():
    """Test LLM's ability to understand and solve SWE-bench problems."""
    print("� Testing SWE-bench Problem Solving...")
    print("=" * 50)
    
    try:
        # Test 1: Load SWE-bench dataset
        print("\n1. Loading SWE-bench dataset...")
        harness = SWEBenchHarness(".")
        
        if harness.dataset and len(harness.dataset) > 0:
            print(f"✅ SUCCESS: Loaded {len(harness.dataset)} SWE-bench instances")
        else:
            print("❌ FAILED: Could not load SWE-bench dataset")
            return False
        
        # Test 2: Get a sample problem
        print("\n2. Getting sample problem...")
        sample_task = harness.dataset[0]  # Get first task
        
        problem_statement = sample_task.get('problem_statement', '')
        repo = sample_task.get('repo', 'unknown')
        
        if problem_statement:
            print(f"✅ SUCCESS: Got problem from {repo}")
            print(f"   Problem preview: {problem_statement[:100]}...")
        else:
            print("❌ FAILED: No problem statement found")
            return False
        
        # Test 3: LLM problem understanding
        print("\n3. Testing LLM problem understanding...")
        understanding_prompt = [
            {"role": "system", "content": "You are a software engineer. Analyze the given problem and provide a brief summary."},
            {"role": "user", "content": f"Problem: {problem_statement[:500]}...\\n\\nProvide a brief summary of what needs to be fixed."}
        ]
        
        understanding_response = llm_client.generate(understanding_prompt)
        
        if understanding_response and understanding_response.content:
            print(f"✅ SUCCESS: LLM understood the problem")
            print(f"   Summary: {understanding_response.content.strip()[:150]}...")
        else:
            print("❌ FAILED: LLM could not understand the problem")
            return False
        
        # Test 4: Solution attempt
        print("\n4. Testing solution generation...")
        solution_prompt = [
            {"role": "system", "content": "You are a software engineer. Generate a code patch to fix the given issue."},
            {"role": "user", "content": f"Problem: {problem_statement[:300]}...\\n\\nGenerate a Python code fix. Start your response with 'PATCH:' and include the modified code."}
        ]
        
        solution_response = llm_client.generate(solution_prompt)
        
        if solution_response and solution_response.content and ("PATCH" in solution_response.content.upper() or "def " in solution_response.content):
            print(f"✅ SUCCESS: LLM generated a solution attempt")
            print(f"   Solution preview: {solution_response.content.strip()[:100]}...")
        else:
            print("❌ FAILED: LLM could not generate a solution")
            return False
        
        print("\n🎉 All SWE problem solving tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: SWE problem solving test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_swe_problem_solving()
    if success:
        print("\n✅ SWE problem solving is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ SWE problem solving has issues!")
        sys.exit(1)
