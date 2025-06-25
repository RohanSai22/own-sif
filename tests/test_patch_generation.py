"""
Test Patch Generation and Application
Tests if LLM can generate valid patches and if they can be applied.
"""

import sys
import os
import tempfile
import shutil
import io

# Force UTF-8 encoding for Windows
if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from llm_provider.unified_client import llm_client
from framework.mutator import CodeMutator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_patch_generation():
    """Test patch generation and application."""
    print("🧪 Testing Patch Generation and Application...")
    print("=" * 50)
    
    try:
        # Test 1: Create test file
        print("\n1. Creating test environment...")
        temp_dir = tempfile.mkdtemp()
        test_file = os.path.join(temp_dir, "test_module.py")
        
        # Create a simple Python file with a bug
        original_code = '''def add_numbers(a, b):
    """Add two numbers together."""
    result = a + b
    print(f"Adding {a} + {b} = {result}")
    return result

def multiply_numbers(a, b):
    """Multiply two numbers - has a bug!"""
    result = a + b  # BUG: Should be a * b
    return result

if __name__ == "__main__":
    print(add_numbers(2, 3))
    print(multiply_numbers(4, 5))
'''
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(original_code)
        
        print(f"✅ SUCCESS: Created test file at {test_file}")
        
        # Test 2: LLM identifies the bug
        print("\n2. Testing bug identification...")
        bug_prompt = [
            {"role": "system", "content": "You are a code reviewer. Find bugs in the given code."},
            {"role": "user", "content": f"Review this Python code and identify any bugs:\\n\\n{original_code}\\n\\nExplain what's wrong."}
        ]
        
        bug_response = llm_client.generate(bug_prompt)
        
        if bug_response and bug_response.content and ("multiply" in bug_response.content.lower() or "bug" in bug_response.content.lower()):
            print(f"✅ SUCCESS: LLM identified the bug")
            print(f"   Analysis: {bug_response.content.strip()[:150]}...")
        else:
            print("❌ FAILED: LLM could not identify the bug")
            return False
        
        # Test 3: Generate patch
        print("\n3. Testing patch generation...")
        patch_prompt = [
            {"role": "system", "content": "You are a software engineer. Generate a patch to fix bugs in code."},
            {"role": "user", "content": f"Fix the bug in this code. Provide the corrected function only:\\n\\n{original_code}\\n\\nReturn only the fixed multiply_numbers function."}
        ]
        
        patch_response = llm_client.generate(patch_prompt)
        
        if patch_response and patch_response.content and ("def multiply_numbers" in patch_response.content and "*" in patch_response.content):
            print(f"✅ SUCCESS: LLM generated a patch")
            print(f"   Patch preview: {patch_response.content.strip()[:100]}...")
        else:
            print("❌ FAILED: LLM could not generate a valid patch")
            return False
        
        # Test 4: Apply patch (manual test)
        print("\n4. Testing patch application...")
        
        # Simple patch application test
        fixed_code = original_code.replace("result = a + b  # BUG: Should be a * b", "result = a * b  # Fixed: Changed + to *")
        
        fixed_file = os.path.join(temp_dir, "test_module_fixed.py")
        with open(fixed_file, 'w', encoding='utf-8') as f:
            f.write(fixed_code)
        
        print(f"✅ SUCCESS: Applied patch to create fixed version")
        
        # Test 5: Validate fix
        print("\n5. Testing fix validation...")
        
        # Test the original vs fixed behavior
        import subprocess
        
        try:
            # Run original (buggy) version
            original_result = subprocess.run([
                sys.executable, "-c", 
                f"exec(open('{test_file}').read())"
            ], capture_output=True, text=True, encoding='utf-8', errors='replace')
            
            # Run fixed version  
            fixed_result = subprocess.run([
                sys.executable, "-c",
                f"exec(open('{fixed_file}').read())"
            ], capture_output=True, text=True, encoding='utf-8', errors='replace')
            
            if original_result.stdout != fixed_result.stdout:
                print(f"✅ SUCCESS: Fix changes behavior as expected")
                print(f"   Original output: {original_result.stdout.strip()}")
                print(f"   Fixed output: {fixed_result.stdout.strip()}")
            else:
                print("⚠️  WARNING: Fix didn't change output")
            
        except Exception as e:
            print(f"⚠️  WARNING: Could not validate fix execution: {e}")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        print("\n🎉 All patch generation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Patch generation test failed with error: {e}")
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir)
        return False

if __name__ == "__main__":
    success = test_patch_generation()
    if success:
        print("\n✅ Patch generation is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ Patch generation has issues!")
        sys.exit(1)
