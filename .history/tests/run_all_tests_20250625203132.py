"""
Master Test Runner
Runs all individual tests and provides a comprehensive report.
"""

import sys
import os
import subprocess
import time

def run_test(test_file, test_name):
    """Run a single test and return results."""
    print(f"\n{'='*60}")
    print(f"🧪 Running {test_name}...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=120  # 2 minute timeout per test
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        success = result.returncode == 0
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return {
            "name": test_name,
            "file": test_file,
            "success": success,
            "duration": duration,
            "output": result.stdout,
            "error": result.stderr
        }
        
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT: {test_name} took longer than 2 minutes")
        return {
            "name": test_name,
            "file": test_file,
            "success": False,
            "duration": 120,
            "output": "",
            "error": "Test timed out"
        }
    except Exception as e:
        print(f"❌ ERROR: Failed to run {test_name}: {e}")
        return {
            "name": test_name,
            "file": test_file,
            "success": False,
            "duration": 0,
            "output": "",
            "error": str(e)
        }

def main():
    """Run all tests and generate report."""
    print("🔥 Prometheus 2.0 - Comprehensive Test Suite")
    print("=" * 60)
    print("Running individual component tests...")
    
    # Define all tests
    tests = [
        ("test_web_search.py", "Web Search Functionality"),
        ("test_llm_connectivity.py", "LLM Connectivity"),
        ("test_swe_problem_solving.py", "SWE Problem Solving"),
        ("test_patch_generation.py", "Patch Generation"),
        ("test_docker_sandbox.py", "Docker Sandbox"),
        ("test_error_handling.py", "Error Handling & Web Search Integration"),
        ("test_agent_evolution.py", "Agent Evolution System (DGM Logic)")
    ]
    
    results = []
    
    # Run each test
    for test_file, test_name in tests:
        test_path = os.path.join(os.path.dirname(__file__), test_file)
        if os.path.exists(test_path):
            result = run_test(test_path, test_name)
            results.append(result)
        else:
            print(f"❌ MISSING: {test_file} not found")
            results.append({
                "name": test_name,
                "file": test_file,
                "success": False,
                "duration": 0,
                "output": "",
                "error": "Test file not found"
            })
    
    # Generate summary report
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY REPORT")
    print(f"{'='*60}")
    
    passed_tests = [r for r in results if r["success"]]
    failed_tests = [r for r in results if not r["success"]]
    
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {len(passed_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success Rate: {len(passed_tests)/len(results)*100:.1f}%")
    
    total_duration = sum(r["duration"] for r in results)
    print(f"Total Duration: {total_duration:.2f} seconds")
    
    # Detailed results
    print(f"\n{'='*60}")
    print("📋 DETAILED RESULTS")
    print(f"{'='*60}")
    
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        duration = f"{result['duration']:.2f}s"
        print(f"{status:8} | {duration:>8} | {result['name']}")
        
        if not result["success"] and result["error"]:
            print(f"         |          | Error: {result['error']}")
    
    # Recommendations
    print(f"\n{'='*60}")
    print("🔧 RECOMMENDATIONS")
    print(f"{'='*60}")
    
    if failed_tests:
        print("❌ Issues found that need attention:")
        for failed in failed_tests:
            print(f"   • {failed['name']}: {failed['error']}")
    else:
        print("✅ All tests passed! System appears to be working correctly.")
    
    print(f"\n{'='*60}")
    print("🎯 NEXT STEPS")
    print(f"{'='*60}")
    
    if failed_tests:
        print("1. Fix the failing components listed above")
        print("2. Re-run individual tests to verify fixes")
        print("3. Run the full system test again")
        print("4. Once all tests pass, proceed with full system testing")
    else:
        print("1. All components are working individually!")
        print("2. Ready for full system integration testing")
        print("3. You can now test the complete Prometheus 2.0 system")
        print("4. Monitor for any issues during full system operation")
    
    # Exit with appropriate code
    if failed_tests:
        print(f"\n❌ Some tests failed. Please address the issues above.")
        sys.exit(1)
    else:
        print(f"\n✅ All tests passed! System is ready for operation.")
        sys.exit(0)

if __name__ == "__main__":
    main()
