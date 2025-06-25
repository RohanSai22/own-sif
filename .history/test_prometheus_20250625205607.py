#!/usr/bin/env python3
"""
Prometheus 2.0 System Test and Verification Script
Tests all major components and verifies fixes.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def print_status(message, status="INFO"):
    """Print colored status message."""
    colors = {
        "INFO": "\033[94m",
        "SUCCESS": "\033[92m", 
        "ERROR": "\033[91m",
        "WARNING": "\033[93m"
    }
    reset = "\033[0m"
    print(f"{colors.get(status, '')}{status}: {message}{reset}")

def test_environment():
    """Test Python environment and dependencies."""
    print_status("Testing Python environment...")
    
    try:
        import rich
        import textual
        import groq
        import duckduckgo_search
        import docker
        import matplotlib
        import pandas
        import numpy
        print_status("All required packages installed", "SUCCESS")
        return True
    except ImportError as e:
        print_status(f"Missing dependency: {e}", "ERROR")
        return False

def test_env_file():
    """Test .env file configuration."""
    print_status("Checking .env configuration...")
    
    env_file = Path("prometheus_dgm/.env")
    if not env_file.exists():
        print_status(".env file not found", "ERROR")
        return False
    
    # Check for required variables
    with open(env_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    required_vars = ["GROQ_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if var not in content or f"{var}=" not in content:
            missing_vars.append(var)
    
    if missing_vars:
        print_status(f"Missing environment variables: {missing_vars}", "ERROR")
        return False
    
    print_status(".env file properly configured", "SUCCESS")
    return True

def test_requirements_fix():
    """Test that the requirements.txt issue is fixed."""
    print_status("Checking requirements.txt fix...")
    
    # Check main requirements.txt
    req_file = Path("prometheus_dgm/requirements.txt")
    if req_file.exists():
        with open(req_file, 'r', encoding='utf-8') as f:
            content = f.read()
        if "pytest-xvs" in content:
            print_status("ERROR: pytest-xvs still in main requirements.txt", "ERROR")
            return False
    
    # Check evaluation harness
    harness_file = Path("prometheus_dgm/evaluation/swe_bench_harness.py")
    if harness_file.exists():
        with open(harness_file, 'r', encoding='utf-8') as f:
            content = f.read()
        if "pytest-xvs" in content:
            print_status("ERROR: pytest-xvs still in swe_bench_harness.py", "ERROR")
            return False
    
    print_status("Requirements.txt issue fixed", "SUCCESS")
    return True

def test_groq_models():
    """Test Groq model configuration."""
    print_status("Checking Groq model configuration...")
    
    llm_file = Path("prometheus_dgm/llm_provider/unified_client.py")
    if not llm_file.exists():
        print_status("unified_client.py not found", "ERROR")
        return False
    
    with open(llm_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for user-specified models
    required_models = [
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct", 
        "qwen/qwen3-32b"
    ]
    
    for model in required_models:
        if model not in content:
            print_status(f"Model {model} not found in configuration", "ERROR")
            return False
    
    # Check test function uses correct model
    if "llama-3.1-8b-instant" in content and "test_model =" in content:
        print_status("Test function still uses old model", "ERROR")
        return False
    
    print_status("Groq models properly configured", "SUCCESS")
    return True

def test_web_search_logic():
    """Test web search logic changes."""
    print_status("Checking web search logic...")
    
    agent_file = Path("prometheus_dgm/agent/agent_core.py")
    if not agent_file.exists():
        print_status("agent_core.py not found", "ERROR")
        return False
    
    with open(agent_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check that web search is not used for general improvement
    if "_research_improvements" in content:
        # Find the method
        method_start = content.find("def _research_improvements")
        if method_start != -1:
            method_end = content.find("def ", method_start + 1)
            if method_end == -1:
                method_end = len(content)
            method_content = content[method_start:method_end]
            
            if "web_search" in method_content and "execute_tool" in method_content:
                print_status("Web search still used in general improvement", "ERROR")
                return False
    
    # Check for error-specific web search
    if "_research_error_solutions" not in content:
        print_status("Error-specific web search method missing", "ERROR")
        return False
    
    print_status("Web search logic properly configured", "SUCCESS")
    return True

def test_gui_creation():
    """Test GUI dashboard creation."""
    print_status("Checking GUI dashboard...")
    
    gui_file = Path("prometheus_dgm/gui_dashboard.py")
    if not gui_file.exists():
        print_status("GUI dashboard not found", "ERROR")
        return False
    
    # Test if GUI can be imported (basic syntax check)
    try:
        import subprocess
        import sys
        prometheus_path = str(Path('prometheus_dgm').absolute()).replace('\\', '/')
        result = subprocess.run(
            [sys.executable, "-c", f"import sys; sys.path.append(r'{prometheus_path}'); from gui_dashboard import PrometheusGUI; print('GUI imports successfully')"],
            capture_output=True, text=True, timeout=10, encoding='utf-8', errors='replace'
        )
        if result.returncode == 0:
            print_status("GUI dashboard created and importable", "SUCCESS")
            return True
        else:
            print_status(f"GUI import error: {result.stderr}", "ERROR")
            return False
    except Exception as e:
        print_status(f"GUI test error: {e}", "ERROR")
        return False

def test_docker_functionality():
    """Test Docker functionality."""
    print_status("Testing Docker availability...")
    
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True, encoding='utf-8', errors='replace')
        if result.returncode == 0:
            print_status(f"Docker available: {result.stdout.strip()}", "SUCCESS")
            return True
        else:
            print_status("Docker not available", "WARNING")
            return False
    except FileNotFoundError:
        print_status("Docker not installed", "WARNING")
        return False

def test_web_search_functionality():
    """Test actual web search functionality."""
    print_status("Testing web search functionality...")
    
    try:
        # Add current directory to path for import
        sys.path.append(str(Path("prometheus_dgm").absolute()))
        
        from tools.base_tools import web_search
        
        # Test search
        results = web_search("Python programming test", max_results=2)
        
        if results and len(results) > 0:
            print_status(f"Web search working - found {len(results)} results", "SUCCESS")
            for i, result in enumerate(results[:1], 1):
                print_status(f"  Result {i}: {result.get('title', 'No title')[:50]}...", "INFO")
            return True
        else:
            print_status("Web search returned no results", "WARNING")
            return False
            
    except Exception as e:
        print_status(f"Web search test failed: {e}", "ERROR")
        return False

def create_run_script():
    """Create a convenience script to run the system."""
    print_status("Creating run script...")
    
    run_script_content = '''#!/usr/bin/env python3
"""
Prometheus 2.0 System Launcher
Choose how to run the system.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_tui():
    """Run with TUI interface."""
    print("🔥 Starting Prometheus 2.0 with TUI...")
    os.chdir("prometheus_dgm")
    subprocess.run([sys.executable, "main.py"])

def run_gui():
    """Run GUI dashboard."""
    print("🔥 Starting Prometheus 2.0 GUI Dashboard...")
    os.chdir("prometheus_dgm")
    subprocess.run([sys.executable, "gui_dashboard.py"])

def run_both():
    """Run both TUI and GUI."""
    import threading
    
    print("🔥 Starting Prometheus 2.0 with both TUI and GUI...")
    
    # Start TUI in background
    tui_thread = threading.Thread(target=run_tui, daemon=True)
    tui_thread.start()
    
    # Start GUI in foreground
    run_gui()

def main():
    """Main launcher."""
    print("🔥 Prometheus 2.0 - Observable Darwinian Gödeli Machine")
    print("=" * 60)
    print("Choose how to run the system:")
    print("1. TUI only (Terminal interface)")
    print("2. GUI only (Graphical dashboard)")
    print("3. Both TUI + GUI")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\\nEnter choice (1-4): ").strip()
            
            if choice == "1":
                run_tui()
                break
            elif choice == "2":
                run_gui()
                break
            elif choice == "3":
                run_both()
                break
            elif choice == "4":
                print("Goodbye! 👋")
                break
            else:
                print("Invalid choice. Please enter 1-4.")
                
        except KeyboardInterrupt:
            print("\\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
'''
    
    with open("run_prometheus.py", "w", encoding='utf-8') as f:
        f.write(run_script_content)
    
    # Make executable on Unix-like systems
    try:
        os.chmod("run_prometheus.py", 0o755)
    except:
        pass
    
    print_status("Run script created: run_prometheus.py", "SUCCESS")

def main():
    """Main test function."""
    print("=" * 60)
    print("🔥 PROMETHEUS 2.0 SYSTEM VERIFICATION")
    print("=" * 60)
    
    tests = [
        ("Environment", test_environment),
        (".env Configuration", test_env_file),
        ("Requirements Fix", test_requirements_fix),
        ("Groq Models", test_groq_models),
        ("Web Search Logic", test_web_search_logic),
        ("GUI Creation", test_gui_creation),
        ("Docker", test_docker_functionality),
        ("Web Search Function", test_web_search_functionality),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\\n{'-' * 40}")
        print(f"Testing: {test_name}")
        print("-" * 40)
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_status(f"Test {test_name} crashed: {e}", "ERROR")
            results[test_name] = False
    
    # Summary
    print("\\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "PASS" if passed_test else "FAIL"
        color = "SUCCESS" if passed_test else "ERROR"
        print_status(f"{test_name:.<30} {status}", color)
        if passed_test:
            passed += 1
    
    print("\\n" + "-" * 60)
    print_status(f"OVERALL: {passed}/{total} tests passed", 
                 "SUCCESS" if passed == total else "WARNING")
    
    if passed == total:
        print_status("🎉 All systems operational! Prometheus 2.0 is ready.", "SUCCESS")
        create_run_script()
        
        print("\\n" + "=" * 60)
        print("QUICK START GUIDE")
        print("=" * 60)
        print("1. Run the system:")
        print("   python run_prometheus.py")
        print("\\n2. Or run components individually:")
        print("   TUI:  cd prometheus_dgm && python main.py")
        print("   GUI:  cd prometheus_dgm && python gui_dashboard.py")
        print("\\n3. The system will:")
        print("   ✓ Use Groq models for LLM generation")
        print("   ✓ Only use web search after errors occur")
        print("   ✓ Evaluate tasks using Docker")
        print("   ✓ Show all activity in real-time")
        
    else:
        print_status("❌ Some issues need to be resolved before running.", "ERROR")
        print("\\nPlease fix the failed tests and run this script again.")
    
    return passed == total

if __name__ == "__main__":
    main()
