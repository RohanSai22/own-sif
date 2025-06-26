#!/usr/bin/env python3
"""
TUI Launcher for Prometheus 2.0
Runs the terminal interface with proper path handling.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the TUI interface."""
    # Get the script directory (where this file is located)
    script_dir = Path(__file__).parent.absolute()
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Add the script directory to Python path
    sys.path.insert(0, str(script_dir))
    
    print("🔥 Prometheus 2.0 - Terminal UI")
    print(f"📁 Working directory: {script_dir}")
    print("🚀 Starting terminal evolution...")
    
    try:
        # Import and run the main evolution
        from main import main as run_evolution
        run_evolution()
    except KeyboardInterrupt:
        print("\n⏹️  Evolution stopped by user")
    except Exception as e:
        print(f"❌ Error running evolution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
