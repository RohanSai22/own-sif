#!/usr/bin/env python3
"""
GUI Launcher for Prometheus 2.0
Runs the GUI dashboard with proper path handling.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the GUI dashboard."""
    # Get the script directory (where this file is located)
    script_dir = Path(__file__).parent.absolute()
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Add the script directory to Python path
    sys.path.insert(0, str(script_dir))
    
    print("🔥 Prometheus 2.0 - GUI Dashboard")
    print(f"📁 Working directory: {script_dir}")
    print("🚀 Starting GUI dashboard...")
    
    try:
        # Import and run the GUI
        from gui_dashboard import main as run_gui
        run_gui()
    except KeyboardInterrupt:
        print("\n⏹️  GUI dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running GUI: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
