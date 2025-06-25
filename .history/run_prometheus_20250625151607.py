#!/usr/bin/env python3
"""
Prometheus 2.0 System Launcher
Choose how to run the system.
"""

import os
import sys
import subprocess
from pathlib import Path

# Ensure UTF-8 encoding for stdout/stderr to prevent UnicodeEncodeError
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def run_tui():
    """Run with TUI interface."""
    print("🔥 Starting Prometheus 2.0 with TUI...")
    prometheus_dir = os.path.join(os.path.dirname(__file__), "prometheus_dgm")
    subprocess.run([sys.executable, "main.py"], cwd=prometheus_dir, encoding='utf-8', errors='replace', text=True)

def run_gui():
    """Run GUI dashboard."""
    print("🔥 Starting Prometheus 2.0 GUI Dashboard...")
    prometheus_dir = os.path.join(os.path.dirname(__file__), "prometheus_dgm")
    subprocess.run([sys.executable, "gui_dashboard.py"], cwd=prometheus_dir, encoding='utf-8', errors='replace', text=True)

def run_both():
    """Run both TUI and GUI."""
    import threading
    import time
    
    print("🔥 Starting Prometheus 2.0 with both TUI and GUI...")
    
    def run_tui_background():
        """Run TUI in background subprocess."""
        try:
            prometheus_dir = os.path.join(os.path.dirname(__file__), "prometheus_dgm")
            # Use Popen for non-blocking background execution
            subprocess.Popen([sys.executable, "main.py"], 
                           cwd=prometheus_dir, 
                           encoding='utf-8', 
                           errors='replace',
                           text=True,
                           # Redirect stdout/stderr to avoid conflicts
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"TUI startup error: {e}")
    
    # Start TUI in background thread
    tui_thread = threading.Thread(target=run_tui_background, daemon=True)
    tui_thread.start()
    
    # Small delay to let TUI start
    time.sleep(3)
    
    # Start GUI in foreground (but also non-blocking so we can return)
    print("Starting GUI in 3 seconds...")
    time.sleep(1)
    print("Starting GUI in 2 seconds...")
    time.sleep(1)
    print("Starting GUI in 1 second...")
    time.sleep(1)
    
    # Start GUI
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
            choice = input("\nEnter choice (1-4): ").strip()
            
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
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
