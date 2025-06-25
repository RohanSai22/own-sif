#!/usr/bin/env python3
"""
Prometheus 2.0 GUI Dashboard
A comprehensive GUI interface for visualizing all agent activity, performance metrics, and evolution.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import threading
import time
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import queue
import pandas as pd
from pathlib import Path
import logging

# Set up logging for the GUI
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrometheusGUI:
    """Comprehensive GUI dashboard for Prometheus 2.0 system."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🔥 Prometheus 2.0 - Observable Darwinian Gödeli Machine")
        self.root.geometry("1800x1200")
        self.root.configure(bg="#1e1e1e")
        
        # Data queues for thread-safe updates
        self.message_queue = queue.Queue()
        self.data_queue = queue.Queue()
        
        # Data storage
        self.agent_performance = []
        self.evaluation_results = []
        self.thought_log = []
        self.action_log = []
        self.docker_log = []
        self.generation_history = []
        self.current_status = "Ready"
        self.is_running = False
        self.prometheus_process = None
        
        # Performance tracking
        self.current_iteration = 0
        self.best_score = 0.0
        self.current_agent_id = "genesis"
        self.success_rate = 0.0
        
        # Create GUI components
        self.setup_styles()
        self.create_widgets()
        self.start_monitoring()
        
    def setup_styles(self):
        """Set up modern dark theme styles."""
        style = ttk.Style()
        
        # Configure dark theme
        style.theme_use('clam')
        
        # Dark theme colors
        bg_color = "#2d2d2d"
        fg_color = "#ffffff"
        select_color = "#404040"
        accent_color = "#0078d4"
        
        style.configure('Dark.TFrame', background=bg_color)
        style.configure('Dark.TLabel', background=bg_color, foreground=fg_color)
        style.configure('Dark.TButton', background=select_color, foreground=fg_color, borderwidth=1)
        style.configure('Dark.TNotebook', background=bg_color, borderwidth=0)
        style.configure('Dark.TNotebook.Tab', background=select_color, foreground=fg_color, padding=[10, 5])
        style.map('Dark.TNotebook.Tab', background=[('selected', accent_color)])
        
    def create_widgets(self):
        """Create all GUI widgets with comprehensive monitoring capabilities."""
        
        # Main container
        main_frame = ttk.Frame(self.root, style='Dark.TFrame')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Header with controls
        self.create_header(main_frame)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame, style='Dark.TNotebook')
        self.notebook.pack(fill='both', expand=True, pady=(10, 0))
        
        # Create tabs
        self.create_overview_tab()
        self.create_performance_tab()
        self.create_logs_tab()
        self.create_evolution_tab()
        self.create_docker_tab()
        self.create_tools_tab()
        
    def create_header(self, parent):
        """Create header with system status and controls."""
        header_frame = ttk.Frame(parent, style='Dark.TFrame')
        header_frame.pack(fill='x', pady=(0, 10))
        
        # Left side - Status info
        status_frame = ttk.Frame(header_frame, style='Dark.TFrame')
        status_frame.pack(side='left', fill='x', expand=True)
        
        # Title
        title_label = ttk.Label(status_frame, text="🔥 Prometheus 2.0 - Observable DGM", 
                               font=('Arial', 16, 'bold'), style='Dark.TLabel')
        title_label.pack(anchor='w')
        
        # Status info
        self.status_frame = ttk.Frame(status_frame, style='Dark.TFrame')
        self.status_frame.pack(anchor='w', pady=(5, 0))
        
        self.iteration_label = ttk.Label(self.status_frame, text=f"Iteration: {self.current_iteration}", style='Dark.TLabel')
        self.iteration_label.pack(side='left', padx=(0, 20))
        
        self.score_label = ttk.Label(self.status_frame, text=f"Best Score: {self.best_score:.3f}", style='Dark.TLabel')
        self.score_label.pack(side='left', padx=(0, 20))
        
        self.agent_label = ttk.Label(self.status_frame, text=f"Agent: {self.current_agent_id}", style='Dark.TLabel')
        self.agent_label.pack(side='left', padx=(0, 20))
        
        self.status_label = ttk.Label(self.status_frame, text=f"Status: {self.current_status}", style='Dark.TLabel')
        self.status_label.pack(side='left')
        
        # Right side - Controls
        controls_frame = ttk.Frame(header_frame, style='Dark.TFrame')
        controls_frame.pack(side='right')
        
        self.start_button = ttk.Button(controls_frame, text="▶ Start Evolution", 
                                      command=self.start_evolution, style='Dark.TButton')
        self.start_button.pack(side='left', padx=(0, 10))
        
        self.stop_button = ttk.Button(controls_frame, text="⏹ Stop", 
                                     command=self.stop_evolution, style='Dark.TButton', state='disabled')
        self.stop_button.pack(side='left', padx=(0, 10))
        
        self.refresh_button = ttk.Button(controls_frame, text="🔄 Refresh", 
                                        command=self.refresh_data, style='Dark.TButton')
        self.refresh_button.pack(side='left')
    
    def create_overview_tab(self):
        """Create overview tab with real-time system status."""
        overview_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(overview_frame, text="📊 Overview")
        
        # Left panel - Current status
        left_panel = ttk.Frame(overview_frame, style='Dark.TFrame')
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Current Status Section
        status_label = ttk.Label(left_panel, text="Current Status", font=('Arial', 12, 'bold'), style='Dark.TLabel')
        status_label.pack(anchor='w', pady=(0, 10))
        
        self.status_text = scrolledtext.ScrolledText(left_panel, height=15, bg="#2d2d2d", fg="#ffffff", 
                                                    insertbackground="#ffffff", wrap='word')
        self.status_text.pack(fill='both', expand=True, pady=(0, 10))
        
        # Right panel - Quick stats
        right_panel = ttk.Frame(overview_frame, style='Dark.TFrame')
        right_panel.pack(side='right', fill='y', padx=(5, 0))
        
        # Quick Stats
        stats_label = ttk.Label(right_panel, text="Quick Stats", font=('Arial', 12, 'bold'), style='Dark.TLabel')
        stats_label.pack(anchor='w', pady=(0, 10))
        
        self.stats_frame = ttk.Frame(right_panel, style='Dark.TFrame')
        self.stats_frame.pack(fill='x')
        
        # Initialize stats
        self.update_quick_stats()
    
    def create_performance_tab(self):
        """Create performance tracking tab with charts."""
        perf_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(perf_frame, text="📈 Performance")
        
        # Create matplotlib figure
        self.perf_fig = Figure(figsize=(12, 8), facecolor='#2d2d2d')
        self.perf_canvas = FigureCanvasTkAgg(self.perf_fig, perf_frame)
        self.perf_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Create subplots for different metrics
        self.score_ax = self.perf_fig.add_subplot(2, 2, 1, facecolor='#2d2d2d')
        self.success_ax = self.perf_fig.add_subplot(2, 2, 2, facecolor='#2d2d2d')
        self.time_ax = self.perf_fig.add_subplot(2, 2, 3, facecolor='#2d2d2d')
        self.generation_ax = self.perf_fig.add_subplot(2, 2, 4, facecolor='#2d2d2d')
        
        # Style the plots
        for ax in [self.score_ax, self.success_ax, self.time_ax, self.generation_ax]:
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
        
        self.perf_fig.tight_layout()
        self.update_performance_charts()
    
    def create_logs_tab(self):
        """Create comprehensive logs tab."""
        logs_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(logs_frame, text="📝 Logs")
        
        # Create sub-notebook for different log types
        log_notebook = ttk.Notebook(logs_frame, style='Dark.TNotebook')
        log_notebook.pack(fill='both', expand=True)
        
        # Agent Thoughts
        thoughts_frame = ttk.Frame(log_notebook, style='Dark.TFrame')
        log_notebook.add(thoughts_frame, text="💭 Agent Thoughts")
        
        self.thoughts_text = scrolledtext.ScrolledText(thoughts_frame, bg="#2d2d2d", fg="#ffffff", 
                                                      insertbackground="#ffffff", wrap='word')
        self.thoughts_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Agent Actions
        actions_frame = ttk.Frame(log_notebook, style='Dark.TFrame')
        log_notebook.add(actions_frame, text="⚡ Agent Actions")
        
        self.actions_text = scrolledtext.ScrolledText(actions_frame, bg="#2d2d2d", fg="#ffffff", 
                                                     insertbackground="#ffffff", wrap='word')
        self.actions_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # LLM Responses
        responses_frame = ttk.Frame(log_notebook, style='Dark.TFrame')
        log_notebook.add(responses_frame, text="🤖 LLM Responses")
        
        self.responses_text = scrolledtext.ScrolledText(responses_frame, bg="#2d2d2d", fg="#ffffff", 
                                                       insertbackground="#ffffff", wrap='word')
        self.responses_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # System Logs
        system_frame = ttk.Frame(log_notebook, style='Dark.TFrame')
        log_notebook.add(system_frame, text="🔧 System Logs")
        
        self.system_text = scrolledtext.ScrolledText(system_frame, bg="#2d2d2d", fg="#ffffff", 
                                                    insertbackground="#ffffff", wrap='word')
        self.system_text.pack(fill='both', expand=True, padx=5, pady=5)
    
    def create_evolution_tab(self):
        """Create evolution tracking tab."""
        evolution_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(evolution_frame, text="🧬 Evolution")
        
        # Generation tree view
        tree_frame = ttk.Frame(evolution_frame, style='Dark.TFrame')
        tree_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        tree_label = ttk.Label(tree_frame, text="Generation History", font=('Arial', 12, 'bold'), style='Dark.TLabel')
        tree_label.pack(anchor='w', pady=(0, 10))
        
        # Treeview for generations
        columns = ('Agent ID', 'Generation', 'Parent', 'Score', 'Delta', 'Created')
        self.generation_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.generation_tree.heading(col, text=col)
            self.generation_tree.column(col, width=120)
        
        tree_scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=self.generation_tree.yview)
        self.generation_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        self.generation_tree.pack(side='left', fill='both', expand=True)
        tree_scrollbar.pack(side='right', fill='y')
        
        # Agent details panel
        details_frame = ttk.Frame(evolution_frame, style='Dark.TFrame')
        details_frame.pack(side='right', fill='y', padx=(5, 0))
        
        details_label = ttk.Label(details_frame, text="Agent Details", font=('Arial', 12, 'bold'), style='Dark.TLabel')
        details_label.pack(anchor='w', pady=(0, 10))
        
        self.agent_details_text = scrolledtext.ScrolledText(details_frame, width=40, height=25, 
                                                           bg="#2d2d2d", fg="#ffffff", 
                                                           insertbackground="#ffffff", wrap='word')
        self.agent_details_text.pack(fill='both', expand=True)
        
        # Bind selection event
        self.generation_tree.bind('<<TreeviewSelect>>', self.on_generation_select)
    
    def create_docker_tab(self):
        """Create Docker evaluation monitoring tab."""
        docker_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(docker_frame, text="🐳 Docker Evaluation")
        
        # Docker status
        status_frame = ttk.Frame(docker_frame, style='Dark.TFrame')
        status_frame.pack(fill='x', pady=(0, 10))
        
        docker_label = ttk.Label(status_frame, text="Docker Evaluation Status", 
                                font=('Arial', 12, 'bold'), style='Dark.TLabel')
        docker_label.pack(anchor='w')
        
        self.docker_status_label = ttk.Label(status_frame, text="Status: Ready", style='Dark.TLabel')
        self.docker_status_label.pack(anchor='w', pady=(5, 0))
        
        # Docker logs
        self.docker_text = scrolledtext.ScrolledText(docker_frame, bg="#2d2d2d", fg="#ffffff", 
                                                    insertbackground="#ffffff", wrap='word')
        self.docker_text.pack(fill='both', expand=True, pady=(10, 0))
    
    def create_tools_tab(self):
        """Create tools monitoring tab."""
        tools_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(tools_frame, text="🛠️ Tools")
        
        # Available tools
        available_frame = ttk.Frame(tools_frame, style='Dark.TFrame')
        available_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        available_label = ttk.Label(available_frame, text="Available Tools", 
                                   font=('Arial', 12, 'bold'), style='Dark.TLabel')
        available_label.pack(anchor='w', pady=(0, 10))
        
        self.tools_listbox = tk.Listbox(available_frame, bg="#2d2d2d", fg="#ffffff", 
                                       selectbackground="#0078d4", height=20)
        self.tools_listbox.pack(fill='both', expand=True)
        
        # Tool usage stats
        usage_frame = ttk.Frame(tools_frame, style='Dark.TFrame')
        usage_frame.pack(side='right', fill='y', padx=(5, 0))
        
        usage_label = ttk.Label(usage_frame, text="Tool Usage", 
                               font=('Arial', 12, 'bold'), style='Dark.TLabel')
        usage_label.pack(anchor='w', pady=(0, 10))
        
        self.tool_usage_text = scrolledtext.ScrolledText(usage_frame, width=40, height=20, 
                                                        bg="#2d2d2d", fg="#ffffff", 
                                                        insertbackground="#ffffff", wrap='word')
        self.tool_usage_text.pack(fill='both', expand=True)
    
    def start_evolution(self):
        """Start the Prometheus evolution process."""
        try:
            if self.is_running:
                messagebox.showwarning("Already Running", "Evolution is already in progress!")
                return
            
            # Check if main.py exists
            main_py_path = Path(__file__).parent / "main.py"
            if not main_py_path.exists():
                messagebox.showerror("Error", "main.py not found in the project directory!")
                return
            
            # Start the process
            self.prometheus_process = subprocess.Popen(
                [sys.executable, str(main_py_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.is_running = True
            self.current_status = "Evolution Running"
            
            # Update UI
            self.start_button.configure(state='disabled')
            self.stop_button.configure(state='normal')
            self.update_status_display()
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self.monitor_process, daemon=True)
            self.monitor_thread.start()
            
            self.add_log_entry("system", "✅ Evolution process started successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start evolution: {e}")
            self.add_log_entry("system", f"❌ Failed to start evolution: {e}")
    
    def stop_evolution(self):
        """Stop the Prometheus evolution process."""
        try:
            if self.prometheus_process and self.prometheus_process.poll() is None:
                self.prometheus_process.terminate()
                # Give it a moment to terminate gracefully
                try:
                    self.prometheus_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.prometheus_process.kill()
            
            self.is_running = False
            self.current_status = "Stopped"
            
            # Update UI
            self.start_button.configure(state='normal')
            self.stop_button.configure(state='disabled')
            self.update_status_display()
            
            self.add_log_entry("system", "⏹ Evolution process stopped")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop evolution: {e}")
            self.add_log_entry("system", f"❌ Failed to stop evolution: {e}")
    
    def monitor_process(self):
        """Monitor the evolution process and capture output."""
        if not self.prometheus_process:
            return
        
        try:
            while self.prometheus_process.poll() is None:
                # Read output
                try:
                    output = self.prometheus_process.stdout.readline()
                    if output:
                        self.add_log_entry("system", output.strip())
                        self.parse_process_output(output.strip())
                except Exception as e:
                    self.add_log_entry("system", f"Error reading output: {e}")
                
                time.sleep(0.1)
            
            # Process finished
            self.is_running = False
            self.current_status = "Finished"
            self.root.after(0, self.update_status_display)
            self.add_log_entry("system", "🏁 Evolution process completed")
            
        except Exception as e:
            self.add_log_entry("system", f"❌ Monitor thread error: {e}")
    
    def parse_process_output(self, output: str):
        """Parse process output to extract meaningful information."""
        try:
            # Look for specific patterns in the output
            if "Starting evolution iteration" in output:
                import re
                match = re.search(r"iteration (\d+)", output)
                if match:
                    self.current_iteration = int(match.group(1))
                    self.root.after(0, self.update_status_display)
            
            elif "agent_" in output and "score" in output.lower():
                # Try to extract performance data
                pass  # Implementation depends on actual log format
            
            elif "Docker" in output:
                self.add_log_entry("docker", output)
            
            elif "web_search" in output or "TOOL" in output:
                self.add_log_entry("actions", output)
            
            elif "thought" in output.lower() or "thinking" in output.lower():
                self.add_log_entry("thoughts", output)
                
        except Exception as e:
            pass  # Ignore parsing errors
    
    def add_log_entry(self, log_type: str, message: str):
        """Add a log entry to the appropriate log panel."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        def update_text_widget():
            try:
                if log_type == "thoughts":
                    self.thoughts_text.insert(tk.END, formatted_message)
                    self.thoughts_text.see(tk.END)
                elif log_type == "actions":
                    self.actions_text.insert(tk.END, formatted_message)
                    self.actions_text.see(tk.END)
                elif log_type == "docker":
                    self.docker_text.insert(tk.END, formatted_message)
                    self.docker_text.see(tk.END)
                elif log_type == "system":
                    self.system_text.insert(tk.END, formatted_message)
                    self.system_text.see(tk.END)
                    # Also add to status
                    self.status_text.insert(tk.END, formatted_message)
                    self.status_text.see(tk.END)
                elif log_type == "responses":
                    self.responses_text.insert(tk.END, formatted_message)
                    self.responses_text.see(tk.END)
            except Exception as e:
                pass  # Ignore GUI update errors
        
        # Schedule GUI update
        self.root.after(0, update_text_widget)
    
    def update_status_display(self):
        """Update the status display in the header."""
        try:
            self.iteration_label.configure(text=f"Iteration: {self.current_iteration}")
            self.score_label.configure(text=f"Best Score: {self.best_score:.3f}")
            self.agent_label.configure(text=f"Agent: {self.current_agent_id}")
            self.status_label.configure(text=f"Status: {self.current_status}")
        except Exception as e:
            pass
    
    def update_quick_stats(self):
        """Update the quick stats panel."""
        try:
            # Clear existing stats
            for widget in self.stats_frame.winfo_children():
                widget.destroy()
            
            # Add current stats
            stats = [
                ("Iterations", str(self.current_iteration)),
                ("Best Score", f"{self.best_score:.3f}"),
                ("Success Rate", f"{self.success_rate:.1%}"),
                ("Current Agent", self.current_agent_id),
                ("Status", self.current_status)
            ]
            
            for i, (label, value) in enumerate(stats):
                label_widget = ttk.Label(self.stats_frame, text=f"{label}:", style='Dark.TLabel')
                label_widget.grid(row=i, column=0, sticky='w', pady=2)
                
                value_widget = ttk.Label(self.stats_frame, text=value, style='Dark.TLabel', font=('Arial', 9, 'bold'))
                value_widget.grid(row=i, column=1, sticky='w', padx=(10, 0), pady=2)
        except Exception as e:
            pass
    
    def update_performance_charts(self):
        """Update performance charts with latest data."""
        try:
            # Clear previous plots
            for ax in [self.score_ax, self.success_ax, self.time_ax, self.generation_ax]:
                ax.clear()
            
            # Score over time
            self.score_ax.set_title("Score Over Time", color='white')
            self.score_ax.set_xlabel("Iteration", color='white')
            self.score_ax.set_ylabel("Score", color='white')
            
            # Success rate over time
            self.success_ax.set_title("Success Rate", color='white')
            self.success_ax.set_xlabel("Iteration", color='white')
            self.success_ax.set_ylabel("Success Rate", color='white')
            
            # Execution time
            self.time_ax.set_title("Execution Time", color='white')
            self.time_ax.set_xlabel("Iteration", color='white')
            self.time_ax.set_ylabel("Time (s)", color='white')
            
            # Generation evolution
            self.generation_ax.set_title("Generation Evolution", color='white')
            self.generation_ax.set_xlabel("Generation", color='white')
            self.generation_ax.set_ylabel("Score Improvement", color='white')
            
            # Style the plots
            for ax in [self.score_ax, self.success_ax, self.time_ax, self.generation_ax]:
                ax.tick_params(colors='white')
                ax.grid(True, alpha=0.3)
            
            self.perf_fig.tight_layout()
            self.perf_canvas.draw()
            
        except Exception as e:
            pass
    
    def on_generation_select(self, event):
        """Handle generation selection in the tree view."""
        try:
            selection = self.generation_tree.selection()
            if not selection:
                return
            
            item = self.generation_tree.item(selection[0])
            agent_id = item['values'][0]
            
            # Load and display agent details
            self.load_agent_details(agent_id)
            
        except Exception as e:
            pass
    
    def load_agent_details(self, agent_id: str):
        """Load and display detailed information about an agent."""
        try:
            self.agent_details_text.delete(1.0, tk.END)
            
            details = f"""Agent Details: {agent_id}
            
Loading agent information...
This would include:
- Source code changes
- Performance metrics
- Tool usage statistics
- Mutation history
- Evaluation results
            """
            
            self.agent_details_text.insert(1.0, details)
            
        except Exception as e:
            pass
    
    def refresh_data(self):
        """Refresh all data displays."""
        try:
            self.update_quick_stats()
            self.update_performance_charts()
            self.load_available_tools()
            self.load_generation_history()
            
        except Exception as e:
            pass
    
    def load_available_tools(self):
        """Load and display available tools."""
        try:
            self.tools_listbox.delete(0, tk.END)
            
            # Add sample tools (replace with actual tool loading)
            tools = [
                "web_search - Search the web for information",
                "read_file - Read file contents",
                "write_file - Write content to file",
                "list_directory - List directory contents",
                "execute_shell_command - Execute shell commands",
                "scrape_and_extract_text - Extract text from web pages"
            ]
            
            for tool in tools:
                self.tools_listbox.insert(tk.END, tool)
                
        except Exception as e:
            pass
    
    def load_generation_history(self):
        """Load and display generation history."""
        try:
            # Clear existing items
            for item in self.generation_tree.get_children():
                self.generation_tree.delete(item)
            
            # Add sample data (replace with actual data loading)
            generations = [
                ("agent_genesis", "0", "-", "0.000", "0.000", "2025-06-25 21:00:00"),
                ("agent_fe45a07d", "1", "agent_genesis", "0.120", "+0.120", "2025-06-25 21:05:00"),
            ]
            
            for gen in generations:
                self.generation_tree.insert("", tk.END, values=gen)
                
        except Exception as e:
            pass
    
    def start_monitoring(self):
        """Start the monitoring and update loop."""
        def update_loop():
            while True:
                try:
                    # Update displays periodically
                    self.root.after(0, self.refresh_data)
                    time.sleep(10)  # Update every 10 seconds
                except Exception as e:
                    break
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=update_loop, daemon=True)
        self.monitor_thread.start()
    
    def run(self):
        """Run the GUI application."""
        try:
            # Initialize data
            self.refresh_data()
            
            # Set up window close handler
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Start the GUI
            self.root.mainloop()
            
        except Exception as e:
            logger.error(f"GUI error: {e}")
            messagebox.showerror("Error", f"GUI error: {e}")
    
    def on_closing(self):
        """Handle window closing."""
        try:
            if self.is_running:
                result = messagebox.askyesno("Confirm Exit", 
                                           "Evolution is still running. Stop it and exit?")
                if result:
                    self.stop_evolution()
                    self.root.destroy()
            else:
                self.root.destroy()
        except Exception as e:
            self.root.destroy()

def main():
    """Main entry point for the GUI."""
    try:
        gui = PrometheusGUI()
        gui.run()
    except Exception as e:
        print(f"Failed to start GUI: {e}")

if __name__ == "__main__":
    main()
        style.configure('Panel.TFrame', background=panel_color, relief='raised', borderwidth=1)
        style.configure('Dark.TLabel', background=bg_color, foreground=fg_color, font=('Segoe UI', 10))
        style.configure('Title.TLabel', background=bg_color, foreground=accent_color, font=('Segoe UI', 16, 'bold'))
        style.configure('Header.TLabel', background=panel_color, foreground=accent_color, font=('Segoe UI', 12, 'bold'))
        style.configure('Dark.TNotebook', background=bg_color, borderwidth=0)
        style.configure('Dark.TNotebook.Tab', background=panel_color, foreground=fg_color, padding=[20, 8])
        
    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, style='Dark.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="🔥 Prometheus 2.0 - Observable Darwinian Gödeli Machine", style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Status bar
        self.create_status_bar(main_frame)
        
        # Main notebook for tabs
        self.notebook = ttk.Notebook(main_frame, style='Dark.TNotebook')
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Create tabs
        self.create_overview_tab()
        self.create_performance_tab()
        self.create_activity_tab()
        self.create_evaluation_tab()
        self.create_system_tab()
        
    def create_status_bar(self, parent):
        """Create the status bar."""
        status_frame = ttk.Frame(parent, style='Panel.TFrame')
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Current status
        self.status_var = tk.StringVar(value="⚡ Starting up...")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, style='Header.TLabel')
        status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Current time
        self.time_var = tk.StringVar()
        time_label = ttk.Label(status_frame, textvariable=self.time_var, style='Dark.TLabel')
        time_label.pack(side=tk.RIGHT, padx=10, pady=5)
        
        self.update_time()
        
    def create_overview_tab(self):
        """Create the overview tab."""
        overview_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(overview_frame, text="📊 Overview")
        
        # Top metrics row
        metrics_frame = ttk.Frame(overview_frame, style='Dark.TFrame')
        metrics_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Performance metrics
        self.create_metric_card(metrics_frame, "Current Agent", "agent_9a4e61f3", "🤖")
        self.create_metric_card(metrics_frame, "Best Score", "0.000", "🏆")
        self.create_metric_card(metrics_frame, "Generation", "1", "🧬")
        self.create_metric_card(metrics_frame, "Tasks Completed", "0", "✅")
        
        # Live activity
        activity_frame = ttk.Frame(overview_frame, style='Panel.TFrame')
        activity_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(activity_frame, text="🧠 Live Agent Activity", style='Header.TLabel').pack(anchor=tk.W, padx=10, pady=5)
        
        self.activity_text = scrolledtext.ScrolledText(
            activity_frame, 
            bg="#1e1e1e", 
            fg="#00d4aa",
            insertbackground="#00d4aa",
            font=('Consolas', 10),
            wrap=tk.WORD,
            height=20
        )
        self.activity_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
    def create_performance_tab(self):
        """Create the performance metrics tab."""
        perf_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(perf_frame, text="📈 Performance")
        
        # Chart frame
        chart_frame = ttk.Frame(perf_frame, style='Panel.TFrame')
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib figure
        self.perf_fig = Figure(figsize=(12, 8), facecolor='#2d2d2d')
        self.perf_canvas = FigureCanvasTkAgg(self.perf_fig, chart_frame)
        self.perf_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initialize performance chart
        self.update_performance_chart()
        
    def create_activity_tab(self):
        """Create the activity log tab."""
        activity_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(activity_frame, text="🔍 Activity Logs")
        
        # Create sub-notebook for different log types
        log_notebook = ttk.Notebook(activity_frame, style='Dark.TNotebook')
        log_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Thoughts log
        thoughts_frame = ttk.Frame(log_notebook, style='Dark.TFrame')
        log_notebook.add(thoughts_frame, text="🧠 Thoughts")
        
        self.thoughts_text = scrolledtext.ScrolledText(
            thoughts_frame,
            bg="#1e1e1e",
            fg="#ffff00",
            font=('Consolas', 9),
            wrap=tk.WORD
        )
        self.thoughts_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Actions log
        actions_frame = ttk.Frame(log_notebook, style='Dark.TFrame')
        log_notebook.add(actions_frame, text="⚡ Actions")
        
        self.actions_text = scrolledtext.ScrolledText(
            actions_frame,
            bg="#1e1e1e",
            fg="#00d4aa",
            font=('Consolas', 9),
            wrap=tk.WORD
        )
        self.actions_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Docker log
        docker_frame = ttk.Frame(log_notebook, style='Dark.TFrame')
        log_notebook.add(docker_frame, text="🐳 Docker")
        
        self.docker_text = scrolledtext.ScrolledText(
            docker_frame,
            bg="#1e1e1e",
            fg="#ff6b6b",
            font=('Consolas', 9),
            wrap=tk.WORD
        )
        self.docker_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def create_evaluation_tab(self):
        """Create the evaluation results tab."""
        eval_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(eval_frame, text="🧪 Evaluation")
        
        # Results table frame
        table_frame = ttk.Frame(eval_frame, style='Panel.TFrame')
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(table_frame, text="📋 Evaluation Results", style='Header.TLabel').pack(anchor=tk.W, padx=10, pady=5)
        
        # Create treeview for results
        columns = ("Task ID", "Status", "Score", "Time", "Error")
        self.eval_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=20)
        
        for col in columns:
            self.eval_tree.heading(col, text=col)
            self.eval_tree.column(col, width=120)
        
        # Scrollbar for treeview
        eval_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.eval_tree.yview)
        self.eval_tree.configure(yscrollcommand=eval_scrollbar.set)
        
        self.eval_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=5)
        eval_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=5)
        
    def create_system_tab(self):
        """Create the system configuration tab."""
        system_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(system_frame, text="⚙️ System")
        
        # Configuration display
        config_frame = ttk.Frame(system_frame, style='Panel.TFrame')
        config_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(config_frame, text="🔧 System Configuration", style='Header.TLabel').pack(anchor=tk.W, padx=10, pady=5)
        
        self.config_text = scrolledtext.ScrolledText(
            config_frame,
            bg="#1e1e1e",
            fg="#ffffff",
            font=('Consolas', 10),
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.config_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Load and display current configuration
        self.load_system_config()
        
    def create_metric_card(self, parent, title, value, emoji):
        """Create a metric display card."""
        card_frame = ttk.Frame(parent, style='Panel.TFrame')
        card_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        # Emoji and title
        header_text = f"{emoji} {title}"
        ttk.Label(card_frame, text=header_text, style='Header.TLabel').pack(pady=(10, 5))
        
        # Value
        value_var = tk.StringVar(value=value)
        ttk.Label(card_frame, textvariable=value_var, style='Dark.TLabel').pack(pady=(0, 10))
        
        # Store reference for updates
        setattr(self, f"{title.lower().replace(' ', '_')}_var", value_var)
        
    def update_performance_chart(self):
        """Update the performance chart."""
        self.perf_fig.clear()
        
        # Create subplots
        ax1 = self.perf_fig.add_subplot(221, facecolor='#1e1e1e')
        ax2 = self.perf_fig.add_subplot(222, facecolor='#1e1e1e')
        ax3 = self.perf_fig.add_subplot(223, facecolor='#1e1e1e')
        ax4 = self.perf_fig.add_subplot(224, facecolor='#1e1e1e')
        
        # Dummy data for now
        generations = list(range(1, 11))
        scores = [0.1 + i * 0.08 for i in range(10)]
        
        # Score evolution
        ax1.plot(generations, scores, color='#00d4aa', linewidth=2, marker='o')
        ax1.set_title('Score Evolution', color='#ffffff')
        ax1.set_xlabel('Generation', color='#ffffff')
        ax1.set_ylabel('Score', color='#ffffff')
        ax1.tick_params(colors='#ffffff')
        ax1.grid(True, alpha=0.3)
        
        # Task success rate
        success_rates = [0.2 + i * 0.05 for i in range(10)]
        ax2.bar(generations, success_rates, color='#ff6b6b', alpha=0.7)
        ax2.set_title('Success Rate', color='#ffffff')
        ax2.set_xlabel('Generation', color='#ffffff')
        ax2.set_ylabel('Success Rate', color='#ffffff')
        ax2.tick_params(colors='#ffffff')
        
        # Error distribution
        error_types = ['Docker', 'Timeout', 'Parse', 'Other']
        error_counts = [15, 8, 5, 2]
        ax3.pie(error_counts, labels=error_types, autopct='%1.1f%%', startangle=90,
                colors=['#ff6b6b', '#ffed4e', '#00d4aa', '#6c5ce7'])
        ax3.set_title('Error Distribution', color='#ffffff')
        
        # Tool usage
        tools = ['web_search', 'read_file', 'write_file', 'git']
        usage = [25, 40, 35, 10]
        ax4.barh(tools, usage, color='#00d4aa', alpha=0.7)
        ax4.set_title('Tool Usage', color='#ffffff')
        ax4.set_xlabel('Usage Count', color='#ffffff')
        ax4.tick_params(colors='#ffffff')
        
        # Style all axes
        for ax in [ax1, ax2, ax3, ax4]:
            ax.spines['bottom'].set_color('#ffffff')
            ax.spines['top'].set_color('#1e1e1e')
            ax.spines['right'].set_color('#1e1e1e')
            ax.spines['left'].set_color('#ffffff')
        
        self.perf_fig.tight_layout()
        self.perf_canvas.draw()
        
    def start_monitoring(self):
        """Start monitoring system for updates."""
        self.monitor_thread = threading.Thread(target=self.monitor_prometheus_logs, daemon=True)
        self.monitor_thread.start()
        
        # Start GUI update timer
        self.process_updates()
        
    def monitor_prometheus_logs(self):
        """Monitor Prometheus log files for updates."""
        log_file = Path("prometheus_dgm/prometheus.log")
        
        if not log_file.exists():
            return
            
        # Monitor log file for changes
        last_size = 0
        
        while True:
            try:
                current_size = log_file.stat().st_size
                
                if current_size > last_size:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        f.seek(last_size)
                        new_lines = f.read()
                        
                    # Parse and queue new log entries
                    for line in new_lines.strip().split('\n'):
                        if line.strip():
                            self.parse_log_line(line)
                    
                    last_size = current_size
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                print(f"Error monitoring logs: {e}")
                time.sleep(5)
                
    def parse_log_line(self, line):
        """Parse a log line and queue appropriate updates."""
        try:
            if " - " in line:
                timestamp_part, message_part = line.split(" - ", 1)
                timestamp = timestamp_part.split(",")[0]
                
                # Determine log type and queue update
                if "agent_core" in line:
                    self.message_queue.put(("thought", f"{timestamp}: {message_part}"))
                elif "evaluation" in line:
                    self.message_queue.put(("docker", f"{timestamp}: {message_part}"))
                else:
                    self.message_queue.put(("action", f"{timestamp}: {message_part}"))
                    
        except Exception:
            # Queue as general activity
            self.message_queue.put(("activity", line))
            
    def process_updates(self):
        """Process queued updates and refresh GUI."""
        try:
            # Process message queue
            while not self.message_queue.empty():
                log_type, message = self.message_queue.get_nowait()
                self.add_log_message(log_type, message)
                
            # Update time
            self.update_time()
            
            # Update status based on activity
            self.update_status()
            
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error processing updates: {e}")
        
        # Schedule next update
        self.root.after(1000, self.process_updates)
        
    def add_log_message(self, log_type, message):
        """Add a message to the appropriate log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        if log_type == "thought":
            self.thoughts_text.insert(tk.END, formatted_message)
            self.thoughts_text.see(tk.END)
            
        elif log_type == "action":
            self.actions_text.insert(tk.END, formatted_message)
            self.actions_text.see(tk.END)
            
        elif log_type == "docker":
            self.docker_text.insert(tk.END, formatted_message)
            self.docker_text.see(tk.END)
            
        elif log_type == "activity":
            self.activity_text.insert(tk.END, formatted_message)
            self.activity_text.see(tk.END)
            
        # Limit text buffer size
        for text_widget in [self.thoughts_text, self.actions_text, self.docker_text, self.activity_text]:
            lines = text_widget.get("1.0", tk.END).count('\n')
            if lines > 1000:
                text_widget.delete("1.0", "100.0")
                
    def update_time(self):
        """Update the time display."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_var.set(f"🕒 {current_time}")
        
    def update_status(self):
        """Update the status display."""
        # This could be enhanced to show real status from the system
        self.status_var.set("⚡ Prometheus 2.0 Running - Evaluating Tasks")
        
    def load_system_config(self):
        """Load and display system configuration."""
        config_info = {
            "System": "Prometheus 2.0 - Observable Darwinian Gödeli Machine",
            "LLM Provider": "Groq",
            "Models": [
                "meta-llama/llama-4-maverick-17b-128e-instruct",
                "meta-llama/llama-4-scout-17b-16e-instruct", 
                "qwen/qwen3-32b"
            ],
            "Web Search": "DuckDuckGo (API-free)",
            "Evaluation": "SWE-bench Lite",
            "Docker": "Enabled",
            "TUI": "Rich/Textual",
            "Created": "2025-06-25"
        }
        
        self.config_text.config(state=tk.NORMAL)
        self.config_text.delete("1.0", tk.END)
        
        config_text = json.dumps(config_info, indent=2)
        self.config_text.insert("1.0", config_text)
        
        self.config_text.config(state=tk.DISABLED)
        
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()

def main():
    """Main entry point."""
    try:
        app = PrometheusGUI()
        app.run()
    except KeyboardInterrupt:
        print("\nGUI shutdown requested")
    except Exception as e:
        print(f"GUI error: {e}")

if __name__ == "__main__":
    main()
