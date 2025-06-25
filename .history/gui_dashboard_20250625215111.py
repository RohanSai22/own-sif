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
