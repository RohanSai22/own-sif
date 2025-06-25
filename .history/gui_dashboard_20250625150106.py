#!/usr/bin/env python3
"""
Prometheus 2.0 GUI Dashboard
A modern GUI interface for visualizing all agent activity and performance metrics.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Any
import queue
import pandas as pd
from pathlib import Path

class PrometheusGUI:
    """Modern GUI dashboard for Prometheus 2.0 system."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🔥 Prometheus 2.0 - Observable Darwinian Gödeli Machine")
        self.root.geometry("1600x1000")
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
        
        # Create GUI components
        self.setup_styles()
        self.create_widgets()
        self.start_monitoring()
        
    def setup_styles(self):
        """Set up modern dark theme styles."""
        style = ttk.Style()
        
        # Configure dark theme
        style.theme_use('clam')
        
        # Define colors
        bg_color = "#1e1e1e"
        fg_color = "#ffffff"
        accent_color = "#00d4aa"
        panel_color = "#2d2d2d"
        
        style.configure('Dark.TFrame', background=bg_color)
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
