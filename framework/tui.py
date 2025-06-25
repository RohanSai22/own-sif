"""Terminal User Interface for Prometheus 2.0 - The Observable Evolution Dashboard."""

import asyncio
import threading
import time
from datetime import datetime
from typing import List, Dict, Optional, Any
from queue import Queue, Empty
from dataclasses import dataclass

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.align import Align
from rich.rule import Rule

@dataclass
class LogEntry:
    """Represents a single log entry."""
    timestamp: datetime
    level: str
    source: str
    message: str
    details: Optional[Dict[str, Any]] = None

@dataclass
class AgentGeneration:
    """Represents an agent generation for tracking."""
    agent_id: str
    parent_id: Optional[str]
    score: float
    delta: float
    iteration: int
    created_at: datetime

class TerminalUI:
    """Rich Terminal User Interface for observing agent evolution."""
    
    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        self.live: Optional[Live] = None
        
        # Data storage
        self.current_iteration = 0
        self.current_task = "Initializing..."
        self.best_score = 0.0
        self.current_agent_id = "genesis"
        self.parent_agent_id = None
        
        # Logs and queues
        self.thought_queue = Queue(maxsize=100)
        self.action_queue = Queue(maxsize=100)
        self.eval_queue = Queue(maxsize=1000)
        self.generations: List[AgentGeneration] = []
        
        # Status tracking
        self.is_running = False
        self.current_status = "Ready"
        self.progress_tasks = {}
        
        self._setup_layout()
    
    def _setup_layout(self):
        """Configure the TUI layout with all panels."""
        # Create main layout structure
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        # Split main area into left and right
        self.layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        # Split left side into top and bottom
        self.layout["left"].split_column(
            Layout(name="thoughts", ratio=1),
            Layout(name="actions", ratio=1),
            Layout(name="evaluation", ratio=1)
        )
        
        # Right side for generations table
        self.layout["right"].split_column(
            Layout(name="status", size=8),
            Layout(name="generations", ratio=1)
        )
    
    def _create_header(self) -> Panel:
        """Create the header panel with system status."""
        title_text = Text("🔥 Prometheus 2.0 - Observable Darwinian Gödeli Machine", style="bold red")
        
        status_info = [
            f"Iteration: {self.current_iteration}",
            f"Best Score: {self.best_score:.3f}",
            f"Agent: {self.current_agent_id}",
            f"Status: {self.current_status}"
        ]
        
        status_text = Text(" | ".join(status_info), style="cyan")
        
        header_content = Align.center(
            Text.assemble(title_text, "\n", status_text)
        )
        
        return Panel(
            header_content,
            title="System Status",
            border_style="bright_blue",
            padding=(0, 1)
        )
    
    def _create_thoughts_panel(self) -> Panel:
        """Create the agent thoughts panel."""
        thoughts = []
        temp_queue = []
        
        # Get recent thoughts
        while not self.thought_queue.empty():
            try:
                entry = self.thought_queue.get_nowait()
                temp_queue.append(entry)
                thoughts.append(entry)
            except Empty:
                break
        
        # Put items back in queue
        for item in temp_queue:
            if not self.thought_queue.full():
                self.thought_queue.put(item)
        
        # Display last 10 thoughts
        content_lines = []
        for entry in thoughts[-10:]:
            timestamp = entry.timestamp.strftime("%H:%M:%S")
            content_lines.append(f"[dim]{timestamp}[/dim] {entry.message}")
        
        if not content_lines:
            content_lines = ["[dim]Agent thoughts will appear here...[/dim]"]
        
        content = "\n".join(content_lines)
        
        return Panel(
            content,
            title="🧠 Agent Inner Monologue",
            border_style="green",
            height=12
        )
    
    def _create_actions_panel(self) -> Panel:
        """Create the actions log panel."""
        actions = []
        temp_queue = []
        
        # Get recent actions
        while not self.action_queue.empty():
            try:
                entry = self.action_queue.get_nowait()
                temp_queue.append(entry)
                actions.append(entry)
            except Empty:
                break
        
        # Put items back in queue
        for item in temp_queue:
            if not self.action_queue.full():
                self.action_queue.put(item)
        
        # Display last 15 actions
        content_lines = []
        for entry in actions[-15:]:
            timestamp = entry.timestamp.strftime("%H:%M:%S")
            if entry.level == "SUCCESS":
                style = "bold green"
            elif entry.level == "ERROR":
                style = "bold red"
            elif entry.level == "TOOL":
                style = "bold yellow"
            else:
                style = "white"
            
            content_lines.append(f"[dim]{timestamp}[/dim] [{style}]{entry.source}[/{style}]: {entry.message}")
        
        if not content_lines:
            content_lines = ["[dim]Agent actions will appear here...[/dim]"]
        
        content = "\n".join(content_lines)
        
        return Panel(
            content,
            title="⚡ Live Action Log",
            border_style="yellow",
            height=12
        )
    
    def _create_evaluation_panel(self) -> Panel:
        """Create the evaluation log panel."""
        eval_logs = []
        temp_queue = []
        
        # Get recent evaluation output
        while not self.eval_queue.empty():
            try:
                entry = self.eval_queue.get_nowait()
                temp_queue.append(entry)
                eval_logs.append(entry)
            except Empty:
                break
        
        # Put items back in queue
        for item in temp_queue:
            if not self.eval_queue.full():
                self.eval_queue.put(item)
        
        # Display last 20 lines
        content_lines = []
        for entry in eval_logs[-20:]:
            timestamp = entry.timestamp.strftime("%H:%M:%S")
            if "PASS" in entry.message or "SUCCESS" in entry.message:
                style = "green"
            elif "FAIL" in entry.message or "ERROR" in entry.message:
                style = "red"
            elif "WARNING" in entry.message:
                style = "yellow"
            else:
                style = "white"
            
            content_lines.append(f"[dim]{timestamp}[/dim] [{style}]{entry.message}[/{style}]")
        
        if not content_lines:
            content_lines = ["[dim]Evaluation output will appear here...[/dim]"]
        
        content = "\n".join(content_lines)
        
        return Panel(
            content,
            title="🐳 Docker Evaluation Log",
            border_style="blue",
            height=12
        )
    
    def _create_status_panel(self) -> Panel:
        """Create the current status panel."""
        status_lines = [
            f"Current Task: {self.current_task}",
            f"Running: {'Yes' if self.is_running else 'No'}",
            "",
            "Recent Performance:",
            f"  Best Score: {self.best_score:.3f}",
            f"  Current Agent: {self.current_agent_id}",
            f"  Parent: {self.parent_agent_id or 'Genesis'}"
        ]
        
        content = "\n".join(status_lines)
        
        return Panel(
            content,
            title="📊 Current Status",
            border_style="magenta"
        )
    
    def _create_generations_table(self) -> Panel:
        """Create the generations performance table."""
        table = Table(show_header=True, header_style="bold blue", show_lines=True)
        table.add_column("Agent ID", style="cyan", width=12)
        table.add_column("Parent", style="dim", width=10)
        table.add_column("Score", justify="right", style="green", width=8)
        table.add_column("Delta", justify="right", style="yellow", width=8)
        table.add_column("Iteration", justify="right", style="white", width=5)
        
        # Show last 10 generations
        for gen in self.generations[-10:]:
            delta_style = "green" if gen.delta >= 0 else "red"
            delta_text = f"+{gen.delta:.3f}" if gen.delta >= 0 else f"{gen.delta:.3f}"
            
            table.add_row(
                gen.agent_id[:10],
                gen.parent_id[:8] if gen.parent_id else "Genesis",
                f"{gen.score:.3f}",
                f"[{delta_style}]{delta_text}[/{delta_style}]",
                str(gen.iteration)
            )
        
        return Panel(
            table,
            title="🧬 Generational Performance",
            border_style="bright_green"
        )
    
    def _create_footer(self) -> Panel:
        """Create the footer with system info."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        footer_text = f"Prometheus 2.0 | {current_time} | Press Ctrl+C to stop"
        
        return Panel(
            Align.center(footer_text),
            style="dim",
            border_style="dim"
        )
    
    def _update_display(self):
        """Update all panels in the layout."""
        self.layout["header"].update(self._create_header())
        self.layout["thoughts"].update(self._create_thoughts_panel())
        self.layout["actions"].update(self._create_actions_panel())
        self.layout["evaluation"].update(self._create_evaluation_panel())
        self.layout["status"].update(self._create_status_panel())
        self.layout["generations"].update(self._create_generations_table())
        self.layout["footer"].update(self._create_footer())
    
    def start(self):
        """Start the TUI display."""
        self.is_running = True
        self._update_display()
        self.live = Live(self.layout, console=self.console, refresh_per_second=4, screen=True)
        self.live.start()
    
    def stop(self):
        """Stop the TUI display."""
        self.is_running = False
        if self.live:
            self.live.stop()
    
    def log_thought(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Log an agent thought."""
        entry = LogEntry(
            timestamp=datetime.now(),
            level="THOUGHT",
            source="Agent",
            message=message,
            details=details
        )
        
        if not self.thought_queue.full():
            self.thought_queue.put(entry)
        
        self._update_display()
    
    def log_action(self, source: str, message: str, level: str = "INFO", details: Optional[Dict[str, Any]] = None):
        """Log an agent action."""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            source=source,
            message=message,
            details=details
        )
        
        if not self.action_queue.full():
            self.action_queue.put(entry)
        
        self._update_display()
    
    def log_eval_output(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Log evaluation output."""
        entry = LogEntry(
            timestamp=datetime.now(),
            level="EVAL",
            source="Docker",
            message=message,
            details=details
        )
        
        if not self.eval_queue.full():
            self.eval_queue.put(entry)
        
        self._update_display()
    
    def update_status(self, status: str):
        """Update the current status."""
        self.current_status = status
        self._update_display()
    
    def update_iteration(self, iteration: int):
        """Update the current iteration."""
        self.current_iteration = iteration
        self._update_display()
    
    def update_task(self, task: str):
        """Update the current task."""
        self.current_task = task
        self._update_display()
    
    def update_agent(self, agent_id: str, parent_id: Optional[str] = None):
        """Update the current agent info."""
        self.current_agent_id = agent_id
        self.parent_agent_id = parent_id
        self._update_display()
    
    def add_generation(self, agent_id: str, parent_id: Optional[str], score: float, iteration: int):
        """Add a new generation to the tracking."""
        # Calculate delta from previous best
        delta = score - self.best_score if self.generations else 0.0
        
        # Update best score if this is better
        if score > self.best_score:
            self.best_score = score
        
        generation = AgentGeneration(
            agent_id=agent_id,
            parent_id=parent_id,
            score=score,
            delta=delta,
            iteration=iteration,
            created_at=datetime.now()
        )
        
        self.generations.append(generation)
        self._update_display()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current state."""
        return {
            "iteration": self.current_iteration,
            "best_score": self.best_score,
            "current_agent": self.current_agent_id,
            "parent_agent": self.parent_agent_id,
            "total_generations": len(self.generations),
            "status": self.current_status,
            "is_running": self.is_running
        }

# Global TUI instance
tui = TerminalUI()

if __name__ == "__main__":
    # Test the TUI
    import time
    import random
    
    try:
        tui.start()
        
        # Simulate some activity
        for i in range(50):
            tui.update_iteration(i)
            tui.log_thought(f"Analyzing problem {i}...")
            time.sleep(0.5)
            
            tui.log_action("web_search", f"Searching for solution to problem {i}", "TOOL")
            time.sleep(0.3)
            
            tui.log_action("write_file", f"Generated solution for problem {i}", "SUCCESS")
            time.sleep(0.2)
            
            tui.log_eval_output(f"Running tests for solution {i}...")
            time.sleep(0.4)
            
            score = random.uniform(0.1, 0.9)
            tui.add_generation(f"agent_{i}", f"agent_{i-1}" if i > 0 else None, score, i)
            
            if i % 10 == 0:
                tui.update_status(f"Completed iteration {i}")
            
    except KeyboardInterrupt:
        tui.stop()
        print("TUI Demo stopped.")
