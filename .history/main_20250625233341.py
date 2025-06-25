"""
Prometheus 2.0 - The Observable Darwinian Gödeli Machine

Main orchestrator that brings together all components for self-improving AI evolution.
"""

import asyncio
import logging
import os
import signal
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Ensure UTF-8 encoding for stdout/stderr to prevent UnicodeEncodeError
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prometheus.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import core components
from config import config, llm_config
from framework.tui import tui
from framework.mutator import CodeMutator
from agent.agent_core import PrometheusAgent
from evaluation.swe_bench_harness import SWEBenchHarness
from archive.agent_archive import AgentArchive
from tools.tool_manager import ToolManager

class PrometheusOrchestrator:
    """Main orchestrator for the Prometheus 2.0 system."""
    
    def __init__(self):
        self.project_root = config.project_root
        self.current_iteration = 0
        self.is_running = False
        self.shutdown_requested = False
        self.population_size = config.population_size
        self.elite_count = max(1, self.population_size // 3)
        self.tournament_size = max(2, self.population_size // 2)
        
        # Initialize components
        self.tui = tui
        self.mutator = CodeMutator(self.project_root)
        self.evaluator = SWEBenchHarness(self.project_root)
        self.archive = AgentArchive(self.project_root)
        
        # Population of agents
        self.population: List[PrometheusAgent] = []
        
        # Performance tracking
        self.best_score = 0.0
        self.stagnation_counter = 0
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    def initialize_system(self) -> bool:
        """Initialize the Prometheus system."""
        try:
            logger.info("Initializing Prometheus 2.0...")
            self.tui.update_status("Initializing system...")
            
            # Check if we have API keys
            if not any([llm_config.openai_api_key, llm_config.groq_api_key, llm_config.gemini_api_key]):
                logger.error("No LLM API keys configured. Please check your .env file.")
                self.tui.log_action("System", "No LLM API keys found", "ERROR")
                return False
            
            # Test LLM connectivity
            self.tui.log_action("System", "Testing LLM connectivity...", "INFO")
            try:
                from llm_provider.unified_client import llm_client
                test_results = llm_client.test_providers()
                
                working_providers = [p for p, r in test_results.items() if r["status"] == "success"]
                if not working_providers:
                    logger.error("No LLM providers are working")
                    self.tui.log_action("System", "No working LLM providers", "ERROR")
                    return False
                
                self.tui.log_action("System", f"LLM providers ready: {', '.join(working_providers)}", "SUCCESS")
                
            except Exception as e:
                logger.error(f"LLM connectivity test failed: {e}")
                self.tui.log_action("System", f"LLM test failed: {e}", "ERROR")
                return False
            
            # Initialize or load existing agent
            self._initialize_agent()
            
            logger.info("Prometheus 2.0 initialization complete")
            self.tui.log_action("System", "Prometheus 2.0 ready for evolution", "SUCCESS")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self.tui.log_action("System", f"Initialization failed: {e}", "ERROR")
            return False
    
    def _initialize_agent(self):
        """Initialize the current agent."""
        # Check if we have existing agents in archive
        parent_id = self.archive.select_parent_for_next_generation()
        
        if parent_id:
            # Create child agent from best parent
            parent_generation = self.archive.generations[parent_id]
            
            self.current_agent = PrometheusAgent(project_root=self.project_root)
            self.current_agent.generation = parent_generation.generation + 1
            self.current_agent.parent_id = parent_id
            
            self.best_score = parent_generation.performance_score
            
            self.tui.log_action("Agent", f"Created agent from parent {parent_id}", "SUCCESS")
            self.tui.update_agent(self.current_agent.agent_id, parent_id)
            
        else:
            # Create genesis agent
            self.current_agent = PrometheusAgent(project_root=self.project_root)
            self.current_agent.generation = 0
            self.current_agent.parent_id = None
            
            self.tui.log_action("Agent", f"Created genesis agent {self.current_agent.agent_id}", "SUCCESS")
            self.tui.update_agent(self.current_agent.agent_id, None)
    
    def _initialize_population(self):
        """Initialize the agent population from archive or create genesis agents."""
        # Select top N agents from archive
        all_gens = list(self.archive.generations.values())
        if len(all_gens) >= self.population_size:
            sorted_gens = sorted(all_gens, key=lambda g: g.performance_score, reverse=True)
            parents = sorted_gens[:self.population_size]
            self.population = []
            for parent in parents:
                agent = PrometheusAgent(project_root=self.project_root)
                agent.generation = parent.generation
                agent.parent_id = parent.parent_id
                agent.agent_id = parent.agent_id
                self.population.append(agent)
        else:
            # Not enough agents, create genesis agents
            self.population = [PrometheusAgent(project_root=self.project_root) for _ in range(self.population_size)]
            for agent in self.population:
                agent.generation = 0
                agent.parent_id = None

    def run_evolution_loop(self):
        """Run the main population-based evolution loop."""
        try:
            self.is_running = True
            self._initialize_population()
            self.tui.log_action("Evolution", "Starting population-based evolution loop", "INFO")
            while self.is_running and not self.shutdown_requested and self.current_iteration < config.max_iterations:
                self.current_iteration += 1
                self.tui.update_iteration(self.current_iteration)
                logger.info(f"Starting evolution iteration {self.current_iteration}")
                self.tui.log_action("Evolution", f"Iteration {self.current_iteration} started", "INFO")
                # Tournament selection: select parents
                all_gens = list(self.archive.generations.values())
                if len(all_gens) < self.population_size:
                    parents = self.population
                else:
                    sorted_gens = sorted(all_gens, key=lambda g: g.performance_score, reverse=True)
                    parents = sorted_gens[:self.population_size]
                # Create next generation
                next_population = []
                for _ in range(self.population_size):
                    import random
                    parent_a, parent_b = random.sample(parents, 2)
                    # Crossover
                    child_code = self.mutator.apply_crossover(
                        parent_a.get_source_code(), parent_b.get_source_code()
                    )
                    # Mutation
                    child_agent = PrometheusAgent(project_root=self.project_root)
                    child_agent.generation = max(parent_a.generation, parent_b.generation) + 1
                    child_agent.parent_id = random.choice([parent_a.agent_id, parent_b.agent_id])
                    # Apply mutation (self-reflection)
                    perf_logs = "Population-based evolution mutation"
                    improved_code_json = child_agent.self_reflect_and_improve(child_code, perf_logs)
                    # Optionally, update child_agent's code here
                    next_population.append(child_agent)
                # Evaluate all children
                evaluation_results = []
                for agent in next_population:
                    results = self.evaluator.evaluate_agent(agent)
                    evaluation_results.append((agent, results))
                # Elitism: keep top P performers
                scored_agents = [(agent, self._calculate_agent_score(results)) for agent, results in evaluation_results]
                scored_agents.sort(key=lambda x: x[1], reverse=True)
                elites = [agent for agent, _ in scored_agents[:self.elite_count]]
                # Archive all children
                for agent, _ in scored_agents:
                    self.archive.archive_agent(agent)
                self.population = elites + [agent for agent, _ in scored_agents[self.elite_count:self.population_size]]
                # Dump state for GUI
                self._dump_state_to_file()
                # Check for stagnation
                best_score = scored_agents[0][1] if scored_agents else 0.0
                if best_score > self.best_score:
                    self.best_score = best_score
                    self.stagnation_counter = 0
                else:
                    self.stagnation_counter += 1
                if self.stagnation_counter >= config.stagnation_limit:
                    self.tui.log_action("Evolution", "Stagnation detected, resetting population", "WARNING")
                    self._initialize_population()
                time.sleep(1)
            self._shutdown_evolution()
        except Exception as e:
            logger.error(f"Evolution loop failed: {e}")
            self.tui.log_action("Evolution", f"Evolution failed: {e}", "ERROR")
            self._shutdown_evolution()
    
    def _dump_state_to_file(self):
        """Dump the current system state to live_state.json for GUI sync."""
        import json
        state = {
            "iteration": self.current_iteration,
            "population": [a.agent_id for a in self.population],
            "generation_history": self.archive.get_evolution_history(),
            "tool_stats": self.mutator.tool_manager.get_tool_usage_stats() if hasattr(self.mutator, 'tool_manager') else {},
            "status": "running" if self.is_running else "stopped"
        }
        with open(os.path.join(self.project_root, "live_state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    
    def _run_evaluation_phase(self) -> List[Any]:
        """Run the evaluation phase on SWE-bench tasks."""
        self.tui.update_status("Running evaluation...")
        self.tui.log_action("Evaluation", "Starting SWE-bench evaluation", "INFO")
        
        try:
            # Get batch of tasks
            batch_size = config.max_concurrent_evaluations
            start_index = (self.current_iteration - 1) * batch_size
            
            # Run batch evaluation
            results = self.evaluator.run_batch_evaluation(
                self.current_agent,
                batch_size=batch_size,
                start_index=start_index
            )
            
            # Log summary
            summary = self.evaluator.get_evaluation_summary(results)
            self.tui.log_action(
                "Evaluation", 
                f"Completed: {summary['success_rate']:.1%} success, {summary['average_score']:.3f} avg score",
                "SUCCESS"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation phase failed: {e}")
            self.tui.log_action("Evaluation", f"Evaluation failed: {e}", "ERROR")
            return []
    
    def _calculate_agent_score(self, evaluation_results: List[Any]) -> float:
        """Calculate overall agent score from evaluation results."""
        if not evaluation_results:
            return 0.0
        
        # Calculate weighted score
        total_score = sum(result.score for result in evaluation_results)
        success_bonus = sum(0.1 for result in evaluation_results if result.success)
        
        base_score = total_score / len(evaluation_results)
        final_score = base_score + (success_bonus / len(evaluation_results))
        
        return min(1.0, final_score)
    
    def _run_self_improvement_phase(self, evaluation_results: List[Any]):
        """Run the self-improvement phase."""
        self.tui.update_status("Self-reflection and improvement...")
        self.tui.log_action("Improvement", "Starting self-reflection", "INFO")
        
        try:
            # Prepare performance logs
            performance_logs = self._generate_performance_logs(evaluation_results)
            
            # Get current source code
            source_code = self.current_agent.get_source_code()
            
            # Agent self-reflection
            self.tui.log_thought("Analyzing my performance and searching for improvements...")
            improvement_json = self.current_agent.self_reflect_and_improve(source_code, performance_logs)
            
            # Apply mutations
            self.tui.update_status("Applying code mutations...")
            mutation_result = self.mutator.apply_patch(improvement_json)
            
            if mutation_result["success"]:
                self.tui.log_action(
                    "Mutation",
                    f"Applied {len(mutation_result['patches_applied'])} mutations",
                    "SUCCESS"
                )
                
                # Create new agent generation
                self._create_next_generation(mutation_result)
            else:
                self.tui.log_action(
                    "Mutation",
                    f"Mutation failed: {mutation_result.get('error', 'Unknown error')}",
                    "ERROR"
                )
                
        except Exception as e:
            logger.error(f"Self-improvement phase failed: {e}")
            self.tui.log_action("Improvement", f"Self-improvement failed: {e}", "ERROR")
    
    def _generate_performance_logs(self, evaluation_results: List[Any]) -> str:
        """Generate performance logs for the agent."""
        if not evaluation_results:
            return "No evaluation results available"
        
        logs = []
        logs.append(f"Evaluation Summary:")
        logs.append(f"- Total tasks: {len(evaluation_results)}")
        logs.append(f"- Successful: {sum(1 for r in evaluation_results if r.success)}")
        logs.append(f"- Average score: {sum(r.score for r in evaluation_results) / len(evaluation_results):.3f}")
        logs.append(f"- Average execution time: {sum(r.execution_time for r in evaluation_results) / len(evaluation_results):.1f}s")
        
        # Add specific failures
        failures = [r for r in evaluation_results if not r.success]
        if failures:
            logs.append("\nFailure Analysis:")
            for failure in failures[:3]:  # Top 3 failures
                logs.append(f"- {failure.instance_id}: {failure.error_message or 'Unknown error'}")
        
        # Add performance trends
        if hasattr(self.current_agent, 'task_results') and self.current_agent.task_results:
            recent_scores = [r.score for r in self.current_agent.task_results[-10:]]
            if len(recent_scores) > 1:
                trend = "improving" if recent_scores[-1] > recent_scores[0] else "declining"
                logs.append(f"\nRecent trend: {trend}")
        
        return "\n".join(logs)
    
    def _create_next_generation(self, mutation_result: Dict[str, Any]):
        """Create the next generation agent after successful mutation."""
        try:
            # Create child agent
            old_agent_id = self.current_agent.agent_id
            self.current_agent = self.current_agent.create_child_agent(mutation_result)
            
            self.tui.log_action(
                "Evolution",
                f"Created new generation: {self.current_agent.agent_id}",
                "SUCCESS"
            )
            
            self.tui.update_agent(self.current_agent.agent_id, old_agent_id)
            
        except Exception as e:
            logger.error(f"Failed to create next generation: {e}")
            self.tui.log_action("Evolution", f"Generation creation failed: {e}", "ERROR")
    
    def _reset_evolution(self):
        """Reset evolution by selecting a different parent or starting fresh."""
        try:
            # Try to select a different parent
            parent_id = self.archive.select_parent_for_next_generation("diverse")
            
            if parent_id and parent_id != self.current_agent.parent_id:
                # Create agent from different lineage
                parent_generation = self.archive.generations[parent_id]
                
                self.current_agent = PrometheusAgent(project_root=self.project_root)
                self.current_agent.generation = parent_generation.generation + 1
                self.current_agent.parent_id = parent_id
                
                self.stagnation_counter = 0
                self.tui.log_action("Evolution", f"Reset to parent {parent_id}", "INFO")
            else:
                # Create completely new genesis agent
                self.current_agent = PrometheusAgent(project_root=self.project_root)
                self.current_agent.generation = 0
                self.current_agent.parent_id = None
                
                self.stagnation_counter = 0
                self.tui.log_action("Evolution", "Reset to new genesis agent", "INFO")
                
        except Exception as e:
            logger.error(f"Evolution reset failed: {e}")
            self.tui.log_action("Evolution", f"Reset failed: {e}", "ERROR")
    
    def _shutdown_evolution(self):
        """Shutdown the evolution process gracefully."""
        logger.info("Shutting down Prometheus 2.0...")
        self.is_running = False
        
        self.tui.update_status("Shutting down...")
        
        # Archive final agent if exists
        if self.current_agent:
            try:
                self.archive.archive_agent(
                    self.current_agent,
                    metadata={
                        "shutdown_iteration": self.current_iteration,
                        "final_agent": True
                    }
                )
                self.tui.log_action("Archive", f"Final agent {self.current_agent.agent_id} archived", "SUCCESS")
            except Exception as e:
                logger.warning(f"Failed to archive final agent: {e}")
        
        # Print final statistics
        stats = self.archive.get_generation_stats()
        self.tui.log_action("System", f"Evolution complete: {stats['total_generations']} generations", "INFO")
        self.tui.log_action("System", f"Best score achieved: {stats['best_score']:.3f}", "INFO")
        
        # Stop TUI
        self.tui.update_status("Shutdown complete")
        time.sleep(2)  # Give time to read final messages
    
    def run(self):
        """Main entry point to run Prometheus 2.0."""
        try:
            # Start TUI
            self.tui.start()
            
            # Initialize system
            if not self.initialize_system():
                logger.error("System initialization failed")
                return False
            
            # Run evolution
            self.run_evolution_loop()
            
            return True
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            self.shutdown_requested = True
            self._shutdown_evolution()
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            self.tui.log_action("System", f"Fatal error: {e}", "ERROR")
            return False
        finally:
            # Ensure TUI is stopped
            try:
                self.tui.stop()
            except:
                pass

def main():
    """Main entry point."""
    print("🔥 Prometheus 2.0 - The Observable Darwinian Gödeli Machine")
    print("Starting evolution...")
    
    try:
        orchestrator = PrometheusOrchestrator()
        success = orchestrator.run()
        
        if success:
            print("Evolution completed successfully")
            return 0
        else:
            print("Evolution failed")
            return 1
            
    except Exception as e:
        print(f"Fatal error: {e}")
        logger.error(f"Fatal error in main: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
