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
import json

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
        
        # Initialize components
        self.tui = tui
        self.mutator = CodeMutator(self.project_root)
        self.evaluator = SWEBenchHarness(self.project_root)
        self.archive = AgentArchive(self.project_root)
        
        # Population-based evolution
        self.population: List[PrometheusAgent] = []
        self.population_scores: List[float] = []
        self.generation = 0
        
        # Current agent (for backward compatibility)
        self.current_agent: Optional[PrometheusAgent] = None
        
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
        """Initialize the current agent (backward compatibility method)."""
        if not self.population:
            # Fallback to single-agent mode
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
        else:
            # Use first agent from population
            self.current_agent = self.population[0]
    
    def run_evolution_loop(self):
        """Run the main population-based evolution loop."""
        try:
            self.is_running = True
            self.tui.log_action("Evolution", "Starting population-based evolution loop", "INFO")
            
            # Initialize population
            self._initialize_population()
            
            while self.is_running and not self.shutdown_requested and self.generation < config.max_iterations:
                self.generation += 1
                self.current_iteration = self.generation  # For compatibility
                self.tui.update_iteration(self.generation)
                
                logger.info(f"Starting evolution generation {self.generation}")
                self.tui.log_action("Evolution", f"Generation {self.generation} started", "INFO")
                
                # 1. Evaluate population
                all_evaluation_results = self._evaluate_population()
                
                if self.shutdown_requested:
                    break
                
                # 2. Archive current population
                self._archive_population(all_evaluation_results)
                
                # 3. Prune archive periodically
                if self.generation % 10 == 0:
                    self.tui.log_action("Archive", "Pruning archive to manage size", "INFO")
                    self.archive.prune_archive(max_generations=50)
                
                # 4. Check for improvement
                current_best_score = max(self.population_scores) if self.population_scores else 0.0
                avg_score = sum(self.population_scores) / len(self.population_scores) if self.population_scores else 0.0
                
                self.tui.log_action("Evolution", f"Generation {self.generation}: Best={current_best_score:.3f}, Avg={avg_score:.3f}", "INFO")
                
                # 5. Check for stagnation
                improvement = current_best_score - self.best_score
                if improvement <= config.score_improvement_threshold:
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = 0
                
                if self.stagnation_counter >= config.stagnation_limit:
                    logger.info("Population stagnation detected, introducing diversity")
                    self.tui.log_action("Evolution", "Stagnation detected, introducing diversity", "INFO")
                    
                    # Add random agents to population for diversity
                    num_random = config.population_size // 3
                    for i in range(num_random):
                        random_agent = PrometheusAgent(project_root=self.project_root)
                        random_agent.generation = self.generation
                        random_agent.parent_id = None
                        self.population[-1-i] = random_agent  # Replace worst agents
                    
                    self.stagnation_counter = 0
                
                if self.shutdown_requested:
                    break
                
                # 6. Selection and reproduction
                if self.generation < config.max_iterations:
                    self.tui.update_status("Selecting parents and creating next generation...")
                    parent_indices = self._select_parents()
                    new_population = self._crossover_and_mutate(parent_indices)
                    
                    self.population = new_population
                    self.population_scores = [0.0] * len(self.population)
                    
                    # Update current agent for compatibility
                    self.current_agent = self.population[0] if self.population else None
                
                # Brief pause between generations
                time.sleep(1)
            
            # Shutdown
            self._shutdown_evolution()
            
        except Exception as e:
            logger.error(f"Evolution loop failed: {e}")
            self.tui.log_action("Evolution", f"Evolution failed: {e}", "ERROR")
            self._shutdown_evolution()
    
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
    
    def _write_live_state(self):
        """Write current state to a file for the GUI to read."""
        try:
            live_state = {
                "timestamp": datetime.now().isoformat(),
                "generation": self.generation,
                "current_iteration": self.current_iteration,
                "is_running": self.is_running,
                "population_size": len(self.population),
                "population_scores": self.population_scores,
                "best_score": self.best_score,
                "stagnation_counter": self.stagnation_counter,
                "current_agent_id": self.current_agent.agent_id if self.current_agent else None,
                "archive_size": len(self.archive.generations),
                "status": "evolving" if self.is_running else "stopped"
            }
            
            live_state_path = os.path.join(self.project_root, "archive", "live_state.json")
            with open(live_state_path, 'w', encoding='utf-8') as f:
                json.dump(live_state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to write live state: {e}")

    def _archive_population(self, all_evaluation_results: List[List[Any]]):
        """Archive the current population and their evaluation results."""
        try:
            self.tui.log_action("Archive", "Archiving current population...", "INFO")
            
            for agent, evaluation_results in zip(self.population, all_evaluation_results):
                # Calculate agent score
                agent_score = self._calculate_agent_score(evaluation_results)
                agent.performance_score = agent_score
                
                # Archive agent
                self.archive.archive_agent(agent, metadata={"generation": self.generation})
            
            # Update population scores
            self.population_scores = [agent.performance_score for agent in self.population]
            
            # Update best score
            self.best_score = max(self.population_scores, default=0.0)
            
            self.tui.log_action("Archive", f"Archived {len(self.population)} agents", "SUCCESS")
            
            # Write live state for GUI
            self._write_live_state()
            
        except Exception as e:
            logger.error(f"Failed to archive population: {e}")
            self.tui.log_action("Archive", f"Archiving failed: {e}", "ERROR")
    
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
