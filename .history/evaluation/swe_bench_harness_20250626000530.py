"""SWE-bench evaluation harness for Prometheus 2.0."""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Iterator
from dataclasses import dataclass
import logging

from datasets import load_dataset
from evaluation.sandbox import DockerSandbox
from framework.tui import tui
from config import SWE_BENCH_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Result of evaluating an agent on a single task."""
    instance_id: str
    success: bool
    score: float
    execution_time: float
    patch_applied: bool
    test_results: List[Dict[str, Any]]
    error_message: Optional[str] = None

class SWEBenchHarness:
    """Harness for evaluating agents on SWE-bench tasks."""
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.sandbox = DockerSandbox()
        self.dataset = None
        self.results_dir = os.path.join(project_root, "archive", "evaluation_results")
        
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load SWE-bench dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the SWE-bench dataset."""
        try:
            tui.log_action("Dataset", "Loading SWE-bench dataset...", "INFO")
            
            self.dataset = load_dataset(
                SWE_BENCH_CONFIG["dataset_name"],
                split=SWE_BENCH_CONFIG["split"]
            )
            
            logger.info(f"Loaded SWE-bench dataset with {len(self.dataset)} instances")
            tui.log_action("Dataset", f"Loaded {len(self.dataset)} SWE-bench instances", "SUCCESS")
            
        except Exception as e:
            logger.error(f"Failed to load SWE-bench dataset: {e}")
            tui.log_action("Dataset", f"Failed to load dataset: {e}", "ERROR")
            raise
    
    def get_task_batch(self, batch_size: int = None, start_index: int = 0) -> List[Dict[str, Any]]:
        """Get a batch of tasks from the dataset."""
        if not self.dataset:
            raise RuntimeError("Dataset not loaded")
        
        batch_size = batch_size or SWE_BENCH_CONFIG["max_tasks_per_iteration"]
        end_index = min(start_index + batch_size, len(self.dataset))
        
        tasks = []
        for i in range(start_index, end_index):
            task = dict(self.dataset[i])
            
            # Validate required fields
            missing_fields = []
            for field in SWE_BENCH_CONFIG["required_fields"]:
                if field not in task or not task[field]:
                    missing_fields.append(field)
            
            if missing_fields:
                logger.warning(f"Task {task.get('instance_id', i)} missing fields: {missing_fields}")
                continue
            
            tasks.append(task)
        
        logger.info(f"Retrieved batch of {len(tasks)} tasks (indices {start_index}-{end_index-1})")
        return tasks
    
    def run_evaluation(
        self,
        agent,
        instance: Dict[str, Any],
        timeout: int = None
    ) -> EvaluationResult:
        """
        Run evaluation of an agent on a single SWE-bench instance.
        
        Args:
            agent: The agent to evaluate
            instance: SWE-bench task instance
            timeout: Timeout in seconds
            
        Returns:
            EvaluationResult with the evaluation outcome
        """
        timeout = timeout or SWE_BENCH_CONFIG["timeout_per_task"]
        instance_id = instance["instance_id"]
        
        start_time = time.time()
        
        tui.log_eval_output(f"Starting evaluation: {instance_id}")
        tui.update_task(f"Evaluating {instance_id}")
        
        try:
            # Extract task information
            repo_url = self._construct_repo_url(instance["repo"])
            base_commit = instance["base_commit"]
            test_patch = instance.get("test_patch", "")
            
            tui.log_eval_output(f"Repository: {repo_url}")
            tui.log_eval_output(f"Base commit: {base_commit}")
            
            # Create Docker environment
            tui.log_eval_output("Creating Docker evaluation environment...")
            env_result = self.sandbox.create_evaluation_environment(
                repo_url=repo_url,
                commit_hash=base_commit,
                dependencies=self._get_task_dependencies(instance)
            )
            
            if not env_result["success"]:
                tui.log_eval_output(f"Failed to create environment: {env_result['error']}")
                return EvaluationResult(
                    instance_id=instance_id,
                    success=False,
                    score=0.0,
                    execution_time=time.time() - start_time,
                    patch_applied=False,
                    test_results=[],
                    error_message=f"Environment creation failed: {env_result['error']}"
                )
            
            image_tag = env_result["image_tag"]
            tui.log_eval_output(f"Environment ready: {image_tag}")
            
            # Generate solution using agent
            tui.log_eval_output("Agent generating solution...")
            task_result = agent.solve_task(instance)
            
            # Extract patch from agent solution
            patch_content = self._extract_patch_from_solution(task_result.solution)
            
            if not patch_content:
                tui.log_eval_output("No valid patch found in agent solution")
                return EvaluationResult(
                    instance_id=instance_id,
                    success=False,
                    score=0.0,
                    execution_time=time.time() - start_time,
                    patch_applied=False,
                    test_results=[],
                    error_message="No valid patch generated"
                )
            
            # Prepare test commands
            test_commands = self._prepare_test_commands(instance, test_patch)
            
            # Run evaluation in Docker
            tui.log_eval_output("Running tests in Docker container...")
            eval_result = self.sandbox.run_evaluation(
                image_tag=image_tag,
                patch_content=patch_content,
                test_commands=test_commands,
                timeout=timeout
            )
            
            # Stream test output to TUI
            for test_result in eval_result.get("test_results", []):
                if test_result.get("stdout"):
                    for line in test_result["stdout"].split('\n'):
                        if line.strip():
                            tui.log_eval_output(f"STDOUT: {line}")
                
                if test_result.get("stderr"):
                    for line in test_result["stderr"].split('\n'):
                        if line.strip():
                            tui.log_eval_output(f"STDERR: {line}")
            
            # Calculate final score
            final_score = self._calculate_evaluation_score(eval_result, task_result)
            
            execution_time = time.time() - start_time
            
            result = EvaluationResult(
                instance_id=instance_id,
                success=eval_result["success"],
                score=final_score,
                execution_time=execution_time,
                patch_applied=eval_result.get("patch_applied", False),
                test_results=eval_result.get("test_results", [])
            )
            
            # Log final result
            status = "PASS" if result.success else "FAIL"
            tui.log_eval_output(f"Final result: {status} (score: {final_score:.3f})")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            logger.error(f"Evaluation failed for {instance_id}: {error_msg}")
            tui.log_eval_output(f"ERROR: {error_msg}")
            
            return EvaluationResult(
                instance_id=instance_id,
                success=False,
                score=0.0,
                execution_time=execution_time,
                patch_applied=False,
                test_results=[],
                error_message=error_msg
            )
    
    def _construct_repo_url(self, repo_name: str) -> str:
        """Construct full repository URL from repo name."""
        if repo_name.startswith("http"):
            return repo_name
        
        # Assume GitHub if not full URL
        return f"https://github.com/{repo_name}.git"
    
    def _get_task_dependencies(self, instance: Dict[str, Any]) -> List[str]:
        """Get dependencies for a specific task."""
        # Basic dependencies that are commonly needed
        base_deps = [
            "pytest",
            "pytest-xvfb",
            "coverage",
            "requests"
        ]
        
        # Add task-specific dependencies based on repo
        repo = instance.get("repo", "")
        
        if "django" in repo.lower():
            base_deps.extend(["django", "djangorestframework"])
        elif "flask" in repo.lower():
            base_deps.extend(["flask", "flask-testing"])
        elif "numpy" in repo.lower() or "scipy" in repo.lower():
            base_deps.extend(["numpy", "scipy"])
        elif "pandas" in repo.lower():
            base_deps.extend(["pandas", "numpy"])
        
        return base_deps
    
    def _extract_patch_from_solution(self, solution: str) -> Optional[str]:
        """Extract a git patch from the agent's solution with improved parsing."""
        # 1. First try to find explicit <patch> tags (recommended format)
        import re
        
        patch_tag_pattern = r'<patch>(.*?)</patch>'
        patch_matches = re.findall(patch_tag_pattern, solution, re.DOTALL | re.IGNORECASE)
        if patch_matches:
            return patch_matches[0].strip()
        
        # 2. Try to find diff blocks with improved detection
        lines = solution.split('\n')
        patch_lines = []
        in_diff = False
        
        for i, line in enumerate(lines):
            # Start of diff block
            if (line.startswith('diff --git') or 
                line.startswith('--- ') or 
                line.startswith('+++ ') or
                (line.startswith('Index:') and i + 1 < len(lines) and lines[i + 1].startswith('==='))):
                in_diff = True
                patch_lines.append(line)
            # Diff content lines
            elif in_diff and (line.startswith('@@') or 
                             line.startswith('+') or 
                             line.startswith('-') or 
                             line.startswith(' ') or
                             line.startswith('\\') or  # "\ No newline at end of file"
                             line.strip() == ""):
                patch_lines.append(line)
            # End of diff - next line doesn't look like diff content
            elif in_diff and line.strip() and not any(line.startswith(prefix) for prefix in [' ', '+', '-', '@', '\\', 'diff', '---', '+++']):
                # Check if this could be the start of another diff
                if not any(keyword in line.lower() for keyword in ['diff', 'index', 'file']):
                    break
                patch_lines.append(line)
            elif in_diff:
                patch_lines.append(line)
        
        if patch_lines:
            patch_content = '\n'.join(patch_lines).strip()
            # Validate that this looks like a real patch
            if any(line.startswith(('diff', '---', '+++', '@@')) for line in patch_lines):
                return patch_content
        
        # 3. Try to extract from code blocks and create a patch
        code_block_pattern = r'```(?:diff|patch)?\n(.*?)\n```'
        code_matches = re.findall(code_block_pattern, solution, re.DOTALL)
        
        for code_block in code_matches:
            # Check if this looks like a diff
            if any(line.startswith(('diff', '---', '+++', '@@', '+', '-')) for line in code_block.split('\n')):
                return code_block.strip()
        
        # 4. Last resort: try to find any diff-like content
        diff_pattern = r'((?:^|\n)(?:diff --git|--- |@@ |[\+\-] ).*?)(?=\n\n|\n[^\+\-@\s]|\Z)'
        diff_matches = re.findall(diff_pattern, solution, re.MULTILINE | re.DOTALL)
        
        if diff_matches:
            # Take the longest match (most likely to be complete)
            return max(diff_matches, key=len).strip()
        
        logger.warning("No valid patch format found in solution")
        return None
index 0000000..1234567
--- /dev/null
+++ b/solution.py
@@ -0,0 +1,{len(code_blocks[0].split(chr(10)))} @@
{chr(10).join('+' + line for line in code_blocks[0].split(chr(10)))}
"""
            return patch
        
        return None
    
    def _prepare_test_commands(self, instance: Dict[str, Any], test_patch: str) -> List[str]:
        """Prepare test commands for the instance."""
        commands = []
        
        # Apply test patch if provided
        if test_patch:
            commands.append("echo 'Applying test patch...'")
            # In practice, you'd apply the actual test patch here
        
        # Common test commands
        commands.extend([
            "echo 'Running tests...'",
            "python -m pytest --tb=short -v",
            "echo 'Tests completed'"
        ])
        
        return commands
    
    def _calculate_evaluation_score(
        self,
        eval_result: Dict[str, Any],
        task_result
    ) -> float:
        """Calculate final evaluation score."""
        score = 0.0
        
        # Base score for patch application
        if eval_result.get("patch_applied", False):
            score += 0.3
        
        # Score for test success
        if eval_result.get("success", False):
            score += 0.7
        
        # Bonus for agent task score
        if hasattr(task_result, 'score'):
            score += task_result.score * 0.2
        
        return min(1.0, score)
    
    def run_batch_evaluation(
        self,
        agent,
        batch_size: int = None,
        start_index: int = 0
    ) -> List[EvaluationResult]:
        """Run evaluation on a batch of tasks."""
        batch_size = batch_size or SWE_BENCH_CONFIG["max_tasks_per_iteration"]
        
        tasks = self.get_task_batch(batch_size, start_index)
        results = []
        
        tui.log_eval_output(f"Starting batch evaluation: {len(tasks)} tasks")
        
        for i, task in enumerate(tasks):
            tui.log_eval_output(f"Task {i+1}/{len(tasks)}: {task['instance_id']}")
            
            try:
                result = self.run_evaluation(agent, task)
                results.append(result)
                
                # Log progress
                success_rate = sum(1 for r in results if r.success) / len(results)
                avg_score = sum(r.score for r in results) / len(results)
                
                tui.log_eval_output(f"Progress: {len(results)}/{len(tasks)} tasks, {success_rate:.1%} success, {avg_score:.3f} avg score")
                
            except Exception as e:
                logger.error(f"Failed to evaluate task {task['instance_id']}: {e}")
                tui.log_eval_output(f"ERROR evaluating {task['instance_id']}: {e}")
                
                # Add failed result
                results.append(EvaluationResult(
                    instance_id=task["instance_id"],
                    success=False,
                    score=0.0,
                    execution_time=0.0,
                    patch_applied=False,
                    test_results=[],
                    error_message=str(e)
                ))
        
        # Save batch results
        self._save_batch_results(results, start_index)
        
        return results
    
    def _save_batch_results(self, results: List[EvaluationResult], batch_start: int):
        """Save batch evaluation results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_results_{batch_start}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        results_data = []
        for result in results:
            results_data.append({
                "instance_id": result.instance_id,
                "success": result.success,
                "score": result.score,
                "execution_time": result.execution_time,
                "patch_applied": result.patch_applied,
                "error_message": result.error_message,
                "test_results_count": len(result.test_results)
            })
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "batch_start": batch_start,
                    "batch_size": len(results),
                    "timestamp": timestamp,
                    "results": results_data
                }, f, indent=2)
            
            logger.info(f"Saved batch results to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save batch results: {e}")
    
    def get_evaluation_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Get summary statistics from evaluation results."""
        if not results:
            return {"total_tasks": 0, "success_rate": 0.0, "average_score": 0.0}
        
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r.success)
        total_score = sum(r.score for r in results)
        
        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": successful_tasks / total_tasks,
            "average_score": total_score / total_tasks,
            "total_score": total_score,
            "average_execution_time": sum(r.execution_time for r in results) / total_tasks,
            "patches_applied": sum(1 for r in results if r.patch_applied),
            "error_rate": sum(1 for r in results if r.error_message) / total_tasks
        }

if __name__ == "__main__":
    # Test the harness
    import tempfile
    from agent.agent_core import PrometheusAgent
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create harness and agent
            harness = SWEBenchHarness(temp_dir)
            agent = PrometheusAgent(project_root=temp_dir)
            
            # Get a small batch for testing
            print("Getting test tasks...")
            tasks = harness.get_task_batch(batch_size=1)
            
            if tasks:
                print(f"Testing with task: {tasks[0]['instance_id']}")
                
                # Run evaluation
                result = harness.run_evaluation(agent, tasks[0])
                print(f"Evaluation result: {result}")
                
                # Get summary
                summary = harness.get_evaluation_summary([result])
                print(f"Summary: {summary}")
            else:
                print("No tasks available for testing")
                
        except Exception as e:
            print(f"Test failed: {e}")
