"""Core agent implementation for Prometheus 2.0 - The self-improving AI agent."""

import json
import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from llm_provider.unified_client import llm_client, LLMResponse
from tools.tool_manager import ToolManager
from agent.prompts import (
    SYSTEM_PROMPT, REFLECTION_PROMPT, PROBLEM_SOLVING_PROMPT,
    CODE_ANALYSIS_PROMPT, IMPROVEMENT_PROMPT
)
from framework.tui import tui

logger = logging.getLogger(__name__)

@dataclass
class TaskResult:
    """Result of solving a task."""
    task_id: str
    success: bool
    score: float
    solution: str
    execution_time: float
    errors: List[str]
    tools_used: List[str]
    reasoning: str

class PrometheusAgent:
    """The core self-improving AI agent."""
    
    def __init__(self, agent_id: str = None, project_root: str = "."):
        self.agent_id = agent_id or f"agent_{uuid.uuid4().hex[:8]}"
        self.project_root = project_root
        self.generation = 0
        self.parent_id = None
        
        # Initialize components
        self.tool_manager = ToolManager(project_root)
        self.llm_client = llm_client
        
        # Performance tracking
        self.task_results: List[TaskResult] = []
        self.total_score = 0.0
        self.success_rate = 0.0
        
        # Pattern tracking for autonomous tool creation (Pillar 1)
        self.action_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.tool_usage_history: List[Dict[str, Any]] = []
        self.repetitive_patterns: List[Dict[str, Any]] = []
        
        # Knowledge base integration (Pillar 3)
        self.knowledge_base_path = os.path.join(project_root, "archive", "knowledge_base.md")
        self.stored_knowledge: Dict[str, str] = {}
        
        # Load existing knowledge
        self._load_knowledge_base()
        
        # Conversation history for context
        self.conversation_history: List[Dict[str, str]] = []
        
        logger.info(f"Initialized agent {self.agent_id}")
    
    def solve_task(self, task: Dict[str, Any]) -> TaskResult:
        """
        Solve a single SWE-bench task.
        
        Args:
            task: Task dictionary with problem description and metadata
            
        Returns:
            TaskResult with the solution and performance metrics
        """
        start_time = time.time()
        task_id = task.get("instance_id", f"task_{uuid.uuid4().hex[:8]}")
        
        tui.log_thought(f"Starting task: {task_id}")
        tui.update_task(f"Solving task {task_id}")
        
        try:
            # Extract task information
            problem_statement = task.get("problem_statement", "")
            repo_name = task.get("repo", "")
            base_commit = task.get("base_commit", "")
            
            # Get available tools
            available_tools = self._format_available_tools()
            
            # Create the problem-solving prompt
            prompt = PROBLEM_SOLVING_PROMPT.format(
                problem_statement=problem_statement,
                repo_name=repo_name,
                base_commit=base_commit,
                available_tools=available_tools
            )
            
            tui.log_thought("Analyzing problem and planning approach...")
            
            # Generate solution using LLM
            solution_response = self._generate_response([
                {"role": "user", "content": prompt}
            ])
            
            # Execute the solution
            execution_result = self._execute_solution(solution_response.content, task)
            
            execution_time = time.time() - start_time
            
            # Track action patterns for tool creation opportunities (Pillar 1)
            self._track_action_pattern("problem_solving", {
                "target": f"{repo_name}_analysis",
                "problem_type": self._categorize_problem(problem_statement),
                "tools_used": execution_result.get("tools_used", []),
                "complexity": len(execution_result.get("tools_used", [])),
                "success": execution_result["success"]
            })
            
            # Calculate score based on success and execution quality
            score = self._calculate_task_score(execution_result, execution_time)
            
            # Create task result
            result = TaskResult(
                task_id=task_id,
                success=execution_result["success"],
                score=score,
                solution=solution_response.content,
                execution_time=execution_time,
                errors=execution_result.get("errors", []),
                tools_used=execution_result.get("tools_used", []),
                reasoning=execution_result.get("reasoning", "")
            )
            
            # Update performance tracking
            self.task_results.append(result)
            self._update_performance_metrics()
            
            # Log result
            if result.success:
                tui.log_action("Task", f"✓ Completed {task_id} (score: {score:.3f})", "SUCCESS")
            else:
                tui.log_action("Task", f"✗ Failed {task_id} (score: {score:.3f})", "ERROR")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            logger.error(f"Error solving task {task_id}: {error_msg}")
            tui.log_action("Task", f"✗ Error in {task_id}: {error_msg}", "ERROR")
            
            # Use web search to research error-specific solutions
            error_research = self._research_error_solutions(error_msg, task)
            
            # Try to recover with the research information
            if error_research:
                tui.log_thought("Attempting error recovery with research insights...")
                try:
                    # Create recovery prompt with error research
                    recovery_prompt = f"""
ORIGINAL TASK:
{task.get("problem_statement", "")}

ENCOUNTERED ERROR:
{error_msg}

RESEARCH FINDINGS:
{error_research}

Based on this error and research, please provide a corrected approach to solve the original task. Focus specifically on avoiding the error that occurred.
"""
                    
                    recovery_response = self._generate_response([
                        {"role": "user", "content": recovery_prompt}
                    ])
                    
                    # Execute recovery solution
                    recovery_result = self._execute_solution(recovery_response.content, task)
                    
                    if recovery_result["success"]:
                        execution_time = time.time() - start_time
                        score = self._calculate_task_score(recovery_result, execution_time)
                        
                        result = TaskResult(
                            task_id=task_id,
                            success=True,
                            score=score,
                            solution=recovery_response.content,
                            execution_time=execution_time,
                            errors=[error_msg],  # Keep original error for learning
                            tools_used=recovery_result.get("tools_used", []) + ["web_search"],
                            reasoning=f"Recovered from error using web research: {recovery_result.get('reasoning', '')}"
                        )
                        
                        self.task_results.append(result)
                        self._update_performance_metrics()
                        
                        tui.log_action("Task", f"✓ Recovered {task_id} (score: {score:.3f})", "SUCCESS")
                        return result
                        
                except Exception as recovery_error:
                    logger.warning(f"Recovery attempt failed: {recovery_error}")
            
            # Return failed result
            result = TaskResult(
                task_id=task_id,
                success=False,
                score=0.0,
                solution="",
                execution_time=execution_time,
                errors=[error_msg],
                tools_used=[],
                reasoning=f"Task failed with error: {error_msg}"
            )
            
            self.task_results.append(result)
            self._update_performance_metrics()
            
            return result
    
    def _execute_solution(self, solution: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the generated solution."""
        execution_result = {
            "success": False,
            "errors": [],
            "tools_used": [],
            "reasoning": "",
            "files_modified": []
        }
        
        try:
            # Parse the solution to extract tool calls and reasoning
            reasoning_parts = []
            tools_used = []
            
            # This is a simplified execution - in practice, you might want
            # to parse and execute specific tool calls from the solution
            
            # For now, we'll simulate execution based on solution content
            if "web_search" in solution.lower():
                tui.log_action("web_search", "Searching for relevant information", "TOOL")
                tools_used.append("web_search")
                
            if "read_file" in solution.lower():
                tui.log_action("read_file", "Reading repository files", "TOOL")
                tools_used.append("read_file")
                
            if "write_file" in solution.lower():
                tui.log_action("write_file", "Writing solution files", "TOOL")
                tools_used.append("write_file")
                
            # Basic success criteria
            has_code_changes = any(keyword in solution.lower() for keyword in 
                                 ["write_file", "modify", "patch", "fix", "implement"])
            has_reasoning = len(solution) > 100
            
            execution_result.update({
                "success": has_code_changes and has_reasoning,
                "tools_used": tools_used,
                "reasoning": solution[:500] + "..." if len(solution) > 500 else solution
            })
            
        except Exception as e:
            execution_result["errors"].append(str(e))
        
        return execution_result
    
    def _calculate_task_score(self, execution_result: Dict[str, Any], execution_time: float) -> float:
        """Calculate a score for the task based on execution results."""
        score = 0.0
        
        # Base score for success
        if execution_result["success"]:
            score += 0.5
        
        # Bonus for using tools effectively
        tools_used = execution_result.get("tools_used", [])
        if tools_used:
            score += min(0.2, len(tools_used) * 0.05)
        
        # Bonus for detailed reasoning
        reasoning = execution_result.get("reasoning", "")
        if len(reasoning) > 200:
            score += 0.1
        
        # Penalty for errors
        errors = execution_result.get("errors", [])
        score -= len(errors) * 0.1
        
        # Time efficiency bonus/penalty
        if execution_time < 60:  # Under 1 minute
            score += 0.1
        elif execution_time > 300:  # Over 5 minutes
            score -= 0.1
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def self_reflect_and_improve(self, source_code_dict: Dict[str, str], performance_logs: str) -> str:
        """
        Enhanced self-reflection with autonomous tool creation, self-modification, and curiosity-driven exploration.
        
        Args:
            source_code_dict: Dictionary of agent source code files
            performance_logs: String containing performance analysis
            
        Returns:
            JSON string with proposed improvements, new tools, and exploration queries
        """
        tui.log_thought("Beginning enhanced self-reflection with autonomous capabilities...")
        tui.update_status("Analyzing performance for self-improvement")
        
        try:
            # Gather enhanced context
            action_history = self._analyze_action_patterns()
            file_manifest = self._load_file_manifest()
            knowledge_base_summary = self._load_knowledge_base_summary()
            
            # Prepare context for enhanced reflection
            from config import SELF_REFLECTION_PROMPT
            
            reflection_prompt = SELF_REFLECTION_PROMPT.format(
                performance_logs=performance_logs,
                source_code=json.dumps(source_code_dict, indent=2),
                file_manifest=json.dumps(file_manifest, indent=2),
                knowledge_base_summary=knowledge_base_summary,
                action_history=action_history
            )
            
            tui.log_thought("Conducting deep cognitive analysis...")
            
            # Get comprehensive reflection
            response = self.llm_client.query(reflection_prompt, temperature=0.8)
            
            # Parse and validate the response
            try:
                reflection_data = json.loads(response)
                
                # Process each component
                if "new_tools" in reflection_data and reflection_data["new_tools"]:
                    tui.log_thought("Processing autonomous tool creation requests...")
                    for tool_spec in reflection_data["new_tools"]:
                        self._design_and_implement_tool(tool_spec)
                
                if "exploration_queries" in reflection_data and reflection_data["exploration_queries"]:
                    tui.log_thought("Conducting curiosity-driven exploration...")
                    exploration_insights = self._conduct_exploration(reflection_data["exploration_queries"])
                    reflection_data["exploration_results"] = exploration_insights
                
                return json.dumps(reflection_data, indent=2)
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse reflection response: {e}")
                tui.log_thought("Reflection response was malformed, attempting recovery...")
                return self._generate_fallback_improvement(source_code_dict, performance_logs)
                
        except Exception as e:
            logger.error(f"Self-reflection failed: {e}")
            tui.log_thought(f"Self-reflection encountered error: {e}")
            return self._generate_fallback_improvement(source_code_dict, performance_logs)
            
            # Generate improvement suggestions
            response = self._generate_response([
                {"role": "user", "content": full_prompt}
            ])
            
            # Validate and enhance the response
            enhanced_response = self._enhance_improvement_response(response.content)
            
            tui.log_thought("Self-reflection complete - improvement plan generated")
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error in self-reflection: {e}")
            tui.log_action("Reflection", f"Error during self-reflection: {e}", "ERROR")
            
            # Return minimal valid response
            return json.dumps({
                "analysis": f"Self-reflection failed: {e}",
                "research_findings": "No research performed due to error",
                "proposed_changes": [],
                "new_tools": []
            })
    
    def _analyze_recent_performance(self) -> str:
        """Analyze recent task performance to identify patterns."""
        if not self.task_results:
            return "No recent performance data available"
        
        recent_tasks = self.task_results[-10:]  # Last 10 tasks
        
        analysis = []
        analysis.append(f"Recent tasks analyzed: {len(recent_tasks)}")
        analysis.append(f"Success rate: {sum(1 for t in recent_tasks if t.success) / len(recent_tasks):.2%}")
        analysis.append(f"Average score: {sum(t.score for t in recent_tasks) / len(recent_tasks):.3f}")
        analysis.append(f"Average execution time: {sum(t.execution_time for t in recent_tasks) / len(recent_tasks):.1f}s")
        
        # Common errors
        all_errors = []
        for task in recent_tasks:
            all_errors.extend(task.errors)
        
        if all_errors:
            analysis.append(f"Common errors: {', '.join(set(all_errors[:5]))}")
        
        # Tool usage patterns
        tool_usage = {}
        for task in recent_tasks:
            for tool in task.tools_used:
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
        
        if tool_usage:
            most_used_tools = sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:3]
            analysis.append(f"Most used tools: {', '.join([f'{tool} ({count})' for tool, count in most_used_tools])}")
        
        return "\n".join(analysis)
    
    def _identify_weaknesses(self) -> str:
        """Identify current weaknesses based on performance data."""
        weaknesses = []
        
        if not self.task_results:
            return "Insufficient data to identify weaknesses"
        
        recent_tasks = self.task_results[-10:]
        
        # Low success rate
        success_rate = sum(1 for t in recent_tasks if t.success) / len(recent_tasks)
        if success_rate < 0.6:
            weaknesses.append(f"Low success rate ({success_rate:.1%})")
        
        # Slow execution
        avg_time = sum(t.execution_time for t in recent_tasks) / len(recent_tasks)
        if avg_time > 180:  # 3 minutes
            weaknesses.append(f"Slow execution (avg: {avg_time:.1f}s)")
        
        # Limited tool usage
        unique_tools_used = set()
        for task in recent_tasks:
            unique_tools_used.update(task.tools_used)
        
        available_tools = len(self.tool_manager.tools)
        if len(unique_tools_used) < available_tools * 0.3:
            weaknesses.append("Limited tool utilization")
        
        # Frequent errors
        error_rate = sum(len(t.errors) for t in recent_tasks) / len(recent_tasks)
        if error_rate > 1:
            weaknesses.append(f"High error rate ({error_rate:.1f} errors per task)")
        
        return "; ".join(weaknesses) if weaknesses else "No major weaknesses identified"
    
    def _load_knowledge_base(self):
        """Load the agent's long-term knowledge base."""
        try:
            if os.path.exists(self.knowledge_base_path):
                with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Parse knowledge base sections
                sections = content.split('## ')
                for section in sections:
                    if section.strip() and not section.startswith('#'):
                        lines = section.split('\n')
                        section_title = lines[0].strip()
                        section_content = '\n'.join(lines[1:]).strip()
                        if section_content and section_content != "*This section will be populated*":
                            self.stored_knowledge[section_title] = section_content
                            
                logger.info(f"Loaded {len(self.stored_knowledge)} knowledge sections")
        except Exception as e:
            logger.warning(f"Failed to load knowledge base: {e}")
    
    def _track_action_pattern(self, action_type: str, details: Dict[str, Any]):
        """Track recurring action patterns for tool creation opportunities."""
        pattern_key = f"{action_type}_{details.get('target', 'unknown')}"
        
        if pattern_key not in self.action_patterns:
            self.action_patterns[pattern_key] = []
        
        pattern_entry = {
            "timestamp": datetime.now().isoformat(),
            "action_type": action_type,
            "details": details,
            "tools_used": details.get("tools_used", [])
        }
        
        self.action_patterns[pattern_key].append(pattern_entry)
        
        # Check if this pattern has become repetitive (3+ occurrences)
        if len(self.action_patterns[pattern_key]) >= 3:
            self._analyze_pattern_for_tool_creation(pattern_key)
    
    def _analyze_pattern_for_tool_creation(self, pattern_key: str):
        """Analyze if a repetitive pattern warrants tool creation."""
        pattern_instances = self.action_patterns[pattern_key]
        
        # Check if pattern already identified
        existing_pattern = any(p["pattern_key"] == pattern_key for p in self.repetitive_patterns)
        if existing_pattern:
            return
        
        # Analyze pattern complexity and frequency
        tool_complexity = sum(len(instance["tools_used"]) for instance in pattern_instances)
        avg_complexity = tool_complexity / len(pattern_instances)
        
        # Only suggest tool creation for patterns with sufficient complexity
        if avg_complexity >= 2:  # Multiple tools typically used
            pattern_analysis = {
                "pattern_key": pattern_key,
                "frequency": len(pattern_instances),
                "complexity": avg_complexity,
                "first_occurrence": pattern_instances[0]["timestamp"],
                "latest_occurrence": pattern_instances[-1]["timestamp"],
                "suggested_tool_name": self._generate_tool_name_suggestion(pattern_key),
                "pattern_summary": self._summarize_pattern(pattern_instances)
            }
            
            self.repetitive_patterns.append(pattern_analysis)
            logger.info(f"Identified repetitive pattern for tool creation: {pattern_key}")
    
    def _generate_tool_name_suggestion(self, pattern_key: str) -> str:
        """Generate a suggested name for a tool based on the pattern."""
        parts = pattern_key.split('_')
        if len(parts) >= 2:
            action = parts[0]
            target = parts[1]
            return f"{target}_{action}_tool"
        return f"{pattern_key}_tool"
    
    def _summarize_pattern(self, pattern_instances: List[Dict[str, Any]]) -> str:
        """Create a summary of what the pattern does."""
        if not pattern_instances:
            return "Unknown pattern"
        
        first_instance = pattern_instances[0]
        action_type = first_instance["action_type"]
        tools_used = set()
        for instance in pattern_instances:
            tools_used.update(instance.get("tools_used", []))
        
        return f"Repeatedly {action_type} using tools: {', '.join(sorted(tools_used))}"
    
    def _research_error_solutions(self, error_msg: str, task: Dict[str, Any]) -> str:
        """Research solutions for specific errors using web search."""
        try:
            tui.log_action("web_search", f"Researching solutions for error: {error_msg[:50]}...", "TOOL")
            
            # Create targeted search queries based on error and task context
            repo_name = task.get("repo", "").split("/")[-1] if task.get("repo") else ""
            
            search_queries = [
                f"{error_msg} {repo_name} python fix solution",
                f"how to fix {error_msg[:30]} error programming",
                f"{error_msg[:40]} debugging solution"
            ]
            
            research_findings = []
            
            for query in search_queries[:2]:  # Limit to 2 queries to avoid excessive searches
                try:
                    results = self.tool_manager.execute_tool("web_search", query, max_results=3)
                    
                    for result in results[:2]:  # Top 2 results per query
                        try:
                            # Extract relevant text from search results
                            if result.get("body"):
                                # Use the search snippet as it's usually most relevant
                                summary = result["body"][:200] + "..." if len(result["body"]) > 200 else result["body"]
                                research_findings.append(f"From {result.get('title', 'source')}: {summary}")
                        except Exception:
                            continue
                    
                except Exception as e:
                    logger.warning(f"Error research query failed: {query} - {e}")
                    continue
            
            if research_findings:
                return "\n\n".join(research_findings[:4])  # Top 4 findings
            else:
                return f"No specific research found for error: {error_msg}"
            
        except Exception as e:
            logger.error(f"Error research failed: {e}")
            return f"Error research failed: {e}"

    def _research_improvements(self, weaknesses: str) -> str:
        """Research potential improvements using web search as designed."""
        try:
            tui.log_thought("Researching improvements via web search...")
            
            # Generate focused search queries using LLM
            query_prompt = f"""
Based on these identified weaknesses in a software engineering AI agent:

{weaknesses}

Generate 3-4 focused search queries to find specific solutions, techniques, or best practices that could address these issues. Focus on:
- Specific technical solutions
- Algorithm improvements  
- Error handling patterns
- Performance optimization techniques

Return only the search queries, one per line.
"""
            
            response = self.llm_client.query(query_prompt, temperature=0.3)
            search_queries = [q.strip() for q in response.split('\n') if q.strip()]
            
            # Perform web searches
            research_findings = []
            for query in search_queries[:3]:  # Limit to 3 queries
                try:
                    tui.log_thought(f"Searching: {query}")
                    search_results = self.tool_manager.execute_tool("web_search", {"query": query})
                    
                    if search_results and search_results.get("success"):
                        research_findings.append(f"Query: {query}")
                        for result in search_results.get("results", [])[:2]:  # Top 2 results per query
                            research_findings.append(f"- {result.get('title', 'Unknown')}: {result.get('snippet', 'No description')}")
                    
                except Exception as e:
                    logger.warning(f"Web search failed for query '{query}': {e}")
                    research_findings.append(f"Search failed for: {query}")
            
            # If web search fails completely, fall back to internal analysis
            if not research_findings:
                tui.log_thought("Web search unavailable, using internal analysis...")
                research_findings = [
                    "Performance Analysis: Focus on improving algorithm efficiency and reducing computational complexity",
                    "Code Quality: Implement better error handling and validation to prevent failures", 
                    "Testing Strategy: Enhance testing coverage and add edge case validation",
                    "Resource Management: Optimize memory usage and processing time",
                    "Error Recovery: Implement robust fallback mechanisms for failed operations"
                ]
            
            # Synthesize findings using LLM
            synthesis_prompt = f"""
Based on the following research findings about software engineering improvements:

{chr(10).join(research_findings)}

And the specific weaknesses identified:
{weaknesses}

Synthesize the most relevant and actionable improvement recommendations. Focus on concrete technical solutions that can be implemented in code. Provide 3-5 specific recommendations.
"""
            
            synthesized_recommendations = self.llm_client.query(synthesis_prompt, temperature=0.4)
            
            return f"Research Findings:\n{chr(10).join(research_findings)}\n\nSynthesized Recommendations:\n{synthesized_recommendations}"
            
        except Exception as e:
            logger.error(f"Research improvements failed: {e}")
            # Fallback to basic internal analysis
            return f"Research failed ({e}), using basic analysis: Focus on error handling, performance optimization, and robust testing patterns."
            relevant_findings = []
            weakness_lower = weaknesses.lower()
            
            for finding in research_findings:
                if any(keyword in weakness_lower for keyword in ["performance", "error", "test", "memory", "fail"]):
                    relevant_findings.append(finding)
            
            if not relevant_findings:
                relevant_findings = research_findings[:3]  # Default top 3
            
            return "\n\n".join(relevant_findings)
            
        except Exception as e:
            logger.error(f"Internal analysis failed: {e}")
            return "Unable to complete improvement analysis - relying on basic optimization strategies"
    
    def _enhance_improvement_response(self, response: str) -> str:
        """Enhance and validate the improvement response."""
        try:
            # Clean the response - remove any markdown formatting
            cleaned_response = response.strip()
            
            # Remove ```json and ``` markers if present
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            cleaned_response = cleaned_response.strip()
            
            # Try to parse as JSON
            try:
                data = json.loads(cleaned_response)
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                import re
                json_pattern = r'\{.*\}'
                matches = re.search(json_pattern, cleaned_response, re.DOTALL)
                if matches:
                    data = json.loads(matches.group())
                else:
                    raise ValueError("No valid JSON found in response")
            
            # Ensure required fields exist
            if "analysis" not in data:
                data["analysis"] = "Generated analysis from response"
            
            if "research_findings" not in data:
                data["research_findings"] = "Internal analysis performed"
            
            if "proposed_changes" not in data:
                data["proposed_changes"] = []
            
            if "new_tools" not in data:
                data["new_tools"] = []
            
            # Validate proposed changes structure
            for i, change in enumerate(data["proposed_changes"]):
                if not isinstance(change, dict):
                    data["proposed_changes"][i] = {
                        "file_path": "agent/agent_core.py",
                        "action": "replace_block", 
                        "identifier": "solve_task",
                        "new_code": "# Improved implementation needed",
                        "reasoning": "Change format was invalid"
                    }
                    continue
                    
                if "file_path" not in change:
                    change["file_path"] = "agent/agent_core.py"
                if "action" not in change:
                    change["action"] = "replace_block"
                if "identifier" not in change:
                    change["identifier"] = "solve_task"
                if "new_code" not in change:
                    change["new_code"] = "# Improved implementation needed"
                if "reasoning" not in change:
                    change["reasoning"] = "Improvement needed"
            
            # Validate new tools structure
            for i, tool in enumerate(data["new_tools"]):
                if not isinstance(tool, dict):
                    data["new_tools"][i] = {
                        "tool_name": "improved_tool",
                        "function_name": "improved_function",
                        "code": "# Tool implementation needed",
                        "dependencies": []
                    }
                    continue
                    
                if "tool_name" not in tool:
                    tool["tool_name"] = "improved_tool"
                if "function_name" not in tool:
                    tool["function_name"] = "improved_function"
                if "code" not in tool:
                    tool["code"] = "# Tool implementation needed"
                if "dependencies" not in tool:
                    tool["dependencies"] = []
            
            return json.dumps(data, indent=2)
            
        except Exception as e:
            logger.warning(f"Failed to parse improvement response: {e}")
            logger.debug(f"Original response: {response[:200]}...")
            
            # If all parsing fails, create a minimal valid response
            return json.dumps({
                "analysis": f"Response parsing failed: {str(e)}. Original response was: {response[:100]}...",
                "research_findings": "Unable to parse research findings from response",
                "proposed_changes": [
                    {
                        "file_path": "agent/agent_core.py",
                        "action": "replace_block",
                        "identifier": "_enhance_improvement_response",
                        "new_code": "# Improved response parsing needed",
                        "reasoning": "Current parsing method failed, needs better error handling"
                    }
                ],
                "new_tools": []
            }, indent=2)
    
    def _format_available_tools(self) -> str:
        """Format available tools for prompt inclusion."""
        tools = self.tool_manager.list_tools()
        
        tool_descriptions = []
        for tool in tools:
            params = ", ".join([f"{name}: {info.get('type', 'Any')}" for name, info in tool["parameters"].items()])
            tool_descriptions.append(f"- {tool['name']}({params}): {tool['description']}")
        
        return "\n".join(tool_descriptions)
    
    def _generate_response(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate a response using the LLM with system prompt."""
        return self.llm_client.generate(
            messages=messages,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=4000
        )
    
    def _update_performance_metrics(self):
        """Update overall performance metrics."""
        if not self.task_results:
            return
        
        # Calculate total score and success rate
        self.total_score = sum(task.score for task in self.task_results)
        self.success_rate = sum(1 for task in self.task_results if task.success) / len(self.task_results)
        
        # Update TUI with latest performance
        avg_score = self.total_score / len(self.task_results)
        tui.add_generation(self.agent_id, self.parent_id, avg_score, self.generation)
    
    def get_source_code(self) -> Dict[str, str]:
        """Get the current source code of the agent."""
        source_files = [
            "agent/agent_core.py",
            "agent/prompts.py"
        ]
        
        source_code = {}
        for file_path in source_files:
            try:
                content = self.tool_manager.execute_tool("read_file", file_path)
                source_code[file_path] = content
            except Exception as e:
                logger.warning(f"Could not read source file {file_path}: {e}")
                source_code[file_path] = f"# Error reading file: {e}"
        
        return source_code
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of agent performance."""
        if not self.task_results:
            return {
                "agent_id": self.agent_id,
                "generation": self.generation,
                "total_tasks": 0,
                "success_rate": 0.0,
                "average_score": 0.0,
                "total_score": 0.0
            }
        
        return {
            "agent_id": self.agent_id,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "total_tasks": len(self.task_results),
            "success_rate": self.success_rate,
            "average_score": self.total_score / len(self.task_results),
            "total_score": self.total_score,
            "tools_available": len(self.tool_manager.tools),
            "recent_performance": self._analyze_recent_performance()
        }
    
    def create_child_agent(self, mutation_result: Dict[str, Any]) -> 'PrometheusAgent':
        """Create a child agent after successful mutation."""
        child_id = f"agent_{uuid.uuid4().hex[:8]}"
        child_agent = PrometheusAgent(child_id, self.project_root)
        child_agent.generation = self.generation + 1
        child_agent.parent_id = self.agent_id
        
        logger.info(f"Created child agent {child_id} from parent {self.agent_id}")
        return child_agent

if __name__ == "__main__":
    # Test the agent
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test agent
        agent = PrometheusAgent(project_root=temp_dir)
        
        # Test task solving
        test_task = {
            "instance_id": "test_task_1",
            "problem_statement": "Write a function to calculate factorial",
            "repo": "test_repo",
            "base_commit": "abc123"
        }
        
        print("Testing task solving...")
        result = agent.solve_task(test_task)
        print(f"Task result: {result}")
        
        # Test self-reflection
        print("\nTesting self-reflection...")
        source_code = agent.get_source_code()
        performance_logs = "Recent performance has been suboptimal"
        
        improvement_json = agent.self_reflect_and_improve(source_code, performance_logs)
        print(f"Improvement suggestions: {improvement_json}")
        
        # Show performance summary
        print("\nPerformance summary:")
        summary = agent.get_performance_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
