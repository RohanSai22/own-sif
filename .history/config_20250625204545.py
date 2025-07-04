"""Configuration settings for Prometheus 2.0 - The Observable Darwinian Gödeli Machine."""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    
    # Default models
    default_model: str = "groq/llama-3.3-70b-versatile"
    fallback_model: str = "groq/llama-3.1-8b-instant"
    
    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 4000

@dataclass
class SystemConfig:
    """Main system configuration."""
    # Project paths
    project_root: str = os.path.dirname(os.path.abspath(__file__))
    archive_dir: str = os.path.join(project_root, "archive", "generations")
    tools_dir: str = os.path.join(project_root, "tools", "generated_tools")
    
    # Evolution parameters
    max_iterations: int = 100
    population_size: int = 5
    mutation_rate: float = 0.3
    
    # Evaluation settings
    swe_bench_timeout: int = 300  # 5 minutes per task
    max_concurrent_evaluations: int = 3
    docker_image_base: str = "python:3.11-slim"
    
    # TUI settings
    refresh_rate: int = 10  # Hz
    log_max_lines: int = 1000
    
    # Performance tracking
    score_improvement_threshold: float = 0.05  # 5% minimum improvement
    stagnation_limit: int = 10  # iterations without improvement before reset

# Global configuration instance
config = SystemConfig()

# LLM configuration
llm_config = LLMConfig(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    groq_api_key=os.getenv("GROQ_API_KEY"),
    gemini_api_key=os.getenv("GEMINI_API_KEY"),
    openrouter_api_key=os.getenv("OPENROUTER_API_KEY")
)

# Ensure required directories exist
os.makedirs(config.archive_dir, exist_ok=True)
os.makedirs(config.tools_dir, exist_ok=True)

# SWE-bench task configuration
SWE_BENCH_CONFIG = {
    "dataset_name": "princeton-nlp/SWE-bench_Lite",
    "split": "test",
    "max_tasks_per_iteration": 5,
    "timeout_per_task": 300,
    "required_fields": ["instance_id", "repo", "base_commit", "patch", "test_patch", "problem_statement"]
}

# Tool creation templates
TOOL_TEMPLATE = '''"""Auto-generated tool: {tool_name}"""

from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

def {function_name}({parameters}) -> Any:
    """
    {docstring}
    
    Auto-generated by Prometheus 2.0 agent.
    """
    try:
        {implementation}
    except Exception as e:
        logger.error(f"Error in {function_name}: {{e}}")
        raise
'''

# Agent prompts and templates
AGENT_SYSTEM_PROMPT = """You are Prometheus 2.0, an advanced self-improving AI agent based on Darwinian Gödeli Machine principles.

Your core capabilities:
1. Solve software engineering problems using available tools
2. Analyze your own performance and identify improvement areas  
3. Search the web for new techniques and solutions
4. Create new tools when existing ones are insufficient
5. Modify your own source code to implement improvements

Your goal is to continuously evolve and improve your problem-solving abilities on the SWE-bench benchmark.

Always think step-by-step and explain your reasoning clearly."""

SELF_REFLECTION_PROMPT = """Analyze your recent performance and propose specific improvements to your source code.

Current Performance Data:
{performance_logs}

Your Current Source Code:
{source_code}

Tasks to complete:
1. Identify specific weaknesses in your current approach
2. Research better methods using web search if needed
3. Design concrete improvements to your code
4. Create new tools if required for missing capabilities

Return your response as a JSON object with this exact structure:
{
    "analysis": "Detailed analysis of current weaknesses",
    "research_findings": "Key insights from web research (if performed)",
    "proposed_changes": [
        {
            "file_path": "relative/path/to/file.py",
            "action": "replace_block",
            "identifier": "function_or_class_name",
            "new_code": "complete new implementation",
            "reasoning": "why this change improves performance"
        }
    ],
    "new_tools": [
        {
            "tool_name": "name_of_new_tool",
            "function_name": "function_name",
            "code": "complete tool implementation",
            "dependencies": ["list", "of", "required", "packages"]
        }
    ]
}"""
