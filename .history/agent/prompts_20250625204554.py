"""Agent prompts and templates for Prometheus 2.0."""

from config import AGENT_SYSTEM_PROMPT, SELF_REFLECTION_PROMPT

# Main system prompt for the agent
SYSTEM_PROMPT = AGENT_SYSTEM_PROMPT

# Self-reflection prompt template
REFLECTION_PROMPT = SELF_REFLECTION_PROMPT

# Problem-solving prompt template
PROBLEM_SOLVING_PROMPT = """You are tasked with solving the following software engineering problem:

PROBLEM DESCRIPTION:
{problem_statement}

REPOSITORY: {repo_name}
BASE COMMIT: {base_commit}

AVAILABLE TOOLS:
{available_tools}

Your approach should be:
1. Understand the problem thoroughly
2. Use web_search to find relevant information if needed
3. Examine the codebase using read_file and list_directory
4. Develop a solution strategy
5. Implement the solution by writing/modifying files
6. Test your solution if possible

Think step by step and explain your reasoning. Always use the available tools to gather information and implement your solution.

Remember: You must generate actual code changes that solve the described problem."""

# Code analysis prompt
CODE_ANALYSIS_PROMPT = """Analyze the following code and identify potential improvements:

CODE:
{code}

CONTEXT:
- File: {file_path}
- Purpose: {purpose}
- Current issues: {issues}

Please provide:
1. Code quality assessment
2. Potential bugs or issues
3. Performance improvements
4. Best practice recommendations
5. Specific code changes needed

Focus on practical, implementable improvements."""

# Tool creation prompt
TOOL_CREATION_PROMPT = """You need to create a new tool to solve a specific problem.

PROBLEM: {problem_description}
EXISTING TOOLS: {existing_tools}

Design and implement a new tool that addresses this problem. The tool should:
1. Have a clear, descriptive name
2. Include proper documentation
3. Handle errors gracefully
4. Follow Python best practices
5. Be reusable for similar problems

Provide the complete tool implementation as Python code."""

# Performance improvement prompt
IMPROVEMENT_PROMPT = """Based on your recent performance data, identify specific areas for improvement:

PERFORMANCE DATA:
{performance_data}

CURRENT APPROACH:
{current_approach}

KNOWN ISSUES:
{known_issues}

Research and propose specific improvements to:
1. Problem-solving methodology
2. Code implementation strategies
3. Tool usage efficiency
4. Error handling and recovery

Use web search to find better approaches if needed. Provide concrete, implementable changes."""
