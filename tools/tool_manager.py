"""Tool manager for Prometheus 2.0 - Dynamic tool creation and management."""

import os
import importlib
import inspect
import sys
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
import logging

from tools.base_tools import (
    web_search, scrape_and_extract_text, read_file, write_file, 
    list_directory, execute_shell_command
)
from config import config, TOOL_TEMPLATE

logger = logging.getLogger(__name__)

@dataclass
class ToolInfo:
    """Information about a tool."""
    name: str
    function: Callable
    description: str
    parameters: Dict[str, Any]
    source_file: Optional[str] = None
    is_generated: bool = False
    usage_count: int = 0

class ToolManager:
    """Manages all available tools for the agent."""
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.tools: Dict[str, ToolInfo] = {}
        self.generated_tools_dir = config.tools_dir
        
        # Ensure generated tools directory exists
        os.makedirs(self.generated_tools_dir, exist_ok=True)
        
        # Add generated tools directory to Python path
        if self.generated_tools_dir not in sys.path:
            sys.path.insert(0, self.generated_tools_dir)
        
        self._register_base_tools()
        self._load_generated_tools()
    
    def _register_base_tools(self):
        """Register the base set of tools."""
        base_tools = [
            {
                "name": "web_search",
                "function": web_search,
                "description": "Search the web for information using DuckDuckGo",
                "parameters": {
                    "query": {"type": "str", "description": "Search query"},
                    "max_results": {"type": "int", "description": "Maximum results to return", "default": 10}
                }
            },
            {
                "name": "scrape_and_extract_text",
                "function": scrape_and_extract_text,
                "description": "Scrape and extract clean text from a webpage",
                "parameters": {
                    "url": {"type": "str", "description": "URL to scrape"},
                    "timeout": {"type": "int", "description": "Request timeout", "default": 10}
                }
            },
            {
                "name": "read_file",
                "function": lambda path: read_file(path, self.project_root),
                "description": "Read contents of a file",
                "parameters": {
                    "file_path": {"type": "str", "description": "Path to file to read"}
                }
            },
            {
                "name": "write_file",
                "function": lambda path, content: write_file(path, content, self.project_root),
                "description": "Write content to a file",
                "parameters": {
                    "file_path": {"type": "str", "description": "Path to file to write"},
                    "content": {"type": "str", "description": "Content to write"}
                }
            },
            {
                "name": "list_directory",
                "function": lambda path=".": list_directory(path, self.project_root),
                "description": "List contents of a directory",
                "parameters": {
                    "dir_path": {"type": "str", "description": "Directory path", "default": "."}
                }
            },
            {
                "name": "execute_shell_command",
                "function": lambda cmd, timeout=30: execute_shell_command(cmd, self.project_root, timeout),
                "description": "Execute a shell command",
                "parameters": {
                    "command": {"type": "str", "description": "Command to execute"},
                    "timeout": {"type": "int", "description": "Timeout in seconds", "default": 30}
                }
            }
        ]
        
        for tool_config in base_tools:
            tool_info = ToolInfo(
                name=tool_config["name"],
                function=tool_config["function"],
                description=tool_config["description"],
                parameters=tool_config["parameters"],
                is_generated=False
            )
            self.tools[tool_config["name"]] = tool_info
        
        logger.info(f"Registered {len(base_tools)} base tools")
    
    def _load_generated_tools(self):
        """Load all generated tools from the tools directory."""
        if not os.path.exists(self.generated_tools_dir):
            return
        
        for filename in os.listdir(self.generated_tools_dir):
            if filename.endswith('.py') and filename != '__init__.py':
                module_name = filename[:-3]  # Remove .py extension
                try:
                    self._load_tool_module(module_name)
                except Exception as e:
                    logger.warning(f"Failed to load generated tool {module_name}: {e}")
        
        logger.info(f"Loaded {sum(1 for t in self.tools.values() if t.is_generated)} generated tools")
    
    def _load_tool_module(self, module_name: str):
        """Load a specific tool module."""
        try:
            # Import the module
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            else:
                importlib.import_module(module_name)
            
            module = sys.modules[module_name]
            
            # Find functions in the module that could be tools
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if not name.startswith('_'):  # Skip private functions
                    # Get function signature
                    sig = inspect.signature(obj)
                    parameters = {}
                    
                    for param_name, param in sig.parameters.items():
                        param_info = {"type": "str"}  # Default type
                        
                        if param.annotation != inspect.Parameter.empty:
                            param_info["type"] = param.annotation.__name__ if hasattr(param.annotation, '__name__') else str(param.annotation)
                        
                        if param.default != inspect.Parameter.empty:
                            param_info["default"] = param.default
                        
                        parameters[param_name] = param_info
                    
                    # Create tool info
                    tool_info = ToolInfo(
                        name=name,
                        function=obj,
                        description=obj.__doc__ or f"Generated tool: {name}",
                        parameters=parameters,
                        source_file=os.path.join(self.generated_tools_dir, f"{module_name}.py"),
                        is_generated=True
                    )
                    
                    self.tools[name] = tool_info
                    logger.info(f"Loaded generated tool: {name}")
        
        except Exception as e:
            logger.error(f"Failed to load tool module {module_name}: {e}")
            raise
    
    def create_new_tool(
        self,
        tool_name: str,
        function_name: str,
        code: str,
        dependencies: Optional[List[str]] = None,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Create a new tool and save it to the generated tools directory.
        
        Args:
            tool_name: Name of the tool file
            function_name: Name of the main function
            code: Implementation code
            dependencies: List of required packages
            description: Tool description
            
        Returns:
            Dictionary with creation results
        """
        try:
            # Clean tool name for filename
            safe_tool_name = tool_name.replace(' ', '_').replace('-', '_')
            tool_file_path = os.path.join(self.generated_tools_dir, f"{safe_tool_name}.py")
            
            # Parse function signature from code to determine parameters
            try:
                # Try to extract function definition
                import ast
                tree = ast.parse(code)
                
                function_def = None
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == function_name:
                        function_def = node
                        break
                
                if function_def:
                    # Extract parameter information
                    parameters = {}
                    for arg in function_def.args.args:
                        if arg.arg != 'self':  # Skip self parameter
                            parameters[arg.arg] = {"type": "Any", "description": f"Parameter {arg.arg}"}
                else:
                    # Fallback to generic parameters
                    parameters = {"*args": {"type": "Any"}, "**kwargs": {"type": "Any"}}
                    
            except:
                # If parsing fails, use generic parameters
                parameters = {"*args": {"type": "Any"}, "**kwargs": {"type": "Any"}}
            
            # Generate the complete tool file content
            tool_content = TOOL_TEMPLATE.format(
                tool_name=tool_name,
                function_name=function_name,
                parameters=", ".join(parameters.keys()) if parameters else "*args, **kwargs",
                docstring=description or f"Auto-generated tool: {tool_name}",
                implementation=code
            )
            
            # Add dependencies imports at the top if specified
            if dependencies:
                import_lines = []
                for dep in dependencies:
                    if '.' in dep:
                        import_lines.append(f"import {dep}")
                    else:
                        import_lines.append(f"import {dep}")
                
                if import_lines:
                    imports = "\n".join(import_lines) + "\n\n"
                    # Insert after the initial docstring
                    lines = tool_content.split('\n')
                    insert_pos = 3  # After the docstring
                    lines.insert(insert_pos, imports)
                    tool_content = '\n'.join(lines)
            
            # Write the tool file
            with open(tool_file_path, 'w', encoding='utf-8') as f:
                f.write(tool_content)
            
            # Load the new tool
            self._load_tool_module(safe_tool_name)
            
            logger.info(f"Created new tool: {tool_name} -> {tool_file_path}")
            
            return {
                "success": True,
                "tool_name": tool_name,
                "function_name": function_name,
                "file_path": tool_file_path,
                "message": f"Successfully created tool '{tool_name}'"
            }
            
        except Exception as e:
            logger.error(f"Failed to create tool {tool_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool_name
            }
    
    def execute_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """Execute a tool by name."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}")
        
        tool_info = self.tools[tool_name]
        tool_info.usage_count += 1
        
        try:
            result = tool_info.function(*args, **kwargs)
            logger.info(f"Executed tool '{tool_name}' successfully")
            return result
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}")
            raise
    
    def get_tool_info(self, tool_name: str) -> Optional[ToolInfo]:
        """Get information about a specific tool."""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools with their information."""
        tools_list = []
        
        for tool_name, tool_info in self.tools.items():
            tools_list.append({
                "name": tool_name,
                "description": tool_info.description,
                "parameters": tool_info.parameters,
                "is_generated": tool_info.is_generated,
                "usage_count": tool_info.usage_count,
                "source_file": tool_info.source_file
            })
        
        return tools_list
    
    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all tools."""
        stats = {
            "total_tools": len(self.tools),
            "base_tools": sum(1 for t in self.tools.values() if not t.is_generated),
            "generated_tools": sum(1 for t in self.tools.values() if t.is_generated),
            "total_usage": sum(t.usage_count for t in self.tools.values()),
            "most_used": None,
            "never_used": []
        }
        
        # Find most used tool
        most_used_tool = max(self.tools.values(), key=lambda t: t.usage_count, default=None)
        if most_used_tool and most_used_tool.usage_count > 0:
            stats["most_used"] = {
                "name": most_used_tool.name,
                "usage_count": most_used_tool.usage_count
            }
        
        # Find never used tools
        stats["never_used"] = [
            tool.name for tool in self.tools.values() if tool.usage_count == 0
        ]
        
        return stats
    
    def remove_tool(self, tool_name: str) -> bool:
        """Remove a generated tool."""
        if tool_name not in self.tools:
            return False
        
        tool_info = self.tools[tool_name]
        
        if not tool_info.is_generated:
            logger.warning(f"Cannot remove base tool: {tool_name}")
            return False
        
        try:
            # Remove from tools dictionary
            del self.tools[tool_name]
            
            # Remove file if it exists
            if tool_info.source_file and os.path.exists(tool_info.source_file):
                os.remove(tool_info.source_file)
            
            logger.info(f"Removed tool: {tool_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove tool {tool_name}: {e}")
            return False
    
    def reload_generated_tools(self):
        """Reload all generated tools from disk."""
        # Remove existing generated tools
        generated_tools = [name for name, tool in self.tools.items() if tool.is_generated]
        for tool_name in generated_tools:
            del self.tools[tool_name]
        
        # Reload from disk
        self._load_generated_tools()
        
        logger.info("Reloaded all generated tools")

if __name__ == "__main__":
    # Test the tool manager
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create tool manager
        manager = ToolManager(temp_dir)
        
        # List available tools
        print("Available tools:")
        for tool in manager.list_tools():
            print(f"- {tool['name']}: {tool['description']}")
        
        # Test executing a tool
        print("\nTesting web search tool:")
        try:
            results = manager.execute_tool("web_search", "Python programming", max_results=2)
            print(f"Found {len(results)} results")
            for result in results:
                print(f"  - {result['title']}")
        except Exception as e:
            print(f"Error: {e}")
        
        # Test creating a new tool
        print("\nCreating a new tool:")
        new_tool_code = '''
def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
'''
        
        result = manager.create_new_tool(
            tool_name="fibonacci_calculator",
            function_name="calculate_fibonacci",
            code=new_tool_code.strip(),
            description="Calculate Fibonacci numbers"
        )
        
        print(f"Tool creation result: {result}")
        
        if result["success"]:
            # Test the new tool
            print("\nTesting new tool:")
            fib_result = manager.execute_tool("calculate_fibonacci", 10)
            print(f"Fibonacci(10) = {fib_result}")
        
        # Show usage stats
        print("\nTool usage statistics:")
        stats = manager.get_tool_usage_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
