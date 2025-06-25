"""Base tools for Prometheus 2.0 - API-free web search and file operations."""

import os
import subprocess
import requests
import time
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import logging

# API-free search
try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

logger = logging.getLogger(__name__)

class WebSearchTool:
    """API-free web search using DuckDuckGo."""
    
    def __init__(self):
        if not DDGS:
            raise ImportError("duckduckgo_search package required for web search")
        self.ddgs = DDGS()
    
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """
        Search the web for information.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with title, href, and body
        """
        try:
            logger.info(f"Searching web for: {query}")
            
            results = []
            search_results = self.ddgs.text(query, max_results=max_results)
            
            for result in search_results:
                results.append({
                    "title": result.get("title", ""),
                    "href": result.get("href", ""),
                    "body": result.get("body", "")
                })
            
            logger.info(f"Found {len(results)} search results")
            return results
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

class WebScrapingTool:
    """Tool for scraping and extracting text from web pages."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scrape_text(self, url: str, timeout: int = 10) -> str:
        """
        Scrape and extract clean text from a webpage.
        
        Args:
            url: URL to scrape
            timeout: Request timeout in seconds
            
        Returns:
            Clean text content from the page
        """
        try:
            logger.info(f"Scraping URL: {url}")
            
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit text length
            if len(text) > 10000:
                text = text[:10000] + "... [truncated]"
            
            logger.info(f"Extracted {len(text)} characters from {url}")
            return text
            
        except requests.RequestException as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return f"Error scraping {url}: {e}"
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return f"Error processing {url}: {e}"

class FileOperationsTool:
    """Tool for file system operations."""
    
    def __init__(self, project_root: str):
        self.project_root = project_root
    
    def read_file(self, file_path: str) -> str:
        """Read contents of a file."""
        try:
            full_path = os.path.join(self.project_root, file_path)
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Read file: {file_path} ({len(content)} chars)")
            return content
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return f"Error reading {file_path}: {e}"
    
    def write_file(self, file_path: str, content: str) -> str:
        """Write content to a file."""
        try:
            full_path = os.path.join(self.project_root, file_path)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Wrote file: {file_path} ({len(content)} chars)")
            return f"Successfully wrote {len(content)} characters to {file_path}"
        except Exception as e:
            logger.error(f"Failed to write {file_path}: {e}")
            return f"Error writing {file_path}: {e}"
    
    def list_directory(self, dir_path: str = ".") -> List[str]:
        """List contents of a directory."""
        try:
            full_path = os.path.join(self.project_root, dir_path)
            items = os.listdir(full_path)
            logger.info(f"Listed directory: {dir_path} ({len(items)} items)")
            return items
        except Exception as e:
            logger.error(f"Failed to list {dir_path}: {e}")
            return []
    
    def file_exists(self, file_path: str) -> bool:
        """Check if a file exists."""
        full_path = os.path.join(self.project_root, file_path)
        return os.path.exists(full_path)
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get information about a file."""
        full_path = os.path.join(self.project_root, file_path)
        
        if not os.path.exists(full_path):
            return {"exists": False}
        
        stat = os.stat(full_path)
        return {
            "exists": True,
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "is_file": os.path.isfile(full_path),
            "is_directory": os.path.isdir(full_path)
        }

class CommandExecutionTool:
    """Tool for executing shell commands safely."""
    
    def __init__(self, project_root: str):
        self.project_root = project_root
    
    def execute_command(self, command: str, timeout: int = 30, capture_output: bool = True) -> Dict[str, Any]:
        """
        Execute a shell command.
        
        Args:
            command: Command to execute
            timeout: Timeout in seconds
            capture_output: Whether to capture stdout/stderr
            
        Returns:
            Dictionary with command results
        """
        try:
            logger.info(f"Executing command: {command}")
            
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.project_root,
                timeout=timeout,
                capture_output=capture_output,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout if capture_output else "",
                "stderr": result.stderr if capture_output else "",
                "command": command
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {command}")
            return {
                "success": False,
                "error": "Command timed out",
                "command": command
            }
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "command": command
            }

class BaseTool:
    """Base class for all tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.usage_count = 0
        self.last_used = None
    
    def __call__(self, *args, **kwargs):
        """Execute the tool and track usage."""
        self.usage_count += 1
        self.last_used = time.time()
        return self.execute(*args, **kwargs)
    
    def execute(self, *args, **kwargs):
        """Override this method in subclasses."""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "usage_count": self.usage_count,
            "last_used": self.last_used
        }

class ToolManager:
    """Manages all available tools and provides unified access."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = project_root
        self.tools = {
            "web_search": WebSearchTool(),
            "web_scraping": WebScrapingTool(),
            "file_operations": FileOperationsTool(project_root),
            "command_execution": CommandExecutionTool(project_root)
        }
        
    def get_tool(self, tool_name: str):
        """Get a tool by name."""
        return self.tools.get(tool_name)
    
    def get_all_tools(self) -> Dict[str, Any]:
        """Get all available tools."""
        return self.tools
    
    def execute_tool(self, tool_name: str, *args, **kwargs):
        """Execute a tool by name with given arguments."""
        if tool_name == "web_search":
            return self.search_web(*args, **kwargs)
        elif tool_name == "web_scraping":
            return self.scrape_url(*args, **kwargs)
        elif tool_name == "file_read":
            return self.read_file(*args, **kwargs)
        elif tool_name == "file_write":
            return self.write_file(*args, **kwargs)
        elif tool_name == "file_list":
            return self.list_directory(*args, **kwargs)
        elif tool_name == "command":
            return self.execute_command(*args, **kwargs)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def search_web(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """Search the web for information."""
        return self.tools["web_search"].search(query, max_results)
    
    def scrape_url(self, url: str, timeout: int = 10) -> str:
        """Scrape and extract text from a webpage."""
        return self.tools["web_scraping"].scrape_text(url, timeout)
    
    def read_file(self, file_path: str) -> str:
        """Read contents of a file."""
        return self.tools["file_operations"].read_file(file_path)
    
    def write_file(self, file_path: str, content: str) -> str:
        """Write content to a file."""
        return self.tools["file_operations"].write_file(file_path, content)
    
    def list_directory(self, dir_path: str = ".") -> List[str]:
        """List contents of a directory."""
        return self.tools["file_operations"].list_directory(dir_path)
    
    def execute_command(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute a shell command."""
        return self.tools["command_execution"].execute_command(command, timeout)

# Convenient wrapper functions for tools
def web_search(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """Search the web for information."""
    tool = WebSearchTool()
    return tool.search(query, max_results)

def scrape_and_extract_text(url: str, timeout: int = 10) -> str:
    """Scrape and extract text from a webpage."""
    tool = WebScrapingTool()
    return tool.scrape_text(url, timeout)

def read_file(file_path: str, project_root: str = ".") -> str:
    """Read contents of a file."""
    tool = FileOperationsTool(project_root)
    return tool.read_file(file_path)

def write_file(file_path: str, content: str, project_root: str = ".") -> str:
    """Write content to a file."""
    tool = FileOperationsTool(project_root)
    return tool.write_file(file_path, content)

def list_directory(dir_path: str = ".", project_root: str = ".") -> List[str]:
    """List contents of a directory."""
    tool = FileOperationsTool(project_root)
    return tool.list_directory(dir_path)

def execute_shell_command(command: str, project_root: str = ".", timeout: int = 30) -> Dict[str, Any]:
    """Execute a shell command."""
    tool = CommandExecutionTool(project_root)
    return tool.execute_command(command, timeout)

if __name__ == "__main__":
    # Test the tools
    import tempfile
    
    # Test web search
    print("Testing web search...")
    try:
        results = web_search("Python machine learning", max_results=3)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']}")
            print(f"   {result['href']}")
            print(f"   {result['body'][:100]}...")
            print()
    except Exception as e:
        print(f"Web search test failed: {e}")
    
    # Test web scraping
    print("Testing web scraping...")
    try:
        text = scrape_and_extract_text("https://httpbin.org/html")
        print(f"Scraped text length: {len(text)}")
        print(f"Sample: {text[:200]}...")
    except Exception as e:
        print(f"Web scraping test failed: {e}")
    
    # Test file operations
    print("Testing file operations...")
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Write a test file
            result = write_file("test.txt", "Hello, world!", temp_dir)
            print(f"Write result: {result}")
            
            # Read the file back
            content = read_file("test.txt", temp_dir)
            print(f"Read content: {content}")
            
            # List directory
            files = list_directory(".", temp_dir)
            print(f"Directory contents: {files}")
            
        except Exception as e:
            print(f"File operations test failed: {e}")
    
    # Test command execution
    print("Testing command execution...")
    try:
        result = execute_shell_command("echo 'Hello from command line'")
        print(f"Command result: {result}")
    except Exception as e:
        print(f"Command execution test failed: {e}")
