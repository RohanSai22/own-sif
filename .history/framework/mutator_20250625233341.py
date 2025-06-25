"""Code mutation system for Prometheus 2.0 - Applies self-modification patches safely."""

import ast
import json
import os
import re
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MutationPatch:
    """Represents a code mutation patch."""
    file_path: str
    action: str  # "replace_block", "insert_after", "insert_before", "delete_block"
    identifier: str  # function/class name or line number
    new_code: str
    reasoning: str
    backup_created: bool = False

class CodeMutator:
    """Handles safe application of code mutations."""
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.backup_dir = os.path.join(project_root, "archive", "backups")
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def apply_patch(self, patch_json: str) -> Dict[str, Any]:
        """
        Apply a JSON patch to modify agent source code.
        
        Args:
            patch_json: JSON string containing mutation instructions
            
        Returns:
            Dictionary with results of the mutation
        """
        try:
            patch_data = json.loads(patch_json)
            
            # Validate patch structure
            if "proposed_changes" not in patch_data:
                raise ValueError("Patch must contain 'proposed_changes' field")
            
            results = {
                "success": False,
                "patches_applied": [],
                "errors": [],
                "backups_created": [],
                "files_modified": []
            }
            
            # Create timestamp for this mutation session
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_backup_dir = os.path.join(self.backup_dir, f"mutation_{timestamp}")
            os.makedirs(session_backup_dir, exist_ok=True)
            
            # Apply each patch
            for i, change in enumerate(patch_data["proposed_changes"]):
                try:
                    patch = MutationPatch(
                        file_path=change["file_path"],
                        action=change["action"],
                        identifier=change["identifier"],
                        new_code=change["new_code"],
                        reasoning=change.get("reasoning", "No reasoning provided")
                    )
                    
                    result = self._apply_single_patch(patch, session_backup_dir)
                    
                    if result["success"]:
                        results["patches_applied"].append({
                            "patch_index": i,
                            "file": patch.file_path,
                            "action": patch.action,
                            "identifier": patch.identifier,
                            "reasoning": patch.reasoning
                        })
                        
                        if patch.file_path not in results["files_modified"]:
                            results["files_modified"].append(patch.file_path)
                        
                        if result.get("backup_path"):
                            results["backups_created"].append(result["backup_path"])
                    else:
                        results["errors"].append({
                            "patch_index": i,
                            "error": result["error"],
                            "file": patch.file_path
                        })
                        
                except Exception as e:
                    results["errors"].append({
                        "patch_index": i,
                        "error": str(e),
                        "file": change.get("file_path", "unknown")
                    })
            
            # Create new tools if specified
            if "new_tools" in patch_data:
                for tool_data in patch_data["new_tools"]:
                    try:
                        tool_result = self._create_new_tool(tool_data, session_backup_dir)
                        if tool_result["success"]:
                            results["patches_applied"].append({
                                "type": "new_tool",
                                "tool_name": tool_data["tool_name"],
                                "file_path": tool_result["file_path"]
                            })
                        else:
                            results["errors"].append({
                                "type": "new_tool",
                                "error": tool_result["error"],
                                "tool_name": tool_data["tool_name"]
                            })
                    except Exception as e:
                        results["errors"].append({
                            "type": "new_tool",
                            "error": str(e),
                            "tool_name": tool_data.get("tool_name", "unknown")
                        })
            
            # Validate all modified files are syntactically correct
            validation_errors = self._validate_modified_files(results["files_modified"])
            if validation_errors:
                results["errors"].extend(validation_errors)
                # Restore from backups if validation fails
                self._restore_from_backups(results["backups_created"])
                results["success"] = False
                results["message"] = "Mutations rolled back due to syntax errors"
            else:
                results["success"] = len(results["patches_applied"]) > 0
                results["message"] = f"Successfully applied {len(results['patches_applied'])} patches"
            
            return results
            
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Invalid JSON in patch: {e}",
                "patches_applied": [],
                "errors": [{"error": f"JSON decode error: {e}"}]
            }
        except Exception as e:
            logger.error(f"Error applying patch: {e}")
            return {
                "success": False,
                "error": str(e),
                "patches_applied": [],
                "errors": [{"error": str(e)}]
            }
    
    def _apply_single_patch(self, patch: MutationPatch, backup_dir: str) -> Dict[str, Any]:
        """Apply a single mutation patch to a file."""
        file_path = os.path.join(self.project_root, patch.file_path)
        
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }
        
        # Create backup
        backup_path = self._create_backup(file_path, backup_dir)
        patch.backup_created = True
        
        try:
            # Read current file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Apply the patch based on action type
            if patch.action == "replace_block":
                new_content = self._replace_code_block(content, patch.identifier, patch.new_code)
            elif patch.action == "insert_after":
                new_content = self._insert_code_after(content, patch.identifier, patch.new_code)
            elif patch.action == "insert_before":
                new_content = self._insert_code_before(content, patch.identifier, patch.new_code)
            elif patch.action == "delete_block":
                new_content = self._delete_code_block(content, patch.identifier)
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {patch.action}"
                }
            
            # Write modified content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return {
                "success": True,
                "backup_path": backup_path,
                "message": f"Applied {patch.action} to {patch.identifier}"
            }
            
        except Exception as e:
            # Restore from backup on error
            if backup_path and os.path.exists(backup_path):
                shutil.copy2(backup_path, file_path)
            
            return {
                "success": False,
                "error": f"Failed to apply patch: {e}"
            }
    
    def _replace_code_block(self, content: str, identifier: str, new_code: str) -> str:
        """Replace a function or class definition with new code."""
        lines = content.split('\n')
        
        # Find the start of the block (function or class definition)
        start_line = None
        for i, line in enumerate(lines):
            # Look for function or class definition
            if re.match(rf'^\s*(def|class)\s+{re.escape(identifier)}\b', line):
                start_line = i
                break
        
        if start_line is None:
            raise ValueError(f"Could not find definition for {identifier}")
        
        # Find the end of the block by tracking indentation
        base_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
        end_line = start_line + 1
        
        while end_line < len(lines):
            line = lines[end_line]
            if line.strip() == "":  # Skip empty lines
                end_line += 1
                continue
            
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= base_indent and line.strip():
                break
            
            end_line += 1
        
        # Replace the block
        new_lines = lines[:start_line] + [new_code] + lines[end_line:]
        return '\n'.join(new_lines)
    
    def _insert_code_after(self, content: str, identifier: str, new_code: str) -> str:
        """Insert code after a specific identifier."""
        lines = content.split('\n')
        
        # Find the identifier
        insert_line = None
        for i, line in enumerate(lines):
            if identifier in line:
                insert_line = i + 1
                break
        
        if insert_line is None:
            raise ValueError(f"Could not find identifier: {identifier}")
        
        # Insert the new code
        new_lines = lines[:insert_line] + [new_code] + lines[insert_line:]
        return '\n'.join(new_lines)
    
    def _insert_code_before(self, content: str, identifier: str, new_code: str) -> str:
        """Insert code before a specific identifier."""
        lines = content.split('\n')
        
        # Find the identifier
        insert_line = None
        for i, line in enumerate(lines):
            if identifier in line:
                insert_line = i
                break
        
        if insert_line is None:
            raise ValueError(f"Could not find identifier: {identifier}")
        
        # Insert the new code
        new_lines = lines[:insert_line] + [new_code] + lines[insert_line:]
        return '\n'.join(new_lines)
    
    def _delete_code_block(self, content: str, identifier: str) -> str:
        """Delete a code block identified by the identifier."""
        lines = content.split('\n')
        
        # Find and remove the block (similar to replace but without replacement)
        start_line = None
        for i, line in enumerate(lines):
            if re.match(rf'^\s*(def|class)\s+{re.escape(identifier)}\b', line):
                start_line = i
                break
        
        if start_line is None:
            raise ValueError(f"Could not find definition for {identifier}")
        
        # Find the end of the block
        base_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
        end_line = start_line + 1
        
        while end_line < len(lines):
            line = lines[end_line]
            if line.strip() == "":
                end_line += 1
                continue
            
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= base_indent and line.strip():
                break
            
            end_line += 1
        
        # Remove the block
        new_lines = lines[:start_line] + lines[end_line:]
        return '\n'.join(new_lines)
    
    def _create_new_tool(self, tool_data: Dict[str, Any], backup_dir: str) -> Dict[str, Any]:
        """Create a new tool file."""
        from config import TOOL_TEMPLATE
        
        try:
            tool_name = tool_data["tool_name"]
            function_name = tool_data["function_name"]
            code = tool_data["code"]
            
            # Generate tool file content
            tool_content = TOOL_TEMPLATE.format(
                tool_name=tool_name,
                function_name=function_name,
                parameters="*args, **kwargs",  # Flexible parameters
                docstring=f"Auto-generated tool: {tool_name}",
                implementation=code
            )
            
            # Write to generated tools directory
            tools_dir = os.path.join(self.project_root, "tools", "generated_tools")
            tool_file_path = os.path.join(tools_dir, f"{tool_name}.py")
            
            with open(tool_file_path, 'w', encoding='utf-8') as f:
                f.write(tool_content)
            
            return {
                "success": True,
                "file_path": tool_file_path,
                "message": f"Created new tool: {tool_name}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create tool: {e}"
            }
    
    def _create_backup(self, file_path: str, backup_dir: str) -> str:
        """Create a backup of a file before modification."""
        rel_path = os.path.relpath(file_path, self.project_root)
        backup_path = os.path.join(backup_dir, rel_path.replace(os.sep, '_'))
        
        # Ensure backup directory exists
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def _validate_modified_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Validate that modified files are syntactically correct."""
        errors = []
        
        for file_path in file_paths:
            full_path = os.path.join(self.project_root, file_path) if not os.path.isabs(file_path) else file_path
            
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Try to parse as Python AST
                ast.parse(content)
                
            except SyntaxError as e:
                errors.append({
                    "file": file_path,
                    "error": f"Syntax error: {e}",
                    "line": e.lineno,
                    "column": e.offset
                })
            except Exception as e:
                errors.append({
                    "file": file_path,
                    "error": f"Validation error: {e}"
                })
        
        return errors
    
    def _restore_from_backups(self, backup_paths: List[str]):
        """Restore files from their backups."""
        for backup_path in backup_paths:
            try:
                # Determine original file path
                backup_name = os.path.basename(backup_path)
                original_rel_path = backup_name.replace('_', os.sep)
                original_path = os.path.join(self.project_root, original_rel_path)
                
                if os.path.exists(backup_path):
                    shutil.copy2(backup_path, original_path)
                    logger.info(f"Restored {original_path} from backup")
                    
            except Exception as e:
                logger.error(f"Failed to restore from backup {backup_path}: {e}")
    
    def get_backup_info(self) -> Dict[str, Any]:
        """Get information about available backups."""
        backups = []
        
        if os.path.exists(self.backup_dir):
            for backup_session in os.listdir(self.backup_dir):
                session_path = os.path.join(self.backup_dir, backup_session)
                if os.path.isdir(session_path):
                    session_info = {
                        "session": backup_session,
                        "created": datetime.fromtimestamp(os.path.getctime(session_path)),
                        "files": []
                    }
                    
                    for backup_file in os.listdir(session_path):
                        file_path = os.path.join(session_path, backup_file)
                        session_info["files"].append({
                            "name": backup_file,
                            "size": os.path.getsize(file_path),
                            "modified": datetime.fromtimestamp(os.path.getmtime(file_path))
                        })
                    
                    backups.append(session_info)
        
        return {
            "backup_directory": self.backup_dir,
            "total_sessions": len(backups),
            "sessions": sorted(backups, key=lambda x: x["created"], reverse=True)
        }
    
    def apply_crossover(self, parent_a_code: Dict[str, str], parent_b_code: Dict[str, str]) -> Dict[str, str]:
        """
        Combine code from two parents to create a new child agent's code.
        For demonstration: take solve_task from parent A and _research_improvements from parent B.
        """
        import re
        child_code = dict(parent_a_code)  # Start with parent A's code
        # For each file, try to replace _research_improvements in A with B's version
        for file_path in child_code:
            if file_path in parent_b_code:
                # Replace _research_improvements in child_code with B's version
                a_content = child_code[file_path]
                b_content = parent_b_code[file_path]
                # Find _research_improvements in both
                def extract_func(content, func_name):
                    pattern = rf'(def {func_name}\s*\(.*?\):[\s\S]*?)(?=^def |^class |\Z)'
                    match = re.search(pattern, content, re.MULTILINE)
                    return match.group(1) if match else None
                b_research = extract_func(b_content, '_research_improvements')
                if b_research:
                    # Replace in a_content
                    pattern = rf'(def _research_improvements\s*\(.*?\):[\s\S]*?)(?=^def |^class |\Z)'
                    a_content_new = re.sub(pattern, b_research, a_content, flags=re.MULTILINE)
                    child_code[file_path] = a_content_new
        return child_code

if __name__ == "__main__":
    # Test the mutator
    import tempfile
    
    # Create test environment
    with tempfile.TemporaryDirectory() as temp_dir:
        mutator = CodeMutator(temp_dir)
        
        # Create a test file
        test_file = os.path.join(temp_dir, "test_agent.py")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write('''def solve_problem(self, problem):
    """Original implementation."""
    return "simple solution"

class TestAgent:
    def __init__(self):
        self.name = "test"
''')
        
        # Test patch
        patch_json = json.dumps({
            "analysis": "Need better problem solving",
            "proposed_changes": [
                {
                    "file_path": "test_agent.py",
                    "action": "replace_block", 
                    "identifier": "solve_problem",
                    "new_code": '''def solve_problem(self, problem):
    """Improved implementation with web search."""
    result = self.web_search(problem)
    return self.analyze_and_solve(result)''',
                    "reasoning": "Add web search capability"
                }
            ]
        })
        
        result = mutator.apply_patch(patch_json)
        print("Mutation result:", result)
        
        # Check the modified file
        with open(test_file, 'r', encoding='utf-8') as f:
            print("Modified file content:")
            print(f.read())
