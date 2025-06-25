"""Docker sandbox for secure evaluation of agent solutions."""

import os
import tempfile
import shutil
import subprocess
from t# Fix Python compatibility issues for older repos
RUN python3 -c "
import sys
import os
import re

# Function to fix Python 3.10+ compatibility issues
def fix_python_compatibility():
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # Fix collections.MutableSequence -> collections.abc.MutableSequence
                    content = re.sub(r'collections\.MutableSequence', 'collections.abc.MutableSequence', content)
                    content = re.sub(r'from collections import.*MutableSequence', 'from collections.abc import MutableSequence', content)
                    
                    # Fix setuptools.dep_util -> distutils.dep_util
                    content = re.sub(r'from setuptools\.dep_util import', 'from distutils.dep_util import', content)
                    content = re.sub(r'setuptools\.dep_util', 'distutils.dep_util', content)
                    
                    # Add necessary imports if missing
                    if 'collections.abc.MutableSequence' in content and 'import collections.abc' not in content and 'from collections.abc import' not in content:
                        # Add import at the top
                        lines = content.split('\n')
                        import_added = False
                        for i, line in enumerate(lines):
                            if line.strip().startswith('import ') or line.strip().startswith('from '):
                                lines.insert(i, 'import collections.abc')
                                import_added = True
                                break
                        if not import_added and lines:
                            lines.insert(0, 'import collections.abc')
                        content = '\n'.join(lines)
                    
                    # Only write if content changed
                    if content != original_content:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f'Fixed compatibility issues in: {filepath}')
                    
                except Exception as e:
                    print(f'Warning: Could not process {filepath}: {e}')
                    continue

fix_python_compatibility()
"port Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DockerSandbox:
    """Secure Docker sandbox for running agent solutions."""
    
    def __init__(self, base_image: str = "python:3.11-slim"):
        self.base_image = base_image
        self.container_name_prefix = "prometheus_eval"
        
        # Check if Docker is available
        self._check_docker_availability()
    
    def _check_docker_availability(self):
        """Check if Docker is installed and running."""
        try:
            result = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                text=True,
                timeout=10,
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode != 0:
                raise RuntimeError("Docker is not running or not accessible")
                
            logger.info("Docker is available and running")
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise RuntimeError(f"Docker is not available: {e}")
    
    def create_evaluation_environment(
        self,
        repo_url: str,
        commit_hash: str,
        dependencies: Optional[List[str]] = None,
        python_version: str = "3.11"
    ) -> Dict[str, Any]:
        """
        Create a Docker environment for evaluating a specific repository.
        
        Args:
            repo_url: Git repository URL
            commit_hash: Specific commit to checkout
            dependencies: Additional Python packages to install
            python_version: Python version to use
            
        Returns:
            Dictionary with environment information
        """
        try:
            # Create temporary directory for Docker context
            temp_dir = tempfile.mkdtemp(prefix="prometheus_docker_")
            
            # Generate Dockerfile
            dockerfile_content = self._generate_dockerfile(
                repo_url, commit_hash, dependencies, python_version
            )
            
            dockerfile_path = os.path.join(temp_dir, "Dockerfile")
            with open(dockerfile_path, 'w', encoding='utf-8') as f:
                f.write(dockerfile_content)
            
            # Create requirements.txt if dependencies provided
            if dependencies:
                requirements_path = os.path.join(temp_dir, "requirements.txt")
                with open(requirements_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(dependencies))
            
            # Build Docker image
            image_tag = f"prometheus_eval_{commit_hash[:8]}"
            build_result = self._build_docker_image(temp_dir, image_tag)
            
            if not build_result["success"]:
                return {
                    "success": False,
                    "error": build_result["error"],
                    "temp_dir": temp_dir
                }
            
            return {
                "success": True,
                "image_tag": image_tag,
                "temp_dir": temp_dir,
                "dockerfile_path": dockerfile_path
            }
            
        except Exception as e:
            logger.error(f"Failed to create evaluation environment: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_dockerfile(
        self,
        repo_url: str,
        commit_hash: str,
        dependencies: Optional[List[str]],
        python_version: str
    ) -> str:
        """Generate Dockerfile for the evaluation environment."""
        
        dockerfile = f"""FROM python:{python_version}-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    build-essential \\
    python3-dev \\
    pkg-config \\
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install setuptools (fix for collections.MutableSequence)
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Clone repository and checkout specific commit
RUN git clone {repo_url} repo
WORKDIR /workspace/repo
RUN git checkout {commit_hash}

# Fix Python compatibility issues for older repos
RUN if [ -f setup.py ]; then \\
    # Fix collections.MutableSequence issue in Python 3.10+
    find . -name "*.py" -type f -exec sed -i 's/collections\\.MutableSequence/collections.abc.MutableSequence/g' {{}} \\; 2>/dev/null || true; \\
    # Fix setuptools.dep_util imports
    find . -name "*.py" -type f -exec sed -i 's/from setuptools\\.dep_util import/from distutils.dep_util import/g' {{}} \\; 2>/dev/null || true; \\
    find . -name "*.py" -type f -exec sed -i 's/setuptools\\.dep_util/distutils.dep_util/g' {{}} \\; 2>/dev/null || true; \\
fi

# Install Python dependencies with fallback strategy
"""
        
        if dependencies:
            dockerfile += """COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
"""
        
        dockerfile += """
# Install any repository-specific dependencies with error handling
RUN if [ -f requirements.txt ]; then \\
    pip install --no-cache-dir -r requirements.txt || true; \\
fi

# Try to install the package in development mode with fallback
RUN if [ -f setup.py ]; then \\
    pip install --no-cache-dir -e . || \\
    pip install --no-cache-dir --no-build-isolation -e . || \\
    python setup.py develop || \\
    echo "Failed to install package in development mode, proceeding anyway"; \\
fi

RUN if [ -f pyproject.toml ]; then \\
    pip install --no-cache-dir -e . || \\
    pip install --no-cache-dir --no-build-isolation -e . || \\
    echo "Failed to install package via pyproject.toml, proceeding anyway"; \\
fi

# Install additional packages that might be needed
RUN pip install --no-cache-dir pytest coverage astropy-helpers extension-helpers || true

# Create entry point script
RUN echo '#!/bin/bash\\n\\
set -e\\n\\
echo "Environment ready"\\n\\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
"""
        
        return dockerfile
    
    def _build_docker_image(self, context_dir: str, image_tag: str) -> Dict[str, Any]:
        """Build Docker image from context directory."""
        try:
            logger.info(f"Building Docker image: {image_tag}")
            
            result = subprocess.run(
                ["docker", "build", "-t", image_tag, "."],
                cwd=context_dir,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes timeout
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully built image: {image_tag}")
                return {
                    "success": True,
                    "image_tag": image_tag,
                    "build_output": result.stdout
                }
            else:
                logger.error(f"Failed to build image: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr,
                    "build_output": result.stdout
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Docker build timed out"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_evaluation(
        self,
        image_tag: str,
        patch_content: str,
        test_commands: List[str],
        timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Run evaluation in Docker container.
        
        Args:
            image_tag: Docker image to use
            patch_content: Code patch to apply
            test_commands: Commands to run for testing
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with evaluation results
        """
        container_name = f"{self.container_name_prefix}_{os.getpid()}"
        
        try:
            # Create temporary directory for patch file
            temp_dir = tempfile.mkdtemp()
            patch_file = os.path.join(temp_dir, "solution.patch")
            
            with open(patch_file, 'w', encoding='utf-8') as f:
                f.write(patch_content)
            
            # Start container
            start_result = self._start_container(image_tag, container_name, temp_dir)
            if not start_result["success"]:
                return start_result
            
            # Apply patch
            patch_result = self._apply_patch_in_container(container_name, "/tmp/solution.patch")
            
            # Run tests
            test_results = []
            for command in test_commands:
                test_result = self._run_command_in_container(
                    container_name, command, timeout
                )
                test_results.append(test_result)
            
            # Calculate overall success
            overall_success = all(result.get("success", False) for result in test_results)
            
            return {
                "success": overall_success,
                "patch_applied": patch_result["success"],
                "test_results": test_results,
                "container_name": container_name
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "container_name": container_name
            }
        finally:
            # Clean up container
            self._cleanup_container(container_name)
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _start_container(self, image_tag: str, container_name: str, temp_dir: str) -> Dict[str, Any]:
        """Start Docker container."""
        try:
            # Mount temp directory for patch file
            mount_arg = f"{temp_dir}:/tmp:ro"
            
            result = subprocess.run([
                "docker", "run", "-d",
                "--name", container_name,
                "-v", mount_arg,
                "--rm",  # Auto-remove when stopped
                image_tag,
                "sleep", "infinity"  # Keep container running
            ], capture_output=True, text=True, timeout=30, encoding='utf-8', errors='replace')
            
            if result.returncode == 0:
                logger.info(f"Started container: {container_name}")
                return {"success": True, "container_id": result.stdout.strip()}
            else:
                return {"success": False, "error": result.stderr}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _apply_patch_in_container(self, container_name: str, patch_path: str) -> Dict[str, Any]:
        """Apply patch inside the container."""
        try:
            # Apply patch using git apply or patch command
            result = subprocess.run([
                "docker", "exec", container_name,
                "bash", "-c", f"cd /workspace/repo && git apply {patch_path} || patch -p1 < {patch_path}"
            ], capture_output=True, text=True, timeout=30, encoding='utf-8', errors='replace')
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _run_command_in_container(
        self,
        container_name: str,
        command: str,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """Run a command inside the container."""
        try:
            result = subprocess.run([
                "docker", "exec", container_name,
                "bash", "-c", f"cd /workspace/repo && {command}"
            ], capture_output=True, text=True, timeout=timeout, encoding='utf-8', errors='replace')
            
            return {
                "success": result.returncode == 0,
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "command": command,
                "error": "Command timed out"
            }
        except Exception as e:
            return {
                "success": False,
                "command": command,
                "error": str(e)
            }
    
    def _cleanup_container(self, container_name: str):
        """Stop and remove container."""
        try:
            # Stop container (it will be auto-removed due to --rm flag)
            subprocess.run([
                "docker", "stop", container_name
            ], capture_output=True, timeout=10, encoding='utf-8', errors='replace')
            
            logger.info(f"Cleaned up container: {container_name}")
            
        except Exception as e:
            logger.warning(f"Failed to cleanup container {container_name}: {e}")
    
    def cleanup_images(self, image_prefix: str = "prometheus_eval"):
        """Clean up Docker images created for evaluation."""
        try:
            # List images with the prefix
            result = subprocess.run([
                "docker", "images", "--format", "{{.Repository}}:{{.Tag}}",
                "--filter", f"reference={image_prefix}*"
            ], capture_output=True, text=True, encoding='utf-8', errors='replace')
            
            if result.returncode == 0 and result.stdout.strip():
                images = result.stdout.strip().split('\n')
                
                # Remove images
                for image in images:
                    subprocess.run([
                        "docker", "rmi", image
                    ], capture_output=True, encoding='utf-8', errors='replace')
                
                logger.info(f"Cleaned up {len(images)} evaluation images")
            
        except Exception as e:
            logger.warning(f"Failed to cleanup images: {e}")

if __name__ == "__main__":
    # Test the sandbox
    sandbox = DockerSandbox()
    
    # Test creating environment (using a public repo for testing)
    print("Testing Docker environment creation...")
    
    try:
        env_result = sandbox.create_evaluation_environment(
            repo_url="https://github.com/psf/requests.git",
            commit_hash="main",
            dependencies=["pytest"]
        )
        
        print(f"Environment creation result: {env_result}")
        
        if env_result["success"]:
            # Test running evaluation
            print("Testing evaluation...")
            
            # Simple patch that adds a comment
            test_patch = """diff --git a/README.md b/README.md
index 1234567..abcdefg 100644
--- a/README.md
+++ b/README.md
@@ -1,3 +1,4 @@
+# Test patch applied
 # Requests: HTTP for Humans™
 
 Requests is an elegant and simple HTTP library for Python, built for human beings.
"""
            
            eval_result = sandbox.run_evaluation(
                image_tag=env_result["image_tag"],
                patch_content=test_patch,
                test_commands=["echo 'Test command executed'", "ls -la"],
                timeout=60
            )
            
            print(f"Evaluation result: {eval_result}")
        
        # Cleanup
        sandbox.cleanup_images()
        
    except Exception as e:
        print(f"Test failed: {e}")
