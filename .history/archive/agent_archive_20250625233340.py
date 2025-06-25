"""Agent archive system for Prometheus 2.0 - Manages generational evolution."""

import json
import os
import shutil
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class AgentGeneration:
    """Represents a single agent generation."""
    agent_id: str
    generation: int
    parent_id: Optional[str]
    created_at: datetime
    performance_score: float
    success_rate: float
    total_tasks: int
    task_results: List[Dict[str, Any]]
    mutations_applied: List[str]
    source_code: str
    metadata: Dict[str, Any]
    schema_version: float = 1.0  # Add schema version for migration

class AgentArchive:
    """Manages the archive of agent generations and their evolution."""
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.archive_dir = os.path.join(project_root, "archive", "generations")
        self.source_archive_dir = os.path.join(project_root, "archive", "source_code")
        
        # Ensure directories exist
        os.makedirs(self.archive_dir, exist_ok=True)
        os.makedirs(self.source_archive_dir, exist_ok=True)
        
        # In-memory cache of generations
        self.generations: Dict[str, AgentGeneration] = {}
        self.generation_tree: Dict[str, List[str]] = {}  # parent_id -> [child_ids]
        
        # Load existing generations
        self._load_archive()
    
    def _load_archive(self):
        """Load existing generations from disk."""
        if not os.path.exists(self.archive_dir):
            return
        
        for filename in os.listdir(self.archive_dir):
            if filename.endswith('.json'):
                try:
                    agent_id = filename[:-5]  # Remove .json extension
                    generation_data = self._load_generation_data(agent_id)
                    
                    if generation_data:
                        # Migration: handle both old and new mutation fields
                        mutations = generation_data.get("mutations_applied", generation_data.get("mutation_changes", []))
                        # Migration: handle schema version
                        schema_version = generation_data.get("schema_version", 1.0)
                        generation = AgentGeneration(
                            agent_id=generation_data["agent_id"],
                            generation=generation_data["generation"],
                            parent_id=generation_data.get("parent_id"),
                            created_at=datetime.fromisoformat(generation_data["created_at"]),
                            performance_score=generation_data["performance_score"],
                            success_rate=generation_data["success_rate"],
                            total_tasks=generation_data["total_tasks"],
                            task_results=generation_data.get("task_results", []),
                            mutations_applied=mutations,
                            source_code=generation_data.get("source_code", ""),
                            metadata=generation_data.get("metadata", {}),
                            schema_version=schema_version
                        )
                        
                        self.generations[agent_id] = generation
                        
                        # Build generation tree
                        parent_id = generation.parent_id
                        if parent_id:
                            if parent_id not in self.generation_tree:
                                self.generation_tree[parent_id] = []
                            self.generation_tree[parent_id].append(agent_id)
                        
                except Exception as e:
                    logger.warning(f"Failed to load generation {filename}: {e}")
        
        logger.info(f"Loaded {len(self.generations)} generations from archive")
    
    def _load_generation_data(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load generation data from JSON file."""
        filepath = os.path.join(self.archive_dir, f"{agent_id}.json")
        
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load generation data for {agent_id}: {e}")
            return None
    
    def archive_agent(
        self,
        agent,
        mutation_changes: List[Dict[str, Any]] = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Archive an agent generation.
        
        Args:
            agent: The agent to archive
            mutation_changes: List of mutations applied to create this agent
            metadata: Additional metadata
            
        Returns:
            True if archiving was successful
        """
        try:
            # Get agent performance summary
            performance = agent.get_performance_summary()
            
            # Get source code
            source_code = agent.get_source_code()
            
            # Create generation record
            generation = AgentGeneration(
                agent_id=agent.agent_id,
                generation=agent.generation,
                parent_id=agent.parent_id,
                created_at=datetime.now(),
                performance_score=performance.get("average_score", 0.0),
                success_rate=performance.get("success_rate", 0.0),
                total_tasks=performance.get("total_tasks", 0),
                task_results=performance.get("task_results", []),
                mutations_applied=mutation_changes or [],
                source_code=source_code,
                metadata=metadata or {}
            )
            
            # Save to archive
            self._save_generation(generation)
            
            # Update in-memory cache
            self.generations[agent.agent_id] = generation
            
            # Update generation tree
            if agent.parent_id:
                if agent.parent_id not in self.generation_tree:
                    self.generation_tree[agent.parent_id] = []
                self.generation_tree[agent.parent_id].append(agent.agent_id)
            
            logger.info(f"Archived agent {agent.agent_id} (generation {agent.generation})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to archive agent {agent.agent_id}: {e}")
            return False
    
    def _archive_source_code(self, agent_id: str, source_code: Dict[str, str]) -> str:
        """Archive source code and return hash."""
        import hashlib
        
        # Create hash of source code
        combined_code = "\n".join(source_code.values())
        source_hash = hashlib.sha256(combined_code.encode()).hexdigest()[:16]
        
        # Save source code
        source_dir = os.path.join(self.source_archive_dir, agent_id)
        os.makedirs(source_dir, exist_ok=True)
        
        for file_path, content in source_code.items():
            # Create subdirectories if needed
            full_path = os.path.join(source_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Also save as pickle for faster loading
        pickle_path = os.path.join(source_dir, "_source_code.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(source_code, f)
        
        return source_hash
    
    def _save_generation(self, generation: AgentGeneration):
        """Save generation data to JSON file."""
        filepath = os.path.join(self.archive_dir, f"{generation.agent_id}.json")
        
        # Convert to dictionary for JSON serialization
        generation_dict = asdict(generation)
        generation_dict["created_at"] = generation.created_at.isoformat()
        generation_dict["schema_version"] = getattr(generation, "schema_version", 1.0)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(generation_dict, f, indent=2)
        
        # Update in-memory cache
        self.generations[generation.agent_id] = generation
        
        # Update generation tree
        if generation.parent_id:
            if generation.parent_id not in self.generation_tree:
                self.generation_tree[generation.parent_id] = []
            self.generation_tree[generation.parent_id].append(generation.agent_id)
    
    def get_best_agent(self, metric: str = "performance_score") -> Optional[AgentGeneration]:
        """Get the best performing agent based on specified metric."""
        if not self.generations:
            return None
        
        return max(self.generations.values(), key=lambda g: getattr(g, metric))
    
    def get_generation_lineage(self, agent_id: str) -> List[AgentGeneration]:
        """Get the lineage of an agent back to the root."""
        lineage = []
        current_id = agent_id
        
        while current_id and current_id in self.generations:
            generation = self.generations[current_id]
            lineage.append(generation)
            current_id = generation.parent_id
        
        return lineage
    
    def get_children(self, agent_id: str) -> List[AgentGeneration]:
        """Get all children of an agent."""
        child_ids = self.generation_tree.get(agent_id, [])
        return [self.generations[child_id] for child_id in child_ids if child_id in self.generations]
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about the archived generations."""
        if not self.generations:
            return {
                "total_generations": 0,
                "best_score": 0.0,
                "average_score": 0.0,
                "success_rate": 0.0,
                "generation_depth": 0
            }
        
        generations = list(self.generations.values())
        
        # Calculate statistics
        total_gens = len(generations)
        best_score = max(g.performance_score for g in generations)
        avg_score = sum(g.performance_score for g in generations) / total_gens
        avg_success_rate = sum(g.success_rate for g in generations) / total_gens
        max_generation = max(g.generation for g in generations)
        
        # Find root agents (no parent)
        roots = [g for g in generations if g.parent_id is None]
        
        return {
            "total_generations": total_gens,
            "best_score": best_score,
            "average_score": avg_score,
            "success_rate": avg_success_rate,
            "generation_depth": max_generation,
            "root_agents": len(roots),
            "branches": len(self.generation_tree)
        }
    
    def get_evolution_history(self) -> List[Dict[str, Any]]:
        """Get the complete evolution history sorted by creation time."""
        if not self.generations:
            return []
        
        history = []
        for generation in sorted(self.generations.values(), key=lambda g: g.created_at):
            history.append({
                "agent_id": generation.agent_id,
                "generation": generation.generation,
                "parent_id": generation.parent_id,
                "created_at": generation.created_at.isoformat(),
                "performance_score": generation.performance_score,
                "success_rate": generation.success_rate,
                "total_tasks": generation.total_tasks,
                "mutations_count": len(generation.mutations_applied)
            })
        
        return history
    
    def select_parent_for_next_generation(
        self,
        selection_strategy: str = "best_score"
    ) -> Optional[str]:
        """
        Select the best parent agent for creating the next generation.
        
        Args:
            selection_strategy: Strategy for selection ("best_score", "recent_best", "diverse")
            
        Returns:
            Agent ID of selected parent, or None if no agents available
        """
        if not self.generations:
            return None
        
        generations = list(self.generations.values())
        
        if selection_strategy == "best_score":
            # Select agent with highest performance score
            best_agent = max(generations, key=lambda g: g.performance_score)
            return best_agent.agent_id
        
        elif selection_strategy == "recent_best":
            # Select best from recent generations
            recent_generations = sorted(generations, key=lambda g: g.created_at, reverse=True)[:5]
            if recent_generations:
                best_recent = max(recent_generations, key=lambda g: g.performance_score)
                return best_recent.agent_id
        
        elif selection_strategy == "diverse":
            # Select from different branches to maintain diversity
            # This is a simplified diversity selection
            branch_representatives = {}
            for gen in generations:
                root = self._find_root_ancestor(gen.agent_id)
                if root not in branch_representatives or gen.performance_score > branch_representatives[root].performance_score:
                    branch_representatives[root] = gen
            
            if branch_representatives:
                best_branch_rep = max(branch_representatives.values(), key=lambda g: g.performance_score)
                return best_branch_rep.agent_id
        
        # Fallback to best score
        best_agent = max(generations, key=lambda g: g.performance_score)
        return best_agent.agent_id
    
    def _find_root_ancestor(self, agent_id: str) -> str:
        """Find the root ancestor of an agent."""
        current_id = agent_id
        
        while current_id and current_id in self.generations:
            generation = self.generations[current_id]
            if generation.parent_id is None:
                return current_id
            current_id = generation.parent_id
        
        return agent_id  # Fallback
    
    def load_agent_source_code(self, agent_id: str) -> Optional[Dict[str, str]]:
        """Load source code for a specific agent."""
        pickle_path = os.path.join(self.source_archive_dir, agent_id, "_source_code.pkl")
        
        if os.path.exists(pickle_path):
            try:
                with open(pickle_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load pickled source for {agent_id}: {e}")
        
        # Fallback to loading individual files
        source_dir = os.path.join(self.source_archive_dir, agent_id)
        if not os.path.exists(source_dir):
            return None
        
        source_code = {}
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, source_dir)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            source_code[rel_path] = f.read()
                    except Exception as e:
                        logger.warning(f"Failed to load {rel_path}: {e}")
        
        return source_code
    
    def prune_archive(self, keep_best: int = 10, keep_recent: int = 5):
        """
        Prune the archive to keep only the best and most recent generations.
        
        Args:
            keep_best: Number of best performing agents to keep
            keep_recent: Number of most recent agents to keep
        """
        if len(self.generations) <= keep_best + keep_recent:
            return  # No pruning needed
        
        # Identify agents to keep
        agents_to_keep = set()
        
        # Keep best performers
        best_agents = sorted(
            self.generations.values(),
            key=lambda g: g.performance_score,
            reverse=True
        )[:keep_best]
        agents_to_keep.update(g.agent_id for g in best_agents)
        
        # Keep most recent
        recent_agents = sorted(
            self.generations.values(),
            key=lambda g: g.created_at,
            reverse=True
        )[:keep_recent]
        agents_to_keep.update(g.agent_id for g in recent_agents)
        
        # Remove agents not in keep list
        agents_to_remove = set(self.generations.keys()) - agents_to_keep
        
        for agent_id in agents_to_remove:
            try:
                # Remove generation file
                gen_file = os.path.join(self.archive_dir, f"{agent_id}.json")
                if os.path.exists(gen_file):
                    os.remove(gen_file)
                
                # Remove source code directory
                source_dir = os.path.join(self.source_archive_dir, agent_id)
                if os.path.exists(source_dir):
                    shutil.rmtree(source_dir)
                
                # Remove from memory
                del self.generations[agent_id]
                
                # Remove from generation tree
                for parent_id, children in self.generation_tree.items():
                    if agent_id in children:
                        children.remove(agent_id)
                
                if agent_id in self.generation_tree:
                    del self.generation_tree[agent_id]
                
                logger.info(f"Pruned agent {agent_id} from archive")
                
            except Exception as e:
                logger.error(f"Failed to prune agent {agent_id}: {e}")
        
        logger.info(f"Archive pruning complete. Kept {len(agents_to_keep)} agents, removed {len(agents_to_remove)}")
    
    def get_generation(self, agent_id: str) -> Optional[AgentGeneration]:
        """Get a specific generation by agent ID."""
        return self.generations.get(agent_id)
    
    def get_all_generations(self) -> List[AgentGeneration]:
        """Get all generations in the archive."""
        return list(self.generations.values())

if __name__ == "__main__":
    # Test the archive system
    import tempfile
    from agent.agent_core import PrometheusAgent
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create archive and test agent
        archive = AgentArchive(temp_dir)
        agent = PrometheusAgent("test_agent", temp_dir)
        
        # Simulate some performance data
        agent.total_score = 0.75
        agent.success_rate = 0.8
        agent.task_results = [None] * 10  # Simulate 10 tasks
        
        # Archive the agent
        print("Archiving agent...")
        success = archive.archive_agent(
            agent,
            mutation_changes=[{"type": "test_mutation", "details": "test change"}],
            metadata={"test": True}
        )
        
        print(f"Archive success: {success}")
        
        # Test retrieval
        print("\nTesting retrieval...")
        best_agent = archive.get_best_agent()
        print(f"Best agent: {best_agent.agent_id if best_agent else None}")
        
        # Test statistics
        print("\nArchive statistics:")
        stats = archive.get_generation_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test evolution history
        print("\nEvolution history:")
        history = archive.get_evolution_history()
        for entry in history:
            print(f"  {entry['agent_id']}: score={entry['performance_score']:.3f}")
        
        # Test parent selection
        print("\nParent selection:")
        parent_id = archive.select_parent_for_next_generation()
        print(f"Selected parent: {parent_id}")
        
        # Test source code loading
        print("\nTesting source code loading...")
        source_code = archive.load_agent_source_code(agent.agent_id)
        if source_code:
            print(f"Loaded source code for {len(source_code)} files")
        else:
            print("No source code found")
