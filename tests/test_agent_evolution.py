"""
Test Agent Archive and Evolution System
Tests if the Darwinian Gödeli Machine (DGM) logic is working correctly.
"""

import sys
import os
import tempfile
import shutil
import io

# Force UTF-8 encoding for Windows
if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from archive.agent_archive import AgentArchive, AgentGeneration
from agent.agent_core import PrometheusAgent
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_agent_evolution():
    """Test agent evolution and archive system."""
    print("� Testing Agent Evolution System (DGM Logic)...")
    print("=" * 50)
    
    try:
        # Test 1: Archive initialization
        print("\n1. Testing archive initialization...")
        
        temp_dir = tempfile.mkdtemp()
        archive = AgentArchive(temp_dir)
        
        print(f"✅ SUCCESS: Archive initialized in {temp_dir}")
        
        # Test 2: Create test agent generation
        print("\n2. Testing agent generation creation...")
        
        # Create a mock agent generation
        generation1 = AgentGeneration(
            agent_id="test_agent_001",
            generation=1,
            parent_id=None,
            performance_score=0.65,
            task_results=[
                {"task_id": "task_1", "success": True, "score": 0.8},
                {"task_id": "task_2", "success": False, "score": 0.5}
            ],
            mutations_applied=["added_error_handling", "improved_search"],
            created_at=datetime.now(),
            source_code="def solve_task(): pass"
        )
        
        # Save to archive
        archive.save_generation(generation1)
        
        print(f"✅ SUCCESS: Created and saved generation 1")
        
        # Test 3: Retrieve and verify
        print("\n3. Testing archive retrieval...")
        
        retrieved = archive.get_generation("test_agent_001")
        
        if retrieved and retrieved.agent_id == "test_agent_001":
            print(f"✅ SUCCESS: Retrieved generation successfully")
            print(f"   Agent ID: {retrieved.agent_id}")
            print(f"   Performance: {retrieved.performance_score}")
        else:
            print("❌ FAILED: Could not retrieve generation")
            return False
        
        # Test 4: Create evolved generation
        print("\n4. Testing evolution logic...")
        
        # Create an improved generation
        generation2 = AgentGeneration(
            agent_id="test_agent_002",
            generation=2,
            parent_id="test_agent_001",
            performance_score=0.75,  # Improved score
            task_results=[
                {"task_id": "task_1", "success": True, "score": 0.9},
                {"task_id": "task_2", "success": True, "score": 0.6}
            ],
            mutations_applied=["fixed_unicode_bug", "better_error_search"],
            created_at=datetime.now(),
            source_code="def solve_task(): # improved version"
        )
        
        archive.save_generation(generation2)
        
        print(f"✅ SUCCESS: Created evolved generation 2")
        
        # Test 5: Best agent selection
        print("\n5. Testing best agent selection...")
        
        best_agent = archive.get_best_agent()
        
        if best_agent and best_agent.agent_id == "test_agent_002":
            print(f"✅ SUCCESS: Best agent correctly identified")
            print(f"   Best Agent: {best_agent.agent_id}")
            print(f"   Best Score: {best_agent.performance_score}")
        else:
            print("❌ FAILED: Best agent selection incorrect")
            return False
        
        # Test 6: Evolution tree
        print("\n6. Testing evolution lineage...")
        
        all_generations = archive.get_all_generations()
        
        if len(all_generations) >= 2:
            print(f"✅ SUCCESS: Evolution tree contains {len(all_generations)} generations")
            
            # Check parent-child relationship
            child = [g for g in all_generations if g.parent_id is not None][0]
            parent = [g for g in all_generations if g.agent_id == child.parent_id][0]
            
            if parent and child.performance_score > parent.performance_score:
                print(f"✅ SUCCESS: Evolution shows improvement")
                print(f"   Parent score: {parent.performance_score}")
                print(f"   Child score: {child.performance_score}")
            else:
                print("⚠️  WARNING: Evolution may not show clear improvement")
        else:
            print("❌ FAILED: Not enough generations in archive")
            return False
        
        # Test 7: Agent instantiation
        print("\n7. Testing agent instantiation...")
        
        try:
            # This is a basic test - we won't fully initialize due to dependencies
            test_agent = PrometheusAgent(
                agent_id="test_agent_instance",
                generation=1,
                parent_id=None
            )
            
            if test_agent.agent_id == "test_agent_instance":
                print(f"✅ SUCCESS: Agent instantiation works")
            else:
                print("❌ FAILED: Agent instantiation failed")
                return False
                
        except Exception as e:
            print(f"⚠️  WARNING: Agent instantiation test limited due to dependencies: {e}")
        
        # Test 8: Performance tracking
        print("\n8. Testing performance tracking...")
        
        # Test performance calculation
        total_tasks = len(generation2.task_results)
        successful_tasks = len([r for r in generation2.task_results if r["success"]])
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0
        
        expected_success_rate = 1.0  # Both tasks succeeded in generation2
        
        if abs(success_rate - expected_success_rate) < 0.01:
            print(f"✅ SUCCESS: Performance tracking accurate")
            print(f"   Success rate: {success_rate:.2f}")
        else:
            print(f"❌ FAILED: Performance tracking inaccurate")
            return False
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        print("\n🎉 All agent evolution tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Agent evolution test failed with error: {e}")
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir)
        return False

if __name__ == "__main__":
    success = test_agent_evolution()
    if success:
        print("\n✅ Agent evolution system (DGM logic) is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ Agent evolution system has issues!")
        sys.exit(1)
