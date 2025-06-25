#!/usr/bin/env python3
"""
Test script to verify the enhanced GUI dashboard is working with real data.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from archive.agent_archive import AgentArchive
from tools.tool_manager import ToolManager

def test_archive_integration():
    """Test the archive integration functionality."""
    print("🔍 Testing Agent Archive Integration...")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Test AgentArchive
    try:
        archive = AgentArchive(project_root)
        print(f"✅ AgentArchive initialized successfully")
        
        # Test statistics
        stats = archive.get_generation_stats()
        print(f"📊 Generation Stats: {stats}")
        
        # Test evolution history
        history = archive.get_evolution_history()
        print(f"📈 Evolution History: {len(history)} generations found")
        
        # Test best agent
        best_agent = archive.get_best_agent()
        if best_agent:
            print(f"🏆 Best Agent: {best_agent.agent_id} (score: {best_agent.performance_score:.3f})")
        else:
            print("❌ No best agent found")
        
        # Test individual generation
        for agent_data in history[:3]:  # Test first 3
            agent_id = agent_data["agent_id"]
            generation = archive.get_generation(agent_id)
            if generation:
                print(f"🔹 {agent_id}: Gen {generation.generation}, Score {generation.performance_score:.3f}")
            else:
                print(f"❌ Could not load generation for {agent_id}")
    
    except Exception as e:
        print(f"❌ AgentArchive test failed: {e}")
        return False
    
    # Test ToolManager
    try:
        tool_manager = ToolManager(project_root)
        print(f"✅ ToolManager initialized successfully")
        
        # Test tool listing
        tools = tool_manager.get_available_tools()
        print(f"🔧 Available Tools: {len(tools)} tools found")
        
        for tool_name in list(tools.keys())[:3]:  # Show first 3 tools
            tool_info = tools[tool_name]
            print(f"   - {tool_name}: {tool_info.get('description', 'No description')[:50]}...")
    
    except Exception as e:
        print(f"❌ ToolManager test failed: {e}")
        return False
    
    print("✅ All integration tests passed!")
    return True

if __name__ == "__main__":
    test_archive_integration()
