"""
Script to update the SQL Query agent to Cypher Query Optimizer in MongoDB.

This script updates the agent configuration in the database to reflect
the new Cypher Query Optimizer role.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from motor.motor_asyncio import AsyncIOMotorClient
from src.infrastructure.config.settings import get_settings
from src.domain.models.prompts import AgentPrompts


async def update_cypher_agent():
    """Update the Cypher Query Optimizer agent in MongoDB."""
    settings = get_settings()
    
    # Connect to MongoDB
    client = AsyncIOMotorClient(settings.mongodb_uri)
    db = client[settings.mongodb_database]
    agents_collection = db["agents"]
    
    # Agent ID for SQL/Cypher Query agent
    agent_id = "a0052146-1080-433a-83b8-03cf66610fb5"
    
    # New agent configuration
    updated_agent = {
        "name": "Cypher Query Optimizer",
        "description": "Expert Neo4j Cypher query optimization specialist. Analyzes queries and provides optimized versions following Neo4j best practices, including index recommendations, traversal optimization, and performance tuning.",
        "system_prompt": AgentPrompts.CYPHER_QUERY_OPTIMIZER,
        "capabilities": [
            "cypher_optimization",
            "neo4j_best_practices",
            "index_recommendations",
            "query_analysis",
            "performance_tuning",
            "graph_traversal_optimization"
        ],
        "temperature": 0.3,  # Lower temperature for more consistent, technical responses
        "max_tokens": 4096,
        "tools": [],
        "metadata": {
            "specialization": "Neo4j Cypher Query Optimization",
            "expertise_level": "expert",
            "supported_versions": ["Neo4j 4.x", "Neo4j 5.x"],
            "query_types": [
                "node_lookups",
                "relationship_traversal",
                "variable_length_paths",
                "aggregations",
                "write_queries",
                "index_usage"
            ]
        }
    }
    
    # Update the agent
    result = await agents_collection.update_one(
        {"id": agent_id},
        {"$set": updated_agent}
    )
    
    if result.matched_count > 0:
        print(f"✅ Successfully updated Cypher Query Optimizer agent (ID: {agent_id})")
        print(f"   Modified: {result.modified_count} document(s)")
        
        # Verify the update
        agent = await agents_collection.find_one({"id": agent_id})
        if agent:
            print(f"\n📋 Updated Agent Details:")
            print(f"   Name: {agent.get('name')}")
            print(f"   Description: {agent.get('description')[:100]}...")
            print(f"   Temperature: {agent.get('temperature')}")
            print(f"   Capabilities: {', '.join(agent.get('capabilities', []))}")
    else:
        print(f"⚠️  Agent with ID {agent_id} not found in database")
        print(f"   Creating new agent...")
        
        # Create the agent if it doesn't exist
        new_agent = {
            "id": agent_id,
            **updated_agent,
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "is_active": True
        }
        
        await agents_collection.insert_one(new_agent)
        print(f"✅ Created new Cypher Query Optimizer agent")
    
    # Close connection
    client.close()
    print("\n✅ Database update completed")


if __name__ == "__main__":
    print("🔄 Updating Cypher Query Optimizer agent in MongoDB...")
    asyncio.run(update_cypher_agent())
