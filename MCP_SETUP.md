# MCP (Model Context Protocol) Setup Guide

## Overview

This AI Agents Platform supports MCP (Model Context Protocol), enabling integration with MCP-compatible clients like Claude Desktop, Cline (VS Code extension), and other AI assistants.

## What is MCP?

**Model Context Protocol** is an open standard that connects AI applications with external tools and data sources. Through MCP, AI assistants can:

- Call tools to perform actions (chat with agents, query documents, optimize queries)
- Access resources (view documents, sessions, agent configurations)
- Use predefined prompts and workflows

## Available MCP Tools

| Tool | Description | Use Case |
|------|-------------|----------|
| `chat_with_agent` | Interact with specialized AI agents | General chat, PDF analysis, Cypher optimization, RAG |
| `optimize_cypher_query` | Optimize Neo4j Cypher queries | Database query performance tuning |
| `query_documents` | Semantic search in uploaded PDFs | Research, document analysis |
| `list_agents` | List all available agents | Discover agent capabilities |

## Configuration

### 1. Enable MCP in Docker

Edit `.env`:

```bash
# Enable MCP server
MCP_ACTIVE=true

# Transport: stdio (local), sse (remote), or both
MCP_TRANSPORT=both

# SSE port (if using SSE transport)
MCP_SSE_PORT=8001

# Server name
MCP_SERVER_NAME=ai-agents-mcp
```

### 2. Restart Docker

```bash
docker compose down
docker compose build
docker compose up -d
```

## Client Setup

### Option 1: Claude Desktop (Recommended)

1. **Install Claude Desktop** from [claude.ai/download](https://claude.ai/download)

2. **Configure MCP Server**

   Create or edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or equivalent path:

   ```json
   {
     "mcpServers": {
       "ai-agents": {
         "command": "docker",
         "args": [
           "exec",
           "-i",
           "ai-agents-app",
           "python",
           "-m",
           "src.infrastructure.mcp.cli"
         ],
         "env": {
           "MCP_ACTIVE": "true",
           "MCP_TRANSPORT": "stdio"
         }
       }
     }
   }
   ```

3. **Restart Claude Desktop**

4. **Verify Installation**
   
   In Claude Desktop, click the 🔌 icon to see available tools. You should see:
   - `chat_with_agent`
   - `optimize_cypher_query`
   - `query_documents`
   - `list_agents`

### Option 2: Cline (VS Code Extension)

1. **Install Cline** from VS Code marketplace

2. **Configure MCP**

   In VS Code settings, add to `settings.json`:

   ```json
   {
     "cline.mcpServers": {
       "ai-agents": {
         "command": "docker",
         "args": [
           "exec",
           "-i",
           "ai-agents-app",
           "python",
           "-m",
           "src.infrastructure.mcp.cli"
         ]
       }
     }
   }
   ```

3. **Reload VS Code**

### Option 3: HTTP Client (SSE Transport)

For remote access, use SSE transport:

```bash
# In .env
MCP_ACTIVE=true
MCP_TRANSPORT=sse
MCP_SSE_PORT=8001
```

Access MCP via HTTP:
```
http://localhost:8001/mcp/sse
```

## Usage Examples

### Example 1: Chat with Cypher Query Optimizer

In Claude Desktop:

```
User: "Can you optimize this Neo4j query for me?"
Claude: "I'll use the optimize_cypher_query tool..."

User: "MATCH (p:Person)-[:KNOWS*]-(f) WHERE p.name = 'John' RETURN f"
Claude: *calls optimize_cypher_query tool*
Claude: "Here's the optimized query: [detailed optimization]"
```

### Example 2: Query Documents

```
User: "Search my uploaded PDFs for information about machine learning"
Claude: *calls query_documents tool*
Claude: "I found 3 relevant sections: [shows results with sources]"
```

### Example 3: General Chat

```
User: "Explain quantum computing"
Claude: *calls chat_with_agent tool with agent_type="conversational"*
Claude: [provides detailed explanation]
```

## Testing MCP Server

### Test from Command Line

```bash
# Enter container
docker exec -it ai-agents-app bash

# Run MCP CLI
python -m src.infrastructure.mcp.cli
```

This starts the MCP server in stdio mode. It will wait for JSON-RPC messages.

### Test Individual Tool

```python
# Python test script
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp():
    server_params = StdioServerParameters(
        command="docker",
        args=["exec", "-i", "ai-agents-app", 
              "python", "-m", "src.infrastructure.mcp.cli"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List tools
            tools = await session.list_tools()
            print(f"Available tools: {[t.name for t in tools]}")
            
            # Call a tool
            result = await session.call_tool(
                "list_agents",
                arguments={}
            )
            print(result)

asyncio.run(test_mcp())
```

## Troubleshooting

### MCP Server Not Starting

**Check logs:**
```bash
docker compose logs app | grep mcp
```

**Verify configuration:**
```bash
docker exec ai-agents-app env | grep MCP
```

### Claude Desktop Not Detecting Server

1. Verify config file location and syntax
2. Check Docker container is running: `docker ps`
3. Restart Claude Desktop
4. Check Claude Desktop logs

### Tool Calls Failing

**Check MongoDB connection:**
```bash
docker compose logs mongodb
```

**Verify services are initialized:**
```bash
docker compose logs app | grep initialized
```

## Architecture

```
┌─────────────────────────────────────┐
│  MCP Client (Claude Desktop)        │
│  - Detects available tools           │
│  - Sends tool call requests          │
└──────────────┬──────────────────────┘
               │ stdio (JSON-RPC)
               │
┌──────────────▼──────────────────────┐
│  MCP Server (Docker Container)      │
│  - Receives tool calls               │
│  - Routes to appropriate service     │
│  - Returns structured responses      │
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┐
       │                │
┌──────▼─────┐  ┌──────▼──────┐
│ ChatService│  │ PDFService  │
│ (Agents)   │  │ (Documents) │
└────────────┘  └─────────────┘
```

## Modes of Operation

### Mode 1: API Only (Default)
```bash
MCP_ACTIVE=false
```
- FastAPI server on port 8000
- No MCP capabilities

### Mode 2: MCP Only
```bash
MCP_ACTIVE=true
MCP_TRANSPORT=stdio
```
- MCP server available via CLI
- No FastAPI routes (use separate instance)

### Mode 3: Dual Mode (Recommended)
```bash
MCP_ACTIVE=true
MCP_TRANSPORT=both
```
- FastAPI server on port 8000 (web/mobile apps)
- MCP server via stdio (Claude Desktop, IDEs)
- MCP server via SSE on port 8001 (remote clients)
- **All share the same services and database**

## Benefits of MCP Integration

1. **AI Assistant Integration**: Use your agents directly in Claude Desktop
2. **IDE Integration**: Access agents while coding in VS Code
3. **Unified Platform**: One codebase serves both API and MCP
4. **No Duplication**: API and MCP share services and data
5. **Flexible Deployment**: Choose transport based on use case

## Security Considerations

- MCP server runs within Docker container
- Uses same authentication/authorization as API
- All tool calls are logged
- Input sanitization via LLMGuard (if enabled)

## Performance

- MCP tools call the same services as API endpoints
- No significant performance overhead
- Concurrent API and MCP calls supported
- MongoDB connection pooling shared

## Future Enhancements

- [ ] Prompts: Expose reusable prompt templates
- [ ] Resources: Add session and document resources
- [ ] Sampling: Allow LLM to sample via MCP
- [ ] Notifications: Push updates to clients
- [ ] Authentication: Token-based auth for SSE

## Support

For issues or questions:
- Check logs: `docker compose logs app`
- Review this guide
- Check MCP specification: https://spec.modelcontextprotocol.io/
