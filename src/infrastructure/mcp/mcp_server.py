"""
MCP Server Implementation.

Provides Model Context Protocol server functionality for AI Agents Platform.
Supports stdio and SSE transports for MCP client integration.
"""

import asyncio
from typing import Any

import structlog
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from src.infrastructure.config.settings import Settings
from src.application.services.chat_service import ChatService
from src.application.services.pdf_ingestion_service import PDFIngestionService
from src.application.services.agent_service import AgentService
from src.domain.dtos.chat_dto import ChatRequestDTO

logger = structlog.get_logger()


class MCPServer:
    """
    MCP Server for AI Agents Platform.
    
    Exposes agent capabilities as MCP tools that can be called by
    MCP-compatible clients like Claude Desktop.
    """
    
    def __init__(
        self,
        settings: Settings,
        chat_service: ChatService,
        pdf_service: PDFIngestionService,
        agent_service: AgentService,
    ):
        """
        Initialize MCP Server.
        
        Args:
            settings: Application settings
            chat_service: Chat service for agent interactions
            pdf_service: PDF ingestion and query service
            agent_service: Agent management service
        """
        self.settings = settings
        self.chat_service = chat_service
        self.pdf_service = pdf_service
        self.agent_service = agent_service
        
        # Initialize MCP server
        self.server = Server(settings.mcp_server_name)
        
        # Register handlers
        self._register_handlers()
        
        logger.info(
            "mcp_server_initialized",
            server_name=settings.mcp_server_name,
            transport=settings.mcp_transport
        )
    
    def _register_handlers(self):
        """Register MCP request handlers."""
        
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available MCP tools."""
            return [
                Tool(
                    name="chat_with_agent",
                    description="Chat with a specific AI agent. Available agents: conversational (general chat), pdf-analyzer (PDF analysis), cypher-query (Neo4j query optimization), rag (document search).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_type": {
                                "type": "string",
                                "enum": ["conversational", "pdf-analyzer", "cypher-query", "rag"],
                                "description": "Type of agent to chat with"
                            },
                            "message": {
                                "type": "string",
                                "description": "Message to send to the agent"
                            },
                            "user_id": {
                                "type": "string",
                                "description": "User identifier"
                            },
                            "session_id": {
                                "type": "string",
                                "description": "Optional session ID for conversation continuity"
                            }
                        },
                        "required": ["agent_type", "message", "user_id"]
                    }
                ),
                Tool(
                    name="optimize_cypher_query",
                    description="Optimize a Neo4j Cypher query following best practices. Returns optimized query with detailed explanations, index recommendations, and performance notes.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The Cypher query to optimize"
                            },
                            "user_id": {
                                "type": "string",
                                "description": "User identifier"
                            }
                        },
                        "required": ["query", "user_id"]
                    }
                ),
                Tool(
                    name="query_documents",
                    description="Semantic search across uploaded PDF documents using RAG. Returns relevant document chunks with sources.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "user_id": {
                                "type": "string",
                                "description": "User identifier"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["query", "user_id"]
                    }
                ),
                Tool(
                    name="list_agents",
                    description="List all available AI agents with their capabilities and descriptions.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Handle tool calls."""
            try:
                if name == "chat_with_agent":
                    return await self._handle_chat_with_agent(arguments)
                elif name == "optimize_cypher_query":
                    return await self._handle_optimize_cypher(arguments)
                elif name == "query_documents":
                    return await self._handle_query_documents(arguments)
                elif name == "list_agents":
                    return await self._handle_list_agents(arguments)
                else:
                    return [TextContent(
                        type="text",
                        text=f"Unknown tool: {name}"
                    )]
            except Exception as e:
                logger.error("mcp_tool_error", tool=name, error=str(e))
                return [TextContent(
                    type="text",
                    text=f"Error executing tool {name}: {str(e)}"
                )]
    
    async def _handle_chat_with_agent(self, args: dict[str, Any]) -> list[TextContent]:
        """Handle chat_with_agent tool call."""
        agent_type = args["agent_type"]
        message = args["message"]
        user_id = args["user_id"]
        session_id = args.get("session_id")
        
        # Map agent type to agent ID
        agent_map = {
            "conversational": "976edc8b-0415-4dfa-9426-3a06c5423508",
            "pdf-analyzer": "5ac370ef-3c25-418b-8a28-ec19ec952ab8",
            "cypher-query": "a0052146-1080-433a-83b8-03cf66610fb5",
            "rag": "1b7b2818-7df6-4bde-b2cf-46c6e2c75ff9"
        }
        
        agent_id = agent_map.get(agent_type)
        if not agent_id:
            return [TextContent(
                type="text",
                text=f"Unknown agent type: {agent_type}"
            )]
        
        request = ChatRequestDTO(
            message=message,
            agent_id=agent_id,
            session_id=session_id
        )
        
        response = await self.chat_service.chat(request, user_id)
        
        result_text = f"""**Agent Response ({agent_type})**

{response.message}

---
Session ID: {response.session_id}
Model: {response.model}
"""
        
        return [TextContent(type="text", text=result_text)]
    
    async def _handle_optimize_cypher(self, args: dict[str, Any]) -> list[TextContent]:
        """Handle optimize_cypher_query tool call."""
        query = args["query"]
        user_id = args["user_id"]
        
        # Use Cypher Query Optimizer agent
        request = ChatRequestDTO(
            message=f"Optimize this Cypher query:\n\n{query}",
            agent_id="a0052146-1080-433a-83b8-03cf66610fb5"
        )
        
        response = await self.chat_service.chat(request, user_id)
        
        return [TextContent(type="text", text=response.message)]
    
    async def _handle_query_documents(self, args: dict[str, Any]) -> list[TextContent]:
        """Handle query_documents tool call."""
        query = args["query"]
        user_id = args["user_id"]
        top_k = args.get("top_k", 5)
        
        try:
            results = await self.pdf_service.query_documents(
                query=query,
                top_k=top_k,
                user_id=user_id
            )
            
            if not results.results:
                return [TextContent(
                    type="text",
                    text="No documents found matching your query."
                )]
            
            result_text = f"**Found {len(results.results)} relevant document chunks**\n\n"
            
            for i, result in enumerate(results.results, 1):
                result_text += f"**Result {i}** (Score: {result.score:.3f})\n"
                result_text += f"Source: {result.metadata.get('source', 'Unknown')}\n"
                result_text += f"Page: {result.metadata.get('page', 'N/A')}\n\n"
                result_text += f"{result.content}\n\n"
                result_text += "---\n\n"
            
            return [TextContent(type="text", text=result_text)]
        except Exception as e:
            logger.error("query_documents_error", error=str(e))
            return [TextContent(
                type="text",
                text=f"Error querying documents: {str(e)}"
            )]
    
    async def _handle_list_agents(self, args: dict[str, Any]) -> list[TextContent]:
        """Handle list_agents tool call."""
        agents = await self.agent_service.get_all()
        
        result_text = "**Available AI Agents**\n\n"
        
        for agent in agents:
            result_text += f"**{agent.name}**\n"
            result_text += f"ID: {agent.id}\n"
            result_text += f"Description: {agent.description}\n"
            result_text += f"Capabilities: {', '.join(agent.capabilities)}\n\n"
        
        return [TextContent(type="text", text=result_text)]
    
    async def run_stdio(self):
        """Run MCP server with stdio transport."""
        logger.info("mcp_server_starting_stdio")
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )
    
    async def run_sse(self):
        """Run MCP server with SSE transport."""
        # SSE implementation would go here
        # For now, we'll focus on stdio which is most common
        logger.info("mcp_sse_not_implemented")
        pass
    
    async def shutdown(self):
        """Shutdown MCP server."""
        logger.info("mcp_server_shutting_down")
