"""
MCP (Model Context Protocol) Infrastructure.

This module provides MCP server implementation for the AI Agents Platform,
enabling integration with MCP-compatible clients like Claude Desktop.
"""

from src.infrastructure.mcp.mcp_server import MCPServer

__all__ = ["MCPServer"]
