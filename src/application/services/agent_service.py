"""
Agent Service - Business logic for agent management.

This module provides the service layer for managing AI agents,
including CRUD operations and agent configuration.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

import structlog

from src.domain.models.agent import Agent, AgentType
from src.domain.dtos.agent_dto import AgentCreateDTO, AgentResponseDTO, AgentUpdateDTO


logger = structlog.get_logger()


class AgentService:
    """
    Service for managing AI agents.
    
    Provides business logic for agent creation, retrieval, updating,
    and deletion. Agents are stored in-memory for this implementation
    but can be extended to use a database.
    
    Attributes:
        agents: In-memory storage for agents.
    """

    def __init__(self):
        """Initialize agent service with default agents."""
        self._agents: dict[UUID, Agent] = {}
        self._initialize_default_agents()

    def _initialize_default_agents(self) -> None:
        """Create default agents available out of the box."""
        default_agents = [
            Agent(
                name="Conversational Assistant",
                agent_type=AgentType.CONVERSATIONAL,
                description="General-purpose conversational AI for Q&A and discussions",
                system_prompt="You are a helpful AI assistant. Answer questions clearly and concisely. If you don't know something, say so honestly.",
                temperature=0.7,
                max_tokens=4096,
            ),
            Agent(
                name="PDF Analyzer",
                agent_type=AgentType.PDF_ANALYZER,
                description="Specialized agent for analyzing and summarizing PDF documents",
                system_prompt="You are a document analysis expert. Analyze the provided document content and provide clear summaries, key insights, and answer questions about the document.",
                temperature=0.3,
                max_tokens=8192,
            ),
            Agent(
                name="SQL Query Assistant",
                agent_type=AgentType.SQL_QUERY,
                description="Agent capable of generating and explaining SQL queries",
                system_prompt="You are a SQL expert. Help users write, optimize, and understand SQL queries. Always explain your queries step by step.",
                temperature=0.2,
                max_tokens=4096,
            ),
            Agent(
                name="RAG Research Agent",
                agent_type=AgentType.RAG,
                description="Retrieval-Augmented Generation agent for knowledge-based Q&A",
                system_prompt="You are a research assistant with access to a knowledge base. Use the retrieved context to answer questions accurately. Always cite your sources when possible.",
                temperature=0.5,
                max_tokens=4096,
            ),
        ]

        for agent in default_agents:
            self._agents[agent.id] = agent
            
        logger.info("default_agents_initialized", count=len(default_agents))

    async def create(self, dto: AgentCreateDTO) -> AgentResponseDTO:
        """
        Create a new agent.
        
        Args:
            dto: Agent creation data.
            
        Returns:
            The created agent as a response DTO.
        """
        agent = Agent(
            name=dto.name,
            agent_type=dto.agent_type,
            description=dto.description,
            system_prompt=dto.system_prompt,
            temperature=dto.temperature,
            max_tokens=dto.max_tokens,
            metadata=dto.metadata,
        )
        
        self._agents[agent.id] = agent
        
        logger.info("agent_created", agent_id=str(agent.id), name=agent.name)
        
        return self._to_response_dto(agent)

    async def get_by_id(self, agent_id: UUID) -> AgentResponseDTO | None:
        """
        Retrieve an agent by ID.
        
        Args:
            agent_id: The agent's unique identifier.
            
        Returns:
            The agent if found, None otherwise.
        """
        agent = self._agents.get(agent_id)
        
        if agent:
            return self._to_response_dto(agent)
        return None

    async def get_all(self) -> list[AgentResponseDTO]:
        """
        Retrieve all agents.
        
        Returns:
            List of all agents.
        """
        return [self._to_response_dto(agent) for agent in self._agents.values()]

    async def get_by_type(self, agent_type: AgentType) -> list[AgentResponseDTO]:
        """
        Retrieve agents by type.
        
        Args:
            agent_type: The type of agents to retrieve.
            
        Returns:
            List of agents of the specified type.
        """
        agents = [
            self._to_response_dto(agent)
            for agent in self._agents.values()
            if agent.agent_type == agent_type
        ]
        return agents

    async def update(self, agent_id: UUID, dto: AgentUpdateDTO) -> AgentResponseDTO | None:
        """
        Update an existing agent.
        
        Args:
            agent_id: The agent's unique identifier.
            dto: Update data.
            
        Returns:
            The updated agent if found, None otherwise.
        """
        agent = self._agents.get(agent_id)
        
        if not agent:
            return None

        if dto.name is not None:
            agent.name = dto.name
        if dto.description is not None:
            agent.description = dto.description
        if dto.system_prompt is not None:
            agent.system_prompt = dto.system_prompt
        if dto.temperature is not None:
            agent.temperature = dto.temperature
        if dto.max_tokens is not None:
            agent.max_tokens = dto.max_tokens
        if dto.metadata is not None:
            agent.metadata = dto.metadata

        agent.updated_at = datetime.utcnow()
        
        logger.info("agent_updated", agent_id=str(agent_id))
        
        return self._to_response_dto(agent)

    async def delete(self, agent_id: UUID) -> bool:
        """
        Delete an agent.
        
        Args:
            agent_id: The agent's unique identifier.
            
        Returns:
            True if deleted, False if not found.
        """
        if agent_id in self._agents:
            del self._agents[agent_id]
            logger.info("agent_deleted", agent_id=str(agent_id))
            return True
        return False

    def get_agent_model(self, agent_id: UUID) -> Agent | None:
        """
        Get the raw Agent model by ID.
        
        Args:
            agent_id: The agent's unique identifier.
            
        Returns:
            The Agent model if found, None otherwise.
        """
        return self._agents.get(agent_id)

    def _to_response_dto(self, agent: Agent) -> AgentResponseDTO:
        """Convert Agent model to response DTO."""
        return AgentResponseDTO(
            id=agent.id,
            name=agent.name,
            agent_type=agent.agent_type.value if isinstance(agent.agent_type, AgentType) else agent.agent_type,
            description=agent.description,
            system_prompt=agent.system_prompt,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens,
            metadata=agent.metadata,
            created_at=agent.created_at,
            updated_at=agent.updated_at,
        )
