"""
Agent Routes - CRUD endpoints for AI agents.

This module provides RESTful endpoints for managing AI agents,
including creation, retrieval, updating, and deletion.
"""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies.services import get_agent_service
from src.application.services.agent_service import AgentService
from src.domain.dtos.agent_dto import AgentCreateDTO, AgentResponseDTO, AgentUpdateDTO
from src.domain.models.agent import AgentType


router = APIRouter(prefix="/agents", tags=["Agents"])


@router.get(
    "",
    response_model=list[AgentResponseDTO],
    status_code=status.HTTP_200_OK,
    summary="List All Agents",
    description="Retrieve a list of all available AI agents.",
)
async def list_agents(
    agent_type: AgentType | None = None,
    service: AgentService = Depends(get_agent_service),
) -> list[AgentResponseDTO]:
    """
    List all agents, optionally filtered by type.
    
    Args:
        agent_type: Optional filter by agent type.
        service: Agent service instance.
        
    Returns:
        List of agents.
    """
    if agent_type:
        return await service.get_by_type(agent_type)
    return await service.get_all()


@router.get(
    "/{agent_id}",
    response_model=AgentResponseDTO,
    status_code=status.HTTP_200_OK,
    summary="Get Agent",
    description="Retrieve a specific agent by its ID.",
)
async def get_agent(
    agent_id: UUID,
    service: AgentService = Depends(get_agent_service),
) -> AgentResponseDTO:
    """
    Get a specific agent by ID.
    
    Args:
        agent_id: The agent's unique identifier.
        service: Agent service instance.
        
    Returns:
        The requested agent.
        
    Raises:
        HTTPException: If agent not found.
    """
    agent = await service.get_by_id(agent_id)
    
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent not found: {agent_id}",
        )
    
    return agent


@router.post(
    "",
    response_model=AgentResponseDTO,
    status_code=status.HTTP_201_CREATED,
    summary="Create Agent",
    description="Create a new AI agent with the specified configuration.",
)
async def create_agent(
    dto: AgentCreateDTO,
    service: AgentService = Depends(get_agent_service),
) -> AgentResponseDTO:
    """
    Create a new agent.
    
    Args:
        dto: Agent creation data.
        service: Agent service instance.
        
    Returns:
        The created agent.
    """
    return await service.create(dto)


@router.patch(
    "/{agent_id}",
    response_model=AgentResponseDTO,
    status_code=status.HTTP_200_OK,
    summary="Update Agent",
    description="Update an existing agent's configuration.",
)
async def update_agent(
    agent_id: UUID,
    dto: AgentUpdateDTO,
    service: AgentService = Depends(get_agent_service),
) -> AgentResponseDTO:
    """
    Update an existing agent.
    
    Args:
        agent_id: The agent's unique identifier.
        dto: Update data.
        service: Agent service instance.
        
    Returns:
        The updated agent.
        
    Raises:
        HTTPException: If agent not found.
    """
    agent = await service.update(agent_id, dto)
    
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent not found: {agent_id}",
        )
    
    return agent


@router.delete(
    "/{agent_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Agent",
    description="Delete an existing agent.",
)
async def delete_agent(
    agent_id: UUID,
    service: AgentService = Depends(get_agent_service),
) -> None:
    """
    Delete an agent.
    
    Args:
        agent_id: The agent's unique identifier.
        service: Agent service instance.
        
    Raises:
        HTTPException: If agent not found.
    """
    deleted = await service.delete(agent_id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent not found: {agent_id}",
        )


@router.get(
    "/types/available",
    response_model=list[dict[str, str]],
    status_code=status.HTTP_200_OK,
    summary="List Agent Types",
    description="Get a list of all available agent types.",
)
async def list_agent_types() -> list[dict[str, str]]:
    """
    List all available agent types.
    
    Returns:
        List of agent types with descriptions.
    """
    return [
        {"type": AgentType.CONVERSATIONAL.value, "description": "General-purpose conversational agent"},
        {"type": AgentType.PDF_ANALYZER.value, "description": "PDF document analysis agent"},
        {"type": AgentType.SQL_QUERY.value, "description": "SQL query generation agent"},
        {"type": AgentType.RAG.value, "description": "Retrieval-Augmented Generation agent"},
    ]
