"""
Base Workflow - Abstract base class for LangGraph workflows.

This module defines the base structure for all agent workflows,
providing common functionality and MongoDB checkpoint integration.
"""

from abc import ABC, abstractmethod
from typing import Any, TypedDict

import structlog
from langgraph.graph import StateGraph
from langgraph.checkpoint.mongodb import MongoDBSaver

from src.infrastructure.config.settings import Settings


logger = structlog.get_logger()


class WorkflowState(TypedDict, total=False):
    """
    Base state for workflow execution.
    
    Attributes:
        messages: List of conversation messages.
        context: Additional context data.
        metadata: Workflow metadata.
        error: Error message if any.
    """
    
    messages: list[dict[str, str]]
    context: dict[str, Any]
    metadata: dict[str, Any]
    error: str | None


class BaseWorkflow(ABC):
    """
    Abstract base class for LangGraph workflows.
    
    Provides common functionality for workflow creation and execution,
    including MongoDB checkpoint persistence for long-term session storage.
    
    Attributes:
        settings: Application settings.
        checkpointer: MongoDB checkpointer for state persistence.
        graph: Compiled LangGraph graph.
    """

    def __init__(self, settings: Settings):
        """
        Initialize base workflow.
        
        Args:
            settings: Application settings.
        """
        self._settings = settings
        self._checkpointer = self._create_checkpointer()
        self._graph = None

    def _create_checkpointer(self) -> MongoDBSaver:
        """
        Create MongoDB checkpointer for state persistence.
        
        Returns:
            Configured MongoDBSaver instance.
        """
        return MongoDBSaver.from_conn_string(
            conn_string=self._settings.mongodb_uri,
            db_name=self._settings.mongodb_database,
            collection_name=self._settings.mongodb_checkpoints_collection,
        )

    @abstractmethod
    def build_graph(self) -> StateGraph:
        """
        Build the workflow graph.
        
        Returns:
            Configured StateGraph instance.
        """
        pass

    def compile(self) -> None:
        """Compile the workflow graph with checkpointer."""
        graph = self.build_graph()
        self._graph = graph.compile(checkpointer=self._checkpointer)
        logger.info("workflow_compiled", workflow=self.__class__.__name__)

    async def run(
        self,
        initial_state: WorkflowState,
        config: dict[str, Any] | None = None,
    ) -> WorkflowState:
        """
        Execute the workflow.
        
        Args:
            initial_state: Initial workflow state.
            config: Optional configuration including thread_id.
            
        Returns:
            Final workflow state after execution.
        """
        if not self._graph:
            self.compile()

        config = config or {}
        if "configurable" not in config:
            config["configurable"] = {}

        result = await self._graph.ainvoke(initial_state, config)
        
        logger.info(
            "workflow_executed",
            workflow=self.__class__.__name__,
            thread_id=config.get("configurable", {}).get("thread_id"),
        )
        
        return result

    async def stream(
        self,
        initial_state: WorkflowState,
        config: dict[str, Any] | None = None,
    ):
        """
        Stream workflow execution.
        
        Args:
            initial_state: Initial workflow state.
            config: Optional configuration.
            
        Yields:
            State updates during execution.
        """
        if not self._graph:
            self.compile()

        config = config or {}
        
        async for event in self._graph.astream(initial_state, config):
            yield event

    async def get_state(self, thread_id: str) -> WorkflowState | None:
        """
        Get the current state for a thread.
        
        Args:
            thread_id: Thread identifier.
            
        Returns:
            Current state if exists, None otherwise.
        """
        if not self._graph:
            self.compile()

        config = {"configurable": {"thread_id": thread_id}}
        state = await self._graph.aget_state(config)
        
        return state.values if state else None

    async def update_state(
        self,
        thread_id: str,
        updates: dict[str, Any],
    ) -> None:
        """
        Update the state for a thread.
        
        Args:
            thread_id: Thread identifier.
            updates: State updates to apply.
        """
        if not self._graph:
            self.compile()

        config = {"configurable": {"thread_id": thread_id}}
        await self._graph.aupdate_state(config, updates)
