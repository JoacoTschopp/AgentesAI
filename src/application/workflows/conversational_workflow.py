"""
Conversational Workflow - LangGraph workflow for conversational agents.

This module implements a conversational AI workflow using LangGraph,
with MongoDB persistence for long-term session storage.
"""

from typing import Any, Literal

import structlog
from langgraph.graph import StateGraph, END

from src.application.workflows.base_workflow import BaseWorkflow, WorkflowState
from src.domain.ports.llm_port import LLMPort
from src.domain.ports.prompt_sanitizer_port import PromptSanitizerPort
from src.infrastructure.config.settings import Settings


logger = structlog.get_logger()


class ConversationalState(WorkflowState):
    """
    State for conversational workflow.
    
    Attributes:
        user_input: Current user input.
        assistant_response: Generated assistant response.
        system_prompt: System prompt for the conversation.
        should_end: Flag to indicate conversation end.
    """
    
    user_input: str
    assistant_response: str
    system_prompt: str
    should_end: bool


class ConversationalWorkflow(BaseWorkflow):
    """
    LangGraph workflow for conversational AI interactions.
    
    Implements a simple but effective conversational flow:
    1. Sanitize input
    2. Generate response
    3. Sanitize output
    4. Return or continue
    
    State is persisted in MongoDB for long-term conversation continuity.
    
    Attributes:
        llm: Language model adapter.
        sanitizer: Prompt sanitizer.
    """

    def __init__(
        self,
        settings: Settings,
        llm: LLMPort,
        sanitizer: PromptSanitizerPort,
    ):
        """
        Initialize conversational workflow.
        
        Args:
            settings: Application settings.
            llm: LLM adapter.
            sanitizer: Prompt sanitizer.
        """
        super().__init__(settings)
        self._llm = llm
        self._sanitizer = sanitizer

    def build_graph(self) -> StateGraph:
        """
        Build the conversational workflow graph.
        
        Returns:
            Configured StateGraph for conversation.
        """
        graph = StateGraph(ConversationalState)

        graph.add_node("sanitize_input", self._sanitize_input_node)
        graph.add_node("generate_response", self._generate_response_node)
        graph.add_node("sanitize_output", self._sanitize_output_node)

        graph.set_entry_point("sanitize_input")
        
        graph.add_edge("sanitize_input", "generate_response")
        graph.add_edge("generate_response", "sanitize_output")
        graph.add_conditional_edges(
            "sanitize_output",
            self._should_end,
            {
                "end": END,
                "continue": END,
            },
        )

        return graph

    async def _sanitize_input_node(self, state: ConversationalState) -> dict[str, Any]:
        """
        Node: Sanitize user input.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated state with sanitized input.
        """
        result = await self._sanitizer.sanitize_input(state["user_input"])
        
        if not result.is_safe:
            logger.warning("input_rejected_in_workflow", issues=result.detected_issues)
            return {
                "error": f"Input rejected: {', '.join(result.detected_issues)}",
                "should_end": True,
            }

        messages = state.get("messages", [])
        messages.append({"role": "user", "content": result.sanitized_text})
        
        return {
            "messages": messages,
            "user_input": result.sanitized_text,
        }

    async def _generate_response_node(self, state: ConversationalState) -> dict[str, Any]:
        """
        Node: Generate LLM response.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated state with assistant response.
        """
        if state.get("error"):
            return {"assistant_response": state["error"]}

        try:
            response = await self._llm.generate(
                messages=state["messages"],
                system_prompt=state.get("system_prompt", "You are a helpful AI assistant."),
            )
            
            return {"assistant_response": response}

        except Exception as e:
            logger.error("llm_generation_failed", error=str(e))
            return {
                "assistant_response": "I apologize, but I encountered an error processing your request.",
                "error": str(e),
            }

    async def _sanitize_output_node(self, state: ConversationalState) -> dict[str, Any]:
        """
        Node: Sanitize LLM output.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated state with sanitized response.
        """
        result = await self._sanitizer.sanitize_output(state["assistant_response"])
        
        messages = state.get("messages", [])
        messages.append({"role": "assistant", "content": result.sanitized_text})
        
        return {
            "messages": messages,
            "assistant_response": result.sanitized_text,
            "should_end": state.get("should_end", False),
        }

    def _should_end(self, state: ConversationalState) -> Literal["end", "continue"]:
        """
        Determine if conversation should end.
        
        Args:
            state: Current workflow state.
            
        Returns:
            "end" or "continue" based on state.
        """
        if state.get("should_end") or state.get("error"):
            return "end"
        return "continue"
