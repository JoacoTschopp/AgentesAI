"""
RAG Workflow - LangGraph workflow for Retrieval-Augmented Generation.

This module implements a RAG workflow using LangGraph,
combining vector store retrieval with LLM generation.
"""

from typing import Any, Literal

import structlog
from langgraph.graph import StateGraph, END

from src.application.workflows.base_workflow import BaseWorkflow, WorkflowState
from src.domain.ports.llm_port import LLMPort
from src.domain.ports.vector_store_port import VectorStorePort
from src.domain.ports.prompt_sanitizer_port import PromptSanitizerPort
from src.infrastructure.config.settings import Settings


logger = structlog.get_logger()


class RAGState(WorkflowState):
    """
    State for RAG workflow.
    
    Attributes:
        query: User query for retrieval.
        retrieved_documents: Documents retrieved from vector store.
        augmented_prompt: Prompt augmented with retrieved context.
        response: Final generated response.
        namespace: Vector store namespace to search.
    """
    
    query: str
    retrieved_documents: list[dict[str, Any]]
    augmented_prompt: str
    response: str
    namespace: str


class RAGWorkflow(BaseWorkflow):
    """
    LangGraph workflow for Retrieval-Augmented Generation.
    
    Implements the RAG pattern:
    1. Sanitize query
    2. Generate query embedding
    3. Retrieve relevant documents
    4. Augment prompt with context
    5. Generate response
    6. Sanitize output
    
    Attributes:
        llm: Language model adapter.
        vector_store: Vector store adapter.
        sanitizer: Prompt sanitizer.
    """

    def __init__(
        self,
        settings: Settings,
        llm: LLMPort,
        vector_store: VectorStorePort,
        sanitizer: PromptSanitizerPort,
    ):
        """
        Initialize RAG workflow.
        
        Args:
            settings: Application settings.
            llm: LLM adapter.
            vector_store: Vector store adapter.
            sanitizer: Prompt sanitizer.
        """
        super().__init__(settings)
        self._llm = llm
        self._vector_store = vector_store
        self._sanitizer = sanitizer

    def build_graph(self) -> StateGraph:
        """
        Build the RAG workflow graph.
        
        Returns:
            Configured StateGraph for RAG.
        """
        graph = StateGraph(RAGState)

        graph.add_node("sanitize_query", self._sanitize_query_node)
        graph.add_node("retrieve_documents", self._retrieve_documents_node)
        graph.add_node("augment_prompt", self._augment_prompt_node)
        graph.add_node("generate_response", self._generate_response_node)
        graph.add_node("sanitize_output", self._sanitize_output_node)

        graph.set_entry_point("sanitize_query")
        
        graph.add_edge("sanitize_query", "retrieve_documents")
        graph.add_edge("retrieve_documents", "augment_prompt")
        graph.add_edge("augment_prompt", "generate_response")
        graph.add_edge("generate_response", "sanitize_output")
        graph.add_edge("sanitize_output", END)

        return graph

    async def _sanitize_query_node(self, state: RAGState) -> dict[str, Any]:
        """
        Node: Sanitize user query.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated state with sanitized query.
        """
        result = await self._sanitizer.sanitize_input(state["query"])
        
        if not result.is_safe:
            logger.warning("query_rejected", issues=result.detected_issues)
            return {
                "error": f"Query rejected: {', '.join(result.detected_issues)}",
            }

        return {"query": result.sanitized_text}

    async def _retrieve_documents_node(self, state: RAGState) -> dict[str, Any]:
        """
        Node: Retrieve relevant documents from vector store.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated state with retrieved documents.
        """
        if state.get("error"):
            return {"retrieved_documents": []}

        try:
            embeddings = await self._llm.get_embeddings([state["query"]])
            query_embedding = embeddings[0]

            results = await self._vector_store.query(
                vector=query_embedding,
                top_k=5,
                namespace=state.get("namespace"),
                include_metadata=True,
            )

            logger.info("documents_retrieved", count=len(results))
            
            return {"retrieved_documents": results}

        except Exception as e:
            logger.error("retrieval_failed", error=str(e))
            return {"retrieved_documents": [], "error": str(e)}

    async def _augment_prompt_node(self, state: RAGState) -> dict[str, Any]:
        """
        Node: Augment prompt with retrieved context.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated state with augmented prompt.
        """
        documents = state.get("retrieved_documents", [])
        
        if not documents:
            augmented = f"Question: {state['query']}\n\nNo relevant context found. Please answer based on your knowledge."
        else:
            context_parts = []
            for i, doc in enumerate(documents, 1):
                content = doc.get("metadata", {}).get("content", doc.get("metadata", {}).get("text", ""))
                if content:
                    context_parts.append(f"[{i}] {content}")
            
            context = "\n\n".join(context_parts)
            augmented = f"""Use the following context to answer the question. If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {state['query']}

Answer:"""

        return {"augmented_prompt": augmented}

    async def _generate_response_node(self, state: RAGState) -> dict[str, Any]:
        """
        Node: Generate response using augmented prompt.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated state with generated response.
        """
        if state.get("error") and not state.get("augmented_prompt"):
            return {"response": f"Error: {state['error']}"}

        try:
            messages = [{"role": "user", "content": state["augmented_prompt"]}]
            
            response = await self._llm.generate(
                messages=messages,
                system_prompt="You are a knowledgeable assistant. Answer questions based on the provided context. Be accurate and cite sources when possible.",
                temperature=0.3,
            )
            
            return {"response": response}

        except Exception as e:
            logger.error("rag_generation_failed", error=str(e))
            return {"response": "I apologize, but I encountered an error generating a response."}

    async def _sanitize_output_node(self, state: RAGState) -> dict[str, Any]:
        """
        Node: Sanitize generated response.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated state with sanitized response.
        """
        result = await self._sanitizer.sanitize_output(state["response"])
        
        messages = state.get("messages", [])
        messages.append({"role": "user", "content": state["query"]})
        messages.append({"role": "assistant", "content": result.sanitized_text})
        
        return {
            "messages": messages,
            "response": result.sanitized_text,
        }
