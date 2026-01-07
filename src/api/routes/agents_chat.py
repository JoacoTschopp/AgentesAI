"""
Agent-Specific Chat Routes - Simplified endpoints for each agent type.

This module provides dedicated endpoints for each predefined agent,
eliminating the need for dynamic agent management.
"""

import os
from pathlib import Path
from typing import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from src.domain.dtos.chat_dto import ChatRequestDTO, ChatResponseDTO
from src.domain.dtos.document_dto import DocumentSummaryResponse
from src.api.dependencies.services import get_chat_service, get_pdf_ingestion_service
from src.application.services.chat_service import ChatService
from src.application.services.pdf_ingestion_service import PDFIngestionService


logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/agents", tags=["Agent Chat"])


@router.post("/conversational/chat", response_model=ChatResponseDTO)
async def chat_with_conversational_agent(
    request: ChatRequestDTO,
    x_user_id: str = Header(..., description="User identifier"),
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponseDTO:
    """
    Chat with the Conversational Assistant agent.
    
    General-purpose conversational AI for Q&A and discussions.
    """
    logger.info(
        "conversational_agent_request",
        user_id=x_user_id,
        message_length=len(request.message)
    )
    
    # Set the predefined conversational agent ID
    request.agent_id = "976edc8b-0415-4dfa-9426-3a06c5423508"
    
    return await chat_service.chat(request, x_user_id)


@router.post("/conversational/chat/stream")
async def stream_chat_with_conversational_agent(
    request: ChatRequestDTO,
    x_user_id: str = Header(..., description="User identifier"),
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """
    Stream chat with the Conversational Assistant agent.
    """
    logger.info(
        "conversational_agent_stream_request",
        user_id=x_user_id,
        message_length=len(request.message)
    )
    
    request.agent_id = "976edc8b-0415-4dfa-9426-3a06c5423508"
    
    async def generate() -> AsyncIterator[str]:
        async for chunk in chat_service.chat_stream(request, x_user_id):
            yield f"data: {chunk.chunk}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/pdf-analyzer/chat", response_model=DocumentSummaryResponse)
async def chat_with_pdf_analyzer(
    file: UploadFile = File(..., description="PDF file to analyze"),
    x_user_id: str = Header(..., description="User identifier"),
    chat_service: ChatService = Depends(get_chat_service),
    pdf_service: PDFIngestionService = Depends(get_pdf_ingestion_service),
) -> DocumentSummaryResponse:
    """
    Analyze and summarize a PDF document.
    
    This endpoint:
    1. Receives a PDF file upload
    2. Ingests the PDF into ChromaDB (or uses existing if already processed)
    3. Generates a detailed summary of the document
    4. Returns the summary with document metadata
    
    Args:
        file: PDF file to analyze
        x_user_id: User identifier
        chat_service: Chat service for summary generation
        pdf_service: PDF ingestion service
        
    Returns:
        Document summary with metadata
    """
    if not file.filename or not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    logger.info(
        "pdf_analyzer_request",
        filename=file.filename,
        user_id=x_user_id
    )
    
    # Create temporary directory for uploads
    upload_dir = Path("./temp_uploads")
    upload_dir.mkdir(exist_ok=True)
    
    # Save uploaded file temporarily
    temp_file_path = upload_dir / file.filename
    
    try:
        # Write uploaded file to disk
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info("pdf_saved_for_analysis", path=str(temp_file_path))
        
        # Step 1: Ingest PDF (will skip if already exists)
        ingestion_result = await pdf_service.ingest_pdf(
            pdf_path=str(temp_file_path),
            user_id=x_user_id,
            additional_metadata={
                "original_filename": file.filename,
                "content_type": file.content_type or "application/pdf",
                "purpose": "analysis"
            }
        )
        
        logger.info("pdf_ingestion_complete", result=ingestion_result)
        
        # Step 2: Retrieve document chunks for summary
        doc_id = ingestion_result["document_id"]
        relevant_chunks = await pdf_service.query_documents(
            query=f"Main topics and key findings in {file.filename}",
            top_k=10,  # Get more chunks for comprehensive summary
            filter_metadata={"document_id": doc_id}
        )
        
        # Step 3: Build context from document chunks
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            metadata = chunk.get('metadata', {})
            text = metadata.get('text', chunk.get('document', ''))
            chunk_index = metadata.get('chunk_index', 0)
            
            context_parts.append(f"[Section {chunk_index}]\n{text}\n")
        
        # Step 4: Generate detailed summary using chat service
        if context_parts:
            context_text = "\n---\n".join(context_parts)
            summary_prompt = f"""Please provide a detailed summary of the following PDF document: "{file.filename}"

Document Content (key sections):
{context_text}

Generate a comprehensive summary that includes:
1. Main topics and themes
2. Key findings and conclusions
3. Important data or statistics mentioned
4. Overall structure and organization
5. Any notable insights or recommendations

Please be thorough and detailed in your summary."""
            
            request = ChatRequestDTO(
                message=summary_prompt,
                agent_id="5ac370ef-3c25-418b-8a28-ec19ec952ab8"
            )
            
            summary_response = await chat_service.chat(request, x_user_id)
            summary = summary_response.message
        else:
            summary = "Unable to generate summary - no content could be extracted from the PDF."
        
        logger.info("summary_generated", document_id=doc_id)
        
        return DocumentSummaryResponse(
            document_id=doc_id,
            filename=file.filename,
            summary=summary,
            total_chunks=ingestion_result.get("total_chunks", 0),
            metadata={
                "status": ingestion_result.get("status"),
                "total_characters": ingestion_result.get("total_characters", 0),
                "message": ingestion_result.get("message", "")
            }
        )
        
    except ValueError as e:
        logger.error("pdf_analysis_failed", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("pdf_analysis_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to analyze PDF")
    finally:
        # Clean up temporary file
        if temp_file_path.exists():
            try:
                os.remove(temp_file_path)
                logger.info("temp_file_cleaned", path=str(temp_file_path))
            except Exception as e:
                logger.warning("temp_file_cleanup_failed", error=str(e))


@router.post("/cypher-query/chat", response_model=ChatResponseDTO)
async def chat_with_cypher_optimizer(
    request: ChatRequestDTO,
    x_user_id: str = Header(..., description="User identifier"),
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponseDTO:
    """
    Optimize Cypher queries for Neo4j graph database.
    
    **Specialized Agent for Neo4j Query Optimization**
    
    This endpoint provides expert-level Cypher query optimization following Neo4j best practices.
    The agent analyzes your queries and provides optimized versions with detailed explanations.
    
    **What This Agent Does:**
    - Analyzes Cypher query performance bottlenecks
    - Identifies missing or inefficient index usage
    - Optimizes relationship traversal patterns
    - Eliminates anti-patterns (cartesian products, unbounded paths, etc.)
    - Provides specific index creation statements
    - Estimates performance improvements
    - Suggests alternative query approaches when beneficial
    
    **Best Use Cases:**
    1. **Query Optimization:** Send a Cypher query and get an optimized version
    2. **Performance Review:** Analyze why a query is slow
    3. **Index Recommendations:** Get index suggestions for your query patterns
    4. **Best Practices:** Learn Neo4j query optimization techniques
    5. **Troubleshooting:** Understand query execution plans
    
    **Example Requests:**
    - "Optimize this query: MATCH (p:Person)-[:KNOWS*]-(f) WHERE p.name = 'John' RETURN f"
    - "Why is this query slow: MATCH (u:User) OPTIONAL MATCH (u)-[:POSTED]->(p:Post) RETURN u, collect(p)"
    - "What indexes should I create for queries on Person.email and Person.age?"
    - "How can I optimize finding shortest paths between two nodes?"
    - "Review this query for performance issues: [your query]"
    
    **Response Format:**
    The agent provides:
    1. Original query analysis with identified issues
    2. Optimized query with explanatory comments
    3. Key improvements with performance impact estimates
    4. Exact CREATE INDEX statements needed
    5. Performance notes and complexity analysis
    6. Additional recommendations for data modeling or alternatives
    
    **Query Patterns Covered:**
    - Node lookups and filtering
    - Relationship traversal optimization
    - Variable-length path queries
    - Aggregations and collections
    - Cartesian product elimination
    - Index usage and hints
    - Memory-efficient patterns
    - Write query optimization
    
    **Neo4j Versions Supported:**
    Best practices for Neo4j 4.x and 5.x
    
    **Tips for Best Results:**
    - Include your complete Cypher query in the message
    - Mention your graph size if relevant (millions of nodes, etc.)
    - Describe performance issues you're experiencing
    - Share PROFILE or EXPLAIN output if available
    - Indicate what you've already tried
    """
    logger.info(
        "cypher_query_optimization_request",
        user_id=x_user_id,
        message_length=len(request.message)
    )
    
    # Use the Cypher Query Optimizer agent ID
    request.agent_id = "a0052146-1080-433a-83b8-03cf66610fb5"
    
    return await chat_service.chat(request, x_user_id)


@router.post("/rag/chat", response_model=ChatResponseDTO)
async def chat_with_rag_agent(
    request: ChatRequestDTO,
    x_user_id: str = Header(..., description="User identifier"),
    chat_service: ChatService = Depends(get_chat_service),
    pdf_service: PDFIngestionService = Depends(get_pdf_ingestion_service),
) -> ChatResponseDTO:
    """
    Chat with the RAG Research Agent.
    
    Retrieval-Augmented Generation agent that:
    1. Retrieves relevant chunks from stored PDFs in ChromaDB
    2. Uses retrieved context to generate informed responses
    3. Provides answers based on your uploaded documents
    
    This endpoint performs semantic search across all PDFs uploaded
    via the /api/v1/pdf/upload endpoint.
    """
    logger.info(
        "rag_agent_request",
        user_id=x_user_id,
        message_length=len(request.message)
    )
    
    # Retrieve relevant document chunks from ChromaDB
    logger.info("retrieving_relevant_documents", query=request.message)
    relevant_docs = await pdf_service.query_documents(
        query=request.message,
        top_k=5,
        filter_metadata=None  # Search across all documents
    )
    
    # Build context from retrieved documents
    context_parts = []
    sources = []
    
    for i, doc in enumerate(relevant_docs, 1):
        metadata = doc.get('metadata', {})
        text = metadata.get('text', doc.get('document', ''))
        filename = metadata.get('filename', 'unknown')
        chunk_index = metadata.get('chunk_index', 0)
        score = doc.get('score', 0)
        
        context_parts.append(f"[Document {i}: {filename}, chunk {chunk_index}, relevance: {score:.2f}]\n{text}\n")
        sources.append({
            "filename": filename,
            "chunk_index": chunk_index,
            "score": score
        })
    
    # Add retrieved context to the request
    if context_parts:
        context_text = "\n---\n".join(context_parts)
        enhanced_message = f"""Based on the following retrieved documents, please answer the question.

Retrieved Context:
{context_text}

Question: {request.message}

Please provide a comprehensive answer based on the retrieved documents. If the documents don't contain relevant information, say so."""
        
        request.message = enhanced_message
        
        logger.info(
            "rag_context_added",
            num_documents=len(relevant_docs),
            context_length=len(context_text)
        )
    else:
        logger.warning("no_relevant_documents_found")
        request.message = f"{request.message}\n\n[Note: No relevant documents were found in the knowledge base for this query.]"
    
    # Set the RAG agent ID
    request.agent_id = "1b7b2818-7df6-4bde-b2cf-46c6e2c75ff9"
    
    # Get response from chat service
    response = await chat_service.chat(request, x_user_id)
    
    # Add source information to metadata
    if sources:
        response.metadata["sources"] = sources
        response.metadata["num_sources"] = len(sources)
    
    return response
