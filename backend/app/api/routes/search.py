"""
Search API Routes - Semantic search and RAG Q&A
"""
import time
from fastapi import APIRouter

from app.models.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResult,
    AskRequest,
    AskResponse
)
from app.services.vector import get_vector_service
from app.services.llm import get_llm_service

router = APIRouter()


@router.post("/semantic", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """
    Semantic search across all documents.
    Uses OpenAI embeddings for similarity matching.
    """
    start_time = time.time()
    
    vector_service = get_vector_service()
    
    # Build filter if mode specified
    filter_metadata = None
    if request.mode_filter:
        filter_metadata = {"mode": request.mode_filter.value}
    
    # Search
    results = await vector_service.search(
        query=request.query,
        n_results=request.n_results,
        filter_metadata=filter_metadata
    )
    
    # Format results
    search_results = [
        SearchResult(
            content=r["content"],
            document_id=r["metadata"].get("document_id", ""),
            filename=r["metadata"].get("filename", "unknown"),
            chunk_index=r["metadata"].get("chunk_index", 0),
            similarity_score=r["similarity_score"],
            metadata=r["metadata"]
        )
        for r in results
    ]
    
    search_time = int((time.time() - start_time) * 1000)
    
    return SearchResponse(
        query=request.query,
        total_results=len(search_results),
        results=search_results,
        search_time_ms=search_time
    )


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    RAG-powered Q&A.
    Searches relevant documents and generates an answer.
    """
    start_time = time.time()
    
    vector_service = get_vector_service()
    llm_service = get_llm_service()
    
    # Build filter if specific documents requested
    filter_metadata = None
    if request.document_ids:
        # For multiple docs, we'd need OR logic - simplify to first doc
        filter_metadata = {"document_id": request.document_ids[0]}
    
    # Search for relevant chunks
    search_results = await vector_service.search(
        query=request.question,
        n_results=5,
        filter_metadata=filter_metadata
    )
    
    # Build context from search results
    context_parts = []
    for i, result in enumerate(search_results):
        context_parts.append(f"""
[Source {i+1}: {result['metadata'].get('filename', 'unknown')}]
{result['content']}
""")
    
    context = "\n---\n".join(context_parts)
    
    # Generate answer
    system_prompt = """You are a helpful assistant that answers questions based on the provided context.
    
Rules:
- Only use information from the provided context
- If the answer isn't in the context, say so
- Cite sources when possible (e.g., "According to [Source 1]...")
- Be concise but comprehensive
- If asked about something not in the context, acknowledge the limitation"""
    
    user_prompt = f"""Context:
{context}

Question: {request.question}

Please provide a comprehensive answer based on the context above."""
    
    response, tokens = await llm_service.simple_prompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3
    )
    
    # Format source results
    sources = [
        SearchResult(
            content=r["content"][:200] + "...",  # Truncate for response
            document_id=r["metadata"].get("document_id", ""),
            filename=r["metadata"].get("filename", "unknown"),
            chunk_index=r["metadata"].get("chunk_index", 0),
            similarity_score=r["similarity_score"],
            metadata={}
        )
        for r in search_results
    ]
    
    response_time = int((time.time() - start_time) * 1000)
    
    return AskResponse(
        question=request.question,
        answer=response,
        sources=sources,
        tokens_used=tokens,
        response_time_ms=response_time
    )
