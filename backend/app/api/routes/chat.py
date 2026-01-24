from fastapi import APIRouter, HTTPException, Depends
from typing import Any
import time

from app.services.vector import get_vector_service, VectorService
from app.services.llm import get_llm_service, LLMService
from app.models.schemas import ChatRequest, ChatResponse, ChatMessage

router = APIRouter()

@router.post("/ask", response_model=ChatResponse)
async def ask_chat(
    request: ChatRequest,
    vector_service: VectorService = Depends(get_vector_service),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Interactive chat with a document using RAG.
    """
    start_time = time.time()
    context_text = ""
    sources = []

    # 1. Retrieve Context (if document_id is provided)
    if request.document_id:
        try:
            # Use correct async search method with filter
            results = await vector_service.search(
                query=request.message,
                n_results=5,
                filter_metadata={"document_id": request.document_id}
            )
            
            # Format context
            context_chunks = []
            for res in results:
                # Similarity is 1 - distance. High similarity is good.
                score = res.get('similarity_score', 0)
                content = res.get('content', '')
                chunk_idx = res.get('metadata', {}).get('chunk_index', '?')
                
                context_chunks.append(f"[Chunk {chunk_idx} | Score: {score:.2f}]\n{content}")
                
                sources.append({
                    "content": content[:200] + "...",
                    "chunk": chunk_idx,
                    "score": score
                })
            
            context_text = "\n\n".join(context_chunks)
            if not context_text:
                print(f"Warning: No relevant chunks found for doc {request.document_id}")
            
        except Exception as e:
            # Fallback if vector search fails
            print(f"Vector search exception: {e}")
            import traceback
            traceback.print_exc()

    # 2. Build System Prompt
    system_prompt = """You are an intelligent AI assistant named Synapse.
Your goal is to answer the user's question based strictly on the provided document context.

Instructions:
- Use the provided context to answer the question.
- If the answer is not in the context, say you don't know based on the document.
- Be concise, professional, and helpful.
- Reference specific parts of the document if possible.

"""
    if context_text:
        system_prompt += f"## Document Context:\n{context_text}\n\n"
    
    # 3. Prepare Messages
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add history (limit to last 10 messages to save context window)
    for msg in request.history[-10:]:
        messages.append({"role": msg.role, "content": msg.content})
        
    # Add current message
    messages.append({"role": "user", "content": request.message})

    # 4. Call LLM
    try:
        response_text, tokens = await llm_service.chat(
            messages=messages,
            temperature=0.3, # Low temp for factual RAG
            max_tokens=1000
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

    # 5. Return Response
    return ChatResponse(
        response=response_text,
        sources=sources,
        processing_time_ms=int((time.time() - start_time) * 1000)
    )
