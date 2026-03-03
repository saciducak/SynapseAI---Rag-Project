"""
Analysis API Routes - Multi-agent document analysis
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime

from app.models.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisMode,
    AnalysisStatus,
    AgentResult as AgentResultSchema
)
from app.services.document import get_document_service
from app.agents.coordinator import get_coordinator, WorkflowType
from app.core.exceptions import DocumentNotFoundError

router = APIRouter()


@router.post("/{document_id}", response_model=AnalysisResponse)
async def analyze_document(document_id: str, request: AnalysisRequest):
    """
    Run multi-agent analysis on a document.
    
    Available modes:
    - document: General document analysis
    - code: Code review and analysis
    - research: Academic paper analysis
    - legal: Legal document analysis
    
    Workflow:
    1. Analyzer → extracts key information
    2. Summarizer → creates multi-level summaries
    3. Recommender → generates action items
    """
    try:
        # Get document content
        doc_service = get_document_service()
        doc = await doc_service.get_document(document_id)
        
    except DocumentNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get coordinator
    coordinator = get_coordinator()
    
    # Determine workflow type
    workflow_type = WorkflowType.FULL
    metadata = {
        "filename": doc["filename"],
        "doc_type": doc["doc_type"],
        "word_count": doc["word_count"]
    }
    
    # Execute workflow: RAG-enhanced for deeper analysis, else parallel or sequential
    if request.use_rag:
        result = await coordinator.execute_rag_workflow(
            mode=request.mode,
            document_id=document_id,
            content=doc["content"],
            metadata=metadata,
            focus_query=request.focus_query
        )
    elif request.parallel_execution:
        result = await coordinator.execute_parallel(
            mode=request.mode,
            content=doc["content"],
            metadata=metadata
        )
    else:
        result = await coordinator.execute_workflow(
            mode=request.mode,
            workflow_type=workflow_type,
            content=doc["content"],
            metadata=metadata,
            user_context=request.user_context
        )
    
    # Format agent results
    agent_results = [
        AgentResultSchema(
            agent_name=r.agent_name,
            role=r.role.value,
            output=r.output,
            confidence=r.confidence,
            tokens_used=r.tokens_used,
            execution_time_ms=r.execution_time_ms
        )
        for r in result.results.values()
    ]
    
    return AnalysisResponse(
        analysis_id=result.workflow_id,
        document_id=document_id,
        mode=result.mode,
        status=AnalysisStatus.COMPLETED if result.success else AnalysisStatus.FAILED,
        started_at=result.started_at,
        completed_at=result.completed_at,
        total_tokens=result.total_tokens,
        total_time_ms=result.total_time_ms,
        agents=agent_results,
        final_output=result.final_output
    )


@router.post("/{document_id}/quick")
async def quick_analysis(document_id: str, mode: AnalysisMode = AnalysisMode.DOCUMENT):
    """
    Quick analysis - only analyzer and summarizer.
    Faster but without detailed recommendations.
    """
    try:
        doc_service = get_document_service()
        doc = await doc_service.get_document(document_id)
    except DocumentNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    
    coordinator = get_coordinator()
    
    result = await coordinator.execute_workflow(
        mode=mode,
        workflow_type=WorkflowType.QUICK,
        content=doc["content"],
        metadata={
            "filename": doc["filename"],
            "doc_type": doc["doc_type"]
        }
    )
    
    return {
        "analysis_id": result.workflow_id,
        "document_id": document_id,
        "mode": mode.value,
        "total_tokens": result.total_tokens,
        "total_time_ms": result.total_time_ms,
        "analysis": result.final_output.get("analyzer", {}),
        "summary": result.final_output.get("summarizer", {})
    }


@router.get("/modes")
async def get_available_modes():
    """Get available analysis modes with descriptions."""
    return {
        "modes": [
            {
                "id": "document",
                "name": "Document Analysis",
                "description": "General document analysis, summarization, and action extraction",
                "icon": "📄"
            },
            {
                "id": "code",
                "name": "Code Review",
                "description": "Code quality analysis, bug detection, and improvement suggestions",
                "icon": "💻"
            },
            {
                "id": "research",
                "name": "Research Paper",
                "description": "Academic paper analysis, methodology extraction, and citation analysis",
                "icon": "📚"
            },
            {
                "id": "legal",
                "name": "Legal Document",
                "description": "Contract analysis, risk assessment, and obligation extraction",
                "icon": "⚖️"
            }
        ]
    }


@router.post("/{document_id}/rag", response_model=AnalysisResponse)
async def analyze_document_with_rag(
    document_id: str, 
    request: AnalysisRequest,
    focus_query: str = None
):
    """
    RAG-Enhanced Analysis with multi-pass workflow and citations.
    
    This endpoint uses the new intelligence layer:
    - Pass 1: Retrieve relevant chunks using hybrid search (semantic + keyword)
    - Pass 2: Analyze with enriched context from VectorDB
    - Pass 3: Generate findings with [Chunk X] citation markers
    
    Use this for higher quality analysis with evidence-based findings.
    """
    try:
        doc_service = get_document_service()
        doc = await doc_service.get_document(document_id)
    except DocumentNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    
    coordinator = get_coordinator()
    
    # Execute RAG-enhanced workflow
    result = await coordinator.execute_rag_workflow(
        mode=request.mode,
        document_id=document_id,
        content=doc["content"],
        metadata={
            "filename": doc["filename"],
            "doc_type": doc["doc_type"],
            "word_count": doc["word_count"]
        },
        focus_query=focus_query
    )
    
    # Format agent results
    agent_results = [
        AgentResultSchema(
            agent_name=r.agent_name,
            role=r.role.value,
            output=r.output,
            confidence=r.confidence,
            tokens_used=r.tokens_used,
            execution_time_ms=r.execution_time_ms
        )
        for r in result.results.values()
    ]
    
    return AnalysisResponse(
        analysis_id=result.workflow_id,
        document_id=document_id,
        mode=result.mode,
        status=AnalysisStatus.COMPLETED if result.success else AnalysisStatus.FAILED,
        started_at=result.started_at,
        completed_at=result.completed_at,
        total_tokens=result.total_tokens,
        total_time_ms=result.total_time_ms,
        agents=agent_results,
        final_output=result.final_output
    )
