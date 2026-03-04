"""
Pydantic Schemas for SynapseAI API
"""
from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime
from enum import Enum


# === Enums ===

class AnalysisMode(str, Enum):
    """Available analysis modes."""
    DOCUMENT = "document"
    CODE = "code"
    RESEARCH = "research"
    LEGAL = "legal"


class AnalysisStatus(str, Enum):
    """Analysis status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# === Request Schemas ===

class AnalysisRequest(BaseModel):
    """Request for document analysis."""
    mode: AnalysisMode = Field(default=AnalysisMode.DOCUMENT, description="Analysis mode")
    user_context: Optional[str] = Field(None, description="Additional context from user")
    parallel_execution: bool = Field(False, description="Run agents in parallel")
    use_rag: bool = Field(True, description="Use RAG-enhanced workflow for deeper, citation-backed analysis")
    focus_query: Optional[str] = Field(None, description="Optional query to focus RAG retrieval (e.g. 'risks and deadlines')")


class SearchRequest(BaseModel):
    """Request for semantic search."""
    query: str = Field(..., min_length=2, max_length=1000, description="Search query")
    n_results: int = Field(5, ge=1, le=20, description="Number of results")
    mode_filter: Optional[AnalysisMode] = Field(None, description="Filter by mode")


class AskRequest(BaseModel):
    """Request for RAG-powered Q&A."""
    question: str = Field(..., min_length=5, max_length=2000, description="Question to ask")
    document_ids: Optional[list[str]] = Field(None, description="Limit to specific documents")


class ChatMessage(BaseModel):
    """Single chat message."""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    """Request for interactive chat."""
    message: str
    document_id: Optional[str] = None
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    """Response for interactive chat."""
    response: str
    sources: list[dict[str, Any]] = []
    processing_time_ms: int


# === Response Schemas ===

class DocumentInfo(BaseModel):
    """Document information."""
    id: str
    filename: str
    doc_type: str
    word_count: int
    chunk_count: int
    uploaded_at: Optional[datetime] = None
    mode: Optional[AnalysisMode] = None


class UploadResponse(BaseModel):
    """Response after document upload."""
    document_id: str
    filename: str
    word_count: int
    chunks_created: int
    processing_time_ms: int
    message: str = "Document uploaded successfully"


class AgentResult(BaseModel):
    """Result from a single agent."""
    agent_name: str
    role: str
    output: dict[str, Any] | str
    confidence: float
    tokens_used: int
    execution_time_ms: int


class Citation(BaseModel):
    """Citation reference to source chunk."""
    marker: str = Field(..., description="Citation marker like [Chunk 5]")
    chunk_index: int
    source_preview: Optional[str] = Field(None, description="Preview of source text")
    similarity_score: Optional[float] = None


class QualityMetrics(BaseModel):
    """Output quality metrics."""
    has_structure: bool = True
    citation_count: int = 0
    completeness: float = Field(0.0, ge=0, le=1)
    depth: str = Field("moderate", pattern="^(shallow|moderate|comprehensive)$")


class AnalysisResponse(BaseModel):
    """Full analysis response with quality metrics."""
    analysis_id: str
    document_id: str
    mode: AnalysisMode
    status: AnalysisStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_tokens: int
    total_time_ms: int
    agents: list[AgentResult]
    final_output: dict[str, Any]
    # v2.0 Premium Output fields
    rag_enabled: bool = Field(False, description="Whether RAG was used")
    confidence_score: Optional[float] = Field(None, ge=0, le=1)
    citations: list[Citation] = Field(default_factory=list)
    quality_metrics: Optional[QualityMetrics] = None


class SearchResult(BaseModel):
    """Single search result."""
    content: str
    document_id: str
    filename: str
    chunk_index: int
    similarity_score: float
    metadata: dict[str, Any] = {}


class SearchResponse(BaseModel):
    """Search response."""
    query: str
    total_results: int
    results: list[SearchResult]
    search_time_ms: int


class AskResponse(BaseModel):
    """RAG Q&A response."""
    question: str
    answer: str
    sources: list[SearchResult]
    tokens_used: int
    response_time_ms: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str
    vector_db_status: str
    total_documents: int
    available_modes: list[str]
