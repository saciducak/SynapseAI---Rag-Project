"""
SynapseAI - Multi-Mode Multi-Agent Decision Support System
FastAPI Main Application
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys

from app.core.config import get_settings
from app.api.routes import documents, analysis, search, chat
from app.services.vector import get_vector_service
from app.models.schemas import HealthResponse


# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>")


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("🚀 Starting SynapseAI...")
    logger.info(f"📊 Environment: {'DEBUG' if settings.DEBUG else 'PRODUCTION'}")
    
    # Initialize vector service
    try:
        vector_service = get_vector_service()
        stats = vector_service.get_stats()
        logger.info(f"✅ Vector DB connected: {stats['total_chunks']} chunks")
    except Exception as e:
        logger.error(f"❌ Vector DB initialization failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down SynapseAI...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="""
## SynapseAI - Multi-Mode Multi-Agent Decision Support System

An intelligent document analysis platform powered by multiple specialized AI agents.

### Features
- 📄 **Document Analysis**: General document processing
- 💻 **Code Review**: Code quality and security analysis
- 📚 **Research Papers**: Academic paper analysis
- ⚖️ **Legal Documents**: Contract and legal document analysis

### Capabilities
- Multi-agent orchestration
- RAG-powered semantic search
- Intelligent summarization
- Action item extraction
    """,
    version=settings.APP_VERSION,
    lifespan=lifespan
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# Include routers
app.include_router(
    documents.router,
    prefix="/api/documents",
    tags=["Documents"]
)

app.include_router(
    analysis.router,
    prefix="/api/analysis",
    tags=["Analysis"]
)

app.include_router(
    search.router,
    prefix="/api/search",
    tags=["Search"]
)

app.include_router(
    chat.router,
    prefix="/api/chat",
    tags=["Chat"]
)


# Health check
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    try:
        vector_service = get_vector_service()
        stats = vector_service.get_stats()
        vector_status = stats["status"]
        total_docs = stats["total_chunks"]
    except Exception:
        vector_status = "error"
        total_docs = 0
    
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        vector_db_status=vector_status,
        total_documents=total_docs,
        available_modes=settings.AVAILABLE_MODES
    )


@app.get("/", tags=["System"])
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "Multi-Mode Multi-Agent Decision Support System",
        "docs": "/docs",
        "health": "/health"
    }


# App init
@app.get("/api/info", tags=["System"])
async def get_info():
    """Get system information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "modes": [
            {"id": "document", "name": "Document Analysis", "icon": "📄"},
            {"id": "code", "name": "Code Review", "icon": "💻"},
            {"id": "research", "name": "Research Paper", "icon": "📚"},
            {"id": "legal", "name": "Legal Document", "icon": "⚖️"}
        ],
        "agents": [
            "Analyzer", "Summarizer", "Recommender",
            "CodeAnalyzer", "ResearchAnalyzer", "LegalAnalyzer"
        ]
    }
