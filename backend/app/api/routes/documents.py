"""
Document API Routes - Upload, manage, and retrieve documents
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pathlib import Path
from typing import Optional

from app.models.schemas import (
    UploadResponse, 
    DocumentInfo, 
    AnalysisMode
)
from app.services.document import get_document_service
from app.services.vector import get_vector_service
from app.core.exceptions import DocumentNotFoundError

router = APIRouter()


ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt', '.md', '.py', '.js', '.ts', '.jsx', '.tsx'}


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    mode: AnalysisMode = Form(default=AnalysisMode.DOCUMENT)
):
    """
    Upload a document for processing.
    
    Supports: PDF, DOCX, TXT, MD, Python, JavaScript files.
    """
    # Validate file type
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Read file content
    content = await file.read()
    
    # Process document
    doc_service = get_document_service()
    result = await doc_service.upload_and_process(
        file_content=content,
        filename=file.filename,
        mode=mode
    )
    
    return UploadResponse(
        document_id=result["document_id"],
        filename=result["filename"],
        word_count=result["word_count"],
        chunks_created=result["chunk_count"],
        processing_time_ms=result["processing_time_ms"]
    )


@router.post("/upload/text", response_model=UploadResponse)
async def upload_text_content(
    content: str,
    title: str = "Untitled",
    mode: AnalysisMode = AnalysisMode.DOCUMENT
):
    """
    Upload raw text content (useful for code snippets, etc).
    """
    doc_service = get_document_service()
    result = await doc_service.upload_content(
        content=content,
        title=title,
        mode=mode
    )
    
    return UploadResponse(
        document_id=result["document_id"],
        filename=result["filename"],
        word_count=result["word_count"],
        chunks_created=result["chunk_count"],
        processing_time_ms=result["processing_time_ms"]
    )


@router.get("/", response_model=list[DocumentInfo])
async def list_documents():
    """List all uploaded documents."""
    doc_service = get_document_service()
    docs = await doc_service.list_documents()
    
    return [
        DocumentInfo(
            id=doc["document_id"],
            filename=doc["filename"],
            doc_type=doc["doc_type"],
            word_count=0,  # Not stored in list
            chunk_count=0,
            uploaded_at=None,
            mode=doc.get("mode")
        )
        for doc in docs
    ]


@router.get("/{document_id}")
async def get_document(document_id: str, include_content: bool = False):
    """Get document information and optionally content."""
    try:
        doc_service = get_document_service()
        doc = await doc_service.get_document(document_id)
        
        response = {
            "document_id": doc["document_id"],
            "filename": doc["filename"],
            "doc_type": doc["doc_type"],
            "word_count": doc["word_count"],
            "chunk_count": doc["chunk_count"],
            "mode": doc.get("mode")
        }
        
        if include_content:
            response["content"] = doc["content"]
        
        return response
        
    except DocumentNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document."""
    doc_service = get_document_service()
    success = await doc_service.delete_document(document_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"message": "Document deleted", "document_id": document_id}
