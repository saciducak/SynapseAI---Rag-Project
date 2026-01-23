"""
Document Service - High-level document management
"""
import uuid
import time
from pathlib import Path
from typing import Optional
import aiofiles
from loguru import logger

from app.core.config import get_settings
from app.core.exceptions import DocumentProcessingError, DocumentNotFoundError
from app.utils.parser import parser, ParsedDocument
from app.utils.chunker import chunker
from app.services.vector import get_vector_service
from app.models.schemas import AnalysisMode


settings = get_settings()


class DocumentService:
    """
    High-level document management service.
    Handles upload, processing, and storage.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.upload_dir = Path(self.settings.UPLOAD_DIR)
        self.upload_dir.mkdir(exist_ok=True)
    
    async def upload_and_process(
        self,
        file_content: bytes,
        filename: str,
        mode: AnalysisMode = AnalysisMode.DOCUMENT
    ) -> dict:
        """
        Upload and process a document.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            mode: Analysis mode for the document
            
        Returns:
            Document info dict
        """
        start_time = time.time()
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Save file temporarily
        file_path = self.upload_dir / f"{document_id}_{filename}"
        
        try:
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_content)
            
            # Parse document
            parsed = parser.parse(file_path)
            
            # Chunk the content
            chunks = chunker.chunk(
                parsed.content,
                metadata={
                    "document_id": document_id,
                    "filename": filename
                },
                is_code=parsed.is_code
            )
            
            # Store in vector database
            vector_service = get_vector_service()
            await vector_service.add_document(
                document_id=document_id,
                chunks=chunks,
                metadata={
                    "document_id": document_id,
                    "filename": filename,
                    "doc_type": parsed.doc_type.value,
                    "word_count": parsed.word_count,
                    "mode": mode.value,
                    **parsed.metadata
                }
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            
            logger.info(f"Document {document_id} processed: {len(chunks)} chunks in {processing_time}ms")
            
            return {
                "document_id": document_id,
                "filename": filename,
                "doc_type": parsed.doc_type.value,
                "word_count": parsed.word_count,
                "chunk_count": len(chunks),
                "processing_time_ms": processing_time,
                "mode": mode.value
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise DocumentProcessingError(f"Failed to process document: {str(e)}")
        
        finally:
            # Cleanup temp file
            if file_path.exists():
                file_path.unlink()
    
    async def upload_content(
        self,
        content: str,
        title: str = "Untitled",
        mode: AnalysisMode = AnalysisMode.DOCUMENT
    ) -> dict:
        """
        Upload raw text content (useful for code snippets, etc).
        """
        start_time = time.time()
        document_id = str(uuid.uuid4())
        
        try:
            # Parse content directly
            parsed = parser.parse_content(content, f"{title}.txt")
            
            # Chunk
            chunks = chunker.chunk(
                parsed.content,
                metadata={
                    "document_id": document_id,
                    "filename": title
                },
                is_code=parsed.is_code
            )
            
            # Store
            vector_service = get_vector_service()
            await vector_service.add_document(
                document_id=document_id,
                chunks=chunks,
                metadata={
                    "document_id": document_id,
                    "filename": title,
                    "doc_type": parsed.doc_type.value,
                    "word_count": parsed.word_count,
                    "mode": mode.value
                }
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                "document_id": document_id,
                "filename": title,
                "doc_type": parsed.doc_type.value,
                "word_count": parsed.word_count,
                "chunk_count": len(chunks),
                "processing_time_ms": processing_time,
                "mode": mode.value
            }
            
        except Exception as e:
            raise DocumentProcessingError(f"Failed to process content: {str(e)}")
    
    async def get_document(self, document_id: str) -> dict:
        """Get document information and content."""
        vector_service = get_vector_service()
        chunks = await vector_service.get_document_chunks(document_id)
        
        if not chunks:
            raise DocumentNotFoundError(f"Document {document_id} not found")
        
        # Reconstruct document
        content = "\n\n".join([c['content'] for c in chunks])
        metadata = chunks[0]['metadata']
        
        return {
            "document_id": document_id,
            "filename": metadata.get('filename', 'unknown'),
            "doc_type": metadata.get('doc_type', 'unknown'),
            "word_count": metadata.get('word_count', 0),
            "chunk_count": len(chunks),
            "mode": metadata.get('mode'),
            "content": content
        }
    
    async def get_document_content(self, document_id: str) -> str:
        """Get reconstructed document content."""
        doc = await self.get_document(document_id)
        return doc['content']
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document."""
        vector_service = get_vector_service()
        return await vector_service.delete_document(document_id)
    
    async def list_documents(self) -> list[dict]:
        """List all documents."""
        vector_service = get_vector_service()
        return await vector_service.list_documents()


# Singleton
_document_service: Optional[DocumentService] = None


def get_document_service() -> DocumentService:
    """Get or create document service instance."""
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service
