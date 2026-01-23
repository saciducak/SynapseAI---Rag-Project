"""
Vector Store Service - ChromaDB + Ollama Embeddings
RAG-powered semantic search and document storage
"""
import httpx
from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from app.core.config import get_settings
from app.utils.chunker import TextChunk


settings = get_settings()


class VectorService:
    """
    Vector database service using ChromaDB with Ollama embeddings.
    Provides semantic search and document storage capabilities.
    """
    
    def __init__(self, collection_name: str = "synapse_documents"):
        self.settings = get_settings()
        self.ollama_url = "http://localhost:11434"
        self.embedding_model = "nomic-embed-text"  # Small, fast embedding model
        
        # Initialize ChromaDB
        self.client = chromadb.Client(ChromaSettings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory=self.settings.CHROMA_PERSIST_DIR
        ))
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"VectorService initialized with collection: {collection_name}")
    
    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings from Ollama."""
        embeddings = []
        
        # Process each text (Ollama processes one at a time)
        for text in texts:
            try:
                response = httpx.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": text
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                result = response.json()
                embeddings.append(result["embedding"])
            except httpx.ConnectError:
                logger.warning("Ollama not available for embeddings, using simple hash")
                # Fallback: simple hash-based embedding (not semantic but works)
                embeddings.append(self._simple_embedding(text))
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                embeddings.append(self._simple_embedding(text))
        
        return embeddings
    
    def _simple_embedding(self, text: str, dim: int = 384) -> list[float]:
        """Simple hash-based embedding fallback."""
        import hashlib
        # Create a deterministic embedding from text hash
        hash_bytes = hashlib.sha256(text.encode()).digest()
        embedding = []
        for i in range(dim):
            byte_idx = i % len(hash_bytes)
            embedding.append((hash_bytes[byte_idx] - 128) / 128.0)
        return embedding
    
    async def add_document(
        self,
        document_id: str,
        chunks: list[TextChunk],
        metadata: dict
    ) -> int:
        """
        Add document chunks to vector store.
        
        Args:
            document_id: Unique document identifier
            chunks: List of text chunks
            metadata: Document metadata
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        # Prepare data
        ids = [f"{document_id}_chunk_{chunk.index}" for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                **metadata,
                "document_id": document_id,
                "chunk_index": chunk.index,
                "token_count": chunk.token_count
            }
            for chunk in chunks
        ]
        
        # Get embeddings
        embeddings = self._get_embeddings(documents)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} chunks for document {document_id}")
        return len(chunks)
    
    async def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[dict] = None
    ) -> list[dict]:
        """
        Semantic search across documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of search results with content, metadata, and similarity score
        """
        # Get query embedding
        query_embedding = self._get_embeddings([query])[0]
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata
        )
        
        # Format results
        formatted = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][i] if results['distances'] else 0
                # Convert distance to similarity score (cosine: 1 - distance)
                similarity = 1 - distance
                
                formatted.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "similarity_score": round(similarity, 4),
                    "id": results['ids'][0][i] if results['ids'] else ""
                })
        
        return formatted
    
    async def get_document_chunks(
        self,
        document_id: str,
        limit: int = 100
    ) -> list[dict]:
        """Get all chunks for a specific document."""
        results = self.collection.get(
            where={"document_id": document_id},
            limit=limit
        )
        
        chunks = []
        if results['documents']:
            for i, doc in enumerate(results['documents']):
                chunks.append({
                    "content": doc,
                    "metadata": results['metadatas'][i] if results['metadatas'] else {},
                    "id": results['ids'][i] if results['ids'] else ""
                })
        
        # Sort by chunk index
        chunks.sort(key=lambda x: x['metadata'].get('chunk_index', 0))
        return chunks
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks of a document."""
        try:
            # Get all chunk IDs for this document
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get collection statistics."""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection.name,
            "status": "connected"
        }
    
    async def list_documents(self) -> list[dict]:
        """List all unique documents in the collection."""
        # Get all metadata
        results = self.collection.get(limit=10000)
        
        # Extract unique documents
        documents = {}
        if results['metadatas']:
            for meta in results['metadatas']:
                doc_id = meta.get('document_id')
                if doc_id and doc_id not in documents:
                    documents[doc_id] = {
                        "document_id": doc_id,
                        "filename": meta.get('filename', 'unknown'),
                        "doc_type": meta.get('doc_type', 'unknown'),
                        "mode": meta.get('mode'),
                    }
        
        return list(documents.values())


# Singleton instance - created on first use
_vector_service: Optional[VectorService] = None


def get_vector_service() -> VectorService:
    """Get or create vector service instance."""
    global _vector_service
    if _vector_service is None:
        _vector_service = VectorService()
    return _vector_service
