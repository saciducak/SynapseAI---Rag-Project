"""
Vector Store Service - ChromaDB + Ollama Embeddings
RAG-powered semantic search and document storage
"""
import httpx
import asyncio
from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger
import hashlib

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
        # Increase timeout for batch operations
        self.timeout = httpx.Timeout(60.0, connect=10.0)
        
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
    
    async def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings from Ollama in batches."""
        embeddings = []
        batch_size = 10  # Ollama handles small batches better
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Process batch concurrently
                tasks = []
                for text in batch:
                    tasks.append(self._fetch_embedding(client, text))
                
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for res in batch_results:
                    if isinstance(res, list):
                        embeddings.append(res)
                    else:
                        logger.error(f"Embedding failed: {res}")
                        # Fallback to hash if embedding fails
                        embeddings.append(self._simple_embedding("error"))

        return embeddings

    async def _fetch_embedding(self, client: httpx.AsyncClient, text: str) -> list[float]:
        """Fetch single embedding (helper for batch)."""
        try:
            response = await client.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            logger.warning(f"Embedding error: {e}, using fallback")
            return self._simple_embedding(text)
    
    def _simple_embedding(self, text: str, dim: int = 768) -> list[float]:
        """Simple hash-based embedding fallback."""
        # Nomic embed text is 768 dim usually, but let's confirm usage
        # Ensuring dimension consistency is key. Nomic is 768.
        # If using llama2/3 default it might be different (4096).
        # We'll stick to 768 as a safe default for this model.
        hash_bytes = hashlib.sha256(text.encode()).digest()
        embedding = []
        while len(embedding) < dim:
             # Extend hash if needed
             hash_bytes = hashlib.sha256(hash_bytes).digest()
             for b in hash_bytes:
                 if len(embedding) < dim:
                     embedding.append((b - 128) / 128.0)
        return embedding
    
    async def add_document(
        self,
        document_id: str,
        chunks: list[TextChunk],
        metadata: dict
    ) -> int:
        """
        Add document chunks to vector store with entity enrichment.
        Embeds both content and extracted keywords for hybrid search.
        """
        if not chunks:
            return 0
        
        # Prepare data with entity enrichment
        ids = [f"{document_id}_chunk_{chunk.index}" for chunk in chunks]
        
        # Use searchable_text which includes entities for better retrieval
        documents = [chunk.searchable_text for chunk in chunks]
        
        metadatas = []
        for chunk in chunks:
            chunk_meta = {
                **metadata,
                "document_id": document_id,
                "chunk_index": chunk.index,
                "token_count": chunk.token_count,
                "content_preview": chunk.content[:200],  # Store content preview
            }
            
            # Add entity data for filtering
            if chunk.entities:
                entity_dict = chunk.entities.to_dict()
                # Flatten for ChromaDB (only scalar/list values)
                chunk_meta["keywords"] = ",".join(entity_dict.get("keywords", [])[:10])
                chunk_meta["persons"] = ",".join(entity_dict.get("persons", [])[:5])
                chunk_meta["organizations"] = ",".join(entity_dict.get("organizations", [])[:5])
                chunk_meta["dates"] = ",".join(entity_dict.get("dates", [])[:5])
                chunk_meta["has_money"] = len(entity_dict.get("monetary_amounts", [])) > 0
                chunk_meta["has_email"] = len(entity_dict.get("emails", [])) > 0
            
            metadatas.append(chunk_meta)
        
        # Get embeddings with batching (embedding searchable_text not raw content)
        logger.info(f"Generating embeddings for {len(documents)} enriched chunks...")
        embeddings = await self._get_embeddings_batch(documents)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=[chunk.content for chunk in chunks],  # Store original content
            embeddings=embeddings,  # But embed searchable_text
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} enriched chunks for document {document_id}")
        return len(chunks)
    
    async def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[dict] = None,
        boost_keywords: bool = True
    ) -> list[dict]:
        """
        Enhanced semantic search with optional keyword boosting.
        When boost_keywords is True, results containing query keywords in metadata get higher scores.
        """
        # Get query embedding (single)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
             query_embedding = await self._fetch_embedding(client, query)
        
        # Search with more results for re-ranking
        fetch_n = n_results * 2 if boost_keywords else n_results
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_n,
            where=filter_metadata
        )
        
        # Format and optionally boost results
        formatted = []
        query_words = set(query.lower().split())
        
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][i] if results['distances'] else 0
                base_similarity = 1 - distance
                
                meta = results['metadatas'][0][i] if results['metadatas'] else {}
                
                # Keyword boost: increase score if query words appear in extracted keywords
                boost = 0.0
                if boost_keywords and meta.get("keywords"):
                    stored_keywords = set(meta["keywords"].lower().split(","))
                    overlap = query_words & stored_keywords
                    boost = len(overlap) * 0.05  # 5% boost per matching keyword
                
                final_score = min(base_similarity + boost, 1.0)  # Cap at 1.0
                
                formatted.append({
                    "content": doc,
                    "metadata": meta,
                    "similarity_score": round(final_score, 4),
                    "base_score": round(base_similarity, 4),
                    "keyword_boost": round(boost, 4),
                    "id": results['ids'][0][i] if results['ids'] else "",
                    "chunk_index": meta.get("chunk_index", 0)
                })
        
        # Re-sort by boosted score
        if boost_keywords:
            formatted.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return formatted[:n_results]
    
    async def hybrid_search(
        self,
        query: str,
        n_results: int = 5,
        document_id: Optional[str] = None,
        entity_filters: Optional[dict] = None
    ) -> list[dict]:
        """
        Hybrid search combining semantic + entity-based filtering.
        
        Args:
            query: Search query
            n_results: Number of results to return
            document_id: Optional filter to a specific document
            entity_filters: Optional filters like {"has_money": True, "organizations": "contains:Google"}
        """
        # Build ChromaDB where clause
        where_clause = {}
        
        if document_id:
            where_clause["document_id"] = document_id
        
        if entity_filters:
            for key, value in entity_filters.items():
                if isinstance(value, bool):
                    where_clause[key] = value
                elif isinstance(value, str) and value.startswith("contains:"):
                    # ChromaDB doesn't support contains natively, we filter post-query
                    pass
                else:
                    where_clause[key] = value
        
        # Perform search with keyword boosting
        results = await self.search(
            query=query,
            n_results=n_results * 2,  # Fetch more for post-filtering
            filter_metadata=where_clause if where_clause else None,
            boost_keywords=True
        )
        
        # Post-filter for "contains:" type filters
        if entity_filters:
            for key, value in entity_filters.items():
                if isinstance(value, str) and value.startswith("contains:"):
                    search_term = value[9:].lower()  # Remove "contains:" prefix
                    results = [
                        r for r in results 
                        if search_term in r["metadata"].get(key, "").lower()
                    ]
        
        return results[:n_results]
    
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
        results = self.collection.get(limit=10000)
        
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
