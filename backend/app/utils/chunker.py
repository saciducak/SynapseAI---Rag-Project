"""
Smart Text Chunker - Semantic-aware text splitting
"""
from dataclasses import dataclass, field
from typing import Optional
import re


@dataclass
class TextChunk:
    """A chunk of text with metadata."""
    content: str
    index: int
    token_count: int
    metadata: dict = field(default_factory=dict)
    
    @property
    def char_count(self) -> int:
        return len(self.content)


class SmartTextChunker:
    """
    Intelligent text chunker that respects semantic boundaries.
    - Keeps paragraphs together when possible
    - Maintains overlap for context preservation
    - Handles code differently than prose
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        respect_code_blocks: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.respect_code_blocks = respect_code_blocks
    
    def chunk(
        self,
        text: str,
        metadata: Optional[dict] = None,
        is_code: bool = False
    ) -> list[TextChunk]:
        """
        Split text into semantic chunks.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            is_code: Whether the text is code
            
        Returns:
            List of TextChunk objects
        """
        metadata = metadata or {}
        
        # Clean text
        text = self._clean_text(text)
        
        if not text or len(text) < self.min_chunk_size:
            return [TextChunk(
                content=text,
                index=0,
                token_count=self._estimate_tokens(text),
                metadata={**metadata, "chunk_index": 0}
            )] if text else []
        
        # Choose splitting strategy
        if is_code:
            chunks = self._chunk_code(text)
        else:
            chunks = self._chunk_prose(text)
        
        # Create TextChunk objects
        return [
            TextChunk(
                content=chunk,
                index=i,
                token_count=self._estimate_tokens(chunk),
                metadata={**metadata, "chunk_index": i, "total_chunks": len(chunks)}
            )
            for i, chunk in enumerate(chunks)
        ]
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        text = re.sub(r' {3,}', '  ', text)
        return text.strip()
    
    def _chunk_prose(self, text: str) -> list[str]:
        """Chunk prose text by paragraphs."""
        # Split by paragraphs
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return self._merge_into_chunks(paragraphs)
    
    def _chunk_code(self, text: str) -> list[str]:
        """Chunk code by logical blocks (functions, classes)."""
        lines = text.split('\n')
        
        # Find natural break points (empty lines, function/class definitions)
        blocks = []
        current_block = []
        
        for line in lines:
            # Check for natural break points
            is_break = (
                not line.strip() or  # Empty line
                line.strip().startswith(('def ', 'class ', 'function ', 'const ', 'let ', 'var ', 'public ', 'private '))
            )
            
            if is_break and current_block and len('\n'.join(current_block)) > self.min_chunk_size:
                blocks.append('\n'.join(current_block))
                current_block = [line] if line.strip() else []
            else:
                current_block.append(line)
        
        if current_block:
            blocks.append('\n'.join(current_block))
        
        return self._merge_into_chunks(blocks)
    
    def _merge_into_chunks(self, blocks: list[str]) -> list[str]:
        """Merge blocks into appropriately sized chunks with overlap."""
        if not blocks:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for block in blocks:
            block_length = len(block)
            
            # If adding this block exceeds chunk size
            if current_length + block_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_content = self._get_overlap_content(current_chunk)
                if overlap_content:
                    current_chunk = [overlap_content, block]
                    current_length = len(overlap_content) + block_length
                else:
                    current_chunk = [block]
                    current_length = block_length
            else:
                current_chunk.append(block)
                current_length += block_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _get_overlap_content(self, chunks: list[str]) -> str:
        """Get overlap content from previous chunk for context."""
        if not chunks:
            return ""
        
        last_block = chunks[-1]
        
        if len(last_block) <= self.chunk_overlap:
            return last_block
        
        # Take last N characters with "..." prefix
        return "..." + last_block[-self.chunk_overlap:]
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count.
        Rough approximation: ~4 characters per token for English.
        """
        return len(text) // 4


# Default chunker instance
chunker = SmartTextChunker()
