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
    - Keeps sentences intact
    - Maintains overlap for context preservation
    - Handles code differently than prose
    """
    
    def __init__(
        self,
        chunk_size: int = 1500, # Increased for better context
        chunk_overlap: int = 300,
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
        """
        metadata = metadata or {}
        
        # Clean text
        text = self._clean_text(text)
        
        if not text:
            return []
            
        if len(text) < self.min_chunk_size:
            return [TextChunk(
                content=text,
                index=0,
                token_count=self._estimate_tokens(text),
                metadata={**metadata, "chunk_index": 0}
            )]
        
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
        # Normalize newlines
        text = text.replace('\r\n', '\n')
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        return text.strip()
    
    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using regex.
        Uses a lookbehind for sentence terminators (.!?) followed by whitespace.
        """
        # Split by sentence terminators, keeping the terminator
        # Pattern: (?<=[.!?])\s+ -> Split after .!? followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _chunk_prose(self, text: str) -> list[str]:
        """
        Chunk prose text by sentences to preserve semantic meaning.
        """
        sentences = self._split_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If a single sentence is huge, we must split it (rare fallback)
            if sentence_len > self.chunk_size:
                # If current chunk has content, save it first
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Treat huge sentence as a whole chunk (or split roughly)
                chunks.append(sentence)
                continue

            # If adding this sentence exceeds chunk size
            if current_length + sentence_len + 1 > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                # Calculate how many sentences from the end of previous chunk to keep
                overlap_len = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    if overlap_len + len(s) > self.chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                
                current_chunk = overlap_sentences + [sentence]
                current_length = overlap_len + sentence_len
            else:
                current_chunk.append(sentence)
                current_length += sentence_len + 1 # +1 for space
        
        # Last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    
    def _chunk_code(self, text: str) -> list[str]:
        """Chunk code by logical blocks (unchanged logic mostly)."""
        # Code usually has newlines as structural elements
        lines = text.split('\n')
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_len = len(line) + 1 # +1 for newline
            
            if current_length + line_len > self.chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                
                # Simple overlap for code (last N lines)
                overlap_lines = []
                start_overlap_idx = max(0, len(current_chunk) - 10) # Last 10 lines as overlap
                overlaps = current_chunk[start_overlap_idx:]
                current_chunk = overlaps + [line]
                current_length = sum(len(l) + 1 for l in current_chunk)
            else:
                current_chunk.append(line)
                current_length += line_len
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return chunks
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count.
        Rough approximation: ~4 characters per token for English.
        """
        return len(text) // 4


# Default chunker instance
chunker = SmartTextChunker()
