"""
Output Formatter - Post-processing for premium quality outputs
Adds confidence scores, citation links, and consistent formatting
"""
import re
from typing import Optional
from dataclasses import dataclass


@dataclass
class FormattedOutput:
    """Container for formatted analysis output."""
    content: dict
    confidence_score: float
    quality_metrics: dict
    citations: list[dict]


class OutputFormatter:
    """
    Formats agent outputs for premium quality display.
    - Adds confidence scoring
    - Formats citation markers as clickable references
    - Normalizes number formatting
    - Calculates quality metrics
    """
    
    # Citation pattern: [Chunk X] or [Chunk 12]
    CITATION_PATTERN = re.compile(r'\[Chunk\s*(\d+)\]', re.IGNORECASE)
    
    # Number patterns for formatting
    MONEY_PATTERN = re.compile(r'(\$|€|₺|£)?\s?([\d,]+(?:\.\d{2})?)')
    
    def __init__(self, locale: str = "en"):
        self.locale = locale
    
    def format_analysis(
        self, 
        analysis_output: dict,
        rag_chunks: Optional[list[dict]] = None
    ) -> FormattedOutput:
        """
        Post-process analysis output for premium quality.
        
        Args:
            analysis_output: Raw output from agents
            rag_chunks: Optional RAG chunks for citation linking
        """
        # Calculate confidence score
        confidence = self._calculate_confidence(analysis_output)
        
        # Extract and format citations
        citations = self._extract_citations(analysis_output, rag_chunks)
        
        # Calculate quality metrics
        quality = self._calculate_quality_metrics(analysis_output, citations)
        
        # Format the content
        formatted_content = self._format_content(analysis_output)
        
        return FormattedOutput(
            content=formatted_content,
            confidence_score=confidence,
            quality_metrics=quality,
            citations=citations
        )
    
    def _calculate_confidence(self, output: dict) -> float:
        """
        Calculate overall confidence score based on output quality.
        """
        score = 0.5  # Base score
        
        # Check for key fields
        if isinstance(output, dict):
            # Analyzer output (nested under "analyzer" in our pipeline)
            analyzer = output.get("analyzer") if isinstance(output.get("analyzer"), dict) else output
            summarizer = output.get("summarizer") if isinstance(output.get("summarizer"), dict) else output
            recommender = output.get("recommender") if isinstance(output.get("recommender"), dict) else output

            if analyzer.get("main_topics"):
                score += 0.1
            if analyzer.get("key_entities") or analyzer.get("entities"):
                score += 0.1
            if analyzer.get("key_insights") or analyzer.get("key_points"):
                score += 0.1
            if analyzer.get("complexity_score"):
                score += 0.05
            
            # Summarizer output
            if summarizer.get("executive_summary"):
                score += 0.1
            if summarizer.get("key_takeaways"):
                score += 0.05

            # Recommender output
            if recommender.get("action_items"):
                score += 0.05
            
            # Evidence presence boosts confidence
            if self._has_evidence(output):
                score += 0.15
        
        return min(round(score, 2), 1.0)
    
    def _has_evidence(self, output: dict, depth: int = 0) -> bool:
        """Check if output contains evidence fields."""
        if depth > 3:
            return False
            
        if isinstance(output, dict):
            if "evidence" in output or "source" in output:
                return True
            for value in output.values():
                if self._has_evidence(value, depth + 1):
                    return True
        elif isinstance(output, list):
            for item in output[:5]:  # Check first 5 items
                if self._has_evidence(item, depth + 1):
                    return True
        
        return False
    
    def _extract_citations(
        self, 
        output: dict, 
        rag_chunks: Optional[list[dict]]
    ) -> list[dict]:
        """
        Extract citation markers from output and link to source chunks.
        """
        citations = []
        text_content = str(output)
        
        # Find all citation markers
        matches = self.CITATION_PATTERN.findall(text_content)
        unique_chunks = sorted(set(int(m) for m in matches))
        
        for chunk_idx in unique_chunks:
            citation = {
                "marker": f"[Chunk {chunk_idx}]",
                "chunk_index": chunk_idx,
                "source_preview": None
            }
            
            # Link to actual chunk if available
            if rag_chunks:
                for chunk in rag_chunks:
                    if chunk.get("chunk_index") == chunk_idx:
                        citation["source_preview"] = chunk.get("content", "")[:200]
                        citation["similarity_score"] = chunk.get("similarity_score", 0)
                        break
            
            citations.append(citation)
        
        return citations
    
    def _calculate_quality_metrics(
        self, 
        output: dict, 
        citations: list[dict]
    ) -> dict:
        """
        Calculate output quality metrics.
        """
        metrics = {
            "has_structure": isinstance(output, dict) and len(output) > 2,
            "citation_count": len(citations),
            "completeness": 0.0,
            "depth": "shallow"
        }
        
        # Completeness: percentage of expected fields present
        expected_fields = {
            "summary", "executive_summary", "key_points", "key_takeaways",
            "main_topics", "entities", "recommendations", "action_items"
        }
        
        if isinstance(output, dict):
            present = len(set(output.keys()) & expected_fields)
            metrics["completeness"] = round(present / len(expected_fields), 2)
            
            # Depth assessment
            total_items = sum(
                len(v) if isinstance(v, list) else 1 
                for v in output.values()
            )
            if total_items > 20:
                metrics["depth"] = "comprehensive"
            elif total_items > 10:
                metrics["depth"] = "moderate"
        
        return metrics
    
    def _format_content(self, output: dict) -> dict:
        """
        Format content for display.
        - Normalize numbers
        - Add display-friendly keys
        """
        if not isinstance(output, dict):
            return {"raw": output}

        # Normalize common list fields that models sometimes return as null
        list_fields = {
            "main_topics", "key_insights",
            "key_takeaways", "critical_numbers", "time_sensitive", "highlights",
            "action_items", "quick_wins", "next_steps", "risks", "decisions_required",
            "bugs", "security_issues", "refactoring_suggestions",
            "authors", "key_findings", "limitations",
            "parties", "obligations", "compliance_flags", "red_lines",
        }

        def normalize(obj):
            if isinstance(obj, dict):
                out = {}
                for k, v in obj.items():
                    if k in list_fields and v is None:
                        out[k] = []
                    else:
                        out[k] = normalize(v)
                return out
            if isinstance(obj, list):
                return [normalize(x) for x in obj]
            return obj

        output = normalize(output)
        
        formatted = {}
        
        for key, value in output.items():
            # Create display-friendly key
            display_key = key.replace("_", " ").title()
            
            if isinstance(value, str):
                # Format numbers in text
                formatted[key] = self._format_numbers(value)
            elif isinstance(value, list):
                formatted[key] = [
                    self._format_numbers(str(item)) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                formatted[key] = value
            
            # Add display key mapping
            formatted[f"_{key}_display"] = display_key
        
        return formatted
    
    def _format_numbers(self, text: str) -> str:
        """Format numbers for better readability."""
        # This is a simple implementation
        # In production, you'd use locale-aware formatting
        return text


def format_output(
    analysis_output: dict, 
    rag_chunks: Optional[list[dict]] = None
) -> FormattedOutput:
    """Convenience function to format output."""
    formatter = OutputFormatter()
    return formatter.format_analysis(analysis_output, rag_chunks)
