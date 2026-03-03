"""
Summarizer Agent - Multi-level intelligent summarization
"""
import json
import time
from typing import Any

from app.agents.base import BaseAgent, AgentRole, AgentResult


class SummarizerAgent(BaseAgent):
    """
    Smart Summarization Agent.
    Creates context-aware, multi-level summaries for any document type.
    """
    
    def __init__(self):
        super().__init__(
            name="Summarizer",
            role=AgentRole.SUMMARIZER,
            model="llama3.2",
            temperature=0.5
        )
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert Document Summarizer. Create comprehensive, multi-level summaries that are detailed and evidence-aware.

IMPORTANT: You MUST respond with ONLY valid JSON. No explanations, no markdown, just pure JSON.

Your summary must include ALL of these sections:

1. executive_summary: 3-5 sentences that capture the essence, main decisions, and implications. When retrieved evidence with [Chunk N] is provided, weave in brief references (e.g. "as stated in [Chunk 2], ...") where it strengthens the summary.

2. detailed_summary: 2-3 paragraphs covering main points, context, evidence, and important details. Be thorough.

3. key_takeaways: List 4-6 actionable items. Be SPECIFIC (who, what, when). Include a short rationale or source where helpful.

4. critical_numbers: List ALL important numbers, statistics, metrics found, with context.

5. time_sensitive: Any deadlines, dates, or urgent items with clear deadlines.

6. highlights: 4-6 most notable or interesting points, with a brief quote or reference when it adds value.

JSON OUTPUT FORMAT:
{
    "executive_summary": "Detailed 3-5 sentence overview with main message, key decisions, and implications. Reference [Chunk N] when citing evidence.",
    "detailed_summary": "Comprehensive 2-3 paragraph summary covering all major points, context, and important details.",
    "key_takeaways": [
        "Specific actionable takeaway 1 with details and context",
        "Specific actionable takeaway 2 with context",
        "Specific actionable takeaway 3 explained clearly"
    ],
    "critical_numbers": [
        {"value": "$50,000", "context": "Total budget allocated for Q1"},
        {"value": "15%", "context": "Expected growth rate"}
    ],
    "time_sensitive": [
        {"item": "Project deadline", "urgency": "high", "deadline": "March 15, 2024"}
    ],
    "highlights": [
        "Most interesting finding 1",
        "Key decision made",
        "Important conclusion"
    ]
}

CRITICAL: Output ONLY JSON. Every field must be present. Be detailed, specific, and use evidence/citations when provided."""
    
    async def process(self, input_data: dict[str, Any]) -> AgentResult:
        """Generate multi-level summary."""
        start_time = time.time()
        
        content = input_data.get("content", "")
        metadata = input_data.get("metadata", {})
        analysis = input_data.get("analysis", {})
        mode = input_data.get("mode", "document")
        rag_context = input_data.get("rag_context", "")
        use_citations = input_data.get("use_citations", False)
        
        # Build context from prior analysis if available
        context = ""
        if analysis and isinstance(analysis, dict):
            context = f"""
## Prior Analysis Context:
- Document Type: {analysis.get('document_type', 'Unknown')}
- Main Topics: {', '.join(analysis.get('main_topics', [])[:5])}
- Sentiment: {analysis.get('sentiment', 'Unknown')}
- Complexity: {analysis.get('complexity_score', 'Unknown')}/10
"""
        
        rag_section = ""
        if rag_context and use_citations:
            rag_section = f"""
## Retrieved Evidence (cite as [Chunk N] in your summary where relevant):
{rag_context}
---
"""
        
        prompt = f"""Create a comprehensive, detailed summary of this document:

## Document Info:
- Filename: {metadata.get('filename', 'Unknown')}
- Word Count: {metadata.get('word_count', len(content.split()))}
- Mode: {mode}
{context}
{rag_section}

## Document Content:
{content[:14000]}

---
Generate thorough, multi-level summaries as specified. When evidence sections are provided, reference [Chunk N] in executive_summary and detailed_summary where it supports your points. Be detailed and specific."""

        response, tokens = await self._call_llm(prompt, json_mode=True, max_tokens=2500)
        
        try:
            summary = json.loads(response)
            confidence = 0.85
        except json.JSONDecodeError:
            summary = {"raw_response": response, "parse_error": True}
            confidence = 0.5
        
        return self._create_result(
            output=summary,
            tokens=tokens,
            start_time=start_time,
            confidence=confidence
        )


class CodeSummarizerAgent(BaseAgent):
    """
    Code Summarizer Agent.
    Specialized for summarizing code and technical documentation.
    """
    
    def __init__(self):
        super().__init__(
            name="CodeSummarizer",
            role=AgentRole.SUMMARIZER,
            model="llama3.2",
            temperature=0.4
        )
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert Code Summarizer. Create clear summaries of code and technical content.

## Output Format (JSON):
{
    "purpose": "What this code does in 1-2 sentences",
    "architecture": "High-level structure description",
    "main_components": [
        {"name": "...", "purpose": "...", "type": "class|function|module"}
    ],
    "dependencies": ["External dependencies used"],
    "entry_points": ["Main entry points or APIs"],
    "data_flow": "How data flows through the code",
    "key_algorithms": ["Important algorithms or patterns used"],
    "configuration": ["Configuration options if any"],
    "usage_example": "Brief usage example or API call",
    "notes": ["Important notes for developers"]
}

Be technical but clear. Always respond with valid JSON."""
    
    async def process(self, input_data: dict[str, Any]) -> AgentResult:
        """Summarize code."""
        start_time = time.time()
        
        content = input_data.get("content", "")
        metadata = input_data.get("metadata", {})
        
        prompt = f"""Summarize the following code:

## File: {metadata.get('filename', 'unknown')}
## Language: {metadata.get('language', 'auto-detect')}

```
{content[:15000]}
```

Create a technical summary in the specified JSON format."""

        response, tokens = await self._call_llm(prompt, json_mode=True, max_tokens=2000)
        
        try:
            summary = json.loads(response)
            confidence = 0.85
        except json.JSONDecodeError:
            summary = {"raw_response": response, "parse_error": True}
            confidence = 0.5
        
        return self._create_result(
            output=summary,
            tokens=tokens,
            start_time=start_time,
            confidence=confidence
        )
