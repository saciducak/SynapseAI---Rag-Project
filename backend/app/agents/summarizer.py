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
        return """You are an expert Document Summarizer. Create comprehensive, multi-level summaries.

IMPORTANT: You MUST respond with ONLY valid JSON. No explanations, no markdown, just pure JSON.

Your summary must include ALL of these sections:

1. executive_summary: 2-3 sentences that capture the essence. A busy executive should understand the document from this alone.

2. detailed_summary: 1-2 paragraphs with main points, context, and important details.

3. key_takeaways: List 3-5 actionable items. Be SPECIFIC, not vague.
   - Wrong: "Improve communication"
   - Right: "Schedule weekly team sync meetings starting next Monday"

4. critical_numbers: List ALL important numbers, statistics, metrics found.

5. time_sensitive: Any deadlines, dates, or urgent items.

6. highlights: 3-5 most notable or interesting points.

JSON OUTPUT FORMAT:
{
    "executive_summary": "Detailed 2-3 sentence overview capturing the main message and purpose of the document.",
    "detailed_summary": "Comprehensive 1-2 paragraph summary covering all major points, decisions, and context.",
    "key_takeaways": [
        "Specific actionable takeaway 1 with details",
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

CRITICAL: Output ONLY JSON. Every field must be present. Be detailed and specific."""
    
    async def process(self, input_data: dict[str, Any]) -> AgentResult:
        """Generate multi-level summary."""
        start_time = time.time()
        
        content = input_data.get("content", "")
        metadata = input_data.get("metadata", {})
        analysis = input_data.get("analysis", {})
        mode = input_data.get("mode", "document")
        
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
        
        prompt = f"""Create a comprehensive summary of this document:

## Document Info:
- Filename: {metadata.get('filename', 'Unknown')}
- Word Count: {metadata.get('word_count', len(content.split()))}
- Mode: {mode}
{context}

## Document Content:
{content[:12000]}

---
Generate summaries at multiple levels as specified. Adapt your style based on the document type and mode."""

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
