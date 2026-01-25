"""
Analyzer Agent - Document analysis and information extraction
"""
import json
import time
from typing import Any

from app.agents.base import BaseAgent, AgentRole, AgentResult


class AnalyzerAgent(BaseAgent):
    """
    Document Analyzer Agent.
    Extracts key information, entities, themes, and structure.
    """
    
    def __init__(self):
        super().__init__(
            name="Analyzer",
            role=AgentRole.ANALYZER,
            model="llama3.2",
            temperature=0.3  # Lower for consistent analysis
        )
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert Document Analysis AI. Your goal is to extract high-quality, actionable intelligence from documents.
        
IMPORTANT: You MUST respond with ONLY valid JSON.
DO NOT use markdown formatting (like ```json), just output the raw JSON object.

Your analysis must be evidence-based and specific. Avoid generic statements.
If a field is not applicable, use null or empty list.

JSON OUTPUT STRUCTURE:
{
    "document_type": "Specific type (e.g., Q3 Financial Report, Python Backend Source Code, SaaS Service Agreement)",
    "main_topics": ["Specific Topic 1", "Specific Topic 2"],
    "summary_abstract": "A high-level executive summary (max 3 sentences)",
    "key_entities": {
        "people": ["Full Name (Role)"],
        "organizations": ["Org Name"],
        "dates": ["YYYY-MM-DD or Context"],
        "locations": ["City, Country"],
        "monetary_values": ["$10,000", "€5M"],
        "technical_terms": ["Term 1", "Term 2"]
    },
    "sentiment": {
        "score": 0.8, // -1.0 to 1.0
        "label": "positive/neutral/negative",
        "reasoning": "Brief explanation"
    },
    "complexity_score": 7, // 1-10 (1=Simple Memo, 10=Quantum Physics Paper)
    "key_insights": [
        {
            "insight": "The core finding or statement",
            "evidence": "Quote or reference from text",
            "importance": "high/medium/low"
        }
    ],
    "structure": "Description of document organization",
    "language": "English/Turkish/etc"
}

CRITICAL: Output ONLY valid JSON."""
    
    async def process(self, input_data: dict[str, Any]) -> AgentResult:
        """Analyze the document with optional RAG context."""
        start_time = time.time()
        
        content = input_data.get("content", "")
        metadata = input_data.get("metadata", {})
        mode = input_data.get("mode", "document")
        rag_context = input_data.get("rag_context", "")
        use_citations = input_data.get("use_citations", False)
        
        # INCREASED CONTEXT WINDOW: 12k -> 30k
        content_preview = content[:30000]
        
        # Build prompt with RAG context if available
        rag_section = ""
        if rag_context:
            rag_section = f"""
## Retrieved Context (Use for Evidence)
The following sections were identified as most relevant. Cite them using [Chunk X] markers.

{rag_context}
---
"""
        
        citation_instruction = ""
        if use_citations:
            citation_instruction = """
IMPORTANT: For each key finding, include an "evidence" field with a direct quote AND a "source" field with the chunk number like "[Chunk 3]".
"""
        
        prompt = f"""Analyze this document thoroughly:
        
## Metadata
- Filename: {metadata.get('filename', 'Unknown')}
- Type: {metadata.get('doc_type', 'Unknown')}
- Word Count: {metadata.get('word_count', len(content.split()))}
- Mode: {mode}
{rag_section}
## Content Snippet
{content_preview[:20000]}
{citation_instruction}
---
Perform a deep spectrum analysis and output the requested JSON."""

        response, tokens = await self._call_llm(prompt, json_mode=True)
        
        try:
            analysis = json.loads(response)
            confidence = 0.95
        except json.JSONDecodeError:
            logger.warning("JSON Decode Error in Analyzer")
            analysis = {"raw_response": response, "parse_error": True, "summary": "Analysis completed but JSON was malformed."}
            confidence = 0.5
        
        return self._create_result(
            output=analysis,
            tokens=tokens,
            start_time=start_time,
            confidence=confidence
        )


class CodeAnalyzerAgent(BaseAgent):
    """
    Code Analyzer Agent.
    Specialized for code review and analysis.
    """
    
    def __init__(self):
        super().__init__(
            name="CodeAnalyzer",
            role=AgentRole.CODE_REVIEWER,
            model="llama3.2",
            temperature=0.2
        )
    
    @property
    def system_prompt(self) -> str:
        return """You are a Principal Software Architect and Security Expert.
        
Your goal is to perform a rigorous code review. Focus on:
1. Security Vulnerabilities (OWASP Top 10)
2. Performance Bottlenecks
3. Architectural Flaws (Anti-patterns)
4. Clean Code Principles (SOLID, DRY)

IMPORTANT: You MUST respond with ONLY valid JSON.

JSON OUTPUT STRUCTURE:
{
    "language": "python/javascript/etc",
    "quality_score": 8, // 0-10
    "summary": "Technical summary of the module's purpose",
    "architecture_analysis": "Assessment of design patterns and structure",
    "bugs": [
        {
            "line": 10, 
            "severity": "high/medium/low", 
            "issue": "Concise issue description", 
            "fix": "Specific code fix or suggestion"
        }
    ],
    "security_issues": [
        {
            "severity": "critical/high", 
            "type": "SQL Injection/XSS/Command Injection", 
            "description": "How the exploit works here", 
            "mitigation": "Secure implementation details"
        }
    ],
    "refactoring_suggestions": [
        {
            "priority": "high", 
            "suggestion": "Extract Function / Rename Variable", 
            "reason": "Cyclomatic complexity is too high"
        }
    ],
    "complexity": "low/medium/high"
}"""
    
    async def process(self, input_data: dict[str, Any]) -> AgentResult:
        """Analyze code."""
        start_time = time.time()
        
        content = input_data.get("content", "")
        metadata = input_data.get("metadata", {})
        
        # 30k context for code is crucial
        prompt = f"""Review the following code file:
        
## File Info:
- Filename: {metadata.get('filename', 'unknown')}
- Language: {metadata.get('language', 'auto-detect')}

## Code Content:
```
{content[:30000]}
```

Provide a strict, senior-level architectural review in JSON."""

        response, tokens = await self._call_llm(prompt, json_mode=True, max_tokens=3000)
        
        try:
            analysis = json.loads(response)
            confidence = 0.9
        except json.JSONDecodeError:
            analysis = {"raw_response": response, "parse_error": True}
            confidence = 0.5
        
        return self._create_result(output=analysis, tokens=tokens, start_time=start_time, confidence=confidence)


class ResearchAnalyzerAgent(BaseAgent):
    """
    Research Paper Analyzer Agent.
    """
    
    def __init__(self):
        super().__init__(
            name="ResearchAnalyzer",
            role=AgentRole.RESEARCHER,
            model="llama3.2",
            temperature=0.3
        )
    
    @property
    def system_prompt(self) -> str:
        return """You are a Distinguished Academic Reviewer. Analyze the research paper for scientific validity and novelty.

IMPORTANT: You MUST respond with ONLY valid JSON.

JSON OUTPUT STRUCTURE:
{
    "title": "Paper Title",
    "authors": ["Author List"],
    "research_question": "The core hypothesis or problem",
    "methodology": {
        "type": "Qualitative/Quantitative/Mixed",
        "description": "Detailed study design",
        "sample_size": "N=..."
    },
    "key_findings": [
        "Finding 1 (Statistically Significant?)",
        "Finding 2"
    ],
    "novelty_score": 8, // 1-10
    "limitations": ["Limitation A", "Limitation B"],
    "implications": "Real-world or theoretical impact",
    "future_work": "Suggested next steps"
}"""
    
    async def process(self, input_data: dict[str, Any]) -> AgentResult:
        start_time = time.time()
        content = input_data.get("content", "")[:30000] # Increased context
        metadata = input_data.get("metadata", {})
        
        prompt = f"""Analyze research paper: {metadata.get('filename', 'unknown')}
        
{content}
        
Extract academic insights in strict JSON."""

        response, tokens = await self._call_llm(prompt, json_mode=True, max_tokens=2500)
        try:
            analysis = json.loads(response)
            confidence = 0.9
        except json.JSONDecodeError:
            analysis = {"raw_response": response, "parse_error": True}
            confidence = 0.5
        return self._create_result(output=analysis, tokens=tokens, start_time=start_time, confidence=confidence)


class LegalAnalyzerAgent(BaseAgent):
    """
    Legal Document Analyzer Agent.
    """
    
    def __init__(self):
        super().__init__(
            name="LegalAnalyzer",
            role=AgentRole.LEGAL_EXPERT,
            model="llama3.2",
            temperature=0.2
        )
    
    @property
    def system_prompt(self) -> str:
        return """You are a Senior Legal Counsel. Analyze the document for risk, compliance, and binding obligations.

IMPORTANT: You MUST respond with ONLY valid JSON.

JSON OUTPUT STRUCTURE:
{
    "document_type": "NDA/MSA/SLA/Lease/etc",
    "parties": [{"name": "Party A", "role": "Provider/Clent"}],
    "term_dates": {"effective": "Date", "expiration": "Date", "renewal": "Terms"},
    "obligations": [
        {"party": "Name", "description": "Must do X", "deadline": "if applicable"}
    ],
    "risk_assessment": {
        "score": "High/Medium/Low",
        "critical_risks": [
            {"clause": "Indemnification", "risk": "Uncapped liability", "severity": "CRITICAL"}
        ]
    },
    "compliance_flags": ["GDPR", "HIPAA", "Governing Law: NY"],
    "red_lines": ["List of clauses that typically require negotiation"]
}"""
    
    async def process(self, input_data: dict[str, Any]) -> AgentResult:
        start_time = time.time()
        content = input_data.get("content", "")[:30000] # Increased context
        
        prompt = f"""Legal analysis required for:
        
{content}
        
Output strict JSON legal assessment."""

        response, tokens = await self._call_llm(prompt, json_mode=True, max_tokens=3000)
        try:
            analysis = json.loads(response)
            analysis["_disclaimer"] = "AI Analysis - Does not constitute legal advice."
            confidence = 0.9
        except json.JSONDecodeError:
            analysis = {"raw_response": response, "parse_error": True}
            confidence = 0.5
        return self._create_result(output=analysis, tokens=tokens, start_time=start_time, confidence=confidence)
