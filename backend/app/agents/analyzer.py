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
        return """You are an expert Document Analyzer. Analyze the document thoroughly and extract ALL useful information.

IMPORTANT: You MUST respond with ONLY valid JSON. No explanations, no markdown, just pure JSON.

Your analysis must include:
1. document_type: What type of document is this? (report, email, contract, memo, proposal, code, research, article, etc.)
2. main_topics: List 3-5 main subjects discussed (be specific, not generic)
3. key_entities: Extract ALL entities found:
   - people: Names of people mentioned
   - organizations: Company/org names
   - dates: Any dates mentioned
   - locations: Places mentioned
   - monetary_values: Any money amounts
4. sentiment: Overall tone (positive/negative/neutral/mixed)
5. complexity_score: 1-10 rating of document complexity
6. key_points: List 3-7 most important takeaways (be detailed and specific)
7. structure: Brief description of how document is organized
8. language: Primary language of document

JSON OUTPUT FORMAT:
{
    "document_type": "specific type here",
    "main_topics": ["Topic 1 - be specific", "Topic 2", "Topic 3"],
    "key_entities": {
        "people": ["Name1", "Name2"],
        "organizations": ["Org1", "Org2"],
        "dates": ["2024-01-15", "Q1 2024"],
        "locations": ["City, Country"],
        "monetary_values": ["$50,000", "€10M"]
    },
    "sentiment": "positive",
    "complexity_score": 6,
    "key_points": [
        "First important finding with details",
        "Second key point with specifics",
        "Third takeaway explained clearly"
    ],
    "structure": "Document has X sections covering Y topics",
    "language": "English"
}

CRITICAL: Output ONLY the JSON object, nothing else. Every field must be present."""
    
    async def process(self, input_data: dict[str, Any]) -> AgentResult:
        """Analyze the document."""
        start_time = time.time()
        
        content = input_data.get("content", "")
        metadata = input_data.get("metadata", {})
        mode = input_data.get("mode", "document")
        
        # Truncate for context limits
        content_preview = content[:12000]
        
        prompt = f"""Analyze the following document:

## Document Metadata:
- Filename: {metadata.get('filename', 'Unknown')}
- Type: {metadata.get('doc_type', 'Unknown')}
- Word Count: {metadata.get('word_count', len(content.split()))}
- Mode: {mode}

## Document Content:
{content_preview}

---
Provide comprehensive analysis in the specified JSON format."""

        response, tokens = await self._call_llm(prompt, json_mode=True)
        
        try:
            analysis = json.loads(response)
            confidence = 0.9
        except json.JSONDecodeError:
            analysis = {"raw_response": response, "parse_error": True}
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
        return """You are an expert Senior Software Architect and Code Auditor.

IMPORTANT: You MUST respond with ONLY valid JSON.

Your goal is to conduct a deep technical review of the code.

JSON OUTPUT STRUCTURE:
{
    "language": "python/javascript/etc",
    "quality_score": 8, // 0-10
    "summary": "Technical summary of what the code does",
    "architecture_analysis": "Assessment of structure, patterns used, and modularity",
    "bugs": [
        {"line": 10, "severity": "high/medium/low", "issue": "Description", "fix": "Suggested fix"}
    ],
    "security_issues": [
        {"severity": "critical/high", "type": "SQL Injection/XSS/etc", "description": "Details", "mitigation": "How to fix"}
    ],
    "refactoring_suggestions": [
        {"priority": "high", "suggestion": "Extract method X", "reason": "Function is too long (50+ lines)"}
    ],
    "complexity": "low/medium/high"
}

CRITICAL: Output ONLY JSON. Be highly technical and critical."""
    
    async def process(self, input_data: dict[str, Any]) -> AgentResult:
        """Analyze code."""
        start_time = time.time()
        
        content = input_data.get("content", "")
        metadata = input_data.get("metadata", {})
        
        prompt = f"""Review the following code:
        
## File Info:
- Filename: {metadata.get('filename', 'unknown')}
- Language: {metadata.get('language', 'auto-detect')}

## Code:
```
{content[:15000]}
```

Provide a thorough, architect-level code review in JSON."""

        response, tokens = await self._call_llm(prompt, json_mode=True, max_tokens=3000)
        
        try:
            analysis = json.loads(response)
            confidence = 0.85
        except json.JSONDecodeError:
            analysis = {"raw_response": response, "parse_error": True}
            confidence = 0.5
        
        return self._create_result(output=analysis, tokens=tokens, start_time=start_time, confidence=confidence)


class ResearchAnalyzerAgent(BaseAgent):
    """
    Research Paper Analyzer Agent.
    Specialized for academic papers and research documents.
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
        return """You are an expert Academic Researcher. Analyze the research paper with academic rigor.

IMPORTANT: You MUST respond with ONLY valid JSON.

JSON OUTPUT STRUCTURE:
{
    "title": "Paper Title",
    "authors": ["Author 1", "Author 2"],
    "research_question": "What is the core problem being solved?",
    "methodology": {
        "type": "Qualitative/Quantitative/Mixed",
        "details": "Description of methods used (e.g. Transformer architecture, Double-blind study)"
    },
    "key_findings": [
        "Finding 1 with statistical significance if available",
        "Finding 2"
    ],
    "novelty": "What is new/unique about this work?",
    "limitations": ["Limitation 1", "Limitation 2"],
    "implications": "Theoretical or practical impact of this work",
    "future_work": "Suggestions for future research mentioned"
}

CRITICAL: Output ONLY JSON. Focus on methodology and validity."""
    
    async def process(self, input_data: dict[str, Any]) -> AgentResult:
        """Analyze research paper."""
        start_time = time.time()
        
        content = input_data.get("content", "")
        metadata = input_data.get("metadata", {})
        
        prompt = f"""Analyze this research paper:

## Document Info:
- Filename: {metadata.get('filename', 'unknown')}

## Content:
{content[:14000]}

Extract academic insights in JSON format."""

        response, tokens = await self._call_llm(prompt, json_mode=True, max_tokens=2500)
        
        try:
            analysis = json.loads(response)
            confidence = 0.85
        except json.JSONDecodeError:
            analysis = {"raw_response": response, "parse_error": True}
            confidence = 0.5
        
        return self._create_result(output=analysis, tokens=tokens, start_time=start_time, confidence=confidence)


class LegalAnalyzerAgent(BaseAgent):
    """
    Legal Document Analyzer Agent.
    Specialized for contracts, regulations, and legal documents.
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
        return """You are an expert Legal Analyst. Analyze the legal document for risks, obligations, and compliance.

IMPORTANT: You MUST respond with ONLY valid JSON.

JSON OUTPUT STRUCTURE:
{
    "document_type": "NDA/SaaS Agreement/Employment Contract/etc",
    "parties": [{"name": "Party A", "role": "Provider"}, {"name": "Party B", "role": "Client"}],
    "key_dates": {"effective_date": "Date", "termination_date": "Date", "renewal": "Auto-renew?"},
    "obligations": [
        {"party": "Party A", "obligation": "Must deliver X by Y"}
    ],
    "rights": [
        {"party": "Party B", "right": "Right to audit"}
    ],
    "risk_analysis": [
        {"risk": "Unlimited Liability Clause", "severity": "CRITICAL", "explanation": "Clause 5 exposes Client to unlimited damages"}
    ],
    "red_flags": ["Missing termination clause", "Ambiguous payment terms"],
    "jurisdiction": "Governing Law (e.g. California, UK)",
    "compliance_check": "GDPR/CCPA mentioned?"
}

DISCLAIMER: This is AI analysis, not legal advice.

CRITICAL: Output ONLY JSON. Be extremely precise with legal terms."""
    
    async def process(self, input_data: dict[str, Any]) -> AgentResult:
        """Analyze legal document."""
        start_time = time.time()
        
        content = input_data.get("content", "")
        metadata = input_data.get("metadata", {})
        
        prompt = f"""Analyze the following legal document:

## Document Info:
- Filename: {metadata.get('filename', 'unknown')}

## Content:
{content[:14000]}

Extract legal-relevant information in the specified JSON format.
Remember: This is for analysis purposes only, not legal advice."""

        response, tokens = await self._call_llm(prompt, json_mode=True, max_tokens=3000)
        
        try:
            analysis = json.loads(response)
            analysis["disclaimer"] = "This analysis is not legal advice. Consult a qualified attorney."
            confidence = 0.8
        except json.JSONDecodeError:
            analysis = {"raw_response": response, "parse_error": True}
            confidence = 0.5
        
        return self._create_result(
            output=analysis,
            tokens=tokens,
            start_time=start_time,
            confidence=confidence
        )
