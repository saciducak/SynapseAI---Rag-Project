"""
Analyzer Agent - Document analysis and information extraction
"""
import json
import time
from typing import Any

from app.agents.base import BaseAgent, AgentRole, AgentResult
from loguru import logger


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
        return """You are an expert Document Analysis AI. Your goal is to extract high-quality, actionable intelligence from documents with clear evidence.

IMPORTANT: You MUST respond with ONLY valid JSON.
DO NOT use markdown formatting (like ```json), just output the raw JSON object.

Your analysis must be EVIDENCE-BASED: for every key finding, include a short direct quote or reference from the text.
When "Retrieved Context" with [Chunk N] sections is provided, prefer citing those chunks (e.g. "evidence": "[Chunk 2] ...").
Be specific and detailed; avoid generic statements. If a field is not applicable, use null or empty list.
Do NOT invent facts, dates, numbers, obligations, risks, or entities. If not explicitly present, leave the relevant fields empty.
Avoid repetitive phrasing; each key_insight should be distinct.

JSON OUTPUT STRUCTURE:
{
    "document_type": "Specific type (e.g., Q3 Financial Report, Python Backend Source Code, SaaS Service Agreement)",
    "main_topics": ["Specific Topic 1", "Specific Topic 2"],
    "summary_abstract": "A detailed 2-4 sentence executive summary that captures the main message, key decisions, and implications.",
    "key_entities": {
        "people": ["Full Name (Role)"],
        "organizations": ["Org Name"],
        "dates": ["YYYY-MM-DD or Context"],
        "locations": ["City, Country"],
        "monetary_values": ["$10,000", "€5M"],
        "technical_terms": ["Term 1", "Term 2"]
    },
    "sentiment": {
        "score": 0.8,
        "label": "positive/neutral/negative",
        "reasoning": "Brief explanation with a short quote if relevant"
    },
    "complexity_score": 7,
    "key_insights": [
        {
            "insight": "The core finding or statement in one clear sentence",
            "evidence": "Direct quote or [Chunk N] reference from the document",
            "importance": "high/medium/low"
        }
    ],
    "structure": "Description of document organization (sections, flow, key transitions)",
    "language": "English/Turkish/etc"
}

CRITICAL: Output ONLY valid JSON. Every key_insight must have an evidence field with a quote or chunk reference."""
    
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
        if use_citations and rag_context:
            citation_instruction = """
IMPORTANT: Use the "Retrieved Context" sections above as your primary evidence. For each key_insight, set "evidence" to a direct quote from the text and include the chunk marker, e.g. "[Chunk 2] ...quote...". Be thorough and cite multiple chunks where relevant.
"""
        
        prompt = f"""Analyze this document thoroughly and in detail:
        
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
            logger.warning("JSON Decode Error in Analyzer; attempting one repair pass")
            # One repair pass using the same schema constraints
            repair_prompt = f"""The previous output was not valid JSON.\n\nReturn ONLY valid JSON matching the required schema. Do not add new keys.\nIf you cannot fill a field from the document, use null/empty list.\n\nInvalid output:\n{response}"""
            repaired, repair_tokens = await self._call_llm(repair_prompt, json_mode=True, max_tokens=1200)
            tokens += repair_tokens
            try:
                analysis = json.loads(repaired)
                confidence = 0.85
            except json.JSONDecodeError:
                analysis = {"raw_response": response, "parse_error": True, "summary": "Analysis completed but JSON was malformed."}
                confidence = 0.5

        # Schema sanity check (reduce "valid JSON but wrong shape")
        required = {"document_type", "main_topics", "summary_abstract", "key_entities", "sentiment", "complexity_score", "key_insights", "structure", "language"}
        if isinstance(analysis, dict) and not analysis.get("parse_error"):
            if not required.issubset(set(analysis.keys())):
                logger.warning("Analyzer schema mismatch; attempting one schema repair pass")
                schema_repair = f"""Return ONLY valid JSON with EXACTLY the required keys:\n{sorted(required)}\n\nDo NOT invent facts; if unknown use null/empty list.\n\nPrevious JSON (wrong shape):\n{json.dumps(analysis, ensure_ascii=False)}"""
                repaired, repair_tokens = await self._call_llm(schema_repair, json_mode=True, max_tokens=1200)
                tokens += repair_tokens
                try:
                    analysis2 = json.loads(repaired)
                    if isinstance(analysis2, dict) and required.issubset(set(analysis2.keys())):
                        analysis = analysis2
                        confidence = min(confidence, 0.85)
                except json.JSONDecodeError:
                    pass
        
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

        # Add stable line numbers to reduce hallucinated locations
        lines = content.splitlines()
        numbered = "\n".join([f"L{idx+1:04d}: {ln}" for idx, ln in enumerate(lines[:2000])])
        
        # 30k context for code is crucial
        prompt = f"""Review the following code file:
        
## File Info:
- Filename: {metadata.get('filename', 'unknown')}
- Language: {metadata.get('language', 'auto-detect')}

## Code Content:
```
{numbered[:30000]}
```

Provide a strict, senior-level architectural review in JSON.
CRITICAL: When reporting issues, use the provided L#### line numbers. If you are not sure, omit the line (use null)."""

        response, tokens = await self._call_llm(prompt, json_mode=True, max_tokens=3000)
        
        try:
            analysis = json.loads(response)
            confidence = 0.9
        except json.JSONDecodeError:
            logger.warning("JSON Decode Error in CodeAnalyzer; attempting one repair pass")
            repair_prompt = f"""The previous output was not valid JSON.\n\nReturn ONLY valid JSON matching the required schema.\n- Use L#### line numbers from the input.\n- Do NOT invent vulnerabilities; if unsure, omit.\n\nInvalid output:\n{response}"""
            repaired, repair_tokens = await self._call_llm(repair_prompt, json_mode=True, max_tokens=1500)
            tokens += repair_tokens
            try:
                analysis = json.loads(repaired)
                confidence = 0.8
            except json.JSONDecodeError:
                analysis = {"raw_response": response, "parse_error": True}
                confidence = 0.5

        required = {"language", "quality_score", "summary", "architecture_analysis", "bugs", "security_issues", "refactoring_suggestions", "complexity"}
        if isinstance(analysis, dict) and not analysis.get("parse_error"):
            if not required.issubset(set(analysis.keys())):
                repair_prompt = f"""Return ONLY valid JSON with EXACTLY these keys:\n{sorted(required)}\n\nRules:\n- Use L#### line numbers from input, or null.\n- Do NOT invent vulnerabilities.\n\nPrevious JSON (wrong shape):\n{json.dumps(analysis, ensure_ascii=False)}"""
                repaired, repair_tokens = await self._call_llm(repair_prompt, json_mode=True, max_tokens=1500)
                tokens += repair_tokens
                try:
                    analysis2 = json.loads(repaired)
                    if isinstance(analysis2, dict) and required.issubset(set(analysis2.keys())):
                        analysis = analysis2
                        confidence = min(confidence, 0.8)
                except json.JSONDecodeError:
                    pass
        
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
DO NOT invent authors, datasets, metrics, or results. If not explicitly present, use null/empty list.
When retrieved context with [Chunk N] is provided, cite it in free-text fields (e.g. "According to [Chunk 3]...").

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
        rag_context = input_data.get("rag_context", "")
        use_citations = input_data.get("use_citations", False)

        rag_section = ""
        if rag_context and use_citations:
            rag_section = f"""
## Retrieved Evidence (cite as [Chunk N]):
{rag_context}
---
"""
        
        prompt = f"""Analyze research paper: {metadata.get('filename', 'unknown')}

{rag_section}
        
{content}
        
Extract academic insights in strict JSON."""

        response, tokens = await self._call_llm(prompt, json_mode=True, max_tokens=2500)
        try:
            analysis = json.loads(response)
            confidence = 0.9
        except json.JSONDecodeError:
            repair_prompt = f"""The previous output was not valid JSON.\n\nReturn ONLY valid JSON matching the required schema. Do not add new keys.\nIf unsupported by the paper/context, use null/empty.\n\nInvalid output:\n{response}"""
            repaired, repair_tokens = await self._call_llm(repair_prompt, json_mode=True, max_tokens=1200)
            tokens += repair_tokens
            try:
                analysis = json.loads(repaired)
                confidence = 0.75
            except json.JSONDecodeError:
                analysis = {"raw_response": response, "parse_error": True}
                confidence = 0.5

        required = {"title", "authors", "research_question", "methodology", "key_findings", "novelty_score", "limitations", "implications", "future_work"}
        if isinstance(analysis, dict) and not analysis.get("parse_error"):
            if not required.issubset(set(analysis.keys())):
                repair_prompt = f"""Return ONLY valid JSON with EXACTLY these keys:\n{sorted(required)}\n\nDo NOT invent missing info. Use null/empty lists.\n\nPrevious JSON (wrong shape):\n{json.dumps(analysis, ensure_ascii=False)}"""
                repaired, repair_tokens = await self._call_llm(repair_prompt, json_mode=True, max_tokens=1500)
                tokens += repair_tokens
                try:
                    analysis2 = json.loads(repaired)
                    if isinstance(analysis2, dict) and required.issubset(set(analysis2.keys())):
                        analysis = analysis2
                        confidence = min(confidence, 0.8)
                except json.JSONDecodeError:
                    pass
            # Hard fallback if still wrong shape
            if not required.issubset(set(analysis.keys())):
                analysis = {
                    "title": None,
                    "authors": [],
                    "research_question": None,
                    "methodology": {"type": None, "description": None, "sample_size": None},
                    "key_findings": [],
                    "novelty_score": 0,
                    "limitations": [],
                    "implications": None,
                    "future_work": None,
                    "_note": "Could not extract a research-paper structure from the document; returning empty fields."
                }
                confidence = 0.4
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
DO NOT invent clauses, parties, dates, or obligations. If not explicitly present, use null/empty list.
When retrieved context with [Chunk N] is provided, cite it in descriptive fields where it supports your claims (e.g. \"[Chunk 5] Termination...\"), and prefer using clause text from the retrieved chunks.

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
        rag_context = input_data.get("rag_context", "")
        use_citations = input_data.get("use_citations", False)

        rag_section = ""
        if rag_context and use_citations:
            rag_section = f"""
## Retrieved Evidence (cite as [Chunk N] in clause/risk descriptions):
{rag_context}
---
"""
        
        prompt = f"""Legal analysis required for:

{rag_section}
        
{content}
        
Output strict JSON legal assessment."""

        response, tokens = await self._call_llm(prompt, json_mode=True, max_tokens=3000)
        try:
            analysis = json.loads(response)
            analysis["_disclaimer"] = "AI Analysis - Does not constitute legal advice."
            confidence = 0.9
        except json.JSONDecodeError:
            repair_prompt = f"""The previous output was not valid JSON.\n\nReturn ONLY valid JSON matching the required schema. Do not add new keys.\nDo NOT invent legal terms; if unsupported, use null/empty.\n\nInvalid output:\n{response}"""
            repaired, repair_tokens = await self._call_llm(repair_prompt, json_mode=True, max_tokens=1500)
            tokens += repair_tokens
            try:
                analysis = json.loads(repaired)
                analysis["_disclaimer"] = "AI Analysis - Does not constitute legal advice."
                confidence = 0.75
            except json.JSONDecodeError:
                analysis = {"raw_response": response, "parse_error": True}
                confidence = 0.5

        required = {"document_type", "parties", "term_dates", "obligations", "risk_assessment", "compliance_flags", "red_lines"}
        if isinstance(analysis, dict) and not analysis.get("parse_error"):
            if not required.issubset(set(analysis.keys())):
                repair_prompt = f"""Return ONLY valid JSON with EXACTLY these keys:\n{sorted(required)}\n\nRules:\n- Do NOT invent parties/clauses/dates.\n- If this is not a legal agreement, set document_type to something like \"not_a_contract\" and keep parties/obligations empty.\n\nPrevious JSON (wrong shape):\n{json.dumps(analysis, ensure_ascii=False)}"""
                repaired, repair_tokens = await self._call_llm(repair_prompt, json_mode=True, max_tokens=1800)
                tokens += repair_tokens
                try:
                    analysis2 = json.loads(repaired)
                    if isinstance(analysis2, dict) and required.issubset(set(analysis2.keys())):
                        analysis = analysis2
                        analysis["_disclaimer"] = "AI Analysis - Does not constitute legal advice."
                        confidence = min(confidence, 0.8)
                except json.JSONDecodeError:
                    pass
            # Hard fallback if still wrong shape
            if not required.issubset(set(analysis.keys())):
                analysis = {
                    "document_type": "not_a_contract",
                    "parties": [],
                    "term_dates": {"effective": None, "expiration": None, "renewal": None},
                    "obligations": [],
                    "risk_assessment": {"score": "Low", "critical_risks": []},
                    "compliance_flags": [],
                    "red_lines": [],
                    "_note": "The document does not appear to contain a recognizable legal agreement structure; returning empty fields."
                }
                analysis["_disclaimer"] = "AI Analysis - Does not constitute legal advice."
                confidence = 0.4
        return self._create_result(output=analysis, tokens=tokens, start_time=start_time, confidence=confidence)
