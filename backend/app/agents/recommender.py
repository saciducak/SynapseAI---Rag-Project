"""
Recommender Agent - Action recommendations and decision support
"""
import json
import time
from typing import Any

from app.agents.base import BaseAgent, AgentRole, AgentResult


class ActionRecommenderAgent(BaseAgent):
    """
    Action Recommendation Agent.
    Extracts actionable items and provides decision support.
    """
    
    def __init__(self):
        super().__init__(
            name="ActionRecommender",
            role=AgentRole.RECOMMENDER,
            model="llama3.2",
            temperature=0.4
        )
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert Action Recommender. Extract actionable items and provide clear, evidence-based recommendations tied to the document content.

IMPORTANT: You MUST respond with ONLY valid JSON. No explanations, no markdown, just pure JSON.

Base your recommendations on the document's stated goals, explicit content, and the analysis/summary provided. Reference specific sections or findings where possible (e.g. "Per the Q1 budget section..." or "Given the risks identified in the analysis...").

Your recommendations must include:

1. action_items: List 4-6 specific actions. Each must have:
   - action: SPECIFIC description (WHO does WHAT by WHEN; reference document content when relevant)
   - priority: high/medium/low
   - category: administrative/technical/strategic/communication

2. quick_wins: 2-4 easy actions that can be done immediately with minimal effort.

3. next_steps: Ordered list of the first 3-4 concrete steps to take right now, with brief rationale.

4. risks: Any potential problems or blockers identified, with mitigation where possible.

5. decisions_required: Decisions that need to be made, with options and a clear recommendation + rationale.

JSON OUTPUT FORMAT:
{
    "action_items": [
        {
            "action": "Schedule a meeting with the finance team to review Q1 budget by Friday",
            "priority": "high",
            "category": "administrative"
        },
        {
            "action": "Update the project timeline document with new milestones",
            "priority": "medium",
            "category": "technical"
        }
    ],
    "quick_wins": [
        "Send email summary to stakeholders today",
        "Create shared folder for project documents"
    ],
    "next_steps": [
        "1. Review the attached budget proposal",
        "2. Schedule kickoff meeting",
        "3. Assign team leads for each workstream"
    ],
    "risks": [
        {
            "risk": "Timeline may slip if resources are not allocated by next week",
            "probability": "medium",
            "impact": "high",
            "mitigation": "Escalate resource request to management"
        }
    ],
    "decisions_required": [
        {
            "decision": "Which vendor to select for the project",
            "options": ["Vendor A - cheaper", "Vendor B - better quality"],
            "recommendation": "Vendor B",
            "rationale": "Quality is more important for this project"
        }
    ]
}

CRITICAL: Output ONLY JSON. Be SPECIFIC and ACTIONABLE. No vague recommendations."""
    
    async def process(self, input_data: dict[str, Any]) -> AgentResult:
        """Generate action recommendations."""
        start_time = time.time()
        
        content = input_data.get("content", "")
        metadata = input_data.get("metadata", {})
        analysis = input_data.get("analysis", {})
        summary = input_data.get("summary", {})
        user_context = input_data.get("user_context", "")
        mode = input_data.get("mode", "document")
        
        # Build context
        context_parts = []
        
        if analysis and isinstance(analysis, dict):
            context_parts.append(f"""
## Document Analysis:
- Type: {analysis.get('document_type', 'Unknown')}
- Topics: {', '.join(analysis.get('main_topics', [])[:5])}
- Key Points: {json.dumps(analysis.get('key_points', [])[:5], ensure_ascii=False)}
""")
        
        if summary and isinstance(summary, dict):
            context_parts.append(f"""
## Summary:
{summary.get('executive_summary', '')}

Key Takeaways: {json.dumps(summary.get('key_takeaways', [])[:5], ensure_ascii=False)}
""")
        
        if user_context:
            context_parts.append(f"""
## User Context:
{user_context}
""")
        
        context = "\n".join(context_parts)
        
        prompt = f"""Based on the following document and analysis, provide detailed, actionable recommendations tied to the content.

## Document Info:
- Filename: {metadata.get('filename', 'Unknown')}
- Mode: {mode}

{context}

## Original Document (use this to ground your recommendations in specific statements):
{content[:10000]}

---
Generate comprehensive action recommendations and decision support in the specified JSON format. Be specific and reference the document's goals, numbers, and deadlines where relevant."""

        response, tokens = await self._call_llm(prompt, json_mode=True, max_tokens=3000)
        
        try:
            recommendations = json.loads(response)
            confidence = 0.8
        except json.JSONDecodeError:
            recommendations = {"raw_response": response, "parse_error": True}
            confidence = 0.5
        
        return self._create_result(
            output=recommendations,
            tokens=tokens,
            start_time=start_time,
            confidence=confidence
        )


class CodeRecommenderAgent(BaseAgent):
    """
    Code Improvement Recommender.
    Provides specific code improvement suggestions.
    """
    
    def __init__(self):
        super().__init__(
            name="CodeRecommender",
            role=AgentRole.RECOMMENDER,
            model="llama3.2",
            temperature=0.3
        )
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert Code Improvement Advisor. Provide specific, actionable code improvements.

## Output Format (JSON):
{
    "improvements": [
        {
            "type": "refactor|optimization|security|readability|testing",
            "priority": "high|medium|low",
            "current_issue": "What's wrong or could be better",
            "suggestion": "Specific improvement suggestion",
            "code_example": "Example code if applicable",
            "effort": "quick_fix|moderate|significant"
        }
    ],
    "architecture_suggestions": [
        {
            "area": "Area of improvement",
            "suggestion": "What to change",
            "benefit": "Expected benefit"
        }
    ],
    "testing_recommendations": [
        "Test case suggestions"
    ],
    "documentation_needs": [
        "What should be documented"
    ],
    "tech_debt": [
        {"item": "Tech debt item", "priority": "high|medium|low"}
    ],
    "estimated_improvement": "Overall improvement estimate after changes"
}

Be specific with code examples where helpful. Always respond with valid JSON."""
    
    async def process(self, input_data: dict[str, Any]) -> AgentResult:
        """Generate code improvement recommendations."""
        start_time = time.time()
        
        content = input_data.get("content", "")
        metadata = input_data.get("metadata", {})
        analysis = input_data.get("analysis", {})
        
        # Build context from code analysis
        context = ""
        if analysis and isinstance(analysis, dict):
            context = f"""
## Code Analysis Results:
- Quality Score: {analysis.get('quality_score', 'Unknown')}/10
- Bugs Found: {len(analysis.get('bugs', []))}
- Security Issues: {len(analysis.get('security_issues', []))}
- Complexity: {analysis.get('complexity', 'Unknown')}
"""
        
        prompt = f"""Based on the following code and analysis, provide improvement recommendations:

## File: {metadata.get('filename', 'unknown')}
{context}

## Code:
```
{content[:12000]}
```

Provide specific, actionable improvements in the specified JSON format."""

        response, tokens = await self._call_llm(prompt, json_mode=True, max_tokens=3000)
        
        try:
            recommendations = json.loads(response)
            confidence = 0.8
        except json.JSONDecodeError:
            recommendations = {"raw_response": response, "parse_error": True}
            confidence = 0.5
        
        return self._create_result(
            output=recommendations,
            tokens=tokens,
            start_time=start_time,
            confidence=confidence
        )
