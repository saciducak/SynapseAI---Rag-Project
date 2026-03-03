"""
Multi-Agent Coordinator - Orchestrates agent workflows with RAG integration
"""
import asyncio
import time
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from app.agents.base import AgentResult
from app.agents.analyzer import (
    AnalyzerAgent, 
    CodeAnalyzerAgent, 
    ResearchAnalyzerAgent,
    LegalAnalyzerAgent
)
from app.agents.summarizer import SummarizerAgent, CodeSummarizerAgent
from app.agents.recommender import ActionRecommenderAgent, CodeRecommenderAgent
from app.models.schemas import AnalysisMode
from app.services.vector import get_vector_service


class WorkflowType(str, Enum):
    """Available workflow types."""
    FULL = "full"
    QUICK = "quick"
    ANALYSIS_ONLY = "analysis_only"
    SUMMARY_ONLY = "summary_only"


@dataclass
class CoordinationResult:
    """Result from multi-agent coordination."""
    workflow_id: str
    mode: AnalysisMode
    workflow_type: WorkflowType
    started_at: datetime
    completed_at: datetime
    total_tokens: int
    total_time_ms: int
    results: dict[str, AgentResult]
    final_output: dict
    success: bool
    errors: list[str] = field(default_factory=list)


class MultiAgentCoordinator:
    """
    Orchestrates multiple agents in coordinated workflows.
    Routes to appropriate agents based on mode.
    """
    
    def __init__(self):
        # Document mode agents
        self.doc_analyzer = AnalyzerAgent()
        self.doc_summarizer = SummarizerAgent()
        self.doc_recommender = ActionRecommenderAgent()
        
        # Code mode agents
        self.code_analyzer = CodeAnalyzerAgent()
        self.code_summarizer = CodeSummarizerAgent()
        self.code_recommender = CodeRecommenderAgent()
        
        # Research mode agents
        self.research_analyzer = ResearchAnalyzerAgent()
        
        # Legal mode agents
        self.legal_analyzer = LegalAnalyzerAgent()
        
        logger.info("MultiAgentCoordinator initialized with all agents")
    
    def _get_agents_for_mode(self, mode: AnalysisMode) -> dict:
        """Get the appropriate agents for the given mode."""
        if mode == AnalysisMode.CODE:
            return {
                "analyzer": self.code_analyzer,
                "summarizer": self.code_summarizer,
                "recommender": self.code_recommender
            }
        elif mode == AnalysisMode.RESEARCH:
            return {
                "analyzer": self.research_analyzer,
                "summarizer": self.doc_summarizer,
                "recommender": self.doc_recommender
            }
        elif mode == AnalysisMode.LEGAL:
            return {
                "analyzer": self.legal_analyzer,
                "summarizer": self.doc_summarizer,
                "recommender": self.doc_recommender
            }
        else:  # DOCUMENT mode (default)
            return {
                "analyzer": self.doc_analyzer,
                "summarizer": self.doc_summarizer,
                "recommender": self.doc_recommender
            }
    
    async def execute_workflow(
        self,
        mode: AnalysisMode,
        workflow_type: WorkflowType,
        content: str,
        metadata: dict,
        user_context: Optional[str] = None
    ) -> CoordinationResult:
        """
        Execute a workflow with the appropriate agents.
        
        Sequential flow: Analyzer → Summarizer → Recommender
        Each agent's output feeds into the next.
        """
        workflow_id = f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        started_at = datetime.now()
        results: dict[str, AgentResult] = {}
        errors: list[str] = []
        total_tokens = 0
        
        # Get agents for this mode
        agents = self._get_agents_for_mode(mode)
        
        # Shared context that grows
        shared_context = {
            "content": content,
            "metadata": metadata,
            "mode": mode.value,
            "user_context": user_context
        }
        
        try:
            # Step 1: Analysis (always runs)
            logger.info(f"[{workflow_id}] Running analyzer...")
            analyzer_result = await agents["analyzer"].process(shared_context)
            results["analyzer"] = analyzer_result
            total_tokens += analyzer_result.tokens_used
            
            # Add analysis to shared context
            if isinstance(analyzer_result.output, dict):
                shared_context["analysis"] = analyzer_result.output
            
            # Step 2: Summarization (if full or summary-only workflow)
            if workflow_type in [WorkflowType.FULL, WorkflowType.SUMMARY_ONLY, WorkflowType.QUICK]:
                logger.info(f"[{workflow_id}] Running summarizer...")
                summarizer_result = await agents["summarizer"].process(shared_context)
                results["summarizer"] = summarizer_result
                total_tokens += summarizer_result.tokens_used
                
                if isinstance(summarizer_result.output, dict):
                    shared_context["summary"] = summarizer_result.output
            
            # Step 3: Recommendations (if full workflow)
            if workflow_type == WorkflowType.FULL:
                logger.info(f"[{workflow_id}] Running recommender...")
                recommender_result = await agents["recommender"].process(shared_context)
                results["recommender"] = recommender_result
                total_tokens += recommender_result.tokens_used
            
        except Exception as e:
            logger.error(f"[{workflow_id}] Workflow error: {e}")
            errors.append(str(e))
        
        completed_at = datetime.now()
        total_time = int((completed_at - started_at).total_seconds() * 1000)
        
        # Compile final output
        final_output = self._compile_output(results, mode)
        
        return CoordinationResult(
            workflow_id=workflow_id,
            mode=mode,
            workflow_type=workflow_type,
            started_at=started_at,
            completed_at=completed_at,
            total_tokens=total_tokens,
            total_time_ms=total_time,
            results=results,
            final_output=final_output,
            success=len(errors) == 0,
            errors=errors
        )
    
    async def execute_parallel(
        self,
        mode: AnalysisMode,
        content: str,
        metadata: dict
    ) -> CoordinationResult:
        """
        Execute agents with parallel optimization.
        Analyzer runs first, then Summarizer + Recommender in parallel.
        """
        workflow_id = f"wf_parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        started_at = datetime.now()
        
        agents = self._get_agents_for_mode(mode)
        
        shared_context = {
            "content": content,
            "metadata": metadata,
            "mode": mode.value
        }
        
        # Run analyzer first (others depend on it)
        logger.info(f"[{workflow_id}] Running analyzer...")
        analyzer_result = await agents["analyzer"].process(shared_context)
        
        if isinstance(analyzer_result.output, dict):
            shared_context["analysis"] = analyzer_result.output
        
        # Run summarizer and recommender in parallel
        logger.info(f"[{workflow_id}] Running summarizer + recommender in parallel...")
        summarizer_task = asyncio.create_task(
            agents["summarizer"].process(shared_context)
        )
        recommender_task = asyncio.create_task(
            agents["recommender"].process(shared_context)
        )
        
        summarizer_result, recommender_result = await asyncio.gather(
            summarizer_task, recommender_task, return_exceptions=True
        )
        
        results = {"analyzer": analyzer_result}
        errors = []
        
        if isinstance(summarizer_result, AgentResult):
            results["summarizer"] = summarizer_result
        else:
            errors.append(f"Summarizer error: {summarizer_result}")
        
        if isinstance(recommender_result, AgentResult):
            results["recommender"] = recommender_result
        else:
            errors.append(f"Recommender error: {recommender_result}")
        
        completed_at = datetime.now()
        total_tokens = sum(r.tokens_used for r in results.values())
        total_time = int((completed_at - started_at).total_seconds() * 1000)
        
        return CoordinationResult(
            workflow_id=workflow_id,
            mode=mode,
            workflow_type=WorkflowType.FULL,
            started_at=started_at,
            completed_at=completed_at,
            total_tokens=total_tokens,
            total_time_ms=total_time,
            results=results,
            final_output=self._compile_output(results, mode),
            success=len(errors) == 0,
            errors=errors
        )
    
    async def execute_rag_workflow(
        self,
        mode: AnalysisMode,
        document_id: str,
        content: str,
        metadata: dict,
        focus_query: Optional[str] = None
    ) -> CoordinationResult:
        """
        Execute RAG-enhanced workflow with multi-pass analysis.
        
        Pass 1: Retrieve most relevant chunks using hybrid search
        Pass 2: Analyze with enriched context (chunks + entities)
        Pass 3: Generate output with citation markers
        
        Args:
            mode: Analysis mode (document, code, research, legal)
            document_id: Document ID for RAG retrieval
            content: Full document content
            metadata: Document metadata
            focus_query: Optional query to focus the analysis
        """
        workflow_id = f"wf_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        started_at = datetime.now()
        results: dict[str, AgentResult] = {}
        errors: list[str] = []
        total_tokens = 0
        rag_results: list = []
        
        agents = self._get_agents_for_mode(mode)
        vector_service = get_vector_service()
        
        try:
            # === PASS 1: RAG Retrieval ===
            logger.info(f"[{workflow_id}] Pass 1: RAG Retrieval...")
            
            # Build search query from content summary or focus query
            search_query = focus_query or self._extract_search_query(content)
            
            # Retrieve more chunks for deeper, evidence-backed analysis (12 instead of 8)
            rag_results = await vector_service.hybrid_search(
                query=search_query,
                n_results=12,
                document_id=document_id
            )
            
            # Format RAG context with citation markers
            rag_context = self._format_rag_context(rag_results)
            
            logger.info(f"[{workflow_id}] Retrieved {len(rag_results)} relevant chunks")
            
            # === PASS 2: Enhanced Analysis ===
            logger.info(f"[{workflow_id}] Pass 2: RAG-Enhanced Analysis...")
            
            shared_context = {
                "content": content[:30000],  # Full content for context
                "metadata": metadata,
                "mode": mode.value,
                "rag_context": rag_context,  # Injected RAG chunks
                "rag_chunks": rag_results,   # Raw chunks for citation
                "use_citations": True         # Flag to enable citations
            }
            
            # Run analyzer with RAG context
            analyzer_result = await agents["analyzer"].process(shared_context)
            results["analyzer"] = analyzer_result
            total_tokens += analyzer_result.tokens_used
            
            if isinstance(analyzer_result.output, dict):
                shared_context["analysis"] = analyzer_result.output
            
            # === PASS 3: Summarization with Citations ===
            logger.info(f"[{workflow_id}] Pass 3: Summary with Citations...")
            
            summarizer_result = await agents["summarizer"].process(shared_context)
            results["summarizer"] = summarizer_result
            total_tokens += summarizer_result.tokens_used
            
            if isinstance(summarizer_result.output, dict):
                shared_context["summary"] = summarizer_result.output
            
            # Recommender
            recommender_result = await agents["recommender"].process(shared_context)
            results["recommender"] = recommender_result
            total_tokens += recommender_result.tokens_used
            
        except Exception as e:
            logger.error(f"[{workflow_id}] RAG Workflow error: {e}")
            errors.append(str(e))
        
        completed_at = datetime.now()
        total_time = int((completed_at - started_at).total_seconds() * 1000)
        
        # Compile output with RAG metadata and citations for frontend
        final_output = self._compile_output(results, mode)
        final_output["rag_enabled"] = True
        final_output["chunks_retrieved"] = len(rag_results)
        final_output["citations"] = [
            {
                "marker": f"[Chunk {r.get('chunk_index', i)}]",
                "chunk_index": r.get("chunk_index", i),
                "source_preview": (r.get("content") or "")[:200],
            }
            for i, r in enumerate(rag_results)
        ]

        return CoordinationResult(
            workflow_id=workflow_id,
            mode=mode,
            workflow_type=WorkflowType.FULL,
            started_at=started_at,
            completed_at=completed_at,
            total_tokens=total_tokens,
            total_time_ms=total_time,
            results=results,
            final_output=final_output,
            success=len(errors) == 0,
            errors=errors
        )
    
    def _extract_search_query(self, content: str, max_length: int = 600) -> str:
        """Extract key phrases from content for RAG search (beginning + middle + end for better coverage)."""
        lines = [ln.strip() for ln in content.split('\n') if ln.strip()]
        query_parts = []
        seen = set()
        
        # Beginning: first 15 substantial lines
        for line in lines[:15]:
            if len(line) > 25 and line not in seen:
                seen.add(line)
                query_parts.append(line[:220])
        
        # Middle: sample from center (for long docs)
        if len(lines) > 30:
            mid = len(lines) // 2
            for line in lines[mid : mid + 5]:
                if len(line) > 25 and line not in seen:
                    seen.add(line)
                    query_parts.append(line[:220])
        
        # End: last 5 substantial lines (conclusions, next steps)
        for line in lines[-5:]:
            if len(line) > 25 and line not in seen:
                seen.add(line)
                query_parts.append(line[:220])
        
        combined = ' '.join(query_parts)[:max_length]
        return combined if combined else content[:500]
    
    def _format_rag_context(self, rag_results: list[dict]) -> str:
        """Format RAG results into context string with citation markers (larger chunks for detail)."""
        if not rag_results:
            return "No relevant context retrieved."
        
        context_parts = []
        for i, result in enumerate(rag_results):
            chunk_idx = result.get("chunk_index", i)
            score = result.get("similarity_score", 0)
            # Allow up to 1000 chars per chunk for richer evidence
            content = result.get("content", "")[:1000]
            
            context_parts.append(
                f"[Chunk {chunk_idx}] (Relevance: {score:.2f})\n{content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _compile_output(
        self,
        results: dict[str, AgentResult],
        mode: AnalysisMode
    ) -> dict:
        """Compile all agent results into a unified output."""
        output = {
            "mode": mode.value,
            "generated_at": datetime.now().isoformat(),
            "agents_used": list(results.keys())
        }
        
        for agent_name, result in results.items():
            if isinstance(result.output, dict):
                output[agent_name] = result.output
            else:
                output[agent_name] = {"raw": result.output}
        
        return output


# Singleton
_coordinator: Optional[MultiAgentCoordinator] = None


def get_coordinator() -> MultiAgentCoordinator:
    """Get or create coordinator instance."""
    global _coordinator
    if _coordinator is None:
        _coordinator = MultiAgentCoordinator()
    return _coordinator
