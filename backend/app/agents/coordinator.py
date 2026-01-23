"""
Multi-Agent Coordinator - Orchestrates agent workflows
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
