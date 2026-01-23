"""
Base Agent - Abstract foundation for all specialized agents
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time

from app.services.llm import get_llm_service


class AgentRole(str, Enum):
    """Agent roles in the system."""
    ANALYZER = "analyzer"
    SUMMARIZER = "summarizer"
    RECOMMENDER = "recommender"
    CODE_REVIEWER = "code_reviewer"
    RESEARCHER = "researcher"
    LEGAL_EXPERT = "legal_expert"
    COORDINATOR = "coordinator"


@dataclass
class AgentResult:
    """Result from an agent execution."""
    agent_name: str
    role: AgentRole
    output: dict[str, Any] | str
    confidence: float
    tokens_used: int
    execution_time_ms: int
    metadata: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    Provides common functionality for LLM interaction and result handling.
    """
    
    def __init__(
        self,
        name: str,
        role: AgentRole,
        model: str = "llama3.2",
        temperature: float = 0.7
    ):
        self.name = name
        self.role = role
        self.model = model
        self.temperature = temperature
        self.llm = get_llm_service()
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Define the agent's system prompt."""
        pass
    
    @abstractmethod
    async def process(self, input_data: dict[str, Any]) -> AgentResult:
        """
        Main processing method.
        Must be implemented by each specialized agent.
        """
        pass
    
    async def _call_llm(
        self,
        user_prompt: str,
        json_mode: bool = False,
        max_tokens: int = 2000
    ) -> tuple[str, int]:
        """Call LLM with the agent's system prompt."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        if json_mode:
            return await self.llm.chat_json(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=max_tokens
            )
        else:
            return await self.llm.chat(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=max_tokens
            )
    
    def _create_result(
        self,
        output: dict[str, Any] | str,
        tokens: int,
        start_time: float,
        confidence: float = 0.8,
        metadata: dict = None
    ) -> AgentResult:
        """Create an AgentResult with timing."""
        return AgentResult(
            agent_name=self.name,
            role=self.role,
            output=output,
            confidence=confidence,
            tokens_used=tokens,
            execution_time_ms=int((time.time() - start_time) * 1000),
            metadata=metadata or {}
        )
