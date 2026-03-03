"""
LLM Service - Ollama Local LLM Wrapper
Handles all LLM interactions with Ollama for free local inference
"""
import httpx
import json
from typing import Optional
from loguru import logger

from app.core.config import get_settings
from app.core.exceptions import LLMError


settings = get_settings()


class LLMService:
    """
    Ollama LLM service wrapper.
    Uses async httpx client for non-blocking I/O.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = "http://localhost:11434"
        self.default_model = "llama3.2"
        # Increase timeout for complex reasoning
        self.timeout = httpx.Timeout(300.0, connect=10.0)
        logger.info(f"LLMService initialized with base_url={self.base_url}, model={self.default_model}")
    
    async def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        response_format: Optional[dict] = None
    ) -> tuple[str, int]:
        """
        Send a chat completion request to Ollama asynchronously.
        """
        # Convert messages to single prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}\n\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n\n")
        
        prompt_parts.append("Assistant: ")
        full_prompt = "".join(prompt_parts)
        
        # Build payload
        # Not: Yerel Llama modelinde gereksiz uzun cevaplar ciddi gecikme yaratıyor.
        # Bu yüzden maksimum token tahminini makul bir üst sınırla sınırlandırıyoruz.
        safe_max_tokens = min(max_tokens, 1024)
        payload = {
            "model": model or self.default_model,
            "prompt": full_prompt,  # Remove hard truncation for better quality
            "stream": False,
            "context": [],  # Clear previous context to avoid confusion
            "options": {
                "temperature": temperature,
                "num_predict": safe_max_tokens,
                "num_ctx": 3072  # Slightly smaller context for daha hızlı çalışma
            }
        }
        
        # Add JSON format if requested
        if response_format and response_format.get("type") == "json_object":
            payload["format"] = "json"
        
        url = f"{self.base_url}/api/generate"
        logger.info(f"🔥 CALLING OLLAMA: {url}")
        logger.info(f"🔥 MODEL: {payload['model']}")
        logger.info(f"🔥 PROMPT LENGTH: {len(full_prompt)} chars")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                logger.info(f"🔥 RESPONSE STATUS: {response.status_code}")
                
                if response.status_code != 200:
                    logger.error(f"🔥 RESPONSE TEXT: {response.text[:500]}")
                    response.raise_for_status()
                
                result = response.json()
                content = result.get("response", "")
                # Fallback token estimation if eval_count missing
                tokens = result.get("eval_count", len(content.split()) * 1.5)
                
                logger.info(f"✅ Ollama success: ~{int(tokens)} tokens")
                return content, int(tokens)
                
        except httpx.ConnectError as e:
            logger.error(f"❌ Connection error: {e}")
            raise LLMError("Ollama not running. Please start Ollama (ollama serve).")
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP error: {e}")
            raise LLMError(f"LLM call failed: {e.response.text}")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {type(e).__name__}: {e}")
            raise LLMError(f"LLM call failed: {str(e)}")
    
    async def chat_json(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> tuple[str, int]:
        """Chat with JSON response format."""
        return await self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
    
    async def simple_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7
    ) -> tuple[str, int]:
        """Simple prompt with system and user messages."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return await self.chat(messages, model=model, temperature=temperature)


# Singleton
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
