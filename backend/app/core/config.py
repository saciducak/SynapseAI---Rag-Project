"""
SynapseAI Core Configuration
"""
from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # App
    APP_NAME: str = "SynapseAI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "llama3.2"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    
    # Security
    SECRET_KEY: str = "dev-secret-key"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # Upload
    MAX_UPLOAD_SIZE_MB: int = 50
    UPLOAD_DIR: str = "./uploads"
    
    # Available Modes
    AVAILABLE_MODES: list[str] = ["document", "code", "research", "legal"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
