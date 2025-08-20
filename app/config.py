"""Application configuration management."""

from pydantic import Field
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    murf_api_key: str = ""
    assemblyai_api_key: str = ""
    gemini_api_key: str = ""
    
    debug: bool = False
    log_level: str = "INFO"
    max_audio_size: int = 50 * 1024 * 1024
    session_timeout: int = 3600
    
    stt_timeout: int = 60
    llm_timeout: int = 90
    tts_timeout: int = 120
    
    max_llm_response: int = 2500
    max_conversation_history: int = 20
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
