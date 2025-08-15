"""Application configuration management."""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Keys
    murf_api_key: str = ""
    assemblyai_api_key: str = ""
    gemini_api_key: str = ""
    
    # Application Settings
    debug: bool = False
    log_level: str = "INFO"
    max_audio_size: int = 50 * 1024 * 1024  # 50MB
    session_timeout: int = 3600  # 1 hour
    
    # Request Timeouts (seconds)
    stt_timeout: int = 60
    llm_timeout: int = 90
    tts_timeout: int = 120
    
    # Text Limits
    max_llm_response_chars: int = 2500
    max_conversation_history: int = 20
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Map environment variables to settings
        fields = {
            "murf_api_key": {"env": "MURF_API_KEY"},
            "assemblyai_api_key": {"env": "ASSEMBLYAI_API_KEY"},
            "gemini_api_key": {"env": "GEMINI_API_KEY"},
            "debug": {"env": "DEBUG"},
            "log_level": {"env": "LOG_LEVEL"},
        }


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Global settings instance
settings = get_settings()
