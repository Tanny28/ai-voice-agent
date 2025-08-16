"""Pydantic models for request/response validation."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class AudioUploadResponse(BaseModel):
    """Response model for audio upload."""
    filename: str
    content_type: str
    size: int
    message: str


class TranscriptionResponse(BaseModel):
    """Response model for audio transcription."""
    transcription: str
    message: str
    confidence: Optional[float] = None


class TTSRequest(BaseModel):
    """Request model for text-to-speech."""
    text: str = Field(..., min_length=1, max_length=3000)


class TTSResponse(BaseModel):
    """Response model for text-to-speech."""
    audio_url: str
    message: str


class ChatMessage(BaseModel):
    """Model for individual chat messages."""
    role: str = Field(..., pattern="^(user|assistant)$")  # Fixed: regex -> pattern
    content: str = Field(..., min_length=1)


class ConversationResponse(BaseModel):
    """Response model for conversational agent."""
    audio_url: str
    transcription: str
    llm_response: str
    message: str
    error_type: Optional[str] = None
    fallback_used: bool = False


class ChatHistoryResponse(BaseModel):
    """Response model for chat history."""
    session_id: str
    history: List[ChatMessage]
    message: str


class HealthCheckResponse(BaseModel):
    """Response model for health checks."""
    status: str
    services: Dict[str, str]
    timestamp: float
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Standardized error response model."""
    detail: str
    error_type: str
    fallback_message: Optional[str] = None
    timestamp: float
