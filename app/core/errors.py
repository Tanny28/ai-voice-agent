"""Custom exceptions for the application."""

from typing import Optional


class VoiceAgentError(Exception):
    """Base exception for voice agent errors."""
    
    def __init__(
        self, 
        message: str, 
        error_type: str = "general_error",
        fallback_message: Optional[str] = None
    ):
        self.message = message
        self.error_type = error_type
        self.fallback_message = fallback_message
        super().__init__(self.message)


class STTServiceError(VoiceAgentError):
    """Speech-to-text service error."""
    
    def __init__(self, message: str, fallback_message: Optional[str] = None):
        super().__init__(
            message=message,
            error_type="stt_error",
            fallback_message=fallback_message or "I'm having trouble understanding your voice right now."
        )


class LLMServiceError(VoiceAgentError):
    """LLM service error."""
    
    def __init__(self, message: str, fallback_message: Optional[str] = None):
        super().__init__(
            message=message,
            error_type="llm_error",
            fallback_message=fallback_message or "I'm experiencing some technical difficulties with my thinking process."
        )


class TTSServiceError(VoiceAgentError):
    """Text-to-speech service error."""
    
    def __init__(self, message: str, fallback_message: Optional[str] = None):
        super().__init__(
            message=message,
            error_type="tts_error",
            fallback_message=fallback_message or "I can understand you, but I'm having trouble speaking right now."
        )


class ServiceUnavailableError(VoiceAgentError):
    """Service unavailable error."""
    
    def __init__(self, service_name: str):
        super().__init__(
            message=f"{service_name} service is unavailable",
            error_type="service_unavailable",
            fallback_message=f"The {service_name} service is currently unavailable. Please try again later."
        )
