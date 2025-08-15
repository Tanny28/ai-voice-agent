"""AssemblyAI speech-to-text service."""

import asyncio
from typing import Optional
import assemblyai as aai

from app.config import settings
from app.core.errors import STTServiceError, ServiceUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)


class AssemblyAIService:
    """AssemblyAI speech-to-text service wrapper."""
    
    def __init__(self):
        """Initialize AssemblyAI service."""
        if not settings.assemblyai_api_key:
            logger.error("AssemblyAI API key not configured")
            raise ServiceUnavailableError("AssemblyAI")
        
        try:
            aai.settings.api_key = settings.assemblyai_api_key
            self.transcriber = aai.Transcriber()
            logger.info("AssemblyAI service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AssemblyAI: {e}")
            raise ServiceUnavailableError("AssemblyAI")
    
    async def transcribe_audio(self, audio_data: bytes) -> str:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Transcribed text
            
        Raises:
            STTServiceError: If transcription fails
        """
        if len(audio_data) == 0:
            raise STTServiceError("Empty audio file provided")
        
        try:
            logger.info(f"Starting transcription for {len(audio_data)} bytes of audio")
            
            # Run transcription in thread pool to avoid blocking
            transcript = await asyncio.wait_for(
                asyncio.to_thread(self.transcriber.transcribe, audio_data),
                timeout=settings.stt_timeout
            )
            
            if transcript.error:
                logger.error(f"AssemblyAI transcription error: {transcript.error}")
                raise STTServiceError(f"Transcription failed: {transcript.error}")
            
            transcribed_text = transcript.text.strip() if transcript.text else ""
            
            if not transcribed_text:
                logger.warning("No speech detected in audio")
                raise STTServiceError(
                    "No speech detected in audio",
                    "I didn't hear anything. Could you please speak more clearly?"
                )
            
            logger.info(f"Transcription successful: {len(transcribed_text)} characters")
            return transcribed_text
            
        except asyncio.TimeoutError:
            logger.error("AssemblyAI transcription timeout")
            raise STTServiceError(
                "Audio transcription timed out",
                "The transcription is taking too long. Please try with shorter audio."
            )
        except STTServiceError:
            raise
        except Exception as e:
            logger.error(f"Unexpected transcription error: {e}")
            raise STTServiceError(f"Transcription service error: {str(e)}")


# Global service instance
assembly_ai_service = AssemblyAIService()
