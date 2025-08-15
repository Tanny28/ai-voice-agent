"""Murf AI text-to-speech service."""

import asyncio
import requests
from typing import Optional

from app.config import settings
from app.core.errors import TTSServiceError, ServiceUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)


class MurfAIService:
    """Murf AI text-to-speech service wrapper."""
    
    def __init__(self):
        """Initialize Murf AI service."""
        if not settings.murf_api_key:
            logger.error("Murf API key not configured")
            raise ServiceUnavailableError("Murf AI")
        
        self.api_key = settings.murf_api_key
        self.base_url = "https://api.murf.ai/v1/speech/generate"
        logger.info("Murf AI service initialized successfully")
    
    async def generate_speech(
        self, 
        text: str, 
        voice_id: str = "en-US-natalie",
        style: str = "Conversational"
    ) -> str:
        """
        Generate speech from text.
        
        Args:
            text: Text to convert to speech
            voice_id: Murf voice ID to use
            style: Voice style to apply
            
        Returns:
            Audio URL from Murf API
            
        Raises:
            TTSServiceError: If speech generation fails
        """
        if not text or not text.strip():
            raise TTSServiceError("Empty text provided for speech generation")
        
        # Truncate text if too long
        if len(text) > 2800:  # Leave buffer for Murf's 3000 char limit
            logger.warning(f"Text too long ({len(text)} chars), truncating")
            text = text[:2800].rsplit(' ', 1)[0] + "..."
        
        try:
            logger.info(f"Generating speech for {len(text)} characters")
            
            headers = {
                "api-key": self.api_key,
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
            
            payload = {
                "voiceId": voice_id,
                "style": style,
                "text": text,
                "format": "MP3",
                "sampleRate": 44100,
                "effect": "none",
            }
            
            # Make request with timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    requests.post, 
                    self.base_url, 
                    headers=headers, 
                    json=payload, 
                    timeout=120
                ),
                timeout=settings.tts_timeout
            )
            
            if response.status_code != 200:
                logger.error(f"Murf API error: {response.status_code} - {response.text}")
                raise TTSServiceError(
                    f"Voice generation failed: {response.text}",
                    "I'm having trouble generating voice right now."
                )
            
            data = response.json()
            audio_url = data.get("audioFile", "")
            
            if not audio_url:
                logger.error("Murf returned no audio URL")
                raise TTSServiceError(
                    "No audio file generated",
                    "Voice generation completed but no audio was created."
                )
            
            logger.info("Speech generation successful")
            return audio_url
            
        except asyncio.TimeoutError:
            logger.error("Murf TTS timeout")
            raise TTSServiceError(
                "Voice generation timed out",
                "Voice generation is taking too long. Please try again."
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Murf network error: {e}")
            raise TTSServiceError(
                f"Voice service network error: {str(e)}",
                "There's a network issue with the voice service."
            )
        except TTSServiceError:
            raise
        except Exception as e:
            logger.error(f"Unexpected TTS error: {e}")
            raise TTSServiceError(f"Voice generation error: {str(e)}")


# Global service instance
murf_ai_service = MurfAIService()
