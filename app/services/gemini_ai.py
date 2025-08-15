"""Google Gemini AI service."""

import asyncio
from typing import List, Dict
import google.generativeai as genai

from app.config import settings
from app.core.errors import LLMServiceError, ServiceUnavailableError
from app.core.logging import get_logger
from app.models.schemas import ChatMessage

logger = get_logger(__name__)


class GeminiAIService:
    """Google Gemini AI service wrapper."""
    
    def __init__(self):
        """Initialize Gemini AI service."""
        if not settings.gemini_api_key:
            logger.error("Gemini API key not configured")
            raise ServiceUnavailableError("Gemini AI")
        
        try:
            genai.configure(api_key=settings.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Gemini AI service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini AI: {e}")
            raise ServiceUnavailableError("Gemini AI")
    
    async def generate_response(
        self, 
        current_message: str, 
        conversation_history: List[ChatMessage]
    ) -> str:
        """
        Generate AI response with conversation context.
        
        Args:
            current_message: Current user message
            conversation_history: Previous conversation messages
            
        Returns:
            Generated AI response text
            
        Raises:
            LLMServiceError: If generation fails
        """
        try:
            logger.info(f"Generating response for message: {current_message[:50]}...")
            
            # Build conversation context
            context = self._build_conversation_context(current_message, conversation_history)
            
            # Generate response with timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(self.model.generate_content, context),
                timeout=settings.llm_timeout
            )
            
            if not response.text:
                logger.error("Gemini returned empty response")
                raise LLMServiceError(
                    "AI generated empty response",
                    "I'm having trouble generating a response right now. Please try again."
                )
            
            response_text = response.text.strip()
            
            # Truncate if too long for TTS
            if len(response_text) > settings.max_llm_response_chars:
                logger.warning(f"Response too long ({len(response_text)} chars), truncating")
                response_text = response_text[:settings.max_llm_response_chars].rsplit(' ', 1)[0] + "..."
            
            logger.info(f"Response generated successfully: {len(response_text)} characters")
            return response_text
            
        except asyncio.TimeoutError:
            logger.error("Gemini response generation timeout")
            raise LLMServiceError(
                "AI response generation timed out",
                "I'm taking too long to think. Please try with a simpler question."
            )
        except LLMServiceError:
            raise
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Gemini API error: {error_msg}")
            
            # Handle specific Gemini errors
            if "API_KEY" in error_msg.upper():
                raise LLMServiceError(
                    "Invalid AI API key",
                    "There's a configuration issue with the AI service."
                )
            elif "SAFETY" in error_msg.upper():
                raise LLMServiceError(
                    "Content blocked by AI safety filters",
                    "I can't respond to that type of content. Please try a different question."
                )
            elif "QUOTA" in error_msg.upper():
                raise LLMServiceError(
                    "AI API quota exceeded",
                    "The AI service is temporarily overloaded. Please try again later."
                )
            else:
                raise LLMServiceError(f"AI service error: {error_msg}")
    
    def _build_conversation_context(
        self, 
        current_message: str, 
        history: List[ChatMessage]
    ) -> str:
        """Build conversation context for the LLM."""
        
        context = "You are a helpful AI assistant. Here's our conversation so far:\n\n"
        
        # Add conversation history
        for message in history[-settings.max_conversation_history:]:
            role = "Human" if message.role == "user" else "Assistant"
            context += f"{role}: {message.content}\n"
        
        context += f"\nHuman: {current_message}\n\n"
        context += "Please provide a helpful and conversational response. "
        context += f"Keep your response under {settings.max_llm_response_chars} characters "
        context += "since it will be converted to speech.\n\nAssistant:"
        
        return context


# Global service instance
gemini_ai_service = GeminiAIService()
