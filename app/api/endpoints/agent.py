"""Conversational agent API endpoints."""

from typing import Dict, Any
from fastapi import APIRouter, File, UploadFile, HTTPException, Path as PathParam
import time

from app.models.schemas import (
    ConversationResponse, 
    ChatHistoryResponse, 
    ChatMessage,
    TTSRequest,
    TTSResponse
)
from app.services.assembly_ai import assembly_ai_service
from app.services.gemini_ai import gemini_ai_service
from app.services.murf_ai import murf_ai_service
from app.core.errors import VoiceAgentError
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/agent", tags=["agent"])

# In-memory session storage (TODO: Replace with persistent storage)
chat_sessions: Dict[str, list] = {}


@router.post("/chat/{session_id}", response_model=ConversationResponse)
async def chat_with_agent(
    session_id: str = PathParam(..., description="Unique session identifier"),
    file: UploadFile = File(..., description="Audio file containing user message")
) -> ConversationResponse:
    """
    Have a conversation with the AI agent.
    
    This endpoint accepts audio input, transcribes it, generates an AI response,
    converts the response to speech, and maintains conversation history.
    """
    start_time = time.time()
    logger.info(f"Starting conversation for session {session_id}")
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    allowed_types = ["audio/webm", "audio/wav", "audio/mp3", "audio/mpeg", "audio/ogg", "audio/m4a"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {allowed_types}"
        )
    
    # Get or initialize conversation history
    history = chat_sessions.get(session_id, [])
    
    try:
        # Step 1: Read and validate audio
        audio_data = await file.read()
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        logger.info(f"Processing {len(audio_data)} bytes of audio")
        
        # Step 2: Transcribe audio
        try:
            user_text = await assembly_ai_service.transcribe_audio(audio_data)
        except VoiceAgentError as e:
            return await _handle_service_error(e, session_id, "transcription")
        
        # Step 3: Add user message to history
        user_message = ChatMessage(role="user", content=user_text)
        history.append(user_message.dict())
        
        # Limit history size
        if len(history) > 40:  # Keep 40 messages (20 exchanges)
            history = history[-40:]
        
        # Step 4: Generate AI response
        try:
            chat_history = [ChatMessage(**msg) for msg in history[:-1]]  # Exclude current message
            ai_response = await gemini_ai_service.generate_response(user_text, chat_history)
        except VoiceAgentError as e:
            # Add fallback response to history
            fallback_message = ChatMessage(role="assistant", content=e.fallback_message)
            history.append(fallback_message.dict())
            chat_sessions[session_id] = history
            
            return await _handle_service_error(e, session_id, "generation", user_text)
        
        # Step 5: Add AI response to history
        ai_message = ChatMessage(role="assistant", content=ai_response)
        history.append(ai_message.dict())
        chat_sessions[session_id] = history
        
        # Step 6: Generate speech
        try:
            audio_url = await murf_ai_service.generate_speech(ai_response)
        except VoiceAgentError as e:
            # Return text response even if TTS fails
            logger.warning(f"TTS failed but returning text response: {e}")
            return ConversationResponse(
                audio_url="",
                transcription=user_text,
                llm_response=ai_response,
                message="Response generated but voice synthesis unavailable",
                error_type=e.error_type
            )
        
        processing_time = time.time() - start_time
        logger.info(f"Conversation completed in {processing_time:.2f} seconds")
        
        return ConversationResponse(
            audio_url=audio_url,
            transcription=user_text,
            llm_response=ai_response,
            message=f"Conversation successful for session {session_id[:8]}..."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in conversation: {e}")
        
        # Try to provide fallback response
        fallback_text = "I'm experiencing technical difficulties. Please try again."
        try:
            fallback_audio = await murf_ai_service.generate_speech(fallback_text)
            return ConversationResponse(
                audio_url=fallback_audio,
                transcription="[Error occurred]",
                llm_response=fallback_text,
                message="Using fallback response due to system error",
                error_type="system_error",
                fallback_used=True
            )
        except:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": f"Complete system failure: {str(e)}",
                    "error_type": "system_failure",
                    "fallback_message": fallback_text
                }
            )


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
async def get_conversation_history(
    session_id: str = PathParam(..., description="Session identifier")
) -> ChatHistoryResponse:
    """Get conversation history for a session."""
    try:
        history = chat_sessions.get(session_id, [])
        chat_messages = [ChatMessage(**msg) for msg in history]
        
        logger.info(f"Retrieved {len(chat_messages)} messages for session {session_id}")
        
        return ChatHistoryResponse(
            session_id=session_id,
            history=chat_messages,
            message=f"Retrieved {len(chat_messages)} messages"
        )
    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")


@router.delete("/history/{session_id}")
async def clear_conversation_history(
    session_id: str = PathParam(..., description="Session identifier")
) -> Dict[str, str]:
    """Clear conversation history for a session."""
    try:
        if session_id in chat_sessions:
            del chat_sessions[session_id]
            logger.info(f"Cleared history for session {session_id}")
            return {"message": f"History cleared for session {session_id[:8]}..."}
        else:
            return {"message": f"No history found for session {session_id[:8]}..."}
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")


@router.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest) -> TTSResponse:
    """Convert text to speech using Murf AI."""
    try:
        logger.info(f"Converting text to speech: {len(request.text)} characters")
        audio_url = await murf_ai_service.generate_speech(request.text)
        
        return TTSResponse(
            audio_url=audio_url,
            message="Audio generated successfully"
        )
    except VoiceAgentError as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": e.message,
                "error_type": e.error_type,
                "fallback_message": e.fallback_message
            }
        )


async def _handle_service_error(
    error: VoiceAgentError, 
    session_id: str, 
    operation: str,
    user_text: str = "[Error occurred]"
) -> ConversationResponse:
    """Handle service errors with fallback responses."""
    logger.warning(f"Service error in {operation}: {error}")
    
    try:
        # Try to provide audio fallback
        fallback_audio = await murf_ai_service.generate_speech(error.fallback_message)
        return ConversationResponse(
            audio_url=fallback_audio,
            transcription=user_text,
            llm_response=error.fallback_message,
            message=f"Using fallback response due to {operation} error",
            error_type=error.error_type,
            fallback_used=True
        )
    except:
        # If even fallback TTS fails, return text-only
        return ConversationResponse(
            audio_url="",
            transcription=user_text,
            llm_response=error.fallback_message,
            message=f"Service error in {operation}, text-only fallback",
            error_type=error.error_type,
            fallback_used=True
        )
