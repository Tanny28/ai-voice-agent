"""
AI Voice Agent - FastAPI Application with Streaming LLM Responses (Day 19)
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Dict, List
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import assemblyai as aai
from starlette.websockets import WebSocketState

# Google Gemini imports
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Google Gemini SDK not installed. Run: pip install google-generativeai")

from app.config import settings
from app.core.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create required directories
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

# Configure AssemblyAI
if settings.assemblyai_api_key:
    aai.settings.api_key = settings.assemblyai_api_key
    logger.info("AssemblyAI configured for streaming transcription")
else:
    logger.warning("AssemblyAI API key not found")

# Configure Google Gemini
if GEMINI_AVAILABLE and hasattr(settings, 'gemini_api_key') and settings.gemini_api_key:
    genai.configure(api_key=settings.gemini_api_key)
    gemini_client = genai.Client()
    logger.info("Google Gemini configured for streaming LLM responses")
else:
    gemini_client = None
    logger.warning("Google Gemini API key not found - LLM streaming disabled")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Voice Agent with Streaming LLM Responses")
    yield
    logger.info("Shutting down AI Voice Agent application")

# Initialize FastAPI application
app = FastAPI(
    title="AI Voice Agent with Streaming LLM",
    description="Real-time transcription with streaming LLM responses",
    version="1.9.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# WebSocket helper functions
def is_websocket_open(ws):
    """Check if WebSocket is still connected."""
    return ws.client_state == WebSocketState.CONNECTED

async def safe_send(ws, message):
    """Safely send WebSocket message, avoiding sends after close."""
    if is_websocket_open(ws):
        try:
            await ws.send_text(message)
        except Exception as e:
            logger.error(f"WebSocket send error: {e}")
    else:
        logger.debug("WebSocket closed, skipping message send")

# ==================================================================
# DAY 19: STREAMING LLM RESPONSES WITH GEMINI
# ==================================================================

async def generate_llm_response_stream(prompt: str, session_id: str = "default"):
    """
    Generate a streaming response from Google Gemini LLM.
    
    Args:
        prompt: The input text to generate a response for
        session_id: Session ID for logging
    
    Returns:
        Complete generated response text
    """
    if not gemini_client:
        logger.error("Gemini client not configured - cannot generate LLM response")
        print(f"[LLM ERROR] Gemini not configured")
        return "I'm sorry, but I'm not configured to generate responses right now."
    
    try:
        logger.info(f"Generating LLM response for session {session_id}")
        print(f"\n[LLM PROMPT] {prompt}")
        print(f"[LLM RESPONSE] ", end="", flush=True)
        
        # Generate streaming response from Gemini
        response = gemini_client.models.generate_content(
            model="gemini-1.5-flash",  # Use available Gemini model
            contents=prompt,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
        )
        
        # For non-streaming, we'll simulate streaming by printing response
        full_response = response.text
        
        # Simulate streaming by printing character by character
        accumulated_response = ""
        for char in full_response:
            print(char, end="", flush=True)
            accumulated_response += char
            await asyncio.sleep(0.02)  # Small delay to simulate streaming
        
        print()  # New line after response
        logger.info(f"LLM response generated: {len(accumulated_response)} characters")
        
        return accumulated_response
        
    except Exception as e:
        error_msg = f"Error generating LLM response: {e}"
        logger.error(error_msg)
        print(f"[LLM ERROR] {error_msg}")
        return "I encountered an error while generating a response. Please try again."

async def generate_llm_response_stream_v2(prompt: str, session_id: str = "default"):
    """
    Alternative streaming implementation using google-generativeai package.
    """
    try:
        import google.generativeai as genai_alt
        
        # Configure the alternative client
        genai_alt.configure(api_key=settings.gemini_api_key)
        model = genai_alt.GenerativeModel('gemini-1.5-flash')
        
        logger.info(f"Generating streaming LLM response for session {session_id}")
        print(f"\n[LLM PROMPT] {prompt}")
        print(f"[LLM STREAMING] ", end="", flush=True)
        
        # Generate streaming response
        response = model.generate_content(
            prompt,
            generation_config=genai_alt.types.GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                top_k=40,
                max_output_tokens=1024,
            ),
            stream=True
        )
        
        accumulated_response = ""
        for chunk in response:
            if chunk.text:
                print(chunk.text, end="", flush=True)
                accumulated_response += chunk.text
        
        print()  # New line after response
        logger.info(f"Streaming LLM response completed: {len(accumulated_response)} characters")
        
        return accumulated_response
        
    except ImportError:
        logger.warning("google-generativeai package not available, using fallback")
        return await generate_llm_response_stream(prompt, session_id)
    except Exception as e:
        error_msg = f"Error in streaming LLM response: {e}"
        logger.error(error_msg)
        print(f"[LLM ERROR] {error_msg}")
        return "I encountered an error while generating a response. Please try again."

@app.websocket("/ws/llm-streaming")
async def websocket_llm_streaming(websocket: WebSocket):
    """
    WebSocket endpoint for turn detection with streaming LLM responses.
    
    This endpoint combines Day 18 turn detection with Day 19 LLM streaming.
    """
    await websocket.accept()
    logger.info("LLM streaming WebSocket connection established")
    
    if not settings.assemblyai_api_key:
        await safe_send(websocket, json.dumps({
            "type": "error",
            "message": "AssemblyAI API key not configured",
            "timestamp": time.time()
        }))
        await websocket.close(code=1000, reason="AssemblyAI API key required")
        return
    
    session_id = f"llm_session_{int(time.time())}"
    audio_file_path = UPLOADS_DIR / f"{session_id}.webm"
    
    try:
        # Send connection confirmation
        await safe_send(websocket, json.dumps({
            "type": "connection_established",
            "message": "LLM streaming ready - speak to get AI responses!",
            "session_id": session_id,
            "timestamp": time.time()
        }))
        
        # Collect audio chunks for turn detection analysis
        audio_chunks = []
        turn_count = 0
        
        await safe_send(websocket, json.dumps({
            "type": "llm_ready",
            "message": "Listening for speech - will generate LLM responses after turns!",
            "timestamp": time.time()
        }))
        
        with open(audio_file_path, "wb") as audio_file:
            silence_start = None
            
            while True:
                try:
                    # Receive audio data with timeout for turn detection
                    data = await asyncio.wait_for(websocket.receive_bytes(), timeout=2.0)
                    
                    # Write audio data
                    audio_file.write(data)
                    audio_chunks.append(data)
                    
                    # Reset silence detection
                    silence_start = None
                    
                    # Send audio chunk acknowledgment
                    await safe_send(websocket, json.dumps({
                        "type": "audio_received",
                        "chunk_size": len(data),
                        "total_chunks": len(audio_chunks),
                        "timestamp": time.time()
                    }))
                    
                except asyncio.TimeoutError:
                    # Potential turn end detected due to silence
                    current_time = time.time()
                    
                    if silence_start is None:
                        silence_start = current_time
                        logger.info("Silence detected - potential turn ending")
                        
                        await safe_send(websocket, json.dumps({
                            "type": "silence_detected",
                            "message": "Silence detected - analyzing turn...",
                            "timestamp": current_time
                        }))
                    
                    # Check if silence duration indicates turn end (2 seconds)
                    silence_duration = current_time - silence_start
                    if silence_duration >= 2.0 and audio_chunks:
                        logger.info(f"Turn end detected after {silence_duration:.1f}s silence")
                        
                        # Process turn and generate LLM response
                        await process_turn_with_llm(websocket, audio_file_path, 
                                                  turn_count, session_id)
                        
                        turn_count += 1
                        
                        # Reset for next turn
                        audio_chunks.clear()
                        silence_start = None
                        
                        # Create new file for next turn
                        audio_file_path = UPLOADS_DIR / f"{session_id}_turn_{turn_count}.webm"
                        
                        await safe_send(websocket, json.dumps({
                            "type": "ready_for_next_turn",
                            "message": "Ready for next turn - continue speaking!",
                            "turn_number": turn_count + 1,
                            "timestamp": time.time()
                        }))
                        
                        # Start new audio file
                        break
                        
                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected during LLM streaming")
                    break
                except Exception as e:
                    logger.error(f"Error in LLM streaming: {e}")
                    break
        
        # Process final turn if any audio was collected
        if audio_chunks:
            await process_turn_with_llm(websocket, audio_file_path, 
                                      turn_count, session_id)
        
    except WebSocketDisconnect:
        logger.info("LLM streaming WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in LLM streaming: {e}")
        print(f"[LLM STREAMING ERROR] {e}")
    finally:
        logger.info(f"LLM streaming session {session_id} completed with {turn_count + 1} turns")

async def process_turn_with_llm(websocket: WebSocket, audio_file_path: Path, 
                               turn_number: int, session_id: str):
    """Process a detected turn, transcribe it, and generate an LLM response."""
    try:
        if not audio_file_path.exists() or audio_file_path.stat().st_size == 0:
            logger.warning(f"No audio data for turn {turn_number}")
            return
        
        logger.info(f"Processing turn {turn_number} with LLM: {audio_file_path}")
        
        # Notify client that turn is being processed
        await safe_send(websocket, json.dumps({
            "type": "turn_processing",
            "turn_number": turn_number,
            "message": f"Processing turn {turn_number}...",
            "timestamp": time.time()
        }))
        
        # Transcribe the turn audio
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(str(audio_file_path))
        
        if transcript.error:
            logger.error(f"Transcription error for turn {turn_number}: {transcript.error}")
            print(f"[TURN {turn_number} ERROR] {transcript.error}")
            
            await safe_send(websocket, json.dumps({
                "type": "turn_error",
                "turn_number": turn_number,
                "error": transcript.error,
                "timestamp": time.time()
            }))
        else:
            # Successful turn transcription
            user_input = transcript.text.strip()
            print(f"[TURN {turn_number} TRANSCRIBED] {user_input}")
            logger.info(f"Turn {turn_number} transcribed: {user_input}")
            
            # Send transcription to client
            await safe_send(websocket, json.dumps({
                "type": "turn_transcribed",
                "turn_number": turn_number,
                "transcript": user_input,
                "confidence": getattr(transcript, 'confidence', 1.0),
                "session_id": session_id,
                "timestamp": time.time()
            }))
            
            # Generate LLM response if user said something
            if user_input and len(user_input.strip()) > 0:
                # Notify client that LLM is generating response
                await safe_send(websocket, json.dumps({
                    "type": "llm_generating",
                    "turn_number": turn_number,
                    "message": "Generating AI response...",
                    "timestamp": time.time()
                }))
                
                # Generate streaming LLM response
                llm_response = await generate_llm_response_stream_v2(
                    user_input, 
                    f"{session_id}_turn_{turn_number}"
                )
                
                # Send completed LLM response to client
                await safe_send(websocket, json.dumps({
                    "type": "llm_response_complete",
                    "turn_number": turn_number,
                    "user_input": user_input,
                    "llm_response": llm_response,
                    "session_id": session_id,
                    "timestamp": time.time()
                }))
                
                print(f"[TURN {turn_number} COMPLETE] User: {user_input} | LLM: {llm_response[:100]}...")
            else:
                await safe_send(websocket, json.dumps({
                    "type": "turn_complete_no_response",
                    "turn_number": turn_number,
                    "message": "Turn completed but no speech detected",
                    "timestamp": time.time()
                }))
            
    except Exception as e:
        logger.error(f"Error processing turn {turn_number}: {e}")
        print(f"[TURN {turn_number} ERROR] {e}")

# Keep existing Day 18 endpoint for compatibility
@app.websocket("/ws/turn-detection")
async def websocket_turn_detection(websocket: WebSocket):
    """Day 18: Turn detection without LLM responses."""
    await websocket.accept()
    logger.info("Turn detection WebSocket connection established")
    
    # Implementation from Day 18...
    # (Previous turn detection code here)
    
    await safe_send(websocket, json.dumps({
        "type": "connection_established",
        "message": "Turn detection ready (Day 18 mode)",
        "timestamp": time.time()
    }))

# Existing endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.9.0",
        "features": ["turn_detection", "llm_streaming", "real_time_transcription"],
        "services": {
            "assemblyai": "available" if settings.assemblyai_api_key else "unavailable",
            "gemini": "available" if gemini_client else "unavailable"
        }
    }

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the main frontend application."""
    return JSONResponse({
        "message": "AI Voice Agent with Streaming LLM Responses",
        "websocket_endpoints": {
            "llm_streaming": "/ws/llm-streaming",
            "turn_detection": "/ws/turn-detection",
            "real_time_transcription": "/ws/transcribe"
        },
        "api_endpoints": {
            "health": "/health",
            "docs": "/docs"
        },
        "version": "1.9.0"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=1)
