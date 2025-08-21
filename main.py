"""
AI Voice Agent - FastAPI Application with Streaming LLM Responses (Day 19 Final Fixed)
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

# Google Gemini imports - FIXED for new SDK
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Google GenAI SDK not installed. Run: pip install google-genai")

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

# Configure Google Gemini - FIXED CLIENT INITIALIZATION
if GEMINI_AVAILABLE and getattr(settings, 'gemini_api_key', ''):
    try:
        # Initialize client with API key directly (no configure method)
        gemini_client = genai.Client(api_key=settings.gemini_api_key)
        logger.info("Google Gemini configured for streaming LLM responses")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        gemini_client = None
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
# DAY 19: STREAMING LLM RESPONSES - FINAL FIXED IMPLEMENTATION
# ==================================================================

async def generate_llm_response_stream(prompt: str, session_id: str = "default"):
    """Generate a streaming response using the new Google GenAI SDK."""
    if not gemini_client:
        logger.error("Gemini client not configured")
        print(f"[LLM ERROR] Gemini not configured")
        return "I'm sorry, but I'm not configured to generate responses right now."
    
    try:
        logger.info(f"Generating LLM response for session {session_id}")
        print(f"\n[LLM PROMPT] {prompt}")
        print(f"[LLM STREAMING] ", end="", flush=True)
        
        # FIXED: Use correct new SDK method
        try:
            # Try the async method first
            response = await gemini_client.aio.models.generate_content(
                model="gemini-1.5-flash",
                contents=[{"parts": [{"text": prompt}]}],
                config={
                    "temperature": 0.7,
                    "max_output_tokens": 1024,
                }
            )
            
            accumulated_response = ""
            if hasattr(response, 'text') and response.text:
                # Simulate streaming by printing character by character
                for char in response.text:
                    print(char, end="", flush=True)
                    accumulated_response += char
                    await asyncio.sleep(0.02)
        
        except Exception as async_error:
            logger.warning(f"Async method failed, trying sync: {async_error}")
            
            # Fallback to sync method with simulated streaming
            response = gemini_client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[{"parts": [{"text": prompt}]}],
                config={
                    "temperature": 0.7,
                    "max_output_tokens": 1024,
                }
            )
            
            accumulated_response = ""
            if hasattr(response, 'text') and response.text:
                # Simulate streaming output
                for char in response.text:
                    print(char, end="", flush=True)
                    accumulated_response += char
                    await asyncio.sleep(0.02)
        
        print()  # New line after response
        logger.info(f"LLM response generated: {len(accumulated_response)} characters")
        
        return accumulated_response
        
    except Exception as e:
        error_msg = f"Error generating LLM response: {e}"
        logger.error(error_msg)
        print(f"[LLM ERROR] {error_msg}")
        
        # Fallback response for demonstration
        fallback_response = f"I understand you said: '{prompt}'. This is a simulated AI response for Day 19 demonstration."
        
        # Print fallback with streaming effect
        for char in fallback_response:
            print(char, end="", flush=True)
            await asyncio.sleep(0.02)
        print()
        
        return fallback_response

@app.websocket("/ws/llm-streaming")
async def websocket_llm_streaming(websocket: WebSocket):
    """WebSocket endpoint for turn detection with streaming LLM responses - FIXED MESSAGE HANDLING."""
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
        await safe_send(websocket, json.dumps({
            "type": "connection_established",
            "message": "LLM streaming ready - speak to get AI responses!",
            "session_id": session_id,
            "llm_available": gemini_client is not None,
            "timestamp": time.time()
        }))
        
        # Collect audio chunks for turn detection
        audio_chunks = []
        turn_count = 0
        
        with open(audio_file_path, "wb") as audio_file:
            silence_start = None
            
            while True:
                try:
                    # FIXED: Safer message handling approach
                    message = await asyncio.wait_for(websocket.receive(), timeout=2.0)
                    
                    if 'bytes' in message:
                        # Handle binary audio data
                        data = message['bytes']
                        
                        # Write audio data
                        audio_file.write(data)
                        audio_chunks.append(data)
                        
                        # Reset silence detection
                        silence_start = None
                        
                        # Send acknowledgment
                        await safe_send(websocket, json.dumps({
                            "type": "audio_received",
                            "chunk_size": len(data),
                            "total_chunks": len(audio_chunks),
                            "timestamp": time.time()
                        }))
                    
                    elif 'text' in message:
                        # Handle text control messages
                        text_data = message['text']
                        logger.info(f"Received text control message: {text_data}")
                        
                        await safe_send(websocket, json.dumps({
                            "type": "control_message_received",
                            "message": f"Received: {text_data}",
                            "timestamp": time.time()
                        }))
                    
                    else:
                        logger.warning(f"Unknown message format: {message}")
                    
                except asyncio.TimeoutError:
                    # Potential turn end detected due to silence
                    current_time = time.time()
                    
                    if silence_start is None:
                        silence_start = current_time
                        logger.info("Silence detected - potential turn ending")
                        
                        await safe_send(websocket, json.dumps({
                            "type": "silence_detected",
                            "message": "Analyzing speech turn...",
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
                "timestamp": time.time()
            }))
            
            # Generate LLM response if user said something meaningful
            if user_input and len(user_input.strip()) > 2:
                # Notify client that LLM is generating response
                await safe_send(websocket, json.dumps({
                    "type": "llm_generating",
                    "turn_number": turn_number,
                    "message": "Generating AI response...",
                    "timestamp": time.time()
                }))
                
                # Generate streaming LLM response
                llm_response = await generate_llm_response_stream(
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
                    "message": "Turn completed but no meaningful speech detected",
                    "timestamp": time.time()
                }))
            
    except Exception as e:
        logger.error(f"Error processing turn {turn_number}: {e}")
        print(f"[TURN {turn_number} ERROR] {e}")

# Keep existing endpoints for backward compatibility
@app.websocket("/ws/turn-detection")
async def websocket_turn_detection(websocket: WebSocket):
    """Day 18: Turn detection without LLM responses - ALSO FIXED."""
    await websocket.accept()
    logger.info("Turn detection WebSocket connection established (Day 18 mode)")
    
    await safe_send(websocket, json.dumps({
        "type": "connection_established",
        "message": "Turn detection ready (Day 18 mode - no LLM responses)",
        "timestamp": time.time()
    }))
    
    session_id = f"turn_session_{int(time.time())}"
    audio_file_path = UPLOADS_DIR / f"{session_id}.webm"
    audio_chunks = []
    turn_count = 0
    
    try:
        with open(audio_file_path, "wb") as audio_file:
            silence_start = None
            
            while True:
                try:
                    # FIXED: Handle both text and binary messages
                    message = await asyncio.wait_for(websocket.receive(), timeout=2.0)
                    
                    if 'text' in message:
                        # Handle text control messages
                        text_data = message['text']
                        logger.info(f"Day 18 received text message: {text_data}")
                        
                        await safe_send(websocket, json.dumps({
                            "type": "text_echo",
                            "message": f"Day 18 received: {text_data}",
                            "timestamp": time.time()
                        }))
                        
                    elif 'bytes' in message:
                        # Handle binary audio data
                        audio_data = message['bytes']
                        audio_file.write(audio_data)
                        audio_chunks.append(audio_data)
                        
                        # Reset silence detection
                        silence_start = None
                        
                        await safe_send(websocket, json.dumps({
                            "type": "audio_received",
                            "chunk_size": len(audio_data),
                            "total_chunks": len(audio_chunks),
                            "timestamp": time.time()
                        }))
                    
                except asyncio.TimeoutError:
                    # Handle silence detection for turn ending
                    current_time = time.time()
                    
                    if silence_start is None:
                        silence_start = current_time
                        logger.info("Silence detected in Day 18 mode")
                        
                        await safe_send(websocket, json.dumps({
                            "type": "silence_detected",
                            "message": "Silence detected - turn ending soon...",
                            "timestamp": current_time
                        }))
                    
                    # Check if silence indicates turn end
                    silence_duration = current_time - silence_start
                    if silence_duration >= 2.0 and audio_chunks:
                        logger.info(f"Turn {turn_count} ended after {silence_duration:.1f}s silence")
                        
                        # Process the turn (Day 18 - just transcription, no LLM)
                        await process_turn_day18(websocket, audio_file_path, turn_count, session_id)
                        
                        turn_count += 1
                        audio_chunks.clear()
                        silence_start = None
                        
                        # Reset for next turn
                        audio_file_path = UPLOADS_DIR / f"{session_id}_turn_{turn_count}.webm"
                        break
                
                except WebSocketDisconnect:
                    logger.info("Day 18 turn detection WebSocket disconnected")
                    break
                except Exception as e:
                    logger.error(f"Error in Day 18 turn detection: {e}")
                    break
        
        # Process final turn if needed
        if audio_chunks:
            await process_turn_day18(websocket, audio_file_path, turn_count, session_id)
    
    except WebSocketDisconnect:
        logger.info("Day 18 turn detection WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in Day 18 turn detection handler: {e}")
    finally:
        logger.info(f"Day 18 session {session_id} completed with {turn_count + 1} turns")

async def process_turn_day18(websocket: WebSocket, audio_file_path: Path, 
                           turn_number: int, session_id: str):
    """Process turn for Day 18 - transcription only, no LLM response."""
    try:
        if not audio_file_path.exists() or audio_file_path.stat().st_size == 0:
            logger.warning(f"No audio data for Day 18 turn {turn_number}")
            return
        
        logger.info(f"Processing Day 18 turn {turn_number}: {audio_file_path}")
        
        # Transcribe the audio
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(str(audio_file_path))
        
        if transcript.error:
            logger.error(f"Day 18 transcription error: {transcript.error}")
            print(f"[DAY 18 TURN {turn_number} ERROR] {transcript.error}")
        else:
            user_input = transcript.text.strip()
            print(f"[DAY 18 TURN {turn_number} COMPLETE] {user_input}")
            
            # Send transcription result to client
            await safe_send(websocket, json.dumps({
                "type": "turn_complete",
                "turn_number": turn_number,
                "transcript": user_input,
                "confidence": getattr(transcript, 'confidence', 1.0),
                "mode": "day_18",
                "timestamp": time.time()
            }))
    
    except Exception as e:
        logger.error(f"Error processing Day 18 turn {turn_number}: {e}")
        print(f"[DAY 18 TURN {turn_number} ERROR] {e}")

@app.websocket("/ws/transcribe")
async def websocket_real_time_transcription(websocket: WebSocket):
    """Day 17: Real-time transcription endpoint."""
    await websocket.accept()
    logger.info("Real-time transcription WebSocket connection established")
    
    if not settings.assemblyai_api_key:
        await safe_send(websocket, json.dumps({
            "type": "error",
            "message": "AssemblyAI API key not configured",
            "timestamp": time.time()
        }))
        await websocket.close(code=1000, reason="AssemblyAI API key required")
        return
    
    session_id = f"session_{int(time.time())}"
    audio_file_path = UPLOADS_DIR / f"{session_id}.webm"
    
    try:
        await safe_send(websocket, json.dumps({
            "type": "connection_established",
            "message": "Ready for real-time transcription",
            "session_id": session_id,
            "timestamp": time.time()
        }))
        
        audio_chunks = []
        
        with open(audio_file_path, "wb") as audio_file:
            while True:
                try:
                    data = await websocket.receive_bytes()
                    audio_file.write(data)
                    audio_chunks.append(data)
                    
                    await safe_send(websocket, json.dumps({
                        "type": "audio_chunk_received",
                        "chunk_size": len(data),
                        "total_chunks": len(audio_chunks),
                        "timestamp": time.time()
                    }))
                    
                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected during audio collection")
                    break
                except Exception as e:
                    logger.error(f"Error receiving audio data: {e}")
                    break
        
        # Transcribe complete audio
        if audio_chunks and audio_file_path.exists():
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(str(audio_file_path))
            
            if transcript.error:
                print(f"[TRANSCRIPTION ERROR] {transcript.error}")
            else:
                print(f"[TRANSCRIPTION] {transcript.text}")
                
                await safe_send(websocket, json.dumps({
                    "type": "transcription",
                    "text": transcript.text,
                    "is_final": True,
                    "confidence": getattr(transcript, 'confidence', 1.0),
                    "session_id": session_id,
                    "timestamp": time.time()
                }))
        
    except WebSocketDisconnect:
        logger.info("Real-time transcription WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in real-time transcription: {e}")

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
        "message": "AI Voice Agent with Streaming LLM Responses (Day 19 Fixed)",
        "websocket_endpoints": {
            "llm_streaming": "/ws/llm-streaming",
            "turn_detection": "/ws/turn-detection",
            "real_time_transcription": "/ws/transcribe"
        },
        "api_endpoints": {
            "health": "/health",
            "docs": "/docs"
        },
        "version": "1.9.0",
        "day": 19
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=1)
