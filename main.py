"""
AI Voice Agent - FastAPI Application with Turn Detection (Day 18)
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

from app.config import settings
from app.core.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create required directories
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)
STATIC_DIR = Path("static")
TEMPLATES_DIR = Path("templates")

# Configure AssemblyAI
if settings.assemblyai_api_key:
    aai.settings.api_key = settings.assemblyai_api_key
    logger.info("AssemblyAI configured for streaming transcription with turn detection")
else:
    logger.warning("AssemblyAI API key not found - streaming transcription disabled")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Voice Agent with Turn Detection")
    yield
    logger.info("Shutting down AI Voice Agent application")

# Initialize FastAPI application
app = FastAPI(
    title="AI Voice Agent with Turn Detection",
    description="Real-time transcription with intelligent turn detection",
    version="1.8.0",
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
# DAY 18: TURN DETECTION WITH ASSEMBLYAI STREAMING
# ==================================================================

@app.websocket("/ws/turn-detection")
async def websocket_turn_detection(websocket: WebSocket):
    """
    WebSocket endpoint for real-time transcription with turn detection.
    
    This endpoint uses AssemblyAI's streaming API to detect when users stop speaking
    and sends notifications to the client for intelligent conversation handling.
    """
    await websocket.accept()
    logger.info("Turn detection WebSocket connection established")
    
    if not settings.assemblyai_api_key:
        await safe_send(websocket, json.dumps({
            "type": "error",
            "message": "AssemblyAI API key not configured",
            "timestamp": time.time()
        }))
        await websocket.close(code=1000, reason="AssemblyAI API key required")
        return
    
    session_id = f"turn_session_{int(time.time())}"
    audio_file_path = UPLOADS_DIR / f"{session_id}.webm"
    
    try:
        # Send connection confirmation
        await safe_send(websocket, json.dumps({
            "type": "connection_established",
            "message": "Turn detection ready - start speaking!",
            "session_id": session_id,
            "timestamp": time.time()
        }))
        
        # Initialize AssemblyAI transcriber with turn detection
        transcriber = aai.Transcriber()
        
        # Collect audio chunks for turn detection analysis
        audio_chunks = []
        turn_count = 0
        
        await safe_send(websocket, json.dumps({
            "type": "turn_detection_ready",
            "message": "Listening for speech turns - speak naturally!",
            "timestamp": time.time()
        }))
        
        with open(audio_file_path, "wb") as audio_file:
            silence_start = None
            last_audio_time = time.time()
            
            while True:
                try:
                    # Receive audio data with timeout for turn detection
                    data = await asyncio.wait_for(websocket.receive_bytes(), timeout=2.0)
                    
                    # Write audio data
                    audio_file.write(data)
                    audio_chunks.append(data)
                    last_audio_time = time.time()
                    
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
                        
                        # Process accumulated audio for this turn
                        await process_turn(websocket, transcriber, audio_file_path, 
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
                    logger.info("WebSocket disconnected during turn detection")
                    break
                except Exception as e:
                    logger.error(f"Error in turn detection: {e}")
                    break
        
        # Process final turn if any audio was collected
        if audio_chunks:
            await process_turn(websocket, transcriber, audio_file_path, 
                             turn_count, session_id)
        
    except WebSocketDisconnect:
        logger.info("Turn detection WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in turn detection: {e}")
        print(f"[TURN DETECTION ERROR] {e}")
    finally:
        logger.info(f"Turn detection session {session_id} completed with {turn_count + 1} turns")

async def process_turn(websocket: WebSocket, transcriber: aai.Transcriber, 
                      audio_file_path: Path, turn_number: int, session_id: str):
    """Process a detected turn and transcribe the audio."""
    try:
        if not audio_file_path.exists() or audio_file_path.stat().st_size == 0:
            logger.warning(f"No audio data for turn {turn_number}")
            return
        
        logger.info(f"Processing turn {turn_number}: {audio_file_path}")
        
        # Notify client that turn is being processed
        await safe_send(websocket, json.dumps({
            "type": "turn_processing",
            "turn_number": turn_number,
            "message": f"Processing turn {turn_number}...",
            "timestamp": time.time()
        }))
        
        # Transcribe the turn audio
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
            print(f"[TURN {turn_number} COMPLETE] {transcript.text}")
            logger.info(f"Turn {turn_number} transcribed: {transcript.text}")
            
            # Send turn completion to client
            await safe_send(websocket, json.dumps({
                "type": "turn_complete",
                "turn_number": turn_number,
                "transcript": transcript.text,
                "confidence": getattr(transcript, 'confidence', 1.0),
                "session_id": session_id,
                "timestamp": time.time(),
                "end_of_turn": True
            }))
            
    except Exception as e:
        logger.error(f"Error processing turn {turn_number}: {e}")
        print(f"[TURN {turn_number} ERROR] {e}")

# ==================================================================
# EXISTING ENDPOINTS (DAY 15-17 COMPATIBILITY)
# ==================================================================

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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Day 15: Basic WebSocket endpoint for text messaging."""
    await websocket.accept()
    
    try:
        welcome_message = {
            "type": "welcome",
            "message": "Connected to AI Voice Agent WebSocket!",
            "timestamp": time.time(),
            "endpoints": {
                "turn_detection": "/ws/turn-detection",
                "transcription": "/ws/transcribe",
                "text_messaging": "/ws"
            }
        }
        await websocket.send_text(json.dumps(welcome_message))
        
        while True:
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                if message_data.get("type") == "ping":
                    response = {"type": "pong", "message": "pong", "timestamp": time.time()}
                else:
                    response = {"type": "echo", "echo": f"Echo: {data}", "timestamp": time.time()}
            except json.JSONDecodeError:
                response = {"type": "echo", "echo": f"Echo: {data}", "timestamp": time.time()}
            
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.8.0",
        "features": ["turn_detection", "real_time_transcription", "websockets"],
        "services": {
            "assemblyai": "available" if settings.assemblyai_api_key else "unavailable"
        }
    }

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the main frontend application."""
    return JSONResponse({
        "message": "AI Voice Agent with Turn Detection",
        "websocket_endpoints": {
            "turn_detection": "/ws/turn-detection",
            "real_time_transcription": "/ws/transcribe",
            "text_messaging": "/ws"
        },
        "api_endpoints": {
            "health": "/health",
            "docs": "/docs"
        },
        "version": "1.8.0"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=1)
