"""
AI Voice Agent - FastAPI Application with Real-Time Transcription (Day 17 Final)
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
    logger.info("AssemblyAI configured for streaming transcription")
else:
    logger.warning("AssemblyAI API key not found - streaming transcription disabled")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Voice Agent with Real-Time Transcription")
    yield
    logger.info("Shutting down AI Voice Agent application")

# Initialize FastAPI application
app = FastAPI(
    title="AI Voice Agent with Real-Time Transcription",
    description="Streaming audio with live AssemblyAI transcription",
    version="1.7.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# Enhanced timeout middleware
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    start_time = asyncio.get_event_loop().time()
    
    try:
        response = await asyncio.wait_for(call_next(request), timeout=180)
        process_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"{request.method} {request.url.path} completed in {process_time:.2f}s")
        return response
        
    except asyncio.TimeoutError:
        process_time = asyncio.get_event_loop().time() - start_time
        logger.error(f"{request.method} {request.url.path} timed out after {process_time:.2f}s")
        
        return JSONResponse(
            {
                "detail": "Request processing exceeded timeout limit. Please try again.",
                "error_type": "timeout",
                "fallback_message": "The request is taking too long to process. Please try again."
            }, 
            status_code=504
        )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ==================================================================
# WEBSOCKET HELPER FUNCTIONS
# ==================================================================

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
# DAY 17: REAL-TIME TRANSCRIPTION WITH ASSEMBLYAI
# ==================================================================

@app.websocket("/ws/transcribe")
async def websocket_real_time_transcription(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio transcription using AssemblyAI.
    
    This endpoint collects audio chunks, saves them to a file, and transcribes
    the complete audio when the recording session ends.
    """
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
    
    # Create temporary file for this session
    session_id = f"session_{int(time.time())}"
    audio_file_path = UPLOADS_DIR / f"{session_id}.webm"
    
    try:
        # Send connection confirmation
        await safe_send(websocket, json.dumps({
            "type": "connection_established",
            "message": "Ready for real-time transcription",
            "session_id": session_id,
            "timestamp": time.time()
        }))
        
        await safe_send(websocket, json.dumps({
            "type": "transcription_ready",
            "message": "Start speaking - audio will be transcribed when you stop!",
            "timestamp": time.time()
        }))
        
        # Collect audio data
        audio_chunks = []
        
        with open(audio_file_path, "wb") as audio_file:
            while True:
                try:
                    # Receive binary audio data from client
                    data = await websocket.receive_bytes()
                    
                    # Save to file
                    audio_file.write(data)
                    audio_chunks.append(data)
                    
                    # Send acknowledgment (safely)
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
        
        # Transcribe the complete audio file
        if audio_chunks and audio_file_path.exists():
            logger.info(f"Transcribing audio file: {audio_file_path}")
            
            try:
                # Initialize transcriber
                transcriber = aai.Transcriber()
                
                # Transcribe the audio file
                transcript = transcriber.transcribe(str(audio_file_path))
                
                if transcript.error:
                    logger.error(f"Transcription error: {transcript.error}")
                    print(f"[TRANSCRIPTION ERROR] {transcript.error}")
                    
                    await safe_send(websocket, json.dumps({
                        "type": "transcription_error",
                        "error": transcript.error,
                        "timestamp": time.time()
                    }))
                else:
                    # Print to server console (as requested for Day 17) ✅
                    print(f"[TRANSCRIPTION] {transcript.text}")
                    logger.info(f"Transcription completed: {transcript.text}")
                    
                    # Only send response if WebSocket is still open
                    await safe_send(websocket, json.dumps({
                        "type": "transcription",
                        "text": transcript.text,
                        "is_final": True,
                        "confidence": getattr(transcript, 'confidence', 1.0),
                        "session_id": session_id,
                        "timestamp": time.time()
                    }))
                    
            except Exception as e:
                logger.error(f"Error during transcription: {e}")
                print(f"[TRANSCRIPTION ERROR] {e}")
                
                await safe_send(websocket, json.dumps({
                    "type": "error",
                    "message": f"Transcription failed: {str(e)}",
                    "timestamp": time.time()
                }))
        
    except WebSocketDisconnect:
        logger.info("Real-time transcription WebSocket disconnected")
        
    except Exception as e:
        logger.error(f"Error in real-time transcription: {e}")
        print(f"[TRANSCRIPTION ERROR] {e}")
        
    finally:
        # Clean up temporary file
        try:
            if audio_file_path.exists():
                file_size = audio_file_path.stat().st_size
                logger.info(f"Session {session_id} completed. Audio file: {file_size} bytes")
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")

# ==================================================================
# DAY 16: AUDIO STREAMING ENDPOINT (COMPATIBILITY)
# ==================================================================

@app.websocket("/ws/audio")
async def websocket_audio_stream(websocket: WebSocket):
    """Day 16: Basic audio streaming endpoint."""
    await websocket.accept()
    timestamp = int(time.time())
    filename = f"audio_stream_{timestamp}.webm"
    file_path = UPLOADS_DIR / filename
    
    try:
        with open(file_path, "wb") as audio_file:
            logger.info(f"Started recording audio stream to: {file_path}")
            
            while True:
                data = await websocket.receive_bytes()
                audio_file.write(data)
                
                await safe_send(websocket, json.dumps({
                    "type": "chunk_received",
                    "chunk_size": len(data),
                    "file": filename,
                    "timestamp": time.time()
                }))
                
    except WebSocketDisconnect:
        logger.info(f"Audio streaming disconnected. File saved: {file_path}")
        
        # Get file size for logging
        try:
            file_size = file_path.stat().st_size
            logger.info(f"Final audio file size: {file_size} bytes")
        except Exception as e:
            logger.error(f"Error checking file size: {e}")

# ==================================================================
# DAY 15: TEXT WEBSOCKET ENDPOINT (COMPATIBILITY)
# ==================================================================

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
                "transcription": "/ws/transcribe",
                "audio_streaming": "/ws/audio",
                "text_messaging": "/ws"
            }
        }
        await websocket.send_text(json.dumps(welcome_message))
        
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received WebSocket message: {data}")
            
            try:
                message_data = json.loads(data)
                if message_data.get("type") == "ping":
                    response = {
                        "type": "pong",
                        "message": "pong",
                        "timestamp": time.time()
                    }
                else:
                    response = {
                        "type": "echo",
                        "original_message": data,
                        "echo": f"Echo: {data}",
                        "timestamp": time.time()
                    }
            except json.JSONDecodeError:
                response = {
                    "type": "echo",
                    "original_message": data,
                    "echo": f"Echo: {data}",
                    "timestamp": time.time()
                }
            
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

# ==================================================================
# API ENDPOINTS
# ==================================================================

@app.get("/audio/files")
async def list_audio_files():
    """List all saved audio files."""
    try:
        files = []
        for file_path in UPLOADS_DIR.glob("*.webm"):
            stat = file_path.stat()
            files.append({
                "filename": file_path.name,
                "size": stat.st_size,
                "created": stat.st_ctime,
                "modified": stat.st_mtime
            })
        
        return {
            "files": sorted(files, key=lambda x: x["created"], reverse=True),
            "total_files": len(files),
            "upload_directory": str(UPLOADS_DIR)
        }
    except Exception as e:
        logger.error(f"Error listing audio files: {e}")
        return {"error": str(e), "files": []}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.7.0",
        "services": {
            "assemblyai": "available" if settings.assemblyai_api_key else "unavailable"
        }
    }

# Static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Serve frontend
@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the main frontend application."""
    index_path = TEMPLATES_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    else:
        return JSONResponse({
            "message": "AI Voice Agent with Real-Time Transcription",
            "websocket_endpoints": {
                "real_time_transcription": "/ws/transcribe",
                "audio_streaming": "/ws/audio",
                "text_messaging": "/ws"
            },
            "api_endpoints": {
                "audio_files": "/audio/files",
                "health": "/health",
                "docs": "/docs"
            },
            "version": "1.7.0"
        })

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception in {request.method} {request.url.path}: {exc}")
    
    return JSONResponse(
        {
            "detail": "An unexpected error occurred",
            "error_type": "internal_server_error",
            "fallback_message": "Something went wrong. Please try again later."
        },
        status_code=500
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=1)
