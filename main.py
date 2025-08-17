"""
AI Voice Agent - FastAPI Application with Audio Streaming (Day 16)
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

# WebSocket Connection Manager (existing from Day 15)
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_data: Dict[WebSocket, dict] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_data[websocket] = {
            "client_id": client_id or f"client_{len(self.active_connections)}",
            "connected_at": time.time(),
            "message_count": 0
        }
        logger.info(f"WebSocket connection established: {self.connection_data[websocket]['client_id']}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            client_info = self.connection_data.get(websocket, {})
            self.active_connections.remove(websocket)
            del self.connection_data[websocket]
            logger.info(f"WebSocket connection closed: {client_info.get('client_id', 'unknown')}")

manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Voice Agent application with Audio Streaming")
    yield
    logger.info("Shutting down AI Voice Agent application")

# Initialize FastAPI application
app = FastAPI(
    title="AI Voice Agent with Audio Streaming",
    description="Real-time audio streaming with WebSocket support",
    version="1.6.0",
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

# ==================================================================
# DAY 16: AUDIO STREAMING WEBSOCKET ENDPOINTS
# ==================================================================

@app.websocket("/ws/audio")
async def websocket_audio_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming.
    
    Receives binary audio data chunks from client and saves to file.
    """
    await websocket.accept()
    logger.info("Audio streaming WebSocket connection established")
    
    # Create unique filename with timestamp
    timestamp = int(time.time())
    filename = f"audio_stream_{timestamp}.webm"
    file_path = UPLOADS_DIR / filename
    
    try:
        with open(file_path, "wb") as audio_file:
            logger.info(f"Started recording audio stream to: {file_path}")
            
            while True:
                # Receive binary audio data from client
                data = await websocket.receive_bytes()
                
                # Write audio chunk to file immediately
                audio_file.write(data)
                
                # Optional: Send acknowledgment back to client
                await websocket.send_text(json.dumps({
                    "type": "chunk_received",
                    "chunk_size": len(data),
                    "timestamp": time.time(),
                    "file": filename
                }))
                
    except WebSocketDisconnect:
        logger.info(f"Audio streaming WebSocket disconnected. Audio saved to: {file_path}")
        
        # Get file size for logging
        try:
            file_size = file_path.stat().st_size
            logger.info(f"Final audio file size: {file_size} bytes")
        except Exception as e:
            logger.error(f"Error checking file size: {e}")
            
    except Exception as e:
        logger.error(f"Error in audio streaming: {e}")
        
        # Clean up partial file if error occurred
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Cleaned up partial file: {file_path}")
        except:
            pass

@app.websocket("/ws/audio-with-metadata")
async def websocket_audio_stream_with_metadata(websocket: WebSocket):
    """
    Enhanced audio streaming endpoint that handles both audio data and metadata.
    """
    await websocket.accept()
    logger.info("Enhanced audio streaming WebSocket connection established")
    
    session_id = f"session_{int(time.time())}"
    base_filename = f"audio_stream_{session_id}"
    current_file = None
    current_file_handle = None
    chunk_count = 0
    
    try:
        while True:
            # Receive message (could be binary audio or text metadata)
            message = await websocket.receive()
            
            if "bytes" in message:
                # Binary audio data
                audio_data = message["bytes"]
                
                # Create new file if needed
                if current_file_handle is None:
                    filename = f"{base_filename}_{chunk_count}.webm"
                    current_file = UPLOADS_DIR / filename
                    current_file_handle = open(current_file, "wb")
                    logger.info(f"Started new audio file: {current_file}")
                
                # Write audio data
                current_file_handle.write(audio_data)
                chunk_count += 1
                
                # Send acknowledgment
                await websocket.send_text(json.dumps({
                    "type": "audio_chunk_received",
                    "chunk_number": chunk_count,
                    "chunk_size": len(audio_data),
                    "file": current_file.name,
                    "timestamp": time.time()
                }))
                
            elif "text" in message:
                # Text metadata/control message
                try:
                    control_data = json.loads(message["text"])
                    
                    if control_data.get("type") == "stop_recording":
                        # Close current file
                        if current_file_handle:
                            current_file_handle.close()
                            current_file_handle = None
                            
                            file_size = current_file.stat().st_size if current_file else 0
                            
                            await websocket.send_text(json.dumps({
                                "type": "recording_stopped",
                                "file": current_file.name if current_file else None,
                                "final_size": file_size,
                                "total_chunks": chunk_count,
                                "timestamp": time.time()
                            }))
                            
                            logger.info(f"Recording stopped. File: {current_file}, Size: {file_size} bytes")
                            
                    elif control_data.get("type") == "start_recording":
                        # Reset for new recording
                        if current_file_handle:
                            current_file_handle.close()
                        current_file_handle = None
                        chunk_count = 0
                        
                        await websocket.send_text(json.dumps({
                            "type": "recording_started",
                            "session_id": session_id,
                            "timestamp": time.time()
                        }))
                        
                        logger.info(f"New recording started for session: {session_id}")
                        
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON metadata received: {message['text']}")
                    
    except WebSocketDisconnect:
        logger.info(f"Enhanced audio streaming WebSocket disconnected")
        
    except Exception as e:
        logger.error(f"Error in enhanced audio streaming: {e}")
        
    finally:
        # Clean up file handle
        if current_file_handle:
            current_file_handle.close()
            logger.info(f"Closed audio file handle for session: {session_id}")

# Existing Day 15 WebSocket endpoint (keep for backward compatibility)
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Original WebSocket endpoint from Day 15 for text messaging."""
    await manager.connect(websocket)
    
    try:
        welcome_message = {
            "type": "welcome",
            "message": "Connected to AI Voice Agent WebSocket!",
            "timestamp": time.time(),
            "connection_id": manager.connection_data[websocket]["client_id"]
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
                        "timestamp": time.time(),
                        "connection_id": manager.connection_data[websocket]["client_id"]
                    }
                else:
                    response = {
                        "type": "echo",
                        "original_message": data,
                        "echo": f"Echo: {data}",
                        "timestamp": time.time(),
                        "connection_id": manager.connection_data[websocket]["client_id"]
                    }
            except json.JSONDecodeError:
                response = {
                    "type": "echo",
                    "original_message": data,
                    "echo": f"Echo: {data}",
                    "timestamp": time.time(),
                    "connection_id": manager.connection_data[websocket]["client_id"]
                }
            
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Health check and file listing endpoints
@app.get("/audio/files")
async def list_audio_files():
    """List all saved audio files."""
    try:
        files = []
        for file_path in UPLOADS_DIR.glob("audio_stream_*.webm"):
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

# Serve frontend
@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the main frontend application."""
    return JSONResponse({
        "message": "AI Voice Agent with Audio Streaming",
        "websocket_endpoints": {
            "text_messaging": "/ws",
            "audio_streaming": "/ws/audio",
            "enhanced_audio": "/ws/audio-with-metadata"
        },
        "api_endpoints": {
            "audio_files": "/audio/files",
            "docs": "/docs"
        }
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=1)
