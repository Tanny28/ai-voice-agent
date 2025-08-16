"""
AI Voice Agent - FastAPI Application with WebSocket Support (Day 15)
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
from app.api.endpoints import agent, health

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create required directories
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

STATIC_DIR = Path("static")
TEMPLATES_DIR = Path("templates")

# WebSocket Connection Manager
class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_data: Dict[WebSocket, dict] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str = None):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_data[websocket] = {
            "client_id": client_id or f"client_{len(self.active_connections)}",
            "connected_at": time.time(),
            "message_count": 0
        }
        logger.info(f"WebSocket connection established: {self.connection_data[websocket]['client_id']}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            client_info = self.connection_data.get(websocket, {})
            self.active_connections.remove(websocket)
            del self.connection_data[websocket]
            logger.info(f"WebSocket connection closed: {client_info.get('client_id', 'unknown')}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        try:
            await websocket.send_text(message)
            # Update message count
            if websocket in self.connection_data:
                self.connection_data[websocket]["message_count"] += 1
        except Exception as e:
            logger.error(f"Error sending message to WebSocket: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        """Send a message to all connected WebSocket clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
                if connection in self.connection_data:
                    self.connection_data[connection]["message_count"] += 1
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
    
    def get_connection_stats(self) -> dict:
        """Get statistics about current connections."""
        return {
            "active_connections": len(self.active_connections),
            "connections_info": [
                {
                    "client_id": data["client_id"],
                    "connected_at": data["connected_at"],
                    "message_count": data["message_count"],
                    "duration": time.time() - data["connected_at"]
                }
                for data in self.connection_data.values()
            ]
        }

# Global connection manager
manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting AI Voice Agent application with WebSocket support")
    logger.info(f"Configuration: Debug={settings.debug}, Log Level={settings.log_level}")
    yield
    # Shutdown
    logger.info("Shutting down AI Voice Agent application")
    # Close all WebSocket connections
    for connection in manager.active_connections.copy():
        try:
            await connection.close(code=1000, reason="Server shutdown")
        except:
            pass

# Initialize FastAPI application
app = FastAPI(
    title="AI Voice Agent with WebSocket",
    description="A production-ready conversational AI system with WebSocket support",
    version="1.5.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# Enhanced timeout middleware with logging
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    """Global request timeout middleware with comprehensive error handling."""
    start_time = asyncio.get_event_loop().time()
    
    try:
        response = await asyncio.wait_for(call_next(request), timeout=180)
        
        # Log request completion time
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
    allow_origins=["*"] if settings.debug else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include existing routers
app.include_router(agent.router)
app.include_router(health.router)

# ==================================================================
# DAY 15: WEBSOCKET ENDPOINTS
# ==================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for real-time communication.
    
    This endpoint accepts WebSocket connections and provides echo functionality.
    Clients can send messages and receive echoed responses.
    """
    await manager.connect(websocket)
    
    try:
        # Send welcome message
        welcome_message = {
            "type": "welcome",
            "message": "Connected to AI Voice Agent WebSocket!",
            "timestamp": time.time(),
            "connection_id": manager.connection_data[websocket]["client_id"]
        }
        await manager.send_personal_message(json.dumps(welcome_message), websocket)
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            logger.info(f"Received WebSocket message: {data}")
            
            try:
                # Try to parse as JSON
                message_data = json.loads(data)
                response = await handle_websocket_message(websocket, message_data)
            except json.JSONDecodeError:
                # Handle plain text messages
                response = {
                    "type": "echo",
                    "original_message": data,
                    "echo": f"Echo: {data}",
                    "timestamp": time.time(),
                    "connection_id": manager.connection_data[websocket]["client_id"]
                }
            
            # Send response back to client
            await manager.send_personal_message(json.dumps(response), websocket)
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

async def handle_websocket_message(websocket: WebSocket, message_data: dict) -> dict:
    """
    Handle structured WebSocket messages.
    
    Args:
        websocket: The WebSocket connection
        message_data: Parsed JSON message data
        
    Returns:
        Response dictionary to send back to client
    """
    message_type = message_data.get("type", "unknown")
    client_id = manager.connection_data[websocket]["client_id"]
    
    if message_type == "ping":
        return {
            "type": "pong",
            "message": "pong",
            "timestamp": time.time(),
            "connection_id": client_id
        }
    
    elif message_type == "echo":
        return {
            "type": "echo_response",
            "original_message": message_data.get("message", ""),
            "echo": f"Echo: {message_data.get('message', '')}",
            "timestamp": time.time(),
            "connection_id": client_id
        }
    
    elif message_type == "status":
        stats = manager.get_connection_stats()
        return {
            "type": "status_response",
            "connection_stats": stats,
            "server_time": time.time(),
            "connection_id": client_id
        }
    
    elif message_type == "broadcast":
        # Broadcast message to all connected clients
        broadcast_msg = {
            "type": "broadcast_message",
            "message": message_data.get("message", ""),
            "from": client_id,
            "timestamp": time.time()
        }
        await manager.broadcast(json.dumps(broadcast_msg))
        
        return {
            "type": "broadcast_sent",
            "message": f"Broadcasted message to {len(manager.active_connections)} clients",
            "timestamp": time.time(),
            "connection_id": client_id
        }
    
    else:
        return {
            "type": "error",
            "message": f"Unknown message type: {message_type}",
            "supported_types": ["ping", "echo", "status", "broadcast"],
            "timestamp": time.time(),
            "connection_id": client_id
        }

@app.get("/ws/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics."""
    stats = manager.get_connection_stats()
    return {
        "websocket_stats": stats,
        "server_time": time.time()
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
        return JSONResponse(
            {
                "message": "AI Voice Agent API with WebSocket support", 
                "docs": "/docs",
                "websocket": "/ws",
                "websocket_stats": "/ws/stats"
            },
            status_code=200
        )

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
    
    # Run with appropriate configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        workers=1,
        log_level=settings.log_level.lower()
    )
