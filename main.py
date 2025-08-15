"""
AI Voice Agent - FastAPI Application Entry Point

A production-ready conversational AI system with voice-to-voice interactions,
chat history, and comprehensive error handling.
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting AI Voice Agent application")
    logger.info(f"Configuration: Debug={settings.debug}, Log Level={settings.log_level}")
    yield
    # Shutdown
    logger.info("Shutting down AI Voice Agent application")


# Initialize FastAPI application
app = FastAPI(
    title="AI Voice Agent",
    description="A production-ready conversational AI system with voice-to-voice interactions",
    version="1.0.0",
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
    allow_headers=["*"],
)

# Include routers
app.include_router(agent.router)
app.include_router(health.router)

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
            {"message": "AI Voice Agent API", "docs": "/docs"},
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
