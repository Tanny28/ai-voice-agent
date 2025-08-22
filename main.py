"""
AI Voice Agent - Day 21: Stream Base64 Audio Data to Client via WebSockets (FIXED)
"""

import asyncio
import json
import time
import base64
import aiohttp
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import assemblyai as aai

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

try:
    from murf import Murf
    MURF_AVAILABLE = True
except ImportError:
    MURF_AVAILABLE = False
    print("Warning: Murf SDK not installed. Run: pip install murf")

from app.config import settings
from app.core.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create required directories
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

# Configure services
if settings.assemblyai_api_key:
    aai.settings.api_key = settings.assemblyai_api_key
    logger.info("AssemblyAI configured for streaming transcription")

if GEMINI_AVAILABLE and hasattr(settings, 'gemini_api_key') and settings.gemini_api_key:
    try:
        gemini_client = genai.Client(api_key=settings.gemini_api_key)
        logger.info("Google Gemini configured for streaming LLM responses")
    except Exception as e:
        gemini_client = None
else:
    gemini_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Voice Agent with Audio Streaming")
    yield
    logger.info("Shutting down AI Voice Agent application")

app = FastAPI(
    title="AI Voice Agent with Audio Streaming",
    description="Day 21: Stream base64 audio data to client via WebSockets",
    version="2.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class MurfTTSClient:
    """Murf TTS client with improved streaming and keepalive for large files."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        if MURF_AVAILABLE:
            self.client = Murf(api_key=api_key)
        else:
            self.client = None
    
    async def download_audio_as_base64(self, audio_url: str):
        """Download audio from URL and convert to base64."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(audio_url) as response:
                    if response.status == 200:
                        audio_data = await response.read()
                        base64_audio = base64.b64encode(audio_data).decode('utf-8')
                        return base64_audio
                    else:
                        print(f"[MURF ERROR] Failed to download audio: HTTP {response.status}")
                        return None
        except Exception as e:
            print(f"[MURF ERROR] Download failed: {e}")
            return None
    
    def chunk_base64_audio(self, base64_audio: str, chunk_size: int = 4096):
        """Split base64 audio into chunks for streaming - INCREASED SIZE."""
        chunks = []
        for i in range(0, len(base64_audio), chunk_size):
            chunk = base64_audio[i:i + chunk_size]
            chunks.append(chunk)
        return chunks
    
    async def synthesize_and_stream(self, text: str, websocket: WebSocket = None):
        """Synthesize text and optionally stream chunks to WebSocket client."""
        if not self.client:
            print(f"[MURF ERROR] Murf SDK not available")
            return None
        
        try:
            print(f"[MURF SENDING] Text: {text[:60]}...")
            
            # Generate speech using Murf
            response = self.client.text_to_speech.generate(
                text=text,
                voice_id="en-US-natalie",
                format="MP3",
                sample_rate=44100.0
            )
            
            print(f"[MURF RESPONSE] Audio generated successfully!")
            
            # Get base64 audio
            base64_audio = None
            if hasattr(response, 'encoded_audio') and response.encoded_audio:
                base64_audio = response.encoded_audio
            elif hasattr(response, 'audio_file') and response.audio_file:
                print(f"[MURF DOWNLOADING] Audio from URL...")
                base64_audio = await self.download_audio_as_base64(response.audio_file)
            
            if not base64_audio:
                print(f"[MURF ERROR] No audio data available")
                return None
            
            print(f"[MURF SUCCESS] Generated base64 audio: {len(base64_audio)} characters")
            
            # DAY 21: Stream audio chunks to client if WebSocket provided
            if websocket:
                await self.stream_audio_chunks(base64_audio, websocket)
            
            return base64_audio
                
        except Exception as e:
            print(f"[MURF ERROR] {e}")
            return None
    
    async def stream_audio_chunks(self, base64_audio: str, websocket: WebSocket):
        """Stream base64 audio as chunks with keepalive for large files - FIXED."""
        try:
            # FIXED: Larger chunks and keepalive for performance
            chunks = self.chunk_base64_audio(base64_audio, chunk_size=4096)  # Increased from 1024
            print(f"[STREAMING] Sending {len(chunks)} audio chunks to client (4KB chunks)...")
            
            for i, chunk in enumerate(chunks):
                # Send chunk to client
                chunk_message = {
                    "type": "audio_chunk",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "data": chunk,
                    "is_final": i == len(chunks) - 1
                }
                
                await websocket.send_text(json.dumps(chunk_message))
                print(f"[STREAMING] Sent chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)")
                
                # FIXED: Send keepalive every 50 chunks to prevent timeout
                if i > 0 and i % 50 == 0:
                    keepalive = {"type": "keepalive", "chunk_progress": i}
                    await websocket.send_text(json.dumps(keepalive))
                    print(f"[KEEPALIVE] Sent at chunk {i}")
                
                # FIXED: Reduced delay for faster streaming
                await asyncio.sleep(0.05)  # Reduced from 0.1
            
            print(f"[STREAMING] ✅ All {len(chunks)} chunks sent to client")
                
        except Exception as e:
            print(f"[STREAMING ERROR] {e}")

# Initialize Murf client
murf_client = None
if MURF_AVAILABLE and hasattr(settings, 'murf_api_key') and settings.murf_api_key:
    murf_client = MurfTTSClient(settings.murf_api_key)
    logger.info("Murf TTS client initialized")

async def generate_llm_response_with_streaming_tts(prompt: str, websocket: WebSocket = None):
    """Generate LLM response and stream TTS audio to client."""
    if not gemini_client:
        response_text = f"Hello! You said '{prompt}'. This is Day 21 audio streaming demo!"
    else:
        try:
            response = await gemini_client.aio.models.generate_content(
                model="gemini-1.5-flash",
                contents=[{"parts": [{"text": prompt}]}],
                config={"temperature": 0.7, "max_output_tokens": 256}  # Reduced to avoid huge responses
            )
            response_text = response.text if hasattr(response, 'text') and response.text else f"I understand you said: '{prompt}'. This is Day 21!"
        except Exception as e:
            logger.warning(f"Gemini error: {e}")
            response_text = f"Hello! You said '{prompt}'. This is Day 21 audio streaming with Murf TTS!"
    
    # Print LLM response
    print(f"\n[LLM RESPONSE] {response_text}")
    
    # Generate TTS and stream to client
    if murf_client:
        print(f"[TTS STREAMING] Converting to speech and streaming to client...")
        await murf_client.synthesize_and_stream(response_text, websocket)
    else:
        print(f"[TTS DISABLED] No Murf client available")
    
    return response_text

@app.websocket("/ws/audio-streaming")
async def websocket_audio_streaming(websocket: WebSocket):
    """Day 21: WebSocket endpoint that streams base64 audio chunks to client."""
    await websocket.accept()
    logger.info("Audio streaming WebSocket connection established")
    
    session_id = f"stream_session_{int(time.time())}"
    
    try:
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "message": "Audio Streaming ready - send text to get TTS audio chunks!",
            "session_id": session_id,
            "timestamp": time.time()
        }))
        
        while True:
            try:
                # Receive text from client
                message = await websocket.receive_text()
                
                # Handle different message types
                if message.startswith('{'):
                    data = json.loads(message)
                    
                    # Skip acknowledgment messages to avoid infinite loop
                    if data.get("type") == "chunk_acknowledgment":
                        print(f"[CLIENT ACK] Received acknowledgment for chunk {data.get('chunk_index', 'unknown')}")
                        continue
                    
                    # Handle text input
                    if data.get("type") == "text_input":
                        text_input = data.get("text", "")
                    elif "text" in data:
                        text_input = data["text"]
                    else:
                        continue
                else:
                    # Plain text message
                    text_input = message
                
                print(f"[CLIENT INPUT] Received: {text_input}")
                
                # Generate LLM response and stream TTS audio
                await generate_llm_response_with_streaming_tts(text_input, websocket)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in audio streaming: {e}")
                try:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Error: {e}",
                        "timestamp": time.time()
                    }))
                except:
                    break
        
    except WebSocketDisconnect:
        logger.info("Audio streaming WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info(f"Audio streaming session {session_id} ended")

@app.get("/test-streaming")
async def test_streaming():
    """Test endpoint for Day 21 audio streaming."""
    if not murf_client:
        return {"error": "Murf client not configured"}
    
    try:
        test_text = "Hello! This is Day 21 of the AI Voice Agents challenge. Testing audio streaming with base64 chunks!"
        
        print(f"\n[DAY 21 TEST] 🎵 Testing audio streaming...")
        base64_audio = await murf_client.synthesize_and_stream(test_text)
        
        if base64_audio:
            chunks = murf_client.chunk_base64_audio(base64_audio)
            return {
                "status": "SUCCESS",
                "message": "✅ Audio streaming test completed!",
                "text_sent": test_text,
                "total_chunks": len(chunks),
                "audio_length": len(base64_audio),
                "chunk_size": 4096,
                "sample_chunk": chunks[0][:100] + "..." if chunks else None
            }
        else:
            return {"error": "Failed to generate audio"}
            
    except Exception as e:
        return {"error": f"Test failed: {e}"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.1.0",
        "day": 21,
        "features": ["audio_streaming", "base64_chunks", "murf_tts", "keepalive"],
        "services": {
            "murf": "available" if murf_client else "unavailable",
            "gemini": "available" if gemini_client else "unavailable"
        }
    }

@app.get("/")
async def root():
    """Day 21: Audio Streaming Demo."""
    return {
        "message": "🎵 Day 21: Audio Streaming to Client via WebSockets (FIXED)",
        "endpoints": {
            "audio_streaming": "/ws/audio-streaming",
            "test_streaming": "/test-streaming",
            "health": "/health"
        },
        "improvements": [
            "✅ 4KB chunks instead of 1KB for better performance",
            "✅ Keepalive messages every 50 chunks prevent timeout",
            "✅ Reduced streaming delay (50ms instead of 100ms)",
            "✅ Better handling of acknowledgment messages",
            "✅ Shorter LLM responses to reduce streaming time"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=1)
