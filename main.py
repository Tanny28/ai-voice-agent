"""
AI Voice Agent - Day 20: Murf TTS Integration (FIXED - Downloads and converts to base64)
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
    logger.info("Starting AI Voice Agent with Murf TTS")
    yield
    logger.info("Shutting down AI Voice Agent application")

app = FastAPI(
    title="AI Voice Agent with Murf TTS",
    description="Day 20: Speech-to-Text with LLM responses converted to speech via Murf",
    version="2.0.0",
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
    """Fixed Murf TTS client that downloads audio and converts to base64."""
    
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
    
    async def synthesize_text(self, text: str):
        """Synthesize text to speech using Murf API and return base64."""
        if not self.client:
            print(f"[MURF ERROR] Murf SDK not available")
            return None
        
        try:
            print(f"[MURF SENDING] Text: {text[:60]}...")
            
            # Use Murf SDK to generate speech
            response = self.client.text_to_speech.generate(
                text=text,
                voice_id="en-US-natalie",
                format="MP3",
                sample_rate=44100.0
            )
            
            print(f"[MURF RESPONSE] Audio generated successfully!")
            print(f"[MURF DETAILS] Length: {response.audio_length_in_seconds}s, Characters: {response.consumed_character_count}")
            
            # Check if we have a direct encoded audio
            if hasattr(response, 'encoded_audio') and response.encoded_audio:
                base64_audio = response.encoded_audio
                print(f"[MURF AUDIO BASE64] {base64_audio[:100]}...")
                print(f"[FULL BASE64 AUDIO]\n{base64_audio}")
                return base64_audio
            
            # If no direct encoded audio, download from URL
            elif hasattr(response, 'audio_file') and response.audio_file:
                print(f"[MURF DOWNLOADING] Audio from URL: {response.audio_file[:50]}...")
                base64_audio = await self.download_audio_as_base64(response.audio_file)
                
                if base64_audio:
                    print(f"[MURF AUDIO BASE64] {base64_audio[:100]}...")
                    print(f"[FULL BASE64 AUDIO]\n{base64_audio}")
                    return base64_audio
                else:
                    print(f"[MURF ERROR] Failed to download and convert audio")
                    return None
            else:
                print(f"[MURF ERROR] No audio data in response")
                return None
                
        except Exception as e:
            print(f"[MURF ERROR] {e}")
            return None

# Initialize Murf client
murf_client = None
if MURF_AVAILABLE and hasattr(settings, 'murf_api_key') and settings.murf_api_key:
    murf_client = MurfTTSClient(settings.murf_api_key)
    logger.info("Murf TTS client initialized")
else:
    logger.warning("Murf TTS client not available")

async def generate_llm_response_with_tts(prompt: str, session_id: str = "default"):
    """Generate LLM response and convert to speech via Murf."""
    if not gemini_client:
        fallback_response = f"Hello! You said '{prompt}'. This is a Day 20 AI voice agent demo with Murf TTS integration!"
    else:
        try:
            response = await gemini_client.aio.models.generate_content(
                model="gemini-1.5-flash",
                contents=[{"parts": [{"text": prompt}]}],
                config={"temperature": 0.7, "max_output_tokens": 512}
            )
            fallback_response = response.text if hasattr(response, 'text') and response.text else f"I understand you said: '{prompt}'. This is a Day 20 demo response."
        except Exception as e:
            logger.warning(f"Gemini error: {e}")
            fallback_response = f"Hello! You said '{prompt}'. This is a Day 20 AI voice agent demo with Murf TTS integration!"
    
    # Print streaming effect
    print(f"\n[LLM PROMPT] {prompt}")
    print(f"[LLM STREAMING] ", end="", flush=True)
    for char in fallback_response:
        print(char, end="", flush=True)
        await asyncio.sleep(0.01)
    print()
    
    # Send to Murf for TTS conversion
    if murf_client:
        print(f"[MURF TTS] Converting LLM response to speech...")
        await murf_client.synthesize_text(fallback_response)
    else:
        print(f"[MURF DISABLED] No Murf client available")
    
    return fallback_response

@app.get("/test-murf")
async def test_murf_tts():
    """Day 20: Test Murf TTS integration with base64 output."""
    if not murf_client:
        return {"error": "Murf client not configured - install murf SDK and add MURF_API_KEY"}
    
    try:
        test_text = "Hello! This is Day 20 of the 30 Days of AI Voice Agents challenge. Testing Murf TTS integration with base64 audio output!"
        
        print(f"\n[DAY 20 TEST] 🧪 Testing Murf TTS with base64 conversion...")
        base64_audio = await murf_client.synthesize_text(test_text)
        
        if base64_audio:
            return {
                "status": "SUCCESS",
                "message": "✅ Murf TTS test completed! Check console for [FULL BASE64 AUDIO] output",
                "text_sent": test_text,
                "audio_length": len(base64_audio),
                "base64_preview": base64_audio[:100] + "...",
                "screenshot_instruction": "📸 Screenshot the [FULL BASE64 AUDIO] output in console for LinkedIn!"
            }
        else:
            return {"error": "❌ Failed to generate or convert audio to base64"}
            
    except Exception as e:
        return {"error": f"❌ Test failed: {e}"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0",
        "day": 20,
        "services": {
            "assemblyai": "available" if settings.assemblyai_api_key else "unavailable",
            "gemini": "available" if gemini_client else "unavailable",
            "murf": "available" if murf_client else "unavailable"
        }
    }

@app.get("/")
async def root():
    """Day 20: AI Voice Agent with Murf TTS."""
    return {
        "message": "🎵 Day 20: AI Voice Agent with Murf TTS Integration - WORKING!",
        "endpoints": {
            "test_murf": "/test-murf",
            "health": "/health"
        },
        "demo_instructions": [
            "1. Visit /test-murf to test integration",
            "2. Check console for [FULL BASE64 AUDIO] output",
            "3. Screenshot the base64 audio for LinkedIn post",
            "4. Success! Your Day 20 challenge is complete!"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=1)
