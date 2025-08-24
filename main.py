"""
AI Voice Agent - Day 23: Complete Conversational Voice Agent (FINAL)
Integrates: Speech-to-Text, LLM, TTS, Chat History, Streaming Audio + FIXED Audio Recording
"""

import asyncio
import json
import time
import base64
import aiohttp
from contextlib import asynccontextmanager
from typing import Dict, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
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

# Configure all services
if settings.assemblyai_api_key:
    aai.settings.api_key = settings.assemblyai_api_key
    logger.info("AssemblyAI configured for speech transcription")

if GEMINI_AVAILABLE and hasattr(settings, 'gemini_api_key') and settings.gemini_api_key:
    try:
        gemini_client = genai.Client(api_key=settings.gemini_api_key)
        logger.info("Google Gemini configured for LLM responses")
    except Exception as e:
        gemini_client = None
else:
    gemini_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Complete AI Voice Agent")
    yield
    logger.info("Shutting down Complete AI Voice Agent")

app = FastAPI(
    title="Complete AI Voice Agent",
    description="Day 23: Full conversational voice agent with STT, LLM, TTS, and streaming",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global chat history storage
chat_sessions: Dict[str, List[Dict]] = {}

class CompleteVoiceAgent:
    """Complete voice agent integrating all services."""
    
    def __init__(self):
        self.murf_client = None
        if MURF_AVAILABLE and hasattr(settings, 'murf_api_key') and settings.murf_api_key:
            self.murf_client = Murf(api_key=settings.murf_api_key)
    
    async def transcribe_audio(self, audio_file_path: str) -> str:
        """Step 1: Transcribe audio to text using AssemblyAI."""
        try:
            print(f"[TRANSCRIPTION] Processing audio file: {audio_file_path}")
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(audio_file_path)
            
            if transcript.error:
                logger.error(f"Transcription error: {transcript.error}")
                return f"Sorry, I couldn't understand the audio. Error: {transcript.error}"
            
            transcribed_text = transcript.text.strip()
            print(f"[TRANSCRIPTION] Result: {transcribed_text}")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return f"Sorry, I couldn't process the audio. Please try again."
    
    async def generate_llm_response(self, user_input: str, chat_history: List[Dict]) -> str:
        """Step 2: Generate LLM response using conversation context."""
        try:
            # Build conversation context
            context = "You are a helpful AI voice assistant. Respond conversationally and concisely.\n\n"
            
            # Add recent chat history for context
            recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
            for msg in recent_history:
                role = "Human" if msg.get("role") == "user" else "Assistant"
                context += f"{role}: {msg.get('text', msg.get('message', ''))}\n"
            
            context += f"Human: {user_input}\nAssistant:"
            
            print(f"[LLM INPUT] {user_input}")
            
            if not gemini_client:
                # Fallback response
                response = f"I understand you said: '{user_input}'. This is a Day 23 complete voice agent demo!"
            else:
                # Generate with Gemini
                response_obj = await gemini_client.aio.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=[{"parts": [{"text": context}]}],
                    config={
                        "temperature": 0.7,
                        "max_output_tokens": 256,  # Keep responses concise for TTS
                    }
                )
                response = response_obj.text if hasattr(response_obj, 'text') and response_obj.text else f"I understand your message about: {user_input}"
            
            print(f"[LLM RESPONSE] {response}")
            return response.strip()
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"I apologize, but I encountered an error processing your request."
    
    async def synthesize_speech(self, text: str) -> str:
        """Step 3: Convert text to speech using Murf and return base64 audio."""
        try:
            if not self.murf_client:
                print("[TTS] Murf client not available, using mock audio")
                # Return mock base64 audio for demo
                mock_audio = base64.b64encode(text.encode('utf-8')).decode('utf-8')
                return mock_audio
            
            print(f"[TTS] Generating speech for: {text[:60]}...")
            
            # Generate speech with Murf
            response = self.murf_client.text_to_speech.generate(
                text=text,
                voice_id="en-US-natalie",
                format="MP3",
                sample_rate=44100.0
            )
            
            # Get base64 audio
            base64_audio = None
            if hasattr(response, 'encoded_audio') and response.encoded_audio:
                base64_audio = response.encoded_audio
            elif hasattr(response, 'audio_file') and response.audio_file:
                base64_audio = await self.download_audio_as_base64(response.audio_file)
            
            if base64_audio:
                print(f"[TTS] Generated audio: {len(base64_audio)} characters")
                return base64_audio
            else:
                print("[TTS] No audio generated, using fallback")
                return base64.b64encode(text.encode('utf-8')).decode('utf-8')
                
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            # Return text as base64 fallback
            return base64.b64encode(text.encode('utf-8')).decode('utf-8')
    
    async def download_audio_as_base64(self, audio_url: str) -> str:
        """Helper: Download audio from URL and convert to base64."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(audio_url) as response:
                    if response.status == 200:
                        audio_data = await response.read()
                        return base64.b64encode(audio_data).decode('utf-8')
                    else:
                        logger.error(f"Audio download failed: HTTP {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Audio download error: {e}")
            return None
    
    async def stream_audio_response(self, websocket: WebSocket, base64_audio: str, response_text: str):
        """Step 4: Stream base64 audio chunks to client."""
        try:
            # Send response text first
            await websocket.send_text(json.dumps({
                "type": "llm_response",
                "text": response_text,
                "timestamp": time.time()
            }))
            
            # Chunk and stream audio
            chunk_size = 4096
            chunks = [base64_audio[i:i + chunk_size] for i in range(0, len(base64_audio), chunk_size)]
            
            print(f"[STREAMING] Sending {len(chunks)} audio chunks...")
            
            for i, chunk in enumerate(chunks):
                chunk_message = {
                    "type": "audio_chunk",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "data": chunk,
                    "is_final": i == len(chunks) - 1
                }
                
                await websocket.send_text(json.dumps(chunk_message))
                
                # Keepalive for large files
                if i > 0 and i % 50 == 0:
                    await websocket.send_text(json.dumps({
                        "type": "keepalive",
                        "chunk_progress": i
                    }))
                
                await asyncio.sleep(0.05)
            
            print(f"[STREAMING] ✅ Completed streaming {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Audio streaming failed: {e}")
    
    async def save_chat_history(self, session_id: str, user_message: str, agent_response: str):
        """Step 5: Save conversation to chat history."""
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        # Add user message
        chat_sessions[session_id].append({
            "role": "user",
            "text": user_message,
            "timestamp": time.time()
        })
        
        # Add agent response
        chat_sessions[session_id].append({
            "role": "agent",
            "text": agent_response,
            "timestamp": time.time()
        })
        
        print(f"[CHAT HISTORY] Session {session_id}: {len(chat_sessions[session_id])} messages")

# Initialize voice agent
voice_agent = CompleteVoiceAgent()

@app.websocket("/ws/complete-voice-agent")
async def complete_voice_agent_endpoint(websocket: WebSocket):
    """Day 23: Complete conversational voice agent WebSocket endpoint."""
    await websocket.accept()
    logger.info("Complete Voice Agent WebSocket connected")
    
    session_id = f"voice_session_{int(time.time())}"
    chat_sessions[session_id] = []
    
    try:
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "message": "Complete Voice Agent ready! Send text or audio for conversation.",
            "session_id": session_id,
            "features": ["STT", "LLM", "TTS", "Chat History", "Audio Streaming"],
            "timestamp": time.time()
        }))
        
        while True:
            try:
                # Receive message from client
                message = await websocket.receive_text()
                
                # Parse message
                if message.startswith('{'):
                    data = json.loads(message)
                    
                    # Skip acknowledgments
                    if data.get("type") == "chunk_acknowledgment":
                        continue
                    
                    # Handle different input types
                    if data.get("type") == "text_input":
                        user_input = data.get("text", "")
                        input_type = "text"
                        
                    elif data.get("type") == "audio_input":
                        # FIXED: Handle audio input
                        print(f"[VOICE AGENT] Received audio input for transcription")
                        
                        # Send processing status
                        await websocket.send_text(json.dumps({
                            "type": "processing",
                            "message": "Transcribing your voice message...",
                            "input_type": "audio",
                            "timestamp": time.time()
                        }))
                        
                        # Process audio data
                        audio_base64 = data.get("audio_data", "")
                        if not audio_base64:
                            continue
                        
                        # Save audio to temporary file for transcription
                        try:
                            audio_data = base64.b64decode(audio_base64)
                            temp_audio_path = UPLOADS_DIR / f"voice_{session_id}_{int(time.time())}.webm"
                            
                            with open(temp_audio_path, "wb") as f:
                                f.write(audio_data)
                            
                            # Transcribe audio
                            user_input = await voice_agent.transcribe_audio(str(temp_audio_path))
                            input_type = "audio"
                            
                            # Clean up temp file
                            temp_audio_path.unlink(missing_ok=True)
                            
                            print(f"[VOICE AGENT] Transcribed: {user_input}")
                            
                            # Send transcription result to client
                            await websocket.send_text(json.dumps({
                                "type": "transcription_result",
                                "text": user_input,
                                "timestamp": time.time()
                            }))
                            
                        except Exception as e:
                            logger.error(f"Audio processing failed: {e}")
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "message": f"Audio processing failed: {e}",
                                "timestamp": time.time()
                            }))
                            continue
                        
                    elif "text" in data:
                        user_input = data["text"]
                        input_type = "text"
                    else:
                        continue
                else:
                    # Plain text input
                    user_input = message
                    input_type = "text"
                
                if not user_input.strip():
                    continue
                
                print(f"[VOICE AGENT] Session {session_id}: Processing {input_type} input: {user_input}")
                
                # Send processing status
                await websocket.send_text(json.dumps({
                    "type": "processing",
                    "message": "Generating AI response...",
                    "input_type": input_type,
                    "timestamp": time.time()
                }))
                
                # COMPLETE PIPELINE EXECUTION
                
                # Step 1: Get current chat history
                current_history = chat_sessions.get(session_id, [])
                
                # Step 2: Generate LLM response
                agent_response = await voice_agent.generate_llm_response(user_input, current_history)
                
                # Step 3: Save to chat history
                await voice_agent.save_chat_history(session_id, user_input, agent_response)
                
                # Step 4: Generate TTS audio
                base64_audio = await voice_agent.synthesize_speech(agent_response)
                
                # Step 5: Stream response and audio to client
                await voice_agent.stream_audio_response(websocket, base64_audio, agent_response)
                
                print(f"[VOICE AGENT] Session {session_id}: Completed full pipeline")
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in voice agent pipeline: {e}")
                try:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Pipeline error: {e}",
                        "timestamp": time.time()
                    }))
                except:
                    break
        
    except WebSocketDisconnect:
        logger.info(f"Voice agent session {session_id} disconnected")
    except Exception as e:
        logger.error(f"Voice agent error: {e}")
    finally:
        # Clean up old sessions (keep last 10)
        if len(chat_sessions) > 10:
            oldest_sessions = sorted(chat_sessions.keys())[:len(chat_sessions) - 10]
            for old_session in oldest_sessions:
                del chat_sessions[old_session]
        
        logger.info(f"Voice agent session {session_id} ended")

@app.post("/upload-audio")
async def upload_audio_for_transcription(file: UploadFile = File(...)):
    """Upload audio file for transcription testing."""
    try:
        # Save uploaded file
        file_path = UPLOADS_DIR / f"upload_{int(time.time())}.wav"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Transcribe
        transcribed_text = await voice_agent.transcribe_audio(str(file_path))
        
        # Clean up
        file_path.unlink(missing_ok=True)
        
        return {
            "status": "success",
            "transcription": transcribed_text,
            "filename": file.filename
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Transcription failed: {e}"}
        )

@app.get("/chat-history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session."""
    history = chat_sessions.get(session_id, [])
    return {
        "session_id": session_id,
        "message_count": len(history),
        "history": history
    }

@app.get("/test-pipeline")
async def test_complete_pipeline():
    """Test the complete pipeline with sample data."""
    try:
        test_input = "Hello, this is a test of the complete voice agent pipeline."
        
        # Test LLM response
        response = await voice_agent.generate_llm_response(test_input, [])
        
        # Test TTS
        audio = await voice_agent.synthesize_speech(response)
        
        return {
            "status": "success",
            "test_input": test_input,
            "llm_response": response,
            "audio_length": len(audio),
            "audio_preview": audio[:100] + "..." if len(audio) > 100 else audio
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Pipeline test failed: {e}"}
        )

@app.get("/health")
async def health_check():
    """Comprehensive health check."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "3.0.0",
        "day": 23,
        "pipeline": "Complete Voice Agent",
        "services": {
            "assemblyai": "available" if settings.assemblyai_api_key else "unavailable",
            "gemini": "available" if gemini_client else "unavailable",
            "murf": "available" if voice_agent.murf_client else "unavailable"
        },
        "active_sessions": len(chat_sessions),
        "features": [
            "Speech-to-Text (AssemblyAI)",
            "LLM Responses (Google Gemini)",
            "Text-to-Speech (Murf)",
            "Chat History Management",
            "Audio Streaming",
            "WebSocket Communication",
            "Voice Recording & Transcription"
        ]
    }

@app.get("/")
async def root():
    """Day 23: Complete Voice Agent Demo."""
    return {
        "message": "🎙️ Day 23: Complete AI Voice Agent (FINAL)",
        "description": "Full conversational voice agent with STT, LLM, TTS, streaming, and voice recording",
        "endpoints": {
            "voice_agent": "/ws/complete-voice-agent",
            "upload_audio": "/upload-audio",
            "chat_history": "/chat-history/{session_id}",
            "test_pipeline": "/test-pipeline",
            "health": "/health"
        },
        "demo_flow": [
            "1. Connect to WebSocket: /ws/complete-voice-agent",
            "2. Send text message OR record voice message",
            "3. Voice messages are transcribed automatically",
            "4. Receive LLM response text",
            "5. Receive streaming audio chunks",
            "6. Chat history automatically saved",
            "7. Full conversational experience with voice!"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=1)
