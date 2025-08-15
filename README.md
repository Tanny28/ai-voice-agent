# 🎤 AI Voice Agent - Modern Conversational Interface

> A production-ready conversational AI system built with FastAPI, featuring voice-to-voice interactions, chat history, and robust error handling.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table of Contents

- Overview  
- Features
- Architecture
- Technologies Used
- Installation
- Configuration
- Usage
- API Documentation
- Project Structure

## 🌟 Overview

This AI Voice Agent is a sophisticated conversational AI system that enables natural voice-to-voice interactions. Built with FastAPI, it integrates multiple AI services for transcription, language understanding, and speech synthesis while maintaining production-grade reliability through comprehensive error handling and fallback mechanisms.

**Key Highlights:**
- 🎯 **Voice Processing**: Complete audio pipeline with transcription and synthesis
- 🧠 **Intelligent Responses**: Context-aware conversations using Google Gemini
- 🛡️ **Production-Ready**: Comprehensive error handling and logging
- ⚡ **High Performance**: Async processing with timeout protection
- 🔄 **Stateful**: Session-based conversation history

## ✨ Features

### Core Features
- Speech-to-text transcription using AssemblyAI
- Natural language processing with Google Gemini
- Text-to-speech synthesis via Murf AI
- Session-based conversation history
- Health monitoring endpoints

### Technical Features
- Comprehensive error handling and fallbacks
- Request timeout management
- Structured logging system
- Environment-based configuration
- CORS and middleware support

## 🏗️ Architecture

The application follows a modular architecture with clear separation of concerns:

- `app/api/endpoints` - API route handlers
- `app/services` - AI service integrations
- `app/core` - Core utilities and error handling
- `app/models` - Data validation schemas
- `static` & `templates` - Frontend assets

## 🛠️ Technologies Used

### Core Framework
- FastAPI 0.104.1
- Uvicorn 0.24.0
- Pydantic 2.5.0

### AI Services
- AssemblyAI 0.17.0
- Google Generative AI 0.3.2
- Murf AI REST API

### Development Tools
- Python 3.11+
- pytest 7.4.3
- black 23.11.0
- mypy 1.7.1

## 🚀 Installation

```sh
# Clone repository
git clone https://github.com/yourusername/ai-voice-agent.git
cd ai-voice-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
```

## ⚙️ Configuration

Create a `.env` file with the following settings:

```sh
# API Keys
MURF_API_KEY=your_murf_api_key
ASSEMBLYAI_API_KEY=your_assemblyai_api_key
GEMINI_API_KEY=your_gemini_api_key

# Application Settings
DEBUG=false
LOG_LEVEL=INFO
```

## 📖 Usage

```sh
# Start the application
uvicorn main:app --host 0.0.0.0 --port 8000

# Access API documentation
open http://localhost:8000/docs  # when DEBUG=true
```

## 📚 API Documentation

### Main Endpoints

- `POST /agent/chat/{session_id}` - Process voice input and generate response
- `GET /agent/history/{session_id}` - Retrieve conversation history
- `DELETE /agent/history/{session_id}` - Clear conversation history
- `POST /agent/tts` - Convert text to speech
- `GET /health` - System health check

## 📁 Project Structure

```
.
├── app/
│   ├── api/
│   │   └── endpoints/
│   │       ├── agent.py
│   │       └── health.py
│   ├── core/
│   │   ├── errors.py
│   │   └── logging.py
│   ├── models/
│   │   └── schemas.py
│   ├── services/
│   │   ├── assembly_ai.py
│   │   ├── gemini_ai.py
│   │   └── murf_ai.py
│   └── config.py
├── static/
├── templates/
├── main.py
├── requirements.txt
└── .env.example
```

---

Built with FastAPI and modern AI services for natural voice

