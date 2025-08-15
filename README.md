# 🎤 AI Voice Agent - Modern Conversational Interface

> A production-ready conversational AI system built with FastAPI, featuring voice-to-voice interactions, chat history, and robust error handling.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)  
- [Features](#features)
- [Architecture](#architecture)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

This AI Voice Agent represents the culmination of a 13-day development journey, evolving from a simple text-to-speech interface into a sophisticated conversational AI system. The application enables natural voice conversations with an AI assistant, featuring persistent chat history, robust error handling, and a modern, responsive user interface.

**Key Highlights:**
- 🎯 **Natural Conversations**: Voice-to-voice interactions with contextual memory
- 🛡️ **Production-Ready**: Comprehensive error handling and fallback mechanisms 
- 🎨 **Modern UI**: Clean, responsive interface with smooth animations
- 📱 **Mobile-Friendly**: Optimized for all device sizes
- 🔄 **Real-time Processing**: Live status updates and seamless audio playback

## ✨ Features

### Core Functionality
- Voice input processing with AssemblyAI
- Intelligent responses from Google Gemini
- High-quality speech synthesis via Murf AI
- Persistent conversation history
- Auto-continue conversation flow

### Technical Features
- Comprehensive error handling
- Intelligent fallback responses
- Session-based persistence
- Health monitoring endpoints
- Request timeout management
- Audio format optimization

### User Experience
- Single-button interface with visual feedback
- Real-time status updates
- Responsive design for all devices
- Smooth animations and transitions
- Accessibility support

## 🛠️ Technologies Used

### Backend
- **FastAPI** - Modern Python web framework
- **Python 3.11+** - Core programming language
- **Uvicorn** - ASGI web server
- **Pydantic** - Data validation

### AI Services
- **AssemblyAI** - Speech-to-text transcription
- **Google Gemini** - Large language model
- **Murf AI** - Text-to-speech synthesis

### Frontend
- Vanilla JavaScript
- HTML5 & CSS3
- MediaRecorder API
- Fetch API

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

Create a `.env` file with:

```sh
MURF_API_KEY=your_murf_api_key_here
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

## 📖 Usage

```sh
# Start development server
uvicorn main:app --reload

# Access application
open http://localhost:8000
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ during the 30 Days of AI Voice Agents Challenge**
