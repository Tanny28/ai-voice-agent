// Simplified Voice Agent Interface - Day 12 Revamp

class VoiceAgent {
  constructor() {
    this.sessionId = null;
    this.mediaRecorder = null;
    this.isRecording = false;
    this.audioChunks = [];
    this.recordButton = null;
    this.audioPlayer = null;
    
    this.init();
  }
  
  init() {
    this.initSession();
    this.setupEventListeners();
    this.loadChatHistory();
  }
  
  initSession() {
    const urlParams = new URLSearchParams(window.location.search);
    this.sessionId = urlParams.get('session_id') || this.generateSessionId();
    
    if (!urlParams.get('session_id')) {
      const newUrl = new URL(window.location);
      newUrl.searchParams.set('session_id', this.sessionId);
      window.history.replaceState({}, '', newUrl);
    }
    
    document.getElementById('currentSessionId').textContent = 
      this.sessionId.substring(0, 8) + '...';
  }
  
  generateSessionId() {
    return 'session_' + Date.now().toString(36) + '_' + 
           Math.random().toString(36).substring(2, 15);
  }
  
  setupEventListeners() {
    this.recordButton = document.getElementById('recordButton');
    this.audioPlayer = document.getElementById('audioPlayer');
    
    this.recordButton.addEventListener('click', () => this.toggleRecording());
    
    document.getElementById('clearHistoryBtn').addEventListener('click', () => {
      this.clearChatHistory();
    });
    
    // Auto-continue after AI response
    this.audioPlayer.addEventListener('ended', () => {
      setTimeout(() => {
        if (confirm('Continue the conversation? Click OK to record your next message.')) {
          this.startRecording();
        }
      }, 1000);
    });
  }
  
  async toggleRecording() {
    if (this.isRecording) {
      this.stopRecording();
    } else {
      await this.startRecording();
    }
  }
  
  async startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.audioChunks = [];
      
      this.mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      
      this.mediaRecorder.ondataavailable = (event) => {
        this.audioChunks.push(event.data);
      };
      
      this.mediaRecorder.onstart = () => {
        this.isRecording = true;
        this.updateRecordButtonState('recording');
        this.showStatus('🎤 Recording... Speak now', 'info');
      };
      
      this.mediaRecorder.onstop = () => {
        this.isRecording = false;
        this.updateRecordButtonState('processing');
        this.processRecording();
      };
      
      this.mediaRecorder.start();
      
    } catch (error) {
      this.showStatus('❌ Microphone access denied. Please allow microphone access and try again.', 'error');
      console.error('Recording error:', error);
    }
  }
  
  stopRecording() {
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
      
      // Stop all tracks to release microphone
      this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }
  }
  
  updateRecordButtonState(state) {
    const button = this.recordButton;
    const icon = button.querySelector('.record-icon');
    const text = button.querySelector('.record-text');
    
    button.setAttribute('data-state', state);
    
    switch (state) {
      case 'idle':
        icon.textContent = '🎤';
        text.textContent = 'Start Recording';
        button.disabled = false;
        break;
      case 'recording':
        icon.textContent = '⏹️';
        text.textContent = 'Stop Recording';
        button.disabled = false;
        break;
      case 'processing':
        icon.textContent = '⏳';
        text.textContent = 'Processing...';
        button.disabled = true;
        break;
    }
  }
  
  async processRecording() {
    const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.webm');
    
    this.showProcessingStatus('Processing your message...');
    
    try {
      const response = await fetch(`/agent/chat/${this.sessionId}`, {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      
      if (response.ok) {
        // Add messages to chat history
        this.addMessageToChat('user', data.transcription);
        this.addMessageToChat('assistant', data.llm_response);
        
        // Handle different response scenarios
        if (data.fallback_used) {
          this.showStatus(`⚠️ ${data.message}`, 'warning');
        } else {
          this.showStatus('✅ Response received', 'success');
        }
        
        // Play AI response
        if (data.audio_url) {
          this.playAudioResponse(data.audio_url);
        } else {
          this.showStatus('📝 Response ready (audio unavailable)', 'warning');
          this.updateRecordButtonState('idle');
        }
        
      } else {
        this.handleAPIError(data, response.status);
      }
      
    } catch (error) {
      this.handleNetworkError(error);
    } finally {
      this.hideProcessingStatus();
    }
  }
  
  playAudioResponse(audioUrl) {
    this.audioPlayer.src = audioUrl;
    this.audioPlayer.load();
    
    // Update button state while AI is speaking
    this.updateRecordButtonState('idle');
    this.showStatus('🔊 AI is responding...', 'info');
    
    this.audioPlayer.play().catch(error => {
      console.error('Audio playback error:', error);
      this.showStatus('🔊 Click to hear response (autoplay blocked)', 'warning');
    });
  }
  
  addMessageToChat(role, content) {
    const historyContainer = document.getElementById('historyContainer');
    const conversationHistory = document.getElementById('conversationHistory');
    
    // Show conversation section if hidden
    conversationHistory.classList.remove('hidden');
    
    // Create message element
    const messageElement = document.createElement('div');
    messageElement.className = `message ${role}`;
    messageElement.textContent = content;
    
    // Add to container
    historyContainer.appendChild(messageElement);
    
    // Scroll to bottom
    historyContainer.scrollTop = historyContainer.scrollHeight;
  }
  
  async loadChatHistory() {
    try {
      const response = await fetch(`/agent/history/${this.sessionId}`);
      
      if (!response.ok) return;
      
      const data = await response.json();
      
      if (data.history && data.history.length > 0) {
        const historyContainer = document.getElementById('historyContainer');
        historyContainer.innerHTML = '';
        
        data.history.forEach(message => {
          this.addMessageToChat(message.role, message.content);
        });
        
        document.getElementById('conversationHistory').classList.remove('hidden');
      }
      
    } catch (error) {
      console.log('No previous conversation history found');
    }
  }
  
  async clearChatHistory() {
    if (!confirm('Are you sure you want to clear the conversation history? This cannot be undone.')) {
      return;
    }
    
    try {
      const response = await fetch(`/agent/history/${this.sessionId}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        // Clear UI
        document.getElementById('historyContainer').innerHTML = '';
        document.getElementById('conversationHistory').classList.add('hidden');
        
        // Generate new session
        this.sessionId = this.generateSessionId();
        const newUrl = new URL(window.location);
        newUrl.searchParams.set('session_id', this.sessionId);
        window.history.replaceState({}, '', newUrl);
        
        document.getElementById('currentSessionId').textContent = 
          this.sessionId.substring(0, 8) + '...';
        
        this.showStatus('🗑️ Conversation cleared. Starting fresh!', 'success');
      } else {
        throw new Error('Failed to clear history');
      }
      
    } catch (error) {
      this.showStatus('❌ Failed to clear conversation history', 'error');
      console.error('Clear history error:', error);
    }
  }
  
  showStatus(message, type = 'info') {
    const statusDisplay = document.getElementById('statusDisplay');
    const statusText = statusDisplay.querySelector('.status-text');
    const statusIcon = statusDisplay.querySelector('.status-icon');
    
    statusText.textContent = message;
    statusDisplay.className = `status-display ${type}`;
    statusDisplay.classList.remove('hidden');
    
    // Update icon based on type
    switch (type) {
      case 'success':
        statusIcon.textContent = '✅';
        break;
      case 'error':
        statusIcon.textContent = '❌';
        break;
      case 'warning':
        statusIcon.textContent = '⚠️';
        break;
      default:
        statusIcon.textContent = 'ℹ️';
    }
    
    // Auto-hide after 5 seconds for non-error messages
    if (type !== 'error') {
      setTimeout(() => {
        statusDisplay.classList.add('hidden');
      }, 5000);
    }
  }
  
  showProcessingStatus(message) {
    const processingStatus = document.getElementById('processingStatus');
    const processingText = processingStatus.querySelector('.processing-text');
    
    processingText.textContent = message;
    processingStatus.classList.remove('hidden');
  }
  
  hideProcessingStatus() {
    document.getElementById('processingStatus').classList.add('hidden');
  }
  
  handleAPIError(data, statusCode) {
    let errorMessage = 'An error occurred';
    
    if (statusCode === 408) {
      errorMessage = '⏰ Processing took too long. Please try with a shorter message.';
    } else if (statusCode === 503) {
      errorMessage = '🔧 Services are temporarily unavailable. Please try again later.';
    } else if (statusCode === 504) {
      errorMessage = '⏰ Request timed out. Please try again.';
    } else if (data && data.detail) {
      if (typeof data.detail === 'object' && data.detail.fallback_message) {
        errorMessage = data.detail.fallback_message;
      } else {
        errorMessage = data.detail;
      }
    }
    
    this.showStatus(errorMessage, 'error');
    this.updateRecordButtonState('idle');
  }
  
  handleNetworkError(error) {
    console.error('Network error:', error);
    
    if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
      this.showStatus('🌐 Network error. Please check your internet connection.', 'error');
    } else {
      this.showStatus('❌ An unexpected error occurred. Please try again.', 'error');
    }
    
    this.updateRecordButtonState('idle');
  }
}

// Initialize the Voice Agent when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new VoiceAgent();
});

// Global error handlers
window.addEventListener('error', (event) => {
  console.error('Global error:', event.error);
});

window.addEventListener('unhandledrejection', (event) => {
  console.error('Unhandled promise rejection:', event.reason);
});
