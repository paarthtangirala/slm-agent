# Human-Centered AI Assistant

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)](https://fastapi.tiangolo.com/)

A local, private, transparent AI assistant following Prof. Jingyi Xie's Human-Centered AI principles. Built with simplicity, privacy, and accessibility in mind.

## 🎯 Human-Centered Design Principles

### ✨ **Transparency**
- Every AI response includes clear reasoning for how it was generated
- Intent detection is explainable with simple heuristics
- System status and capabilities are always visible
- No black-box operations

### ♿ **Accessibility** 
- Voice input/output support for hands-free interaction
- Multiple response length options (brief, medium, detailed)
- Clear error messages with actionable suggestions
- Works entirely offline after setup

### 🧠 **Personalization**
- Local learning of user preferences (tone, length, voice settings)
- Feedback system for continuous improvement
- Adapts responses based on user interaction patterns
- No external profiling or tracking

### 🔒 **Privacy**
- **Local-first**: All processing happens on your machine
- **No cloud dependencies**: Works without internet after setup
- **Opt-in web search**: External search only when explicitly enabled
- **Local knowledge base**: Your documents stay on your device

### 🔄 **Feedback**
- User rating system for every response
- Correction mechanism for improving future responses
- Transparent feedback stats and usage analytics
- Continuous learning from user preferences

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai) installed and running
- Git

### Installation

1. **Clone and setup**
   ```bash
   git clone <repository-url>
   cd slm-personal-agent
   cp .env.example .env
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_hcai.txt
   ```

3. **Start Ollama and download model**
   ```bash
   ollama serve
   ollama pull phi3:mini
   ```

4. **Run the assistant**
   ```bash
   python run_hcai.py
   ```

5. **Access the web interface**
   - Open browser to: **http://localhost:8000**
   - Beautiful web interface with document upload capabilities

## 📚 Adding Your Knowledge

Upload documents directly through the web interface:

1. Click **"Add Documents"** in the knowledge base section
2. Select PDF, TXT, MD, or RST files
3. Click **"Upload & Process"**
4. Ask questions about your documents!

**Or use command line:**
```bash
python ingest_hcai.py docs/
python ingest_hcai.py document.pdf
```

## 🔌 API Endpoints

### `/assist` - Core Intelligence
Main endpoint for all assistance tasks with transparent reasoning.

```json
{
  "text": "What are the HCAI principles?",
  "tone": "friendly",
  "length": "medium",
  "use_web": false
}
```

### `/memory` - Personalization
Manage user preferences and provide feedback.

### `/voice` - Accessibility
Voice input and output for hands-free interaction.

### `/upload` - Document Management
Upload documents through the web interface.

## 🛠 Configuration

Edit `.env` to customize:

```bash
PORT=8000
OLLAMA_MODEL=phi3:mini
DEFAULT_WEB_SEARCH=false
DEFAULT_TONE=friendly
DEFAULT_LENGTH=medium
```

## 🧪 Optional Features

**Voice Capabilities:**
```bash
pip install faster-whisper pyttsx3
```

**Web Search (Privacy-Aware):**
```bash
pip install duckduckgo-search
```

## 🏗 Architecture

```
app_hcai/
├── main.py           # FastAPI app with core endpoints
├── types.py          # Transparent request/response models  
├── intent.py         # Explainable intent detection
├── prompts.py        # Human-centered prompt templates
├── llm_local.py      # Ollama client wrapper
├── memory.py         # Local preference storage
├── rag_local.py      # Local knowledge base
├── tts_voice.py      # Voice processing (optional)
└── static/           # Web interface
    └── index.html    # Beautiful web UI
```

## 📊 Privacy & Data

**What Stays Local:**
- All conversations and preferences
- Personal document knowledge base  
- AI model processing (via Ollama)
- User feedback and ratings

**What's Optional (Opt-in):**
- Web search via DuckDuckGo (privacy-focused)
- Voice processing (all local)

**What's Never Sent Anywhere:**
- Your documents and conversations
- Personal preferences and feedback
- Usage patterns or analytics

## 🤝 Contributing

This project follows Human-Centered AI principles:

1. **Every change must improve transparency, accessibility, personalization, privacy, or feedback**
2. **Keep dependencies minimal** - only add what's essential
3. **Maintain local-first design** - no required cloud services
4. **Document the reasoning** - explain design decisions
5. **Test with real users** - validate accessibility improvements

## 📖 Learn More

- [Ollama Documentation](https://ollama.ai/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com)

## 📜 License

MIT License - See LICENSE file for details.

---

**Built with ❤️ following Human-Centered AI principles**

*"Technology should augment human capabilities while respecting human values, privacy, and autonomy."*