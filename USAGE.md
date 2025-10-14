# SLM Personal Agent - Usage Guide

## 🚀 Quick Start

Your SLM Personal Agent is now running! Here's how to use it:

### 1. Server Status
- ✅ **Server**: Running on http://localhost:8000
- ✅ **Model**: phi3:mini via Ollama
- ✅ **Database**: ChromaDB with 1 document indexed

### 2. Available Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Summarize Text
```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Your long text here..."}'
```

#### Draft Email
```bash
curl -X POST http://localhost:8000/draft-email \
  -H "Content-Type: application/json" \
  -d '{
    "recipient": "john@example.com",
    "subject": "Meeting Request",
    "context": "Need to schedule a team meeting",
    "tone": "professional"
  }'
```

#### Web Search (Demo Mode)
```bash
curl -X POST http://localhost:8000/web-search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "AI research 2025",
    "max_results": 3
  }'
```

#### Query Local Documents
```bash
curl -X POST http://localhost:8000/local-query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What features does this agent have?",
    "max_results": 3
  }'
```

### 3. Adding Your Documents

1. **Add files** to the `docs/` folder:
   - Supported formats: `.txt`, `.md`, `.py`, `.js`, `.json`, `.yaml`
   - Example: `docs/my-notes.md`, `docs/project-info.txt`

2. **Ingest documents**:
   ```bash
   python ingest.py
   ```

3. **Query your documents**:
   ```bash
   curl -X POST http://localhost:8000/local-query \
     -H "Content-Type: application/json" \
     -d '{"query": "Tell me about my project"}'
   ```

### 4. Switching Models

Edit `.env` file:
```env
# Fast & lightweight (current)
OLLAMA_MODEL=phi3:mini

# Better reasoning
OLLAMA_MODEL=mistral:7b

# Best quality
OLLAMA_MODEL=llama3.1:8b
```

Then restart the server.

### 5. Test Script

Run the comprehensive test:
```bash
python test_agent.py
```

## 🎯 Use Cases

- **Note Summarization**: Paste meeting notes, get key points
- **Email Assistant**: Draft professional emails quickly  
- **Document Q&A**: Ask questions about your local files
- **Research Helper**: Get AI summaries of topics (demo mode)

## 🔧 Customization

- **Prompts**: Edit `app/main.py` to customize AI responses
- **Models**: Try different Ollama models for various quality/speed trade-offs
- **Documents**: Add your personal knowledge base to `docs/`

Your local AI agent is ready to boost your productivity! 🤖✨