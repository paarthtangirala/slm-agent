# SLM Personal Agent - Starter Kit

Local-first personal AI agent using small language models (SLMs) for everyday tasks like note summarization, email drafting, web search, and local document Q&A.

## 🛠️ Setup

### 1. Install Ollama & Pull Model

**macOS/Linux/Windows:** https://ollama.com/download

Pull a small model:
```bash
ollama pull phi3:mini          # Recommended: 3.8GB, fast
# OR
ollama pull mistral:7b         # Alternative: 4.1GB, stronger
# OR  
ollama pull llama3.1:8b        # Alternative: 4.7GB, broader reasoning
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Ingest Local Documents (Optional)

Add your documents to the `docs/` folder, then run:
```bash
python ingest.py
```

### 4. Start the Server

```bash
uvicorn app.main:app --reload
```

Server runs at: http://localhost:8000

## 🧪 Test with curl

### Health Check
```bash
curl http://localhost:8000/health
```

### Summarize Text
```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Artificial intelligence has revolutionized many industries. Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions. Natural language processing has enabled computers to understand and generate human language with remarkable accuracy. Computer vision systems can recognize objects and faces in images. These technologies are being applied in healthcare for diagnosis, in finance for fraud detection, in transportation for autonomous vehicles, and in entertainment for personalized recommendations."
  }'
```

### Draft Email
```bash
curl -X POST http://localhost:8000/draft-email \
  -H "Content-Type: application/json" \
  -d '{
    "recipient": "john@example.com",
    "subject": "Project Update Meeting",
    "context": "Need to schedule a meeting to discuss the Q4 project progress and upcoming deadlines",
    "tone": "professional"
  }'
```

### Web Search (Real or Mock)
```bash
curl -X POST http://localhost:8000/web-search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "AI research 2025",
    "max_results": 3
  }'
```

**🔥 NEW: Real Search APIs!**
- **SerpAPI**: 100 free searches/month (Google results)
- **Bing API**: 1,000 free searches/month  
- **Smart fallbacks**: Always works, even without API keys

See `SEARCH_API_SETUP.md` for setup instructions.

### Local Document Query
```bash
# First ingest some documents
python ingest.py

# Then query them
curl -X POST http://localhost:8000/local-query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key features of this agent?",
    "max_results": 3
  }'
```

## 🔁 Swapping Models

Change the `.env` file and restart:
```bash
# Edit .env
OLLAMA_MODEL=mistral:7b

# Restart server
uvicorn app.main:app --reload
```

## 📁 Project Structure

```
slm-agent/
├── app/
│   └── main.py          # FastAPI application
├── docs/                # Local documents for ingestion
│   └── sample.md       # Sample document
├── chromadb/           # Vector database (created after ingestion)
├── ingest.py           # Document ingestion script
├── requirements.txt    # Dependencies
├── .env               # Configuration
└── README.md          # This file
```

## 🚀 Next Steps

1. **Add more documents** to `docs/` folder and run `python ingest.py`
2. **Customize prompts** in `app/main.py` for your specific needs
3. **Try different models** by changing `OLLAMA_MODEL` in `.env`
4. **Add logging** to track usage and build fine-tuning datasets
5. **Build a web UI** using React/Vue that calls these endpoints

## 🛡️ Privacy & Safety

- Everything runs locally - no data leaves your machine
- Web search uses DuckDuckGo (privacy-focused)
- Email drafts are generated but not sent automatically
- ChromaDB stores document embeddings locally

## ⚡ Performance Tips

- **phi3:mini**: Fastest, lowest memory usage
- **mistral:7b**: Better for complex reasoning
- **llama3.1:8b**: Best overall quality, higher resource usage

Your SLM Personal Agent is ready! 🚀