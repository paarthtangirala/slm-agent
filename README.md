# 🤖 SLM Personal Agent - Level 6+ AI Assistant

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)](https://fastapi.tiangolo.com/)

Advanced SLM Personal Agent with **Integration Hub**, **Security Tools**, **Data Visualization**, and **Personalization Engine**. Multi-modal AI assistant with voice interface, workflow automation, and enterprise-grade features.

## ✨ Level 6+ Features

### 🔗 **Integration Hub**
- **6 External Services**: GitHub, Slack, Google Drive, Notion, Trello, Google Calendar
- **OAuth2 & API Key Authentication**: Secure connection management
- **Service Status Monitoring**: Real-time integration health tracking
- **Automated Workflows**: Intelligent service suggestions and automation

### 🔒 **Security Tools**
- **Advanced Data Classification**: Automatic PII and sensitive data detection
- **AES-256 Encryption**: End-to-end data protection
- **Comprehensive Audit Logging**: Full activity tracking and compliance
- **Real-time Anomaly Detection**: Security threat monitoring
- **Security Dashboard**: Live metrics and threat assessment

### 📊 **Visual Data Explorer**
- **Interactive Charts**: Bar, line, scatter, pie, heatmap, box, violin, area charts
- **Plotly Integration**: Rich, interactive data visualizations
- **Dashboard Creation**: Combine multiple visualizations
- **Sample Data Generation**: Built-in datasets for testing and demos
- **Data Quality Assessment**: Automated analysis and recommendations

### 🧠 **Enhanced Memory & Personalization**
- **AI-Powered Learning**: Automatic user preference detection
- **Communication Style Analysis**: Adapts to formal/casual/technical styles
- **Behavioral Pattern Recognition**: Learns from interaction patterns
- **Personalized Insights**: Smart recommendations and usage analytics
- **Confidence Scoring**: Tracks learning accuracy and reliability

### 🎯 **Additional Advanced Features**
- **Voice Interface**: Speech recognition and text-to-speech
- **Workflow Automation**: Template-based task automation
- **Smart Reminders**: Intelligent notification system
- **Knowledge Graph**: Enhanced document processing and insights
- **Multi-Modal AI**: 9+ specialized conversation modes

## 🚀 Quick Start

### 1. Install Ollama & Pull Model

**Download:** https://ollama.com/download

```bash
# Recommended for SLM Personal Agent
ollama pull phi3:mini          # 3.8GB, optimized for speed
# OR
ollama pull mistral:7b         # 4.1GB, enhanced reasoning
# OR  
ollama pull llama3.1:8b        # 4.7GB, comprehensive capabilities
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Setup (Optional)

Create `.env` file for API keys:
```bash
# Web Search (Optional)
SERPAPI_API_KEY=your_serpapi_key

# Integration Hub (Optional)
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret
SLACK_CLIENT_ID=your_slack_client_id
# ... other service credentials
```

### 4. Start the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

🌐 **Access at:** http://localhost:8000

## 🔧 Core Capabilities

### 💬 **Multi-Modal Chat Modes**
- **General**: Everyday conversations and assistance
- **Research**: Deep analysis and fact-finding
- **Creative**: Content creation and brainstorming
- **Technical**: Code analysis and system design
- **Educational**: Learning and teaching assistance
- **Email**: Professional communication drafting
- **Summarization**: Document and content summarization
- **Web Search**: Real-time information retrieval
- **Code Analysis**: Development assistance and debugging

### 📁 **Document Processing**
- **Supported Formats**: PDF, DOCX, TXT, CSV, XLSX
- **Intelligent Extraction**: Automatic content analysis
- **Knowledge Base**: Searchable document repository
- **Vector Search**: Semantic document retrieval

### 🎤 **Voice Capabilities**
- **Speech-to-Text**: Real-time voice input processing
- **Text-to-Speech**: Natural voice output
- **Voice Commands**: Hands-free operation
- **Multiple Voice Options**: Customizable speech synthesis

## 📊 API Endpoints

### Core Chat
```bash
# Multi-modal conversation
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "mode": "general"}'
```

### Integration Hub
```bash
# Available integrations
curl -X GET "http://localhost:8000/integrations/available"

# User integrations
curl -X GET "http://localhost:8000/integrations/user"
```

### Security Tools
```bash
# Security dashboard
curl -X GET "http://localhost:8000/security/dashboard"

# Data classification
curl -X POST "http://localhost:8000/security/classify" \
  -H "Content-Type: application/json" \
  -d '{"data_type": "document", "data_id": "doc1", "content": "sensitive data"}'
```

### Data Visualization
```bash
# Create visualization
curl -X POST "http://localhost:8000/data/visualize" \
  -H "Content-Type: application/json" \
  -d '{"name": "Sales Chart", "chart_type": "bar", "data_source": "uploaded", "config": {...}}'

# Get visualizations
curl -X GET "http://localhost:8000/data/visualizations"
```

### Personalization
```bash
# User dashboard
curl -X GET "http://localhost:8000/personalization/dashboard"

# Personalized suggestions
curl -X GET "http://localhost:8000/personalization/suggestions"
```

## 🏗️ Architecture

```
slm-personal-agent/
├── app/
│   ├── main.py                    # FastAPI application
│   ├── database.py               # Async SQLAlchemy setup
│   ├── integration_hub.py        # External service connections
│   ├── security_tools.py         # Data protection & audit
│   ├── data_visualizer.py        # Interactive charts & dashboards
│   ├── personalization_engine.py # AI-powered preference learning
│   ├── voice_interface.py        # Speech recognition & TTS
│   ├── workflows.py              # Automation templates
│   ├── knowledge_graph.py        # Document processing
│   ├── recommendations.py        # Smart suggestions
│   └── reminders.py              # Notification system
├── static/
│   ├── index.html                # Web interface
│   └── visualizations/           # Generated charts
├── uploads/                      # Document storage
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 🔧 Configuration

### Database
- **SQLite**: Local data storage with async SQLAlchemy
- **ChromaDB**: Vector database for document embeddings
- **Automatic Migrations**: Schema updates handled automatically

### Security
- **Data Classification**: Automatic PII and sensitive content detection
- **Encryption**: AES-256 for data at rest
- **Audit Logging**: Comprehensive activity tracking
- **Access Control**: Role-based permissions

### Personalization
- **Learning Engine**: AI-powered preference detection
- **Confidence Scoring**: Tracks learning accuracy
- **Behavioral Analysis**: Usage pattern recognition
- **Adaptive UI**: Interface customization based on preferences

## 🧪 Testing

### Web Interface Testing
1. Open http://localhost:8000
2. Test each feature tab:
   - **Chat**: Multi-modal conversations
   - **Integrations**: Service connections
   - **Security**: Data protection dashboard
   - **Visualizations**: Chart creation
   - **Personalization**: User insights

### API Testing
```bash
# Health check
curl http://localhost:8000/health

# Integration status
curl http://localhost:8000/integrations/available

# Security dashboard
curl http://localhost:8000/security/dashboard

# Sample data generation
curl http://localhost:8000/data/sample/sales
```

## 🚧 Development

### Adding New Integrations
1. Define service config in `integration_hub.py`
2. Implement OAuth2 or API key authentication
3. Add service actions and webhooks
4. Update UI integration cards

### Creating Custom Visualizations
1. Add chart type to `ChartType` enum in `data_visualizer.py`
2. Implement Plotly chart creation logic
3. Add configuration options
4. Update frontend chart selector

### Extending Personalization
1. Add new preference categories to `PreferenceCategory`
2. Implement learning algorithms in `personalization_engine.py`
3. Create insight generation logic
4. Update dashboard displays

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your enhancement
4. Add tests and documentation
5. Submit a pull request

## 🔗 Links

- **Ollama**: https://ollama.com/
- **FastAPI**: https://fastapi.tiangolo.com/
- **SQLAlchemy**: https://www.sqlalchemy.org/
- **Plotly**: https://plotly.com/python/
- **ChromaDB**: https://www.trychroma.com/

---

