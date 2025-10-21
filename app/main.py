from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import chromadb
from sentence_transformers import SentenceTransformer
import os
import asyncio
from typing import List, Optional
import logging
from datetime import datetime
from dotenv import load_dotenv
import json
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from .workflows import initialize_workflow_engine, workflow_engine
from .knowledge_graph import knowledge_graph
from .recommendations import recommendations_engine, Recommendation
from .voice_interface import voice_interface, VoiceRequest
from .ollama_client import call_ollama, check_ollama_health
from .reminders import reminder_system, ReminderStatus, Priority, ReminderType
from .integration_hub import integration_hub, IntegrationAction
from .security_tools import security_tools, AuditEventType, DataCategory, SecurityLevel
from .data_visualizer import data_explorer, ChartType, DataSource, VisualizationConfig
from .personalization_engine import personalization_engine, PreferenceCategory, PersonalizationInsight

load_dotenv()

app = FastAPI(title="SLM Personal Agent", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")

# Search API configuration
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
BING_SEARCH_KEY = os.getenv("BING_SEARCH_KEY") 
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "mock")

chroma_client = chromadb.PersistentClient(path="./chromadb")
collection = chroma_client.get_or_create_collection("local_docs")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_memory_manager():
    """Lazy load memory manager"""
    if not hasattr(get_memory_manager, '_manager'):
        from .database import memory_manager
        get_memory_manager._manager = memory_manager
    return get_memory_manager._manager

def get_doc_processor():
    """Lazy load document processor"""
    if not hasattr(get_doc_processor, '_processor'):
        from .document_processor import doc_processor
        get_doc_processor._processor = doc_processor
    return get_doc_processor._processor

# Workflow function wrappers
async def web_search_function(params):
    """Wrapper for web search functionality"""
    query = params.get("query", "")
    max_results = params.get("max_results", 5)
    
    # Use existing web search logic
    try:
        search_results = []
        provider_used = SEARCH_PROVIDER
        
        if SEARCH_PROVIDER == "serpapi" and SERPAPI_KEY:
            try:
                search_results = await search_with_serpapi(query, max_results)
                provider_used = "serpapi"
            except Exception as e:
                search_results = get_mock_results(query, max_results)
                provider_used = "mock"
        else:
            search_results = get_mock_results(query, max_results)
            provider_used = "mock"
        
        # Create context for AI summarization
        results_text = "\n".join([
            f"- {r['title']}: {r.get('snippet', 'No description available')}"
            for r in search_results
        ])
        
        system_prompt = "You are a web search summarizer. Analyze the search results and provide a concise, informative summary."
        prompt = f"""Based on these search results for "{query}", provide a brief summary:

{results_text}

Summary:"""
        
        summary = await call_ollama(prompt, system_prompt)
        
        return {
            "query": query,
            "results": search_results,
            "summary": summary.strip(),
            "provider": provider_used
        }
    except Exception as e:
        return {"error": str(e)}

async def summarize_function(params):
    """Wrapper for text summarization"""
    text = params.get("text", "")
    system_prompt = "You are a concise summarization assistant. Create clear, structured summaries that capture the key points."
    
    prompt = f"""Please summarize the following text in 2-3 bullet points, focusing on the most important information:

Text: {text}

Summary:"""
    
    try:
        summary = await call_ollama(prompt, system_prompt)
        return {"summary": summary.strip(), "original_length": len(text)}
    except Exception as e:
        return {"error": str(e)}

async def draft_email_function(params):
    """Wrapper for email drafting"""
    recipient = params.get("recipient", "")
    subject = params.get("subject", "")
    context = params.get("context", "")
    tone = params.get("tone", "professional")
    
    system_prompt = f"You are an email drafting assistant. Write professional emails in a {tone} tone. Include proper greeting, body, and closing."
    
    prompt = f"""Draft an email with the following details:

To: {recipient}
Subject: {subject}
Context: {context}
Tone: {tone}

Please write a complete email draft:"""
    
    try:
        draft = await call_ollama(prompt, system_prompt)
        return {
            "draft": draft.strip(),
            "recipient": recipient,
            "subject": subject,
            "tone": tone
        }
    except Exception as e:
        return {"error": str(e)}

async def local_query_function(params):
    """Wrapper for local document querying"""
    query = params.get("query", "")
    max_results = params.get("max_results", 3)
    
    try:
        query_embedding = embedding_model.encode([query])
        
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=max_results
        )
        
        if not results['documents'][0]:
            return {
                "query": query,
                "answer": "No relevant documents found in the local knowledge base.",
                "sources": []
            }
        
        context_docs = results['documents'][0]
        sources = results['metadatas'][0] if results['metadatas'][0] else []
        
        context = "\n\n".join([f"Document: {doc}" for doc in context_docs])
        
        system_prompt = "You are a helpful assistant that answers questions based on provided context. Only use information from the given documents."
        
        prompt = f"""Based on the following documents, answer the question: "{query}"

Context:
{context}

Answer:"""
        
        answer = await call_ollama(prompt, system_prompt)
        
        source_files = [meta.get('source', 'Unknown') for meta in sources] if sources else []
        
        return {
            "query": query,
            "answer": answer.strip(),
            "sources": source_files,
            "num_documents_used": len(context_docs)
        }
    except Exception as e:
        return {"error": str(e)}

async def analyze_code_function(params):
    """Wrapper for code analysis"""
    code = params.get("code", "")
    language = params.get("language", "auto")
    analysis_type = params.get("analysis_type", "comprehensive")
    
    if language == "auto":
        language = detect_programming_language(code)
    
    if analysis_type == "comprehensive":
        system_prompt = f"""You are a senior {language} developer and code reviewer. Provide comprehensive code analysis including:
1. Code quality and best practices
2. Potential bugs or issues
3. Performance considerations
4. Documentation suggestions
5. Security considerations"""
    else:
        system_prompt = f"You are a {language} code analyst. Provide general code analysis."
    
    prompt = f"""Analyze this {language} code:

```{language}
{code}
```

Provide analysis based on: {analysis_type}"""
    
    try:
        analysis = await call_ollama(prompt, system_prompt)
        return {
            "analysis": analysis.strip(),
            "language": language,
            "analysis_type": analysis_type,
            "code_length": len(code)
        }
    except Exception as e:
        return {"error": str(e)}

async def visualize_data_function(params):
    """Wrapper for data visualization"""
    data_source = params.get("data_source", "")
    chart_type = params.get("chart_type", "auto")
    analysis_focus = params.get("analysis_focus", "trends")
    
    system_prompt = """You are a data analyst expert. Analyze data and provide:
1. Key insights and patterns
2. Statistical summary
3. Visualization recommendations
4. Data quality assessment
5. Actionable recommendations"""
    
    prompt = f"""Analyze this data and provide insights:

Data:
{data_source[:2000]}{"..." if len(data_source) > 2000 else ""}

Chart type preference: {chart_type}
Analysis focus: {analysis_focus}

Provide comprehensive data analysis including patterns, insights, and visualization suggestions."""
    
    try:
        analysis = await call_ollama(prompt, system_prompt)
        return {
            "analysis": analysis.strip(),
            "data_source": data_source,
            "chart_type": chart_type,
            "analysis_focus": analysis_focus,
            "data_preview": data_source[:500] + "..." if len(data_source) > 500 else data_source
        }
    except Exception as e:
        return {"error": str(e)}

async def extract_meeting_notes_function(params):
    """Wrapper for meeting notes extraction"""
    notes = params.get("notes", "")
    extract_actions = params.get("extract_actions", True)
    extract_decisions = params.get("extract_decisions", True)
    
    system_prompt = """You are a meeting notes analyst. Extract and organize information from meeting notes:
1. Action items (who, what, when)
2. Key decisions made
3. Important discussion points
4. Follow-up items
5. Participants and roles"""
    
    extraction_focus = []
    if extract_actions:
        extraction_focus.append("action items with owners and deadlines")
    if extract_decisions:
        extraction_focus.append("decisions made and their implications")
    
    focus_text = ", ".join(extraction_focus) if extraction_focus else "key information"
    
    prompt = f"""Analyze these meeting notes and extract {focus_text}:

Meeting Notes:
{notes}

Please organize the information clearly with sections for action items, decisions, key points, and any follow-up needed."""
    
    try:
        analysis = await call_ollama(prompt, system_prompt)
        return {
            "analysis": analysis.strip(),
            "original_length": len(notes),
            "extracted_actions": extract_actions,
            "extracted_decisions": extract_decisions
        }
    except Exception as e:
        return {"error": str(e)}

async def analyze_research_paper_function(params):
    """Wrapper for research paper analysis"""
    content = params.get("content", "")
    focus_areas = params.get("focus_areas", ["methodology", "findings", "conclusions"])
    
    system_prompt = """You are a research analyst expert. Analyze academic papers and provide:
1. Executive summary
2. Methodology overview
3. Key findings and results
4. Significance and implications
5. Limitations and future work"""
    
    focus_areas_text = ", ".join(focus_areas)
    
    prompt = f"""Analyze this research paper content, focusing on: {focus_areas_text}

Paper Content:
{content[:3000]}{"..." if len(content) > 3000 else ""}

Provide a comprehensive analysis including methodology, findings, significance, and implications."""
    
    try:
        analysis = await call_ollama(prompt, system_prompt)
        return {
            "analysis": analysis.strip(),
            "focus_areas": focus_areas,
            "content_length": len(content),
            "content_preview": content[:500] + "..." if len(content) > 500 else content
        }
    except Exception as e:
        return {"error": str(e)}

@app.on_event("startup")
async def startup_event():
    """Initialize app - database will be lazy loaded"""
    # Initialize workflow engine with agent functions
    agent_functions = {
        "web_search": web_search_function,
        "summarize": summarize_function,
        "draft_email": draft_email_function,
        "local_query": local_query_function,
        "analyze_code": analyze_code_function,
        "visualize_data": visualize_data_function,
        "extract_meeting_notes": extract_meeting_notes_function,
        "analyze_research_paper": analyze_research_paper_function
    }
    global workflow_engine
    workflow_engine = initialize_workflow_engine(agent_functions)
    
    # Initialize integration hub
    await integration_hub.initialize()
    
    # Initialize security tools
    await security_tools.initialize()
    
    # Initialize data explorer
    await data_explorer.initialize()
    
    # Initialize personalization engine
    await personalization_engine.initialize()
    
    logger.info("SLM Personal Agent started with workflow engine, integration hub, security tools, data explorer, and personalization engine")

class TextInput(BaseModel):
    text: str

class EmailRequest(BaseModel):
    recipient: str
    subject: str
    context: str
    tone: Optional[str] = "professional"

class SearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5

class LocalQueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 3

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    mode: str = "chat"

class WorkflowRequest(BaseModel):
    template_id: str
    parameters: dict

class WorkflowExecutionRequest(BaseModel):
    workflow_id: str
    input_parameters: dict


@app.get("/")
async def root():
    """Serve the main frontend"""
    return FileResponse('static/index.html')

@app.post("/chat")
async def chat_with_memory(request: ChatRequest):
    """Enhanced chat with conversation memory"""
    try:
        memory_manager = get_memory_manager()
        await memory_manager.initialize()  # Initialize on first use
        
        # Create new conversation if none provided
        if not request.conversation_id:
            conversation_id = await memory_manager.create_conversation()
        else:
            conversation_id = request.conversation_id
        
        # Add user message to memory
        await memory_manager.add_message(
            conversation_id=conversation_id,
            role="user",
            content=request.message,
            mode=request.mode
        )
        
        # Get conversation context for better responses
        context = await memory_manager.get_conversation_context(conversation_id, max_messages=6)
        
        # Enhanced system prompt with context
        system_prompt = f"""You are a helpful AI assistant. You have access to the conversation history below. 
Use this context to provide more relevant and coherent responses. Refer to previous messages when appropriate.

Conversation History:
{context}

Current mode: {request.mode}
"""
        
        # Generate response based on mode
        if request.mode == "chat":
            prompt = f"Human: {request.message}\n\nAssistant:"
        else:
            prompt = request.message
        
        response = await call_ollama(prompt, system_prompt)
        
        # Add assistant response to memory
        await memory_manager.add_message(
            conversation_id=conversation_id,
            role="assistant", 
            content=response,
            mode=request.mode
        )
        
        # Process conversation for knowledge graph (async, don't wait)
        try:
            asyncio.create_task(knowledge_graph.process_conversation(
                conversation_id, request.message, response
            ))
        except Exception as e:
            logger.warning(f"Knowledge graph processing failed: {e}")
        
        # Learn from interaction for personalization (async, don't wait)
        try:
            asyncio.create_task(personalization_engine.learn_from_interaction(
                user_id="default",
                message=request.message,
                response=response,
                mode=request.mode,
                metadata={"conversation_id": conversation_id}
            ))
        except Exception as e:
            logger.warning(f"Personalization learning failed: {e}")
        
        # Process conversation for reminders and follow-ups (async, don't wait)
        try:
            asyncio.create_task(reminder_system.process_conversation_for_reminders(
                conversation_id, request.message, response
            ))
        except Exception as e:
            logger.warning(f"Reminder processing failed: {e}")
        
        # Update conversation title if this is the first exchange
        messages = await memory_manager.get_conversation_messages(conversation_id)
        if len(messages) <= 2:  # First user + assistant message
            title = await memory_manager.generate_conversation_title(conversation_id)
            await memory_manager.update_conversation_title(conversation_id, title)
        
        return {
            "response": response,
            "conversation_id": conversation_id,
            "mode": request.mode
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations")
async def get_conversations():
    """Get list of conversations"""
    try:
        memory_manager = get_memory_manager()
        await memory_manager.initialize()
        conversations = await memory_manager.get_conversations()
        return {"conversations": conversations}
    except Exception as e:
        logger.error(f"Get conversations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str):
    """Get messages from a specific conversation"""
    try:
        memory_manager = get_memory_manager()
        await memory_manager.initialize()
        messages = await memory_manager.get_conversation_messages(conversation_id)
        return {"messages": messages, "conversation_id": conversation_id}
    except Exception as e:
        logger.error(f"Get messages error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversations")
async def create_conversation():
    """Create a new conversation"""
    try:
        memory_manager = get_memory_manager()
        await memory_manager.initialize()
        conversation_id = await memory_manager.create_conversation()
        return {"conversation_id": conversation_id}
    except Exception as e:
        logger.error(f"Create conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    try:
        memory_manager = get_memory_manager()
        await memory_manager.initialize()
        success = await memory_manager.delete_conversation(conversation_id)
        if success:
            return {"message": "Conversation deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
    except Exception as e:
        logger.error(f"Delete conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        doc_processor = get_doc_processor()
        
        # Check if file type is supported
        if not doc_processor.is_supported(file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Supported: {', '.join([ext for exts in doc_processor.supported_types.values() for ext in exts])}"
            )
        
        # Save the uploaded file
        file_metadata = await doc_processor.save_upload(file)
        
        # Extract text content
        text_content = await doc_processor.extract_text(
            file_metadata["path"], 
            file_metadata["type"]
        )
        
        # Chunk the text for embedding
        chunks = doc_processor.chunk_text(text_content)
        
        if chunks:
            # Generate embeddings for chunks
            embeddings = embedding_model.encode(chunks)
            
            # Prepare metadata for each chunk
            chunk_ids = [f"{file_metadata['saved_as']}_{i}" for i in range(len(chunks))]
            metadatas = [{
                "source": file_metadata["filename"],
                "file_path": file_metadata["path"],
                "file_type": file_metadata["type"],
                "chunk_index": i,
                "uploaded_at": file_metadata["uploaded_at"],
                "file_hash": file_metadata["hash"]
            } for i in range(len(chunks))]
            
            # Add to ChromaDB
            collection.add(
                embeddings=embeddings.tolist(),
                documents=chunks,
                metadatas=metadatas,
                ids=chunk_ids
            )
            
            logger.info(f"Processed {file.filename}: {len(chunks)} chunks added to knowledge base")
        
        return {
            "message": "File uploaded and processed successfully",
            "filename": file_metadata["filename"],
            "type": file_metadata["type"],
            "size": file_metadata["size"],
            "chunks_created": len(chunks),
            "text_preview": text_content[:500] + "..." if len(text_content) > 500 else text_content
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    try:
        doc_processor = get_doc_processor()
        files = await doc_processor.list_uploaded_files()
        
        # Get collection stats
        try:
            total_chunks = collection.count()
        except:
            total_chunks = 0
        
        return {
            "files": files,
            "total_files": len(files),
            "total_chunks": total_chunks
        }
        
    except Exception as e:
        logger.error(f"List documents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document and its embeddings"""
    try:
        doc_processor = get_doc_processor()
        
        # Find the file
        files = await doc_processor.list_uploaded_files()
        target_file = None
        for file in files:
            if file["name"] == filename:
                target_file = file
                break
        
        if not target_file:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Delete from file system
        deleted = await doc_processor.delete_file(target_file["path"])
        
        if deleted:
            # Delete from ChromaDB (find chunks by filename)
            try:
                # Get all documents and find ones from this file
                results = collection.get()
                ids_to_delete = []
                
                for i, metadata in enumerate(results['metadatas']):
                    if metadata and metadata.get('source') == filename:
                        ids_to_delete.append(results['ids'][i])
                
                if ids_to_delete:
                    collection.delete(ids=ids_to_delete)
                    logger.info(f"Deleted {len(ids_to_delete)} chunks for {filename}")
                
            except Exception as e:
                logger.warning(f"Error deleting embeddings for {filename}: {e}")
            
            return {"message": f"Document {filename} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete file")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/supported-types")
async def get_supported_types():
    """Get list of supported file types"""
    doc_processor = get_doc_processor()
    return {
        "supported_types": doc_processor.supported_types,
        "all_extensions": [ext for exts in doc_processor.supported_types.values() for ext in exts]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    ollama_healthy = await check_ollama_health()
    return {
        "status": "healthy" if ollama_healthy else "unhealthy", 
        "ollama_connected": ollama_healthy
    }

@app.post("/summarize")
async def summarize_text(request: TextInput):
    """Summarize the provided text"""
    system_prompt = "You are a concise summarization assistant. Create clear, structured summaries that capture the key points."
    
    prompt = f"""Please summarize the following text in 2-3 bullet points, focusing on the most important information:

Text: {request.text}

Summary:"""
    
    try:
        summary = await call_ollama(prompt, system_prompt)
        logger.info(f"Summarization completed for text length: {len(request.text)}")
        return {"summary": summary.strip(), "original_length": len(request.text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/draft-email")
async def draft_email(request: EmailRequest):
    """Draft an email based on the provided context"""
    system_prompt = f"You are an email drafting assistant. Write professional emails in a {request.tone} tone. Include proper greeting, body, and closing."
    
    prompt = f"""Draft an email with the following details:

To: {request.recipient}
Subject: {request.subject}
Context: {request.context}
Tone: {request.tone}

Please write a complete email draft:"""
    
    try:
        draft = await call_ollama(prompt, system_prompt)
        logger.info(f"Email draft created for recipient: {request.recipient}")
        return {
            "draft": draft.strip(),
            "recipient": request.recipient,
            "subject": request.subject,
            "tone": request.tone
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def search_with_serpapi(query: str, max_results: int = 5) -> List[dict]:
    """Search using SerpAPI (Google Search)"""
    if not SERPAPI_KEY:
        raise ValueError("SERPAPI_KEY not configured")
    
    try:
        search = GoogleSearch({
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": min(max_results, 10),
            "safe": "active"
        })
        
        results = search.get_dict()
        
        if "error" in results:
            raise ValueError(f"SerpAPI error: {results['error']}")
        
        search_results = []
        organic_results = results.get("organic_results", [])
        
        for result in organic_results[:max_results]:
            search_results.append({
                "title": result.get("title", ""),
                "url": result.get("link", ""),
                "snippet": result.get("snippet", "")
            })
        
        return search_results
    except Exception as e:
        logger.error(f"SerpAPI search error: {e}")
        raise

async def search_with_bing(query: str, max_results: int = 5) -> List[dict]:
    """Search using Bing Search API"""
    if not BING_SEARCH_KEY:
        raise ValueError("BING_SEARCH_KEY not configured")
    
    try:
        headers = {"Ocp-Apim-Subscription-Key": BING_SEARCH_KEY}
        params = {
            "q": query,
            "count": min(max_results, 10),
            "safeSearch": "Moderate",
            "textFormat": "Raw"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.bing.microsoft.com/v7.0/search",
                headers=headers,
                params=params,
                timeout=10.0
            )
            response.raise_for_status()
            
            data = response.json()
            search_results = []
            
            for result in data.get("webPages", {}).get("value", [])[:max_results]:
                search_results.append({
                    "title": result.get("name", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("snippet", "")
                })
            
            return search_results
    except Exception as e:
        logger.error(f"Bing search error: {e}")
        raise

def get_mock_results(query: str, max_results: int = 5) -> List[dict]:
    """Get mock search results for demo purposes"""
    mock_results = [
        {
            "title": f"Latest research on {query} - 2025 Advances",
            "url": "https://example.com/research-2025",
            "snippet": f"Recent developments in {query} show promising results with new applications..."
        },
        {
            "title": f"Industry applications of {query}",
            "url": "https://example.com/industry-apps",
            "snippet": f"How {query} is transforming various industries including healthcare, finance..."
        },
        {
            "title": f"Future trends in {query}",
            "url": "https://example.com/future-trends",
            "snippet": f"Experts predict significant growth in {query} with emerging technologies..."
        }
    ]
    return mock_results[:max_results]

@app.post("/web-search")
async def web_search(request: SearchRequest):
    """Perform web search and summarize results"""
    try:
        search_results = []
        provider_used = SEARCH_PROVIDER
        
        # Try different search providers based on configuration
        if SEARCH_PROVIDER == "serpapi" and SERPAPI_KEY:
            try:
                search_results = await search_with_serpapi(request.query, request.max_results)
                provider_used = "serpapi"
            except Exception as e:
                logger.warning(f"SerpAPI failed, trying fallback: {e}")
                if BING_SEARCH_KEY:
                    search_results = await search_with_bing(request.query, request.max_results)
                    provider_used = "bing"
                else:
                    search_results = get_mock_results(request.query, request.max_results)
                    provider_used = "mock"
        
        elif SEARCH_PROVIDER == "bing" and BING_SEARCH_KEY:
            try:
                search_results = await search_with_bing(request.query, request.max_results)
                provider_used = "bing"
            except Exception as e:
                logger.warning(f"Bing search failed, trying fallback: {e}")
                if SERPAPI_KEY:
                    search_results = await search_with_serpapi(request.query, request.max_results)
                    provider_used = "serpapi"
                else:
                    search_results = get_mock_results(request.query, request.max_results)
                    provider_used = "mock"
        
        else:
            # Use mock results as fallback
            search_results = get_mock_results(request.query, request.max_results)
            provider_used = "mock"
        
        if not search_results:
            return {
                "query": request.query,
                "results": [],
                "summary": "No search results found.",
                "provider": provider_used
            }
        
        # Create context for AI summarization
        results_text = "\n".join([
            f"- {r['title']}: {r.get('snippet', 'No description available')}"
            for r in search_results
        ])
        
        system_prompt = "You are a web search summarizer. Analyze the search results and provide a concise, informative summary."
        prompt = f"""Based on these search results for "{request.query}", provide a brief summary:

{results_text}

Summary:"""
        
        summary = await call_ollama(prompt, system_prompt)
        
        logger.info(f"Web search completed for query: {request.query} using {provider_used}")
        
        response = {
            "query": request.query,
            "results": search_results,
            "summary": summary.strip(),
            "provider": provider_used
        }
        
        if provider_used == "mock":
            response["note"] = "Using mock results. Configure SERPAPI_KEY or BING_SEARCH_KEY for real search."
        
        return response
        
    except Exception as e:
        logger.error(f"Web search error: {e}")
        raise HTTPException(status_code=500, detail=f"Web search failed: {e}")

@app.post("/local-query")
async def local_query(request: LocalQueryRequest):
    """Query local documents using vector similarity"""
    try:
        query_embedding = embedding_model.encode([request.query])
        
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=request.max_results
        )
        
        if not results['documents'][0]:
            return {
                "query": request.query,
                "answer": "No relevant documents found in the local knowledge base.",
                "sources": []
            }
        
        context_docs = results['documents'][0]
        sources = results['metadatas'][0] if results['metadatas'][0] else []
        
        context = "\n\n".join([f"Document: {doc}" for doc in context_docs])
        
        system_prompt = "You are a helpful assistant that answers questions based on provided context. Only use information from the given documents."
        
        prompt = f"""Based on the following documents, answer the question: "{request.query}"

Context:
{context}

Answer:"""
        
        answer = await call_ollama(prompt, system_prompt)
        
        source_files = [meta.get('source', 'Unknown') for meta in sources] if sources else []
        
        logger.info(f"Local query completed: {request.query}")
        return {
            "query": request.query,
            "answer": answer.strip(),
            "sources": source_files,
            "num_documents_used": len(context_docs)
        }
        
    except Exception as e:
        logger.error(f"Local query error: {e}")
        raise HTTPException(status_code=500, detail=f"Local query failed: {e}")

# Level 4: Specialized Analysis Endpoints

class CodeAnalysisRequest(BaseModel):
    code: str
    language: Optional[str] = "auto"
    analysis_type: str = "comprehensive"  # 'comprehensive', 'syntax', 'documentation', 'security'

class DataVisualizationRequest(BaseModel):
    data_source: str  # file name or data
    chart_type: str = "auto"  # 'auto', 'bar', 'line', 'pie', 'scatter'
    analysis_focus: Optional[str] = "trends"

class MeetingNotesRequest(BaseModel):
    notes: str
    extract_actions: bool = True
    extract_decisions: bool = True

class ResearchPaperRequest(BaseModel):
    content: str
    focus_areas: List[str] = ["methodology", "findings", "conclusions"]

@app.post("/analyze-code")
async def analyze_code(request: CodeAnalysisRequest):
    """Analyze code for syntax, documentation, and best practices"""
    try:
        # Detect language if auto
        language = request.language
        if language == "auto":
            language = detect_programming_language(request.code)
        
        # Create analysis prompt based on type
        if request.analysis_type == "comprehensive":
            system_prompt = f"""You are a senior {language} developer and code reviewer. Provide comprehensive code analysis including:
1. Code quality and best practices
2. Potential bugs or issues
3. Performance considerations
4. Documentation suggestions
5. Security considerations"""
        elif request.analysis_type == "syntax":
            system_prompt = f"You are a {language} syntax checker. Focus on syntax errors, warnings, and code correctness."
        elif request.analysis_type == "documentation":
            system_prompt = f"You are a documentation expert for {language}. Focus on code documentation, comments, and readability."
        elif request.analysis_type == "security":
            system_prompt = f"You are a security analyst for {language} code. Focus on security vulnerabilities and best practices."
        else:
            system_prompt = f"You are a {language} code analyst. Provide general code analysis."
        
        prompt = f"""Analyze this {language} code:

```{language}
{request.code}
```

Provide analysis based on: {request.analysis_type}"""
        
        analysis = await call_ollama(prompt, system_prompt)
        
        logger.info(f"Code analysis completed for {language} code ({request.analysis_type})")
        return {
            "analysis": analysis.strip(),
            "language": language,
            "analysis_type": request.analysis_type,
            "code_length": len(request.code)
        }
        
    except Exception as e:
        logger.error(f"Code analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Code analysis failed: {e}")

@app.post("/visualize-data")
async def visualize_data(request: DataVisualizationRequest):
    """Generate data insights and visualization suggestions"""
    try:
        # Check if data_source is a filename (uploaded document)
        doc_processor = get_doc_processor()
        data_content = ""
        
        files = await doc_processor.list_uploaded_files()
        source_file = None
        for file in files:
            if file["name"] == request.data_source:
                source_file = file
                break
        
        if source_file:
            # Extract data from uploaded file
            data_content = await doc_processor.extract_text(source_file["path"], source_file["type"])
        else:
            # Treat as direct data input
            data_content = request.data_source
        
        system_prompt = """You are a data analyst expert. Analyze data and provide:
1. Key insights and patterns
2. Statistical summary
3. Visualization recommendations
4. Data quality assessment
5. Actionable recommendations"""
        
        prompt = f"""Analyze this data and provide insights:

Data:
{data_content[:2000]}{"..." if len(data_content) > 2000 else ""}

Chart type preference: {request.chart_type}
Analysis focus: {request.analysis_focus}

Provide comprehensive data analysis including patterns, insights, and visualization suggestions."""
        
        analysis = await call_ollama(prompt, system_prompt)
        
        logger.info(f"Data visualization analysis completed for: {request.data_source}")
        return {
            "analysis": analysis.strip(),
            "data_source": request.data_source,
            "chart_type": request.chart_type,
            "analysis_focus": request.analysis_focus,
            "data_preview": data_content[:500] + "..." if len(data_content) > 500 else data_content
        }
        
    except Exception as e:
        logger.error(f"Data visualization error: {e}")
        raise HTTPException(status_code=500, detail=f"Data visualization failed: {e}")

@app.post("/extract-meeting-notes")
async def extract_meeting_notes(request: MeetingNotesRequest):
    """Extract action items, decisions, and key points from meeting notes"""
    try:
        system_prompt = """You are a meeting notes analyst. Extract and organize information from meeting notes:
1. Action items (who, what, when)
2. Key decisions made
3. Important discussion points
4. Follow-up items
5. Participants and roles"""
        
        extraction_focus = []
        if request.extract_actions:
            extraction_focus.append("action items with owners and deadlines")
        if request.extract_decisions:
            extraction_focus.append("decisions made and their implications")
        
        focus_text = ", ".join(extraction_focus) if extraction_focus else "key information"
        
        prompt = f"""Analyze these meeting notes and extract {focus_text}:

Meeting Notes:
{request.notes}

Please organize the information clearly with sections for action items, decisions, key points, and any follow-up needed."""
        
        analysis = await call_ollama(prompt, system_prompt)
        
        logger.info("Meeting notes analysis completed")
        return {
            "analysis": analysis.strip(),
            "original_length": len(request.notes),
            "extracted_actions": request.extract_actions,
            "extracted_decisions": request.extract_decisions
        }
        
    except Exception as e:
        logger.error(f"Meeting notes extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Meeting notes extraction failed: {e}")

@app.post("/analyze-research-paper")
async def analyze_research_paper(request: ResearchPaperRequest):
    """Analyze research papers and extract key findings"""
    try:
        system_prompt = """You are a research analyst expert. Analyze academic papers and provide:
1. Executive summary
2. Methodology overview
3. Key findings and results
4. Significance and implications
5. Limitations and future work"""
        
        focus_areas = ", ".join(request.focus_areas)
        
        prompt = f"""Analyze this research paper content, focusing on: {focus_areas}

Paper Content:
{request.content[:3000]}{"..." if len(request.content) > 3000 else ""}

Provide a comprehensive analysis including methodology, findings, significance, and implications."""
        
        analysis = await call_ollama(prompt, system_prompt)
        
        logger.info(f"Research paper analysis completed, focus: {focus_areas}")
        return {
            "analysis": analysis.strip(),
            "focus_areas": request.focus_areas,
            "content_length": len(request.content),
            "content_preview": request.content[:500] + "..." if len(request.content) > 500 else request.content
        }
        
    except Exception as e:
        logger.error(f"Research paper analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Research paper analysis failed: {e}")

def detect_programming_language(code: str) -> str:
    """Simple language detection based on code patterns"""
    code_lower = code.lower()
    
    # Python
    if 'def ' in code or 'import ' in code or 'from ' in code or 'print(' in code:
        return "python"
    # JavaScript/TypeScript
    elif 'function' in code or 'const ' in code or 'let ' in code or 'var ' in code:
        return "javascript"
    # Java
    elif 'public class' in code or 'public static void main' in code:
        return "java"
    # C/C++
    elif '#include' in code or 'int main(' in code:
        return "c"
    # Go
    elif 'package main' in code or 'func main(' in code:
        return "go"
    # Rust
    elif 'fn main(' in code or 'let mut' in code:
        return "rust"
    # SQL
    elif 'select ' in code_lower or 'insert ' in code_lower or 'update ' in code_lower:
        return "sql"
    # HTML
    elif '<html' in code_lower or '<!doctype' in code_lower:
        return "html"
    # CSS
    elif '{' in code and '}' in code and ':' in code and ';' in code:
        return "css"
    else:
        return "unknown"

# Knowledge Graph Endpoints

@app.get("/knowledge/stats")
async def get_knowledge_stats():
    """Get statistics about the personal knowledge graph"""
    try:
        stats = await knowledge_graph.get_graph_stats()
        return {"stats": stats}
    except Exception as e:
        logger.error(f"Knowledge stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge/insights")
async def get_knowledge_insights():
    """Get AI-generated insights from the knowledge graph"""
    try:
        insights = await knowledge_graph.generate_insights()
        return {"insights": insights}
    except Exception as e:
        logger.error(f"Knowledge insights error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge/related/{entity_name}")
async def get_related_entities(entity_name: str, max_results: int = 5):
    """Find entities related to a given entity"""
    try:
        related = await knowledge_graph.find_related_entities(entity_name, max_results)
        return {"entity": entity_name, "related": related}
    except Exception as e:
        logger.error(f"Related entities error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge/process-document")
async def process_document_knowledge(document_name: str, content: str = ""):
    """Process a document to extract knowledge (manual trigger)"""
    try:
        if not content:
            # Try to read from uploaded documents
            doc_processor = get_doc_processor()
            files = await doc_processor.list_uploaded_files()
            
            target_file = None
            for file in files:
                if file["name"] == document_name:
                    target_file = file
                    break
            
            if target_file:
                content = await doc_processor.extract_text(target_file["path"], target_file["type"])
            else:
                raise HTTPException(status_code=404, detail="Document not found")
        
        await knowledge_graph.process_document(document_name, content)
        return {"message": f"Successfully processed {document_name} for knowledge extraction"}
    except Exception as e:
        logger.error(f"Process document knowledge error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Voice Interface Endpoints

@app.get("/voice/capabilities")
async def get_voice_capabilities():
    """Get voice interface capabilities"""
    try:
        capabilities = voice_interface.get_capabilities()
        return {"capabilities": capabilities}
    except Exception as e:
        logger.error(f"Voice capabilities error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voice/voices")
async def get_available_voices():
    """Get list of available TTS voices"""
    try:
        voices = await voice_interface.get_available_voices()
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Get voices error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe audio file to text using speech recognition"""
    try:
        # Log detailed information about the uploaded file
        logger.info(f"Received file: filename={audio.filename}, content_type={audio.content_type}, size={audio.size}")
        
        # More flexible content type checking
        allowed_types = ['audio/', 'video/', 'application/'] # WebM can be various types
        if audio.content_type and not any(audio.content_type.startswith(t) for t in allowed_types):
            logger.warning(f"Unexpected content type: {audio.content_type}, but proceeding with transcription")
        
        # Check if file has content
        if audio.size == 0:
            raise HTTPException(status_code=400, detail="Audio file is empty")
        
        logger.info(f"Starting transcription for: {audio.filename}")
        text = await voice_interface.transcribe_audio(audio)
        logger.info(f"Transcription successful: '{text}'")
        return {"text": text, "filename": audio.filename}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice/speak")
async def text_to_speech(request: VoiceRequest):
    """Convert text to speech and return audio file"""
    try:
        audio_path = await voice_interface.text_to_speech(request)
        
        # Return the audio file
        return FileResponse(
            audio_path, 
            media_type='audio/wav',
            filename=f"speech_{hash(request.text) % 10000}.wav",
            headers={"Cache-Control": "no-cache"}
        )
    except Exception as e:
        logger.error(f"Text-to-speech error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class VoiceChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    voice: str = "default"
    speed: float = 1.0

@app.post("/voice/chat")
async def voice_chat_text(request: VoiceChatRequest):
    """Voice chat with text input and conversation memory"""
    try:
        # Create ChatRequest object for chat_with_memory
        chat_request = ChatRequest(
            message=request.message,
            conversation_id=request.conversation_id,
            mode="voice"
        )
        
        # Process with AI chat using conversation memory
        response_data = await chat_with_memory(chat_request)
        response_text = response_data["response"]
        conversation_id = response_data["conversation_id"]
        
        # Convert response to speech
        tts_request = VoiceRequest(text=response_text, voice=request.voice, speed=request.speed)
        audio_path = await voice_interface.text_to_speech(tts_request)
        
        # Return both text and audio
        return {
            "response": response_text,
            "conversation_id": conversation_id,
            "audio_url": f"/voice/audio/{audio_path.split('/')[-1]}",
            "audio_file": audio_path
        }
    except Exception as e:
        logger.error(f"Voice chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice/chat-audio")
async def voice_chat_audio(audio_file: UploadFile = File(...), voice: str = "default", speed: float = 1.0):
    """Complete voice interaction: transcribe input, process with AI, return speech"""
    try:
        # Step 1: Transcribe audio to text
        user_text = await voice_interface.transcribe_audio(audio_file)
        logger.info(f"Transcribed: {user_text}")
        
        # Step 2: Process with AI chat
        system_prompt = "You are a helpful AI assistant. Provide concise, clear responses suitable for voice interaction."
        response_text = await call_ollama(f"Human: {user_text}\n\nAssistant:", system_prompt)
        
        # Step 3: Convert response to speech
        tts_request = VoiceRequest(text=response_text, voice=voice, speed=speed)
        audio_path = await voice_interface.text_to_speech(tts_request)
        
        # Return both text and audio
        return {
            "user_text": user_text,
            "assistant_text": response_text,
            "audio_url": f"/voice/audio/{audio_path.split('/')[-1]}",
            "audio_file": audio_path
        }
    except Exception as e:
        logger.error(f"Voice chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voice/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve generated audio files"""
    import tempfile
    audio_path = os.path.join(tempfile.gettempdir(), filename)
    
    if os.path.exists(audio_path):
        return FileResponse(audio_path, media_type='audio/wav')
    else:
        raise HTTPException(status_code=404, detail="Audio file not found")

# Workflow Management Endpoints

@app.get("/workflows/templates")
async def get_workflow_templates():
    """Get all available workflow templates"""
    try:
        if workflow_engine is None:
            raise HTTPException(status_code=503, detail="Workflow engine not initialized")
        templates = workflow_engine.get_workflow_templates()
        return {"templates": templates}
    except Exception as e:
        logger.error(f"Get workflow templates error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workflows/create")
async def create_workflow(request: WorkflowRequest):
    """Create a new workflow from a template"""
    try:
        workflow_id = workflow_engine.create_workflow_from_template(
            request.template_id, 
            request.parameters
        )
        return {"workflow_id": workflow_id, "message": "Workflow created successfully"}
    except Exception as e:
        logger.error(f"Create workflow error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workflows/execute")
async def execute_workflow(request: WorkflowExecutionRequest):
    """Execute a workflow with input parameters"""
    try:
        execution_id = await workflow_engine.execute_workflow(
            request.workflow_id, 
            request.input_parameters
        )
        return {"execution_id": execution_id, "message": "Workflow execution started"}
    except Exception as e:
        logger.error(f"Execute workflow error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflows/execution/{execution_id}")
async def get_workflow_execution(execution_id: str):
    """Get the status and results of a workflow execution"""
    try:
        execution = workflow_engine.get_execution_status(execution_id)
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        response = {
            "execution_id": execution.id,
            "workflow_id": execution.workflow_id,
            "status": execution.status,
            "current_step": execution.current_step,
            "started_at": execution.started_at,
            "completed_at": execution.completed_at,
            "error_message": execution.error_message
        }
        
        if execution.status == "completed":
            results = workflow_engine.get_execution_results(execution_id)
            response["results"] = results
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get workflow execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workflows/quick-execute")
async def quick_execute_workflow(request: dict):
    """Quick execute a workflow template with parameters in one call"""
    try:
        template_id = request.get("template_id")
        parameters = request.get("parameters", {})
        
        if not template_id:
            raise HTTPException(status_code=400, detail="template_id is required")
        
        # Create and execute workflow in one step
        workflow_id = workflow_engine.create_workflow_from_template(template_id, parameters)
        execution_id = await workflow_engine.execute_workflow(workflow_id, parameters)
        
        # Get results
        execution = workflow_engine.get_execution_status(execution_id)
        results = workflow_engine.get_execution_results(execution_id) if execution.status == "completed" else None
        
        return {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "status": execution.status,
            "results": results,
            "message": "Workflow executed successfully" if execution.status == "completed" else "Workflow execution in progress"
        }
    except Exception as e:
        logger.error(f"Quick execute workflow error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Smart Recommendations Endpoints

@app.get("/recommendations")
async def get_recommendations(limit: int = 5):
    """Get personalized recommendations for the user"""
    try:
        recommendations = await recommendations_engine.get_recommendations(limit)
        
        return {
            "recommendations": [
                {
                    "id": rec.id,
                    "type": rec.type,
                    "title": rec.title,
                    "description": rec.description,
                    "confidence": rec.confidence,
                    "reasoning": rec.reasoning,
                    "action_data": rec.action_data,
                    "priority": rec.priority
                }
                for rec in recommendations
            ],
            "total": len(recommendations),
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Get recommendations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations/refresh")
async def refresh_recommendations():
    """Force refresh of recommendations cache"""
    try:
        # Clear cache
        recommendations_engine.recommendations_cache = {}
        
        # Generate fresh recommendations
        recommendations = await recommendations_engine.get_recommendations()
        
        return {
            "message": "Recommendations refreshed successfully",
            "count": len(recommendations),
            "refreshed_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Refresh recommendations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommendations/insights")
async def get_user_insights():
    """Get user pattern insights and analytics"""
    try:
        # Get user patterns
        patterns = await recommendations_engine._analyze_user_patterns()
        
        # Get conversation insights
        conv_insights = await recommendations_engine._analyze_recent_conversations()
        
        # Get knowledge graph insights
        kg_insights = await recommendations_engine._get_knowledge_graph_insights()
        
        return {
            "user_patterns": {
                pattern_name: {
                    "type": pattern.pattern_type,
                    "frequency": pattern.frequency,
                    "last_occurrence": pattern.last_occurrence.isoformat(),
                    "context": pattern.context
                }
                for pattern_name, pattern in patterns.items()
            },
            "conversation_insights": conv_insights,
            "knowledge_graph_insights": kg_insights,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Get user insights error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Task Reminders & Follow-up Endpoints

@app.get("/reminders")
async def get_reminders(limit: int = 50):
    """Get all active reminders"""
    try:
        reminders = await reminder_system.get_active_reminders(limit)
        return {
            "reminders": reminders,
            "total": len(reminders)
        }
    except Exception as e:
        logger.error(f"Get reminders error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reminders/due")
async def get_due_reminders(hours_ahead: int = 24):
    """Get reminders that are due within specified hours"""
    try:
        due_reminders = await reminder_system.get_due_reminders(hours_ahead)
        return {
            "due_reminders": due_reminders,
            "total": len(due_reminders),
            "hours_ahead": hours_ahead
        }
    except Exception as e:
        logger.error(f"Get due reminders error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reminders/{reminder_id}/complete")
async def complete_reminder(reminder_id: str, notes: str = ""):
    """Mark a reminder as completed"""
    try:
        success = await reminder_system.update_reminder_status(
            reminder_id, ReminderStatus.COMPLETED, notes
        )
        
        if success:
            return {"message": "Reminder marked as completed", "reminder_id": reminder_id}
        else:
            raise HTTPException(status_code=404, detail="Reminder not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Complete reminder error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reminders/{reminder_id}/snooze")
async def snooze_reminder(reminder_id: str, hours: int = 24):
    """Snooze a reminder for specified hours"""
    try:
        success = await reminder_system.snooze_reminder(reminder_id, hours)
        
        if success:
            return {
                "message": f"Reminder snoozed for {hours} hours",
                "reminder_id": reminder_id,
                "snoozed_hours": hours
            }
        else:
            raise HTTPException(status_code=404, detail="Reminder not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Snooze reminder error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reminders/{reminder_id}/cancel")
async def cancel_reminder(reminder_id: str, reason: str = ""):
    """Cancel a reminder"""
    try:
        success = await reminder_system.update_reminder_status(
            reminder_id, ReminderStatus.CANCELLED, reason
        )
        
        if success:
            return {"message": "Reminder cancelled", "reminder_id": reminder_id}
        else:
            raise HTTPException(status_code=404, detail="Reminder not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cancel reminder error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reminders/stats")
async def get_reminder_statistics():
    """Get reminder statistics and analytics"""
    try:
        stats = await reminder_system.get_reminder_statistics()
        return {
            "statistics": stats,
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Get reminder stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reminders/check-overdue")
async def check_overdue_reminders():
    """Manually check and update overdue reminders"""
    try:
        await reminder_system.check_overdue_reminders()
        return {"message": "Overdue reminders checked and updated"}
    except Exception as e:
        logger.error(f"Check overdue reminders error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== Integration Hub Endpoints ==============

class IntegrationRequest(BaseModel):
    service_id: str
    api_key: Optional[str] = None
    additional_config: Optional[dict] = None

class IntegrationActionRequest(BaseModel):
    service_id: str
    action_type: str
    parameters: dict

@app.get("/integrations/available")
async def get_available_integrations():
    """Get list of available integrations"""
    try:
        integrations = await integration_hub.get_available_integrations()
        return {
            "integrations": integrations,
            "count": len(integrations)
        }
    except Exception as e:
        logger.error(f"Get available integrations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/integrations/user")
async def get_user_integrations():
    """Get user's active integrations"""
    try:
        integrations = await integration_hub.get_user_integrations()
        return {
            "integrations": integrations,
            "count": len(integrations)
        }
    except Exception as e:
        logger.error(f"Get user integrations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/integrations/oauth/start")
async def start_oauth_integration(service_id: str, redirect_uri: str, state: str = None):
    """Start OAuth2 integration flow"""
    try:
        auth_url = await integration_hub.start_oauth_flow(service_id, redirect_uri, state)
        return {
            "auth_url": auth_url,
            "service_id": service_id,
            "redirect_uri": redirect_uri
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Start OAuth integration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/integrations/oauth/complete")
async def complete_oauth_integration(service_id: str, code: str, state: str = None):
    """Complete OAuth2 integration flow"""
    try:
        success = await integration_hub.complete_oauth_flow(service_id, code, state)
        
        if success:
            return {
                "message": f"{service_id} integration completed successfully",
                "service_id": service_id,
                "status": "active"
            }
        else:
            raise HTTPException(status_code=400, detail="OAuth flow completion failed")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Complete OAuth integration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/integrations/api-key")
async def add_api_key_integration(request: IntegrationRequest):
    """Add an API key-based integration"""
    try:
        if not request.api_key:
            raise HTTPException(status_code=400, detail="API key is required")
        
        success = await integration_hub.add_api_key_integration(
            request.service_id, 
            request.api_key, 
            request.additional_config
        )
        
        if success:
            return {
                "message": f"{request.service_id} integration added successfully",
                "service_id": request.service_id,
                "status": "active"
            }
        else:
            raise HTTPException(status_code=400, detail="Integration setup failed")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Add API key integration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/integrations/action")
async def execute_integration_action(request: IntegrationActionRequest):
    """Execute an action with an integrated service"""
    try:
        action = await integration_hub.execute_integration_action(
            request.service_id,
            request.action_type,
            request.parameters
        )
        
        return {
            "action": {
                "id": action.id,
                "service_id": action.service_id,
                "action_type": action.action_type,
                "name": action.name,
                "description": action.description,
                "result": action.result,
                "executed_at": action.executed_at.isoformat() if action.executed_at else None
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Execute integration action error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/integrations/{service_id}")
async def disconnect_integration(service_id: str):
    """Disconnect an integration"""
    try:
        success = await integration_hub.disconnect_integration(service_id)
        
        if success:
            return {
                "message": f"{service_id} integration disconnected successfully",
                "service_id": service_id
            }
        else:
            raise HTTPException(status_code=404, detail="Integration not found")
    except Exception as e:
        logger.error(f"Disconnect integration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/integrations/suggestions")
async def get_integration_suggestions():
    """Get AI-powered integration suggestions based on recent conversations"""
    try:
        # Get recent conversation context
        memory_manager = get_memory_manager()
        conversations = await memory_manager.get_conversations(5)
        
        conversation_text = ""
        for conv in conversations[:3]:  # Use last 3 conversations
            messages = await memory_manager.get_conversation_messages(conv["id"], 10)
            for msg in messages:
                if msg["role"] == "user":
                    conversation_text += f"User: {msg['content']}\n"
        
        suggestions = await integration_hub.get_integration_suggestions(conversation_text)
        
        return {
            "suggestions": suggestions,
            "count": len(suggestions),
            "based_on": "recent conversation analysis"
        }
    except Exception as e:
        logger.error(f"Get integration suggestions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== Security and Privacy Tools Endpoints ==============

class DataClassificationRequest(BaseModel):
    data_type: str
    data_id: str
    content: str

class SecurityEventRequest(BaseModel):
    event_type: str
    action: str
    resource: Optional[str] = None
    event_data: Optional[dict] = None

@app.post("/security/classify")
async def classify_data(request: DataClassificationRequest):
    """Classify data and determine security requirements"""
    try:
        classification = await security_tools.classify_data(
            request.data_type,
            request.data_id,
            request.content
        )
        
        # Log the classification event
        await security_tools.log_security_event(
            AuditEventType.DATA_ACCESS,
            "data_classification",
            resource=f"{request.data_type}:{request.data_id}",
            event_data={"classification": classification}
        )
        
        return {
            "classification": classification,
            "message": "Data classified successfully"
        }
    except Exception as e:
        logger.error(f"Data classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/security/encrypt")
async def encrypt_data(data: dict):
    """Encrypt sensitive data"""
    try:
        text_to_encrypt = data.get("text", "")
        if not text_to_encrypt:
            raise HTTPException(status_code=400, detail="Text to encrypt is required")
        
        encrypted_text = await security_tools.encrypt_data(text_to_encrypt)
        
        # Log encryption event
        await security_tools.log_security_event(
            AuditEventType.DATA_ACCESS,
            "data_encryption",
            event_data={"data_length": len(text_to_encrypt)}
        )
        
        return {
            "encrypted_data": encrypted_text,
            "message": "Data encrypted successfully"
        }
    except Exception as e:
        logger.error(f"Data encryption error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/security/decrypt")
async def decrypt_data(data: dict):
    """Decrypt sensitive data"""
    try:
        encrypted_text = data.get("encrypted_data", "")
        if not encrypted_text:
            raise HTTPException(status_code=400, detail="Encrypted data is required")
        
        decrypted_text = await security_tools.decrypt_data(encrypted_text)
        
        # Log decryption event (high risk)
        await security_tools.log_security_event(
            AuditEventType.DATA_ACCESS,
            "data_decryption",
            event_data={"high_risk": True}
        )
        
        return {
            "decrypted_data": decrypted_text,
            "message": "Data decrypted successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Data decryption error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/security/anonymize")
async def anonymize_data(data: dict):
    """Anonymize data while preserving utility"""
    try:
        text_to_anonymize = data.get("text", "")
        anonymization_level = data.get("level", "standard")
        
        if not text_to_anonymize:
            raise HTTPException(status_code=400, detail="Text to anonymize is required")
        
        anonymized_text = await security_tools.anonymize_data(text_to_anonymize, anonymization_level)
        
        # Log anonymization event
        await security_tools.log_security_event(
            AuditEventType.DATA_ACCESS,
            "data_anonymization",
            event_data={
                "level": anonymization_level,
                "original_length": len(text_to_anonymize),
                "anonymized_length": len(anonymized_text)
            }
        )
        
        return {
            "anonymized_data": anonymized_text,
            "level": anonymization_level,
            "message": "Data anonymized successfully"
        }
    except Exception as e:
        logger.error(f"Data anonymization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/security/log-event")
async def log_security_event(request: SecurityEventRequest):
    """Log a security event for audit trail"""
    try:
        event_type = AuditEventType(request.event_type)
        
        event_id = await security_tools.log_security_event(
            event_type,
            request.action,
            request.resource,
            request.event_data
        )
        
        return {
            "event_id": event_id,
            "message": "Security event logged successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid event type: {e}")
    except Exception as e:
        logger.error(f"Security event logging error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/security/dashboard")
async def get_security_dashboard():
    """Get security dashboard with metrics and alerts"""
    try:
        dashboard_data = await security_tools.get_security_dashboard()
        return {
            "dashboard": dashboard_data,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Security dashboard error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/security/scan")
async def run_security_scan():
    """Run comprehensive security scan"""
    try:
        scan_results = await security_tools.run_security_scan()
        return {
            "scan": scan_results,
            "message": "Security scan completed"
        }
    except Exception as e:
        logger.error(f"Security scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/security/audit-logs")
async def get_audit_logs(limit: int = 50, event_type: str = None):
    """Get audit logs with optional filtering"""
    try:
        await security_tools.initialize()
        
        async with security_tools.async_session() as session:
            from sqlalchemy import select, desc
            from .security_tools import AuditLog
            
            query = select(AuditLog).order_by(desc(AuditLog.timestamp))
            
            if event_type:
                query = query.where(AuditLog.event_type == event_type)
            
            query = query.limit(limit)
            result = await session.execute(query)
            logs = result.scalars().all()
            
            return {
                "logs": [
                    {
                        "id": log.id,
                        "event_type": log.event_type,
                        "action": log.action,
                        "resource": log.resource,
                        "risk_score": log.risk_score,
                        "anomaly_detected": log.anomaly_detected,
                        "security_level": log.security_level,
                        "timestamp": log.timestamp.isoformat(),
                        "event_data": log.event_data
                    }
                    for log in logs
                ],
                "count": len(logs),
                "filters": {"event_type": event_type, "limit": limit}
            }
    except Exception as e:
        logger.error(f"Audit logs retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== Visual Data Explorer Endpoints ==============

class VisualizationRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    chart_type: str
    data_source: str = "uploaded"
    config: dict
    data: Optional[list] = None  # For uploaded data

class DataAnalysisRequest(BaseModel):
    data_type: str = "sales"  # sales, users, stocks, conversations
    rows: int = 100

@app.post("/data/analyze")
async def analyze_sample_data(request: DataAnalysisRequest):
    """Generate and analyze sample data"""
    try:
        # Generate sample data
        if request.data_type == "conversations":
            data = await data_explorer.get_conversation_analytics()
        else:
            data = await data_explorer.generate_sample_data(request.data_type, request.rows)
        
        # Analyze the data
        analysis = await data_explorer.analyze_data(data, request.data_type)
        
        return {
            "analysis": analysis,
            "sample_data": data.head(10).to_dict('records'),  # First 10 rows
            "message": "Data analysis completed successfully"
        }
    except Exception as e:
        logger.error(f"Data analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data/visualize")
async def create_visualization(request: VisualizationRequest):
    """Create a data visualization"""
    try:
        import pandas as pd
        
        # Handle data source
        if request.data_source == "uploaded" and request.data:
            # Convert uploaded data to DataFrame
            data = pd.DataFrame(request.data)
        elif request.data_source == "conversations":
            data = await data_explorer.get_conversation_analytics()
        else:
            # Generate sample data
            data_type = request.data_source if request.data_source in ["sales", "users", "stocks"] else "sales"
            data = await data_explorer.generate_sample_data(data_type, 100)
        
        # Create visualization config
        config = VisualizationConfig(
            chart_type=ChartType(request.chart_type),
            title=request.config.get("title", request.name),
            x_column=request.config.get("x_column"),
            y_column=request.config.get("y_column"),
            color_column=request.config.get("color_column"),
            size_column=request.config.get("size_column"),
            aggregation=request.config.get("aggregation"),
            theme=request.config.get("theme", "plotly"),
            width=request.config.get("width", 800),
            height=request.config.get("height", 600),
            interactive=request.config.get("interactive", True)
        )
        
        # Map frontend data source to backend enum
        data_source_mapping = {
            "sales": DataSource.GENERATED,
            "users": DataSource.GENERATED, 
            "stocks": DataSource.GENERATED,
            "conversations": DataSource.CONVERSATIONS,
            "uploaded": DataSource.UPLOADED,
            "integrations": DataSource.INTEGRATIONS,
            "security_logs": DataSource.SECURITY_LOGS,
            "knowledge_graph": DataSource.KNOWLEDGE_GRAPH,
            "generated": DataSource.GENERATED
        }
        
        mapped_data_source = data_source_mapping.get(request.data_source, DataSource.GENERATED)
        
        # Create visualization
        viz_id = await data_explorer.create_visualization(
            data=data,
            config=config,
            name=request.name,
            description=request.description,
            data_source=mapped_data_source
        )
        
        # Get the created visualization
        visualization = await data_explorer.get_visualization(viz_id)
        
        return {
            "visualization": visualization,
            "message": "Visualization created successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Visualization creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/visualizations")
async def get_saved_visualizations(limit: int = 20):
    """Get list of saved visualizations"""
    try:
        visualizations = await data_explorer.get_saved_visualizations(limit)
        return {
            "visualizations": visualizations,
            "count": len(visualizations)
        }
    except Exception as e:
        logger.error(f"Get visualizations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/visualizations/{viz_id}")
async def get_visualization(viz_id: str):
    """Get a specific visualization"""
    try:
        visualization = await data_explorer.get_visualization(viz_id)
        
        if not visualization:
            raise HTTPException(status_code=404, detail="Visualization not found")
        
        return {
            "visualization": visualization
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get visualization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data/dashboard")
async def create_dashboard(data: dict):
    """Create a dashboard from multiple visualizations"""
    try:
        visualization_ids = data.get("visualization_ids", [])
        title = data.get("title", "Data Dashboard")
        
        if not visualization_ids:
            raise HTTPException(status_code=400, detail="At least one visualization ID is required")
        
        dashboard_path = await data_explorer.create_dashboard(visualization_ids, title)
        
        return {
            "dashboard_path": dashboard_path.replace(str(data_explorer.output_dir), "/static/visualizations"),
            "title": title,
            "visualization_count": len(visualization_ids),
            "message": "Dashboard created successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Dashboard creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/sample/{data_type}")
async def get_sample_data(data_type: str, rows: int = 100):
    """Generate sample data for visualization testing"""
    try:
        if data_type not in ["sales", "users", "stocks", "conversations"]:
            raise HTTPException(status_code=400, detail="Invalid data type")
        
        if data_type == "conversations":
            data = await data_explorer.get_conversation_analytics()
        else:
            data = await data_explorer.generate_sample_data(data_type, rows)
        
        return {
            "data": data.to_dict('records'),
            "columns": list(data.columns),
            "rows": len(data),
            "data_type": data_type
        }
    except Exception as e:
        logger.error(f"Sample data generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== Enhanced Memory & Personalization Endpoints ==============

@app.get("/personalization/dashboard")
async def get_personalization_dashboard(user_id: str = "default"):
    """Get comprehensive personalization dashboard"""
    try:
        dashboard = await personalization_engine.get_personalization_dashboard(user_id)
        return {
            "dashboard": dashboard,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Personalization dashboard error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/personalization/suggestions")
async def get_personalized_suggestions(user_id: str = "default", context: str = ""):
    """Get AI-powered personalized suggestions"""
    try:
        suggestions = await personalization_engine.get_personalized_suggestions(user_id, context)
        return {
            "suggestions": [
                {
                    "type": s.insight_type,
                    "title": s.title,
                    "description": s.description,
                    "confidence": s.confidence,
                    "actionable": s.actionable,
                    "recommendation": s.recommendation,
                    "data": s.data
                }
                for s in suggestions
            ],
            "count": len(suggestions)
        }
    except Exception as e:
        logger.error(f"Personalized suggestions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/personalization/session/start")
async def start_personalization_session(user_id: str = "default", mode: str = "chat"):
    """Start a new personalization learning session"""
    try:
        session_id = await personalization_engine.start_session(user_id, mode)
        return {
            "session_id": session_id,
            "user_id": user_id,
            "mode": mode,
            "started_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Start personalization session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/personalization/session/{session_id}/end")
async def end_personalization_session(session_id: str):
    """End a personalization learning session"""
    try:
        await personalization_engine.end_session(session_id)
        return {
            "session_id": session_id,
            "ended_at": datetime.utcnow().isoformat(),
            "message": "Session analysis completed"
        }
    except Exception as e:
        logger.error(f"End personalization session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/personalization/export")
async def export_personalization_data(user_id: str = "default"):
    """Export all user personalization data"""
    try:
        data = await personalization_engine.export_user_data(user_id)
        return {
            "export": data,
            "exported_at": datetime.utcnow().isoformat(),
            "user_id": user_id
        }
    except Exception as e:
        logger.error(f"Export personalization data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)