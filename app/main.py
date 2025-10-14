from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import chromadb
from sentence_transformers import SentenceTransformer
import os
from typing import List, Optional
import logging
from dotenv import load_dotenv
import json
from bs4 import BeautifulSoup
from serpapi import GoogleSearch

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

@app.on_event("startup")
async def startup_event():
    """Initialize app - database will be lazy loaded"""
    logger.info("SLM Personal Agent started")

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

async def call_ollama(prompt: str, system_prompt: str = "") -> str:
    """Call Ollama API with the given prompt"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        
        try:
            response = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise HTTPException(status_code=500, detail=f"Ollama API error: {e}")

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
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            return {"status": "healthy", "ollama_connected": True}
    except:
        return {"status": "unhealthy", "ollama_connected": False}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)