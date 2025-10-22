"""
Human-Centered AI Assistant - Main Application
Focused on transparency, accessibility, privacy, and simplicity

Following Prof. Jingyi Xie's HCAI principles:
- Transparency: Every response explains reasoning
- Accessibility: Voice input/output support
- Personalization: Local preference learning
- Privacy: Local-first, no external data by default
- Feedback: User rating system for continuous improvement
"""

import os
import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import our HCAI modules
from .types import (
    AssistRequest, AssistResponse, 
    MemoryRequest, MemoryResponse,
    VoiceRequest, VoiceResponse
)
from .intent import detect_intent, explain_intent_reasoning
from .prompts import get_summarize_prompt, get_email_prompt, get_query_prompt
from .llm_local import local_llm
from .memory import local_memory
from .rag_local import local_rag

# Optional voice capabilities
try:
    from .tts_voice import voice_processor
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

# Configure logging for transparency
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI with human-centered design
app = FastAPI(
    title="Human-Centered AI Assistant",
    description="Local, private, transparent AI assistant focused on human needs",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENABLE_DOCS", "false").lower() == "true" else None
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for web interface
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.on_event("startup")
async def startup_event():
    """Initialize services with health checks"""
    logger.info("🤖 Starting Human-Centered AI Assistant")
    
    # Check LLM health
    health = await local_llm.check_health()
    if health["status"] == "healthy":
        logger.info(f"✅ LLM ready: {health['model']}")
    else:
        logger.warning(f"⚠️ LLM not available: {health.get('suggestion', 'Check Ollama')}")
    
    # Check RAG health
    rag_stats = local_rag.get_collection_stats()
    if rag_stats["status"] == "ready":
        logger.info(f"✅ Knowledge base ready: {rag_stats['document_count']} documents")
    elif rag_stats["status"] == "empty":
        logger.info("📚 Knowledge base empty - run ingest.py to add documents")
    else:
        logger.warning(f"⚠️ Knowledge base: {rag_stats.get('message', 'Not available')}")
    
    # Check voice capabilities
    if VOICE_AVAILABLE:
        logger.info("🎤 Voice capabilities enabled")
    else:
        logger.info("🔇 Voice capabilities disabled (install speech deps)")

@app.get("/api")
async def api_status():
    """Health check with transparent status"""
    llm_health = await local_llm.check_health()
    rag_stats = local_rag.get_collection_stats()
    
    return {
        "service": "Human-Centered AI Assistant",
        "status": "healthy",
        "principles": ["transparency", "accessibility", "personalization", "privacy", "feedback"],
        "endpoints": ["/assist", "/memory", "/voice"],
        "llm": {
            "status": llm_health["status"],
            "model": llm_health.get("model", "unknown")
        },
        "knowledge_base": {
            "status": rag_stats["status"],
            "documents": rag_stats.get("document_count", 0)
        },
        "voice": {"available": VOICE_AVAILABLE},
        "privacy": "local-first",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/assist", response_model=AssistResponse)
async def assist(request: AssistRequest) -> AssistResponse:
    """
    Core assistance endpoint with transparent reasoning
    
    Human-Centered Design:
    - Single endpoint for all assistance tasks
    - Transparent intent detection
    - Clear reasoning for every response
    - Actionable next suggestions
    - Privacy-conscious web search opt-in
    """
    try:
        # Apply user preferences
        user_prefs = local_memory.get_preferences()
        
        # Override with user preferences if not specified
        if not request.tone:
            request.tone = user_prefs.get("tone", "friendly")
        if not request.length:
            request.length = user_prefs.get("length", "medium")
        if user_prefs.get("disable_web", False):
            request.use_web = False
        
        # Detect intent with transparency
        task_type = detect_intent(request)
        intent_reasoning = explain_intent_reasoning(request, task_type)
        
        logger.info(f"Processing {task_type} task: {request.text[:100]}...")
        
        # Route to appropriate handler
        if task_type == "summarize":
            response = await _handle_summarize(request)
        elif task_type == "email":
            response = await _handle_email(request)
        elif task_type == "query":
            response = await _handle_query(request)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Add transparent reasoning
        response.reasoning = f"{intent_reasoning} {response.reasoning}"
        
        logger.info(f"Completed {task_type} task successfully")
        return response
        
    except Exception as e:
        logger.error(f"Assist error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unable to process request: {str(e)}. Please check if Ollama is running."
        )

async def _handle_summarize(request: AssistRequest) -> AssistResponse:
    """Handle summarization with transparency"""
    system_prompt, user_prompt = get_summarize_prompt(request.text, request.length)
    
    # Generate response
    llm_result = await local_llm.generate(system_prompt, user_prompt, temperature=0.3)
    
    # Parse response (summary + reasoning)
    response_parts = llm_result["response"].split("\n\n")
    summary = response_parts[0] if response_parts else llm_result["response"]
    
    return AssistResponse(
        task_type="summarize",
        output_text=summary,
        reasoning=f"Summarized {len(request.text)} characters into {request.length} format, focusing on key facts and conclusions.",
        next_suggestion="Save this summary to your notes or ask follow-up questions about specific points.",
        meta={
            "original_length": len(request.text),
            "summary_length": len(summary),
            "compression_ratio": round(len(summary) / len(request.text), 2),
            "model_stats": {
                "duration": llm_result["duration_seconds"],
                "tokens": llm_result["total_tokens"]
            }
        }
    )

async def _handle_email(request: AssistRequest) -> AssistResponse:
    """Handle email drafting with tone awareness"""
    bullets = request.bullets or [request.text]
    system_prompt, user_prompt = get_email_prompt(bullets, request.tone)
    
    # Generate response
    llm_result = await local_llm.generate(system_prompt, user_prompt, temperature=0.4)
    
    # Extract email content
    email_content = llm_result["response"]
    
    return AssistResponse(
        task_type="email",
        output_text=email_content,
        reasoning=f"Drafted email in {request.tone} tone with {len(bullets)} key points, following professional email structure.",
        next_suggestion="Review the email, make any personal adjustments, then send to your recipient.",
        meta={
            "tone": request.tone,
            "bullet_points": len(bullets),
            "model_stats": {
                "duration": llm_result["duration_seconds"],
                "tokens": llm_result["total_tokens"]
            }
        }
    )

async def _handle_query(request: AssistRequest) -> AssistResponse:
    """Handle queries with local knowledge and optional web search"""
    # Search local knowledge base first  
    local_results = local_rag.search(request.text, max_results=3, min_relevance=0.2)
    
    # Build context from local sources
    context_parts = []
    citations = []
    
    for i, result in enumerate(local_results, 1):
        context_parts.append(f"[Source {i}: {result['source']}]\n{result['content']}")
        citations.append({
            "source": result["source"],
            "relevance": result["relevance_score"],
            "type": "local"
        })
    
    # Add web search if enabled and local results are insufficient
    web_sources = []
    if request.use_web and len(local_results) < 2:
        web_sources = await _search_web(request.text, max_results=2)
        for i, source in enumerate(web_sources, len(local_results) + 1):
            context_parts.append(f"[Web Source {i}: {source['title']}]\n{source['snippet']}")
            citations.append({
                "source": source["url"],
                "title": source["title"],
                "type": "web"
            })
    
    # Generate response with context
    context = "\n\n".join(context_parts) if context_parts else "No relevant context found."
    system_prompt, user_prompt = get_query_prompt(request.text, context, request.use_web)
    
    llm_result = await local_llm.generate(system_prompt, user_prompt, temperature=0.2)
    
    # Determine reasoning based on sources
    if local_results and web_sources:
        reasoning = f"Found {len(local_results)} local sources and {len(web_sources)} web sources. Combined information to provide comprehensive answer."
    elif local_results:
        reasoning = f"Used {len(local_results)} documents from your local knowledge base with {local_results[0]['relevance_score']:.2f} relevance."
    elif web_sources:
        reasoning = f"No local sources found. Used {len(web_sources)} web search results to answer your question."
    else:
        reasoning = "No relevant sources found. Provided general response based on AI knowledge (may be incomplete)."
    
    next_suggestion = "Ask follow-up questions or add relevant documents to your knowledge base for better future answers." if not local_results else "Ask follow-up questions about specific aspects of this topic."
    
    return AssistResponse(
        task_type="query",
        output_text=llm_result["response"],
        reasoning=reasoning,
        next_suggestion=next_suggestion,
        meta={
            "sources": citations,
            "local_results": len(local_results),
            "web_results": len(web_sources),
            "web_search_used": request.use_web,
            "model_stats": {
                "duration": llm_result["duration_seconds"],
                "tokens": llm_result["total_tokens"]
            }
        }
    )

async def _search_web(query: str, max_results: int = 3) -> list:
    """Optional web search with privacy awareness"""
    try:
        # Import only when needed for privacy
        from duckduckgo_search import DDGS
        
        with DDGS() as ddgs:
            results = []
            for result in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", "")
                })
            return results
            
    except ImportError:
        logger.warning("Web search not available - install duckduckgo-search")
        return []
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return []

@app.get("/memory", response_model=MemoryResponse)
async def get_memory() -> MemoryResponse:
    """Get user preferences and feedback stats"""
    prefs = local_memory.get_preferences()
    stats = local_memory.get_feedback_stats()
    
    return MemoryResponse(
        preferences=prefs,
        feedback_count=stats["total_feedback"],
        last_updated=datetime.utcnow().isoformat()
    )

@app.post("/memory", response_model=MemoryResponse)
async def update_memory(request: MemoryRequest) -> MemoryResponse:
    """Update user preferences or add feedback"""
    
    # Handle feedback submission
    if request.feedback:
        feedback = request.feedback
        feedback_id = local_memory.add_feedback(
            task_type=feedback.get("task_type", "unknown"),
            user_input=feedback.get("user_input", ""),
            ai_response=feedback.get("ai_response", ""),
            rating=feedback.get("rating", 3),
            correction=feedback.get("correction")
        )
        logger.info(f"Stored user feedback: {feedback_id}")
    
    # Handle preference updates
    updates = {}
    for field in ["tone", "length", "voice", "disable_web"]:
        value = getattr(request, field, None)
        if value is not None:
            updates[field] = value
    
    if updates:
        prefs = local_memory.update_preferences(updates)
        logger.info(f"Updated preferences: {list(updates.keys())}")
    else:
        prefs = local_memory.get_preferences()
    
    stats = local_memory.get_feedback_stats()
    
    return MemoryResponse(
        preferences=prefs,
        feedback_count=stats["total_feedback"],
        last_updated=datetime.utcnow().isoformat()
    )

@app.post("/voice", response_model=VoiceResponse)
async def voice_assist(
    audio: Optional[UploadFile] = File(None),
    text: Optional[str] = None,
    enable_tts: bool = True
) -> VoiceResponse:
    """
    Voice accessibility endpoint
    
    Human-Centered Design:
    - Voice input for accessibility
    - Text-to-speech output
    - Graceful fallback when voice deps missing
    """
    
    if not VOICE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Voice capabilities not available. Install: pip install faster-whisper pyttsx3"
        )
    
    transcription = None
    
    # Handle voice input
    if audio:
        try:
            # Save uploaded audio temporarily
            audio_path = f"/tmp/{audio.filename}"
            with open(audio_path, "wb") as buffer:
                content = await audio.read()
                buffer.write(content)
            
            # Transcribe audio
            transcription = await voice_processor.transcribe(audio_path)
            
            # Clean up
            os.unlink(audio_path)
            
            # Use transcription as input
            input_text = transcription
            
        except Exception as e:
            logger.error(f"Voice transcription error: {e}")
            raise HTTPException(status_code=400, detail="Failed to process audio")
    
    elif text:
        input_text = text
    else:
        raise HTTPException(status_code=400, detail="Provide either audio file or text")
    
    # Process through main assist endpoint
    assist_request = AssistRequest(text=input_text)
    assist_response = await assist(assist_request)
    
    # Generate TTS if requested
    audio_url = None
    if enable_tts and VOICE_AVAILABLE:
        try:
            audio_url = await voice_processor.text_to_speech(assist_response.output_text)
        except Exception as e:
            logger.error(f"TTS error: {e}")
    
    return VoiceResponse(
        transcription=transcription,
        assist_response=assist_response,
        audio_url=audio_url,
        tts_available=VOICE_AVAILABLE
    )

@app.get("/")
async def root():
    """Serve the web interface"""
    static_file = Path(__file__).parent / "static" / "index.html"
    if static_file.exists():
        return FileResponse(str(static_file))
    else:
        return await api_status()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process documents for the knowledge base
    
    Human-Centered Design:
    - Web-based document upload
    - Transparent processing feedback
    - Support for common document formats
    """
    try:
        # Validate file type
        allowed_extensions = {'.txt', '.md', '.rst', '.pdf'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"Unsupported file type: {file_ext}",
                    "supported": list(allowed_extensions)
                }
            )
        
        # Read file content
        content = await file.read()
        
        # Save temporary file for processing
        temp_dir = Path("./temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / file.filename
        
        with open(temp_file, 'wb') as f:
            f.write(content)
        
        # Process the document
        if file_ext == '.pdf':
            # Handle PDF processing
            try:
                import pypdf
                text_content = []
                with open(temp_file, 'rb') as pdf_file:
                    pdf_reader = pypdf.PdfReader(pdf_file)
                    for page_num, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        if text.strip():
                            text_content.append(text)
                doc_content = "\n\n".join(text_content)
            except ImportError:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error": "PDF processing not available. Install pypdf."
                    }
                )
        else:
            # Handle text files
            try:
                doc_content = content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    doc_content = content.decode('latin1')
                except UnicodeDecodeError:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "success": False,
                            "error": "Cannot decode file. Use UTF-8 or Latin1 encoding."
                        }
                    )
        
        # Add to knowledge base
        success = local_rag.add_document(
            content=doc_content,
            source=file.filename,
            metadata={
                "file_type": file_ext,
                "file_size": len(content),
                "upload_method": "web_interface"
            }
        )
        
        # Clean up temp file
        temp_file.unlink()
        
        if success:
            # Get updated stats
            stats = local_rag.get_collection_stats()
            
            return JSONResponse(content={
                "success": True,
                "message": f"Successfully processed {file.filename}",
                "document": {
                    "filename": file.filename,
                    "size": len(content),
                    "content_length": len(doc_content),
                    "words": len(doc_content.split()),
                    "type": file_ext
                },
                "knowledge_base": {
                    "total_documents": stats.get("document_count", 0),
                    "status": stats.get("status", "unknown")
                }
            })
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Failed to add document to knowledge base"
                }
            )
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Upload failed: {str(e)}"
            }
        )

@app.get("/documents")
async def list_documents():
    """List documents in the knowledge base"""
    try:
        documents = local_rag.list_documents(limit=50)
        stats = local_rag.get_collection_stats()
        
        return JSONResponse(content={
            "success": True,
            "documents": documents,
            "total_count": stats.get("document_count", 0),
            "status": stats.get("status", "unknown")
        })
    except Exception as e:
        logger.error(f"List documents error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

@app.get("/health")
async def health_check():
    """Comprehensive health check for transparency"""
    return await api_status()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"🚀 Starting Human-Centered AI Assistant on port {port}")
    logger.info("📚 HCAI Principles: Transparency, Accessibility, Personalization, Privacy, Feedback")
    
    uvicorn.run(
        "app_hcai.main:app",
        host="0.0.0.0",
        port=port,
        reload=True if os.getenv("ENV") == "development" else False
    )