"""
Local RAG (Retrieval-Augmented Generation) for Human-Centered AI Assistant
Privacy-first document storage and semantic search
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

# Optional ChromaDB with graceful fallback
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# Optional sentence transformers with graceful fallback
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False


class LocalRAGManager:
    """
    Local document storage and retrieval system
    
    Human-Centered Design:
    - All data stays on user's machine
    - Transparent search and retrieval
    - Clear feedback on what's being used
    - Graceful degradation when dependencies missing
    """
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None
        
        # Initialize components if available
        if CHROMA_AVAILABLE and EMBEDDING_AVAILABLE:
            self._initialize_chroma()
            self._initialize_embeddings()
        else:
            logger.warning("ChromaDB or sentence-transformers not available - RAG features disabled")
    
    def _initialize_chroma(self):
        """Initialize ChromaDB for local vector storage"""
        try:
            chroma_path = self.data_dir / "chroma_db"
            chroma_path.mkdir(exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=Settings(
                    anonymized_telemetry=False,  # Privacy-first
                    allow_reset=True
                )
            )
            
            # Get or create collection
            collection_name = "user_documents"
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
            except ValueError:
                # Collection doesn't exist, create it
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": "User documents for local RAG"}
                )
            
            logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.chroma_client = None
            self.collection = None
    
    def _initialize_embeddings(self):
        """Initialize sentence transformer model for embeddings"""
        try:
            # Use a lightweight, fast model for local processing
            model_name = "all-MiniLM-L6-v2"  # Small, efficient, good quality
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Embedding model loaded: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
    
    def is_available(self) -> bool:
        """Check if RAG functionality is available"""
        return (
            CHROMA_AVAILABLE and 
            EMBEDDING_AVAILABLE and 
            self.chroma_client is not None and 
            self.collection is not None and 
            self.embedding_model is not None
        )
    
    def add_document(
        self, 
        content: str, 
        source: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a document to the local knowledge base
        
        Args:
            content: Document text content
            source: Source identifier (filename, URL, etc.)
            metadata: Additional metadata about the document
            
        Returns:
            Success status
        """
        if not self.is_available():
            logger.warning("RAG not available - document not added")
            return False
        
        try:
            # Create document ID from content hash for deduplication
            content_hash = hashlib.md5(content.encode()).hexdigest()
            doc_id = f"{source}_{content_hash[:8]}"
            
            # Prepare metadata
            doc_metadata = {
                "source": source,
                "content_length": len(content),
                "added_at": datetime.utcnow().isoformat(),
                **(metadata or {})
            }
            
            # Check if document already exists
            try:
                existing = self.collection.get(ids=[doc_id])
                if existing['ids']:
                    logger.info(f"Document already exists: {source}")
                    return True
            except:
                pass  # Document doesn't exist, continue
            
            # Generate embedding
            embedding = self.embedding_model.encode(content).tolist()
            
            # Add to collection
            self.collection.add(
                ids=[doc_id],
                documents=[content],
                embeddings=[embedding],
                metadatas=[doc_metadata]
            )
            
            logger.info(f"Added document: {source} ({len(content)} chars)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document {source}: {e}")
            return False
    
    def search(
        self, 
        query: str, 
        max_results: int = 5,
        min_relevance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search documents with semantic similarity
        
        Args:
            query: Search query text
            max_results: Maximum number of results to return
            min_relevance: Minimum relevance score (0.0 to 1.0)
            
        Returns:
            List of relevant documents with metadata
        """
        if not self.is_available():
            logger.warning("RAG not available - no search results")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results
            )
            
            # Format results with transparency
            formatted_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    # ChromaDB uses cosine distance, convert to similarity score
                    # For cosine distance: similarity = (2 - distance) / 2
                    # This maps distance [0, 2] to similarity [1, 0]
                    if distance <= 2.0:
                        relevance_score = max(0.0, (2.0 - distance) / 2.0)
                    else:
                        relevance_score = 0.0
                    
                    if relevance_score >= min_relevance:
                        formatted_results.append({
                            "content": doc,
                            "source": metadata.get("source", "unknown"),
                            "relevance_score": round(relevance_score, 3),
                            "metadata": metadata,
                            "content_preview": doc[:200] + "..." if len(doc) > 200 else doc
                        })
            
            logger.info(f"Search query: '{query}' returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the document collection for transparency
        """
        if not self.is_available():
            return {
                "status": "unavailable",
                "message": "Install chromadb and sentence-transformers to enable RAG",
                "install_command": "pip install chromadb sentence-transformers"
            }
        
        try:
            # Get collection info
            collection_info = self.collection.get()
            doc_count = len(collection_info['ids']) if collection_info['ids'] else 0
            
            if doc_count == 0:
                return {
                    "status": "empty",
                    "document_count": 0,
                    "message": "No documents in knowledge base",
                    "suggestion": "Use ingest_hcai.py to add documents"
                }
            
            # Calculate some basic stats
            sources = [meta.get("source", "unknown") for meta in collection_info['metadatas']]
            unique_sources = len(set(sources))
            
            return {
                "status": "ready",
                "document_count": doc_count,
                "unique_sources": unique_sources,
                "storage_location": str(self.data_dir / "chroma_db"),
                "embedding_model": "all-MiniLM-L6-v2",
                "last_updated": "recent"  # Could be more specific
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "status": "error",
                "error": str(e),
                "suggestion": "Check ChromaDB installation"
            }
    
    def list_documents(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        List documents in the knowledge base for user transparency
        """
        if not self.is_available():
            return []
        
        try:
            collection_info = self.collection.get(limit=limit)
            
            documents = []
            if collection_info['ids']:
                for i, (doc_id, metadata) in enumerate(zip(
                    collection_info['ids'],
                    collection_info['metadatas']
                )):
                    documents.append({
                        "id": doc_id,
                        "source": metadata.get("source", "unknown"),
                        "content_length": metadata.get("content_length", 0),
                        "added_at": metadata.get("added_at", "unknown"),
                        "file_type": metadata.get("file_type", "unknown")
                    })
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    def delete_document(self, source: str) -> bool:
        """
        Delete documents by source for user control
        """
        if not self.is_available():
            return False
        
        try:
            # Get all documents with matching source
            collection_info = self.collection.get()
            
            ids_to_delete = []
            for doc_id, metadata in zip(collection_info['ids'], collection_info['metadatas']):
                if metadata.get("source") == source:
                    ids_to_delete.append(doc_id)
            
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} documents from source: {source}")
                return True
            else:
                logger.warning(f"No documents found for source: {source}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete documents from {source}: {e}")
            return False


# Global instance for application use
local_rag = LocalRAGManager()