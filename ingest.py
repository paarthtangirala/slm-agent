#!/usr/bin/env python3
"""
Local document ingestion script for the SLM Personal Agent.
Embeds documents from the ./docs folder into ChromaDB for local querying.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path
from typing import List, Dict
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentIngestor:
    def __init__(self, docs_path: str = "./docs", db_path: str = "./chromadb"):
        self.docs_path = Path(docs_path)
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection("local_docs")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
                
        return chunks
    
    def read_file(self, file_path: Path) -> str:
        """Read text content from various file types"""
        try:
            if file_path.suffix.lower() in ['.txt', '.md', '.py', '.js', '.json', '.yaml', '.yml']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                logger.warning(f"Unsupported file type: {file_path}")
                return ""
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return ""
    
    def get_file_hash(self, file_path: Path) -> str:
        """Generate hash for file content to detect changes"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def is_file_processed(self, file_path: Path, file_hash: str) -> bool:
        """Check if file with this hash was already processed"""
        try:
            results = self.collection.get(
                where={"source": str(file_path), "file_hash": file_hash}
            )
            return len(results['ids']) > 0
        except:
            return False
    
    def ingest_file(self, file_path: Path) -> int:
        """Ingest a single file into the vector database"""
        logger.info(f"Processing: {file_path}")
        
        content = self.read_file(file_path)
        if not content.strip():
            logger.warning(f"Empty or unreadable file: {file_path}")
            return 0
        
        file_hash = self.get_file_hash(file_path)
        
        if self.is_file_processed(file_path, file_hash):
            logger.info(f"File already processed (no changes): {file_path}")
            return 0
        
        # Remove old versions of this file
        try:
            old_results = self.collection.get(where={"source": str(file_path)})
            if old_results['ids']:
                self.collection.delete(ids=old_results['ids'])
                logger.info(f"Removed {len(old_results['ids'])} old chunks for {file_path}")
        except:
            pass
        
        chunks = self.chunk_text(content)
        if not chunks:
            return 0
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks)
        
        # Prepare metadata
        chunk_ids = [f"{file_path.stem}_{i}_{file_hash[:8]}" for i in range(len(chunks))]
        metadatas = [{
            "source": str(file_path),
            "chunk_index": i,
            "file_hash": file_hash,
            "file_size": len(content)
        } for i in range(len(chunks))]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadatas,
            ids=chunk_ids
        )
        
        logger.info(f"Added {len(chunks)} chunks from {file_path}")
        return len(chunks)
    
    def ingest_directory(self) -> Dict[str, int]:
        """Ingest all supported files from the docs directory"""
        if not self.docs_path.exists():
            logger.error(f"Docs directory not found: {self.docs_path}")
            self.docs_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created docs directory: {self.docs_path}")
            return {}
        
        supported_extensions = {'.txt', '.md', '.py', '.js', '.json', '.yaml', '.yml'}
        results = {}
        total_chunks = 0
        
        for file_path in self.docs_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                chunks_added = self.ingest_file(file_path)
                results[str(file_path)] = chunks_added
                total_chunks += chunks_added
        
        logger.info(f"Ingestion complete. Total chunks added: {total_chunks}")
        return results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection.name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}

def main():
    """Main ingestion function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest documents for SLM Personal Agent")
    parser.add_argument("--docs-path", default="./docs", help="Path to documents directory")
    parser.add_argument("--db-path", default="./chromadb", help="Path to ChromaDB storage")
    parser.add_argument("--stats", action="store_true", help="Show collection statistics")
    
    args = parser.parse_args()
    
    ingestor = DocumentIngestor(args.docs_path, args.db_path)
    
    if args.stats:
        stats = ingestor.get_collection_stats()
        print(f"Collection Statistics: {stats}")
        return
    
    logger.info("Starting document ingestion...")
    results = ingestor.ingest_directory()
    
    if results:
        print("\nIngestion Results:")
        for file_path, chunks in results.items():
            print(f"  {file_path}: {chunks} chunks")
    else:
        print("No files found to ingest. Add .txt, .md, .py, .js, .json, .yaml files to ./docs/")
    
    stats = ingestor.get_collection_stats()
    print(f"\nFinal Collection Stats: {stats}")

if __name__ == "__main__":
    main()