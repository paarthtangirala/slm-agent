#!/usr/bin/env python3
"""
Document Ingestion for Human-Centered AI Assistant
Process and add documents to local knowledge base with transparency

Usage:
    python ingest_hcai.py [directory_or_file]
    python ingest_hcai.py docs/
    python ingest_hcai.py sample.pdf
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# Import HCAI modules
from app_hcai.rag_local import local_rag

# Optional PDF processing
try:
    import pypdf
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

logger = logging.getLogger(__name__)

class DocumentIngestor:
    """
    Human-Centered document processor
    
    Design Principles:
    - Transparent processing status
    - Local-only storage
    - Clear error messages
    - Support for common formats
    """
    
    def __init__(self):
        self.supported_extensions = {'.txt', '.md', '.rst'}
        if PDF_AVAILABLE:
            self.supported_extensions.add('.pdf')
    
    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single file with transparency"""
        if file_path.suffix.lower() not in self.supported_extensions:
            return {
                "status": "skipped",
                "reason": f"Unsupported file type: {file_path.suffix}",
                "suggestion": f"Supported: {', '.join(self.supported_extensions)}"
            }
        
        try:
            # Extract text based on file type
            if file_path.suffix.lower() == '.pdf':
                content = self._extract_pdf_text(file_path)
            else:
                content = self._extract_text_file(file_path)
            
            if not content.strip():
                return {
                    "status": "skipped",
                    "reason": "File is empty or contains no text"
                }
            
            # Add to knowledge base
            success = local_rag.add_document(
                content=content,
                source=str(file_path.name),
                metadata={
                    "file_type": file_path.suffix.lower(),
                    "file_size": file_path.stat().st_size,
                    "character_count": len(content)
                }
            )
            
            if success:
                return {
                    "status": "success",
                    "characters": len(content),
                    "words": len(content.split()),
                    "file_size": file_path.stat().st_size
                }
            else:
                return {
                    "status": "failed",
                    "reason": "Could not add to knowledge base"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "reason": str(e)
            }
    
    def _extract_text_file(self, file_path: Path) -> str:
        """Extract text from text-based files"""
        try:
            # Try UTF-8 first, fallback to other encodings
            for encoding in ['utf-8', 'latin1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            
            raise Exception("Could not decode file with common encodings")
            
        except Exception as e:
            raise Exception(f"Error reading text file: {e}")
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF files"""
        if not PDF_AVAILABLE:
            raise Exception("PDF support not available. Install: pip install pypdf")
        
        try:
            text_content = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            text_content.append(f"[Page {page_num + 1}]\n{text}")
                    except Exception as e:
                        logger.warning(f"Could not extract text from page {page_num + 1}: {e}")
                        continue
            
            if not text_content:
                raise Exception("No text could be extracted from PDF")
            
            return "\n\n".join(text_content)
            
        except Exception as e:
            raise Exception(f"Error processing PDF: {e}")
    
    def process_directory(self, directory: Path, recursive: bool = True) -> Dict[str, Any]:
        """Process all supported files in a directory"""
        results = {
            "processed": [],
            "skipped": [],
            "errors": [],
            "summary": {}
        }
        
        # Find all files
        if recursive:
            files = list(directory.rglob("*"))
        else:
            files = list(directory.glob("*"))
        
        # Filter to regular files only
        files = [f for f in files if f.is_file()]
        
        print(f"📁 Processing {len(files)} files from {directory}")
        
        for file_path in files:
            print(f"  📄 {file_path.name}...", end=" ")
            
            result = self.process_file(file_path)
            result["file"] = file_path.name
            
            if result["status"] == "success":
                results["processed"].append(result)
                print(f"✅ {result['characters']} chars")
            elif result["status"] == "skipped":
                results["skipped"].append(result)
                print(f"⏭️  {result['reason']}")
            else:
                results["errors"].append(result)
                print(f"❌ {result['reason']}")
        
        # Generate summary
        results["summary"] = {
            "total_files": len(files),
            "successful": len(results["processed"]),
            "skipped": len(results["skipped"]),
            "errors": len(results["errors"]),
            "total_characters": sum(r["characters"] for r in results["processed"]),
            "total_words": sum(r["words"] for r in results["processed"])
        }
        
        return results

def main():
    """Main ingestion function with clear user feedback"""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Check if RAG is available
    if not local_rag.is_available():
        print("❌ Knowledge base not available.")
        print("💡 Install dependencies: pip install sentence-transformers chromadb")
        sys.exit(1)
    
    # Get collection stats
    stats = local_rag.get_collection_stats()
    print(f"📚 Knowledge Base Status: {stats['status']}")
    if stats.get('document_count', 0) > 0:
        print(f"   Current documents: {stats['document_count']}")
    
    # Determine input path
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    else:
        # Default to docs directory
        input_path = Path("docs")
        if not input_path.exists():
            input_path = Path(".")
    
    if not input_path.exists():
        print(f"❌ Path not found: {input_path}")
        print("💡 Usage: python ingest_hcai.py [directory_or_file]")
        sys.exit(1)
    
    # Process input
    ingestor = DocumentIngestor()
    
    print(f"🚀 Starting Human-Centered AI document ingestion")
    print(f"📍 Target: {input_path.absolute()}")
    print(f"📝 Supported formats: {', '.join(ingestor.supported_extensions)}")
    print()
    
    if input_path.is_file():
        # Process single file
        print(f"📄 Processing single file: {input_path.name}")
        result = ingestor.process_file(input_path)
        
        if result["status"] == "success":
            print(f"✅ Successfully processed: {result['characters']} characters, {result['words']} words")
        else:
            print(f"❌ Failed: {result['reason']}")
            if "suggestion" in result:
                print(f"💡 {result['suggestion']}")
    
    else:
        # Process directory
        results = ingestor.process_directory(input_path)
        
        # Print summary
        summary = results["summary"]
        print(f"\n📊 Ingestion Summary:")
        print(f"   Total files: {summary['total_files']}")
        print(f"   ✅ Successful: {summary['successful']}")
        print(f"   ⏭️  Skipped: {summary['skipped']}")
        print(f"   ❌ Errors: {summary['errors']}")
        print(f"   📝 Total content: {summary['total_characters']:,} characters, {summary['total_words']:,} words")
        
        # Show errors if any
        if results["errors"]:
            print(f"\n⚠️  Error Details:")
            for error in results["errors"]:
                print(f"   {error['file']}: {error['reason']}")
    
    # Final knowledge base stats
    final_stats = local_rag.get_collection_stats()
    print(f"\n📚 Updated Knowledge Base: {final_stats.get('document_count', 0)} documents")
    print("🎉 Ingestion complete! Your AI assistant now has access to this knowledge.")
    print("💡 Use the /assist endpoint with queries to search your documents.")

if __name__ == "__main__":
    main()