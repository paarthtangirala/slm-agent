#!/usr/bin/env python3
"""
Human-Centered AI Assistant - Main Entry Point
Simple script to start the HCAI assistant with proper module loading
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    try:
        from app_hcai.main import app
        import uvicorn
        
        port = int(os.getenv("PORT", 8000))
        
        print("🚀 Starting Human-Centered AI Assistant")
        print("📚 HCAI Principles: Transparency, Accessibility, Personalization, Privacy, Feedback")
        print(f"🌐 Access at: http://localhost:{port}")
        print()
        
        uvicorn.run(
            "app_hcai.main:app",
            host="0.0.0.0",
            port=port,
            reload=True if os.getenv("ENV") == "development" else False
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install dependencies: pip install -r requirements_hcai.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Startup error: {e}")
        sys.exit(1)