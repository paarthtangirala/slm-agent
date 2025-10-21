"""
Ollama client utilities for the SLM Personal Agent
"""

import httpx
import logging
import os

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

async def call_ollama(prompt: str, system_prompt: str = None, model: str = "phi3:mini") -> str:
    """Call Ollama API with the given prompt"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            data = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            
            if system_prompt:
                data["system"] = system_prompt
            
            response = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=data)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
    except Exception as e:
        logger.error(f"Ollama API error: {e}")
        raise Exception(f"Failed to call Ollama: {e}")

async def check_ollama_health() -> bool:
    """Check if Ollama is available and healthy"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            return response.status_code == 200
    except Exception:
        return False