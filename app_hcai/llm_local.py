"""
Local LLM Client for Human-Centered AI Assistant
Simple, transparent Ollama integration with clear error handling
"""

import os
import time
import httpx
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class LocalLLMClient:
    """
    Local Large Language Model client using Ollama
    
    Human-Centered Design:
    - Clear error messages with actionable suggestions
    - Transparent timing and token counting
    - Health checks for system status
    - No external data transmission by default
    """
    
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "phi3:mini")
        self.timeout = 120  # 2 minutes for complex requests
        
    async def check_health(self) -> Dict[str, Any]:
        """
        Check if Ollama is running and model is available
        
        Returns transparent health status for user visibility
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Check if Ollama is running
                response = await client.get(f"{self.base_url}/api/tags")
                
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [m["name"] for m in models]
                    
                    if self.model in model_names:
                        return {
                            "status": "healthy",
                            "model": self.model,
                            "available_models": len(model_names),
                            "ollama_url": self.base_url
                        }
                    else:
                        return {
                            "status": "model_missing",
                            "model": self.model,
                            "available_models": model_names,
                            "suggestion": f"Run: ollama pull {self.model}"
                        }
                else:
                    return {
                        "status": "ollama_error",
                        "error": f"HTTP {response.status_code}",
                        "suggestion": "Check if Ollama is running"
                    }
                    
        except httpx.ConnectError:
            return {
                "status": "connection_failed",
                "ollama_url": self.base_url,
                "suggestion": "Start Ollama with: ollama serve"
            }
        except Exception as e:
            return {
                "status": "unknown_error",
                "error": str(e),
                "suggestion": "Check Ollama installation and configuration"
            }
    
    async def generate(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate response using local LLM with transparent metrics
        
        Args:
            system_prompt: System instructions for the model
            user_prompt: User's actual request
            temperature: Creativity level (0.0 = deterministic, 1.0 = creative)
            
        Returns:
            Response with timing, token count, and other transparency metrics
        """
        start_time = time.time()
        
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "options": {
                    "temperature": temperature,
                    "num_predict": 2048,  # Max response length
                },
                "stream": False  # Get complete response at once
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    duration = time.time() - start_time
                    
                    # Extract response text
                    response_text = result.get("message", {}).get("content", "")
                    
                    # Calculate metrics for transparency
                    input_tokens = len(system_prompt.split()) + len(user_prompt.split())
                    output_tokens = len(response_text.split())
                    
                    return {
                        "status": "success",
                        "response": response_text,
                        "duration_seconds": round(duration, 2),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                        "model": self.model,
                        "temperature": temperature,
                        "prompt_tokens_estimate": input_tokens,
                        "completion_tokens_estimate": output_tokens
                    }
                else:
                    error_detail = response.text
                    logger.error(f"Ollama API error: {response.status_code} - {error_detail}")
                    
                    return {
                        "status": "api_error",
                        "error": f"HTTP {response.status_code}",
                        "detail": error_detail,
                        "suggestion": "Check Ollama logs for details"
                    }
                    
        except httpx.TimeoutException:
            duration = time.time() - start_time
            logger.error(f"Request timeout after {duration:.1f} seconds")
            
            return {
                "status": "timeout",
                "duration_seconds": round(duration, 2),
                "suggestion": f"Request took longer than {self.timeout}s. Try a shorter input or simpler request."
            }
            
        except httpx.ConnectError:
            return {
                "status": "connection_failed",
                "suggestion": "Ollama may not be running. Start with: ollama serve"
            }
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Unexpected error: {e}")
            
            return {
                "status": "error",
                "error": str(e),
                "duration_seconds": round(duration, 2),
                "suggestion": "Check Ollama status and model availability"
            }
    
    async def list_models(self) -> Dict[str, Any]:
        """
        List available models for user transparency
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])
                    
                    return {
                        "status": "success",
                        "models": [
                            {
                                "name": m["name"],
                                "size": m.get("size", 0),
                                "modified": m.get("modified_at", "")
                            }
                            for m in models
                        ],
                        "current_model": self.model
                    }
                else:
                    return {
                        "status": "error",
                        "error": f"HTTP {response.status_code}",
                        "suggestion": "Check Ollama service"
                    }
                    
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "suggestion": "Ensure Ollama is running"
            }


# Global instance for application use
local_llm = LocalLLMClient()