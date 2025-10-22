"""
Request and Response Types for Human-Centered AI Assistant
Transparent, well-documented Pydantic models for API communication
"""

from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field


class AssistRequest(BaseModel):
    """
    Main assistance request with transparent parameters
    
    Human-Centered Design:
    - Clear, descriptive field names
    - Optional parameters with sensible defaults
    - User preferences can override defaults
    """
    text: str = Field(..., description="The user's input text or question")
    tone: Optional[Literal["friendly", "professional", "casual"]] = Field(
        None, description="Response tone preference"
    )
    length: Optional[Literal["brief", "medium", "detailed"]] = Field(
        None, description="Response length preference"
    )
    bullets: Optional[List[str]] = Field(
        None, description="Bullet points for email drafting"
    )
    use_web: bool = Field(
        False, description="Enable web search (privacy consideration)"
    )


class AssistResponse(BaseModel):
    """Transparent response with reasoning and next steps"""
    task_type: Literal["summarize", "email", "query"]
    output_text: str = Field(..., description="Main response content")
    reasoning: str = Field(..., description="Human explanation of approach")
    next_suggestion: str = Field(..., description="Suggested next action")
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Sources, citations, confidence scores"
    )


class MemoryRequest(BaseModel):
    """User preference updates and feedback submission"""
    # Preference updates
    tone: Optional[Literal["friendly", "professional", "casual"]] = None
    length: Optional[Literal["brief", "medium", "detailed"]] = None
    voice: Optional[Literal["male", "female", "system"]] = None
    disable_web: Optional[bool] = None
    
    # Feedback submission
    feedback: Optional[Dict[str, Any]] = Field(
        None,
        description="User feedback: {task_type, user_input, ai_response, rating, correction}"
    )


class MemoryResponse(BaseModel):
    """Current user preferences and feedback statistics"""
    preferences: Dict[str, Any] = Field(..., description="Current user preferences")
    feedback_count: int = Field(..., description="Total feedback submissions")
    last_updated: str = Field(..., description="Last preference update timestamp")


class VoiceRequest(BaseModel):
    """Voice interaction request"""
    text: Optional[str] = Field(None, description="Text input (alternative to audio)")
    enable_tts: bool = Field(True, description="Generate text-to-speech response")


class VoiceResponse(BaseModel):
    """Voice interaction response with accessibility features"""
    transcription: Optional[str] = Field(None, description="Audio transcription")
    assist_response: AssistResponse = Field(..., description="Main AI response")
    audio_url: Optional[str] = Field(None, description="TTS audio file URL")
    tts_available: bool = Field(..., description="TTS capability status")