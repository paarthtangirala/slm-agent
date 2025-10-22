"""
Explainable Intent Detection for Human-Centered AI Assistant
Transparent, rule-based classification without black boxes
"""

import re
from typing import Literal
from .types import AssistRequest


def detect_intent(request: AssistRequest) -> Literal["summarize", "email", "query"]:
    """
    Detect user intent using transparent heuristics
    
    Human-Centered Design:
    - Simple, explainable rules
    - No complex ML models
    - Clear precedence order
    - Easy to debug and modify
    """
    text = request.text.lower().strip()
    
    # Email intent: explicit keywords + bullet points provided
    email_keywords = [
        "email", "draft", "write to", "send to", "compose",
        "message", "letter", "correspondence", "memo"
    ]
    
    if any(keyword in text for keyword in email_keywords) or request.bullets:
        return "email"
    
    # Summarization intent: explicit summary requests
    summary_keywords = [
        "summarize", "summary", "tldr", "tl;dr", "brief", "overview",
        "main points", "key points", "gist", "essence", "recap",
        "condense", "distill", "highlight"
    ]
    
    if any(keyword in text for keyword in summary_keywords):
        # Additional check: long input text suggests summarization
        if len(request.text) > 500:  # Transparent threshold
            return "summarize"
        # Or explicit summary command structure
        if re.search(r'(summarize|summary of|tldr|overview of)', text):
            return "summarize"
    
    # Default: query/question answering
    return "query"


def explain_intent_reasoning(request: AssistRequest, detected_intent: str) -> str:
    """
    Provide human-readable explanation of intent detection
    
    Transparency principle: User should understand how decisions are made
    """
    text = request.text.lower().strip()
    
    if detected_intent == "email":
        if request.bullets:
            return f"Detected email drafting task because bullet points were provided ({len(request.bullets)} points)."
        else:
            # Find which keyword triggered email detection
            email_keywords = ["email", "draft", "write to", "send to", "compose", "message", "letter"]
            found_keyword = next((kw for kw in email_keywords if kw in text), "email-related")
            return f"Detected email drafting task because text contains '{found_keyword}'."
    
    elif detected_intent == "summarize":
        summary_keywords = ["summarize", "summary", "tldr", "brief", "overview", "main points"]
        found_keyword = next((kw for kw in summary_keywords if kw in text), "summary-related")
        
        reasoning = f"Detected summarization task because text contains '{found_keyword}'"
        if len(request.text) > 500:
            reasoning += f" and input is long ({len(request.text)} characters)"
        return reasoning + "."
    
    else:  # query
        return "Detected general query/question because no specific email or summary keywords were found."


def get_intent_confidence(request: AssistRequest, detected_intent: str) -> float:
    """
    Simple confidence scoring for transparency
    
    Returns confidence score between 0.0 and 1.0
    """
    text = request.text.lower().strip()
    
    if detected_intent == "email":
        if request.bullets:
            return 0.95  # Very confident when bullets provided
        email_keywords = ["email", "draft", "write to", "send to", "compose"]
        matches = sum(1 for kw in email_keywords if kw in text)
        return min(0.7 + (matches * 0.1), 0.9)
    
    elif detected_intent == "summarize":
        summary_keywords = ["summarize", "summary", "tldr", "brief", "overview"]
        matches = sum(1 for kw in summary_keywords if kw in text)
        confidence = 0.6 + (matches * 0.15)
        
        # Boost confidence for long text
        if len(request.text) > 500:
            confidence += 0.2
            
        return min(confidence, 0.95)
    
    else:  # query - default case
        return 0.8  # Moderate confidence as this is the fallback


def get_task_suggestions(detected_intent: str) -> list:
    """
    Provide helpful suggestions based on detected intent
    
    Human-Centered Design: Proactive assistance
    """
    if detected_intent == "email":
        return [
            "Consider adding bullet points for key topics",
            "Specify the tone (professional, friendly, casual)",
            "Mention the recipient context if relevant"
        ]
    
    elif detected_intent == "summarize":
        return [
            "Choose length preference (brief, medium, detailed)",
            "Specify focus areas if you want targeted summary",
            "Consider saving the summary for future reference"
        ]
    
    else:  # query
        return [
            "Enable web search for latest information",
            "Ask follow-up questions for clarification",
            "Add relevant documents to knowledge base for better answers"
        ]