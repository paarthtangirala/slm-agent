"""
Human-Centered Prompt Templates
Clear, transparent prompts that explain their reasoning
"""

from typing import Tuple


def get_summarize_prompt(text: str, length: str = "medium") -> Tuple[str, str]:
    """
    Generate summarization prompts with transparency
    
    Returns: (system_prompt, user_prompt)
    """
    
    # Length guidelines (transparent to user)
    length_guides = {
        "brief": "1-2 sentences maximum, focus only on the core message",
        "medium": "1-2 paragraphs, include key points and conclusions", 
        "detailed": "3-4 paragraphs, include context, key points, and implications"
    }
    
    system_prompt = f"""You are a helpful AI assistant that creates clear, accurate summaries.

TASK: Summarize the provided text following these guidelines:
- Length: {length_guides.get(length, length_guides['medium'])}
- Focus on facts, key insights, and actionable information
- Use clear, accessible language
- Maintain accuracy - don't add information not in the original

TRANSPARENCY: Explain your summarization approach briefly at the end.
"""

    user_prompt = f"""Please summarize this text:

{text}

Length requested: {length}
"""

    return system_prompt, user_prompt


def get_email_prompt(bullet_points: list, tone: str = "professional") -> Tuple[str, str]:
    """
    Generate email drafting prompts with tone awareness
    
    Returns: (system_prompt, user_prompt)
    """
    
    tone_guides = {
        "professional": "formal language, respectful tone, clear structure",
        "friendly": "warm but appropriate language, personable tone",
        "casual": "relaxed language, conversational tone, less formal structure"
    }
    
    system_prompt = f"""You are a helpful AI assistant that drafts clear, effective emails.

TASK: Create an email based on the provided bullet points
- Tone: {tone} ({tone_guides.get(tone, 'professional style')})
- Include proper email structure (greeting, body, closing)
- Make the content clear and actionable
- Keep it concise but complete

FORMAT: Write just the email content (greeting through closing).
Do not include subject line unless specifically requested.
"""

    # Format bullet points clearly
    bullets_text = "\n".join([f"• {point}" for point in bullet_points])
    
    user_prompt = f"""Draft an email using these key points:

{bullets_text}

Tone: {tone}
"""

    return system_prompt, user_prompt


def get_query_prompt(question: str, context: str = "", use_web: bool = False) -> Tuple[str, str]:
    """
    Generate query/question answering prompts with context awareness
    
    Returns: (system_prompt, user_prompt)
    """
    
    context_instruction = ""
    if context.strip():
        context_instruction = """
CONTEXT PROVIDED: Use the context below to inform your answer. Cite sources when possible.
Prefer information from the provided context over general knowledge.
"""
    else:
        context_instruction = """
NO CONTEXT PROVIDED: Answer based on your general knowledge.
Be clear about limitations and suggest where user could find more current information.
"""
    
    web_instruction = ""
    if use_web:
        web_instruction = """
WEB SEARCH ENABLED: Some web search results may be included in context.
Clearly distinguish between local knowledge and web sources.
"""
    
    system_prompt = f"""You are a helpful AI assistant that provides accurate, helpful answers.

TASK: Answer the user's question clearly and completely
- Be accurate and honest about what you know/don't know
- Provide actionable information when possible
- Use clear, accessible language
- Structure longer answers with headers or bullet points

{context_instruction}{web_instruction}

TRANSPARENCY: If you're uncertain or information might be outdated, say so clearly.
"""

    user_prompt = f"""Question: {question}"""
    
    if context.strip():
        user_prompt += f"""

Available Context:
{context}
"""
    
    return system_prompt, user_prompt


def get_feedback_learning_prompt(feedback_data: dict) -> Tuple[str, str]:
    """
    Generate prompts for learning from user feedback
    
    Human-Centered Design: Continuous improvement through user input
    """
    
    system_prompt = """You are analyzing user feedback to improve AI responses.

TASK: Suggest specific improvements based on user feedback
- Identify what worked well and what didn't
- Propose concrete changes for similar future requests
- Focus on actionable improvements
- Consider user preferences and communication style

Be specific and constructive in your analysis.
"""
    
    user_prompt = f"""Analyze this feedback:

Original User Input: {feedback_data.get('user_input', 'N/A')}
AI Response: {feedback_data.get('ai_response', 'N/A')}
User Rating: {feedback_data.get('rating', 'N/A')}/5
User Correction/Comment: {feedback_data.get('correction', 'None provided')}

What improvements should be made for similar requests?
"""
    
    return system_prompt, user_prompt