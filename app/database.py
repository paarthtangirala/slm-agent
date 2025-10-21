"""
Database models and memory system for SLM Personal Agent
"""

import asyncio
import uuid
from datetime import datetime
from typing import List, Optional
from sqlalchemy import Column, String, DateTime, Text, Integer, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import relationship, sessionmaker
import json
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    mode = Column(String, default="chat")  # chat, email, search, docs, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to messages
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    mode = Column(String, nullable=False)  # 'chat', 'email', 'search', 'docs'
    meta_data = Column(Text)  # JSON string for additional data
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to conversation
    conversation = relationship("Conversation", back_populates="messages")

class MemoryManager:
    def __init__(self, database_url: str = "sqlite+aiosqlite:///./conversations.db"):
        self.engine = create_async_engine(database_url)
        self.async_session = async_sessionmaker(self.engine, expire_on_commit=False)
    
    async def initialize(self):
        """Create database tables and handle migrations"""
        async with self.engine.begin() as conn:
            # First, try to create all tables (this will create new tables but not modify existing ones)
            await conn.run_sync(Base.metadata.create_all)
            
            # Handle migration for adding mode column to existing conversations table
            try:
                # Try to check if mode column exists
                result = await conn.execute("PRAGMA table_info(conversations)")
                columns = await result.fetchall()
                has_mode_column = any(col[1] == 'mode' for col in columns)
                
                if not has_mode_column:
                    # Add mode column with default value
                    await conn.execute("ALTER TABLE conversations ADD COLUMN mode VARCHAR DEFAULT 'chat'")
                    logger.info("Added mode column to conversations table")
            except Exception as e:
                # If conversations table doesn't exist yet, it will be created by create_all above
                logger.info(f"Migration info: {e}")
                pass
    
    async def create_conversation(self, title: str = "New Conversation", mode: str = "chat") -> str:
        """Create a new conversation and return its ID"""
        async with self.async_session() as session:
            conversation = Conversation(title=title, mode=mode)
            session.add(conversation)
            await session.commit()
            return conversation.id
    
    async def get_conversations(self, limit: int = 20) -> List[dict]:
        """Get list of recent conversations"""
        async with self.async_session() as session:
            from sqlalchemy import select, desc
            
            stmt = select(Conversation).order_by(desc(Conversation.updated_at)).limit(limit)
            result = await session.execute(stmt)
            conversations = result.scalars().all()
            
            return [
                {
                    "id": conv.id,
                    "title": conv.title,
                    "mode": conv.mode,
                    "created_at": conv.created_at.isoformat(),
                    "updated_at": conv.updated_at.isoformat()
                }
                for conv in conversations
            ]
    
    async def get_conversation_messages(self, conversation_id: str, limit: int = 50) -> List[dict]:
        """Get messages from a conversation"""
        async with self.async_session() as session:
            from sqlalchemy import select
            
            stmt = select(Message).where(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at).limit(limit)
            
            result = await session.execute(stmt)
            messages = result.scalars().all()
            
            return [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "mode": msg.mode,
                    "metadata": json.loads(msg.meta_data) if msg.meta_data else {},
                    "created_at": msg.created_at.isoformat()
                }
                for msg in messages
            ]
    
    async def add_message(self, conversation_id: str, role: str, content: str, 
                         mode: str = "chat", metadata: dict = None) -> str:
        """Add a message to a conversation"""
        async with self.async_session() as session:
            message = Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                mode=mode,
                meta_data=json.dumps(metadata) if metadata else None
            )
            session.add(message)
            
            # Update conversation timestamp
            from sqlalchemy import select, update
            stmt = update(Conversation).where(
                Conversation.id == conversation_id
            ).values(updated_at=datetime.utcnow())
            await session.execute(stmt)
            
            await session.commit()
            return message.id
    
    async def update_conversation_title(self, conversation_id: str, title: str):
        """Update conversation title"""
        async with self.async_session() as session:
            from sqlalchemy import update
            stmt = update(Conversation).where(
                Conversation.id == conversation_id
            ).values(title=title, updated_at=datetime.utcnow())
            await session.execute(stmt)
            await session.commit()
    
    async def delete_conversation(self, conversation_id: str):
        """Delete a conversation and all its messages"""
        async with self.async_session() as session:
            from sqlalchemy import select
            stmt = select(Conversation).where(Conversation.id == conversation_id)
            result = await session.execute(stmt)
            conversation = result.scalar_one_or_none()
            
            if conversation:
                await session.delete(conversation)
                await session.commit()
                return True
            return False
    
    async def get_conversation_context(self, conversation_id: str, max_messages: int = 10) -> str:
        """Get recent conversation context for AI prompts"""
        messages = await self.get_conversation_messages(conversation_id, max_messages)
        
        context_parts = []
        for msg in messages[-max_messages:]:  # Get last N messages
            role_prefix = "Human" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role_prefix}: {msg['content']}")
        
        return "\n".join(context_parts)
    
    async def generate_conversation_title(self, conversation_id: str) -> str:
        """Generate a title based on the first few messages"""
        messages = await self.get_conversation_messages(conversation_id, 3)
        
        if not messages:
            return "New Conversation"
        
        # Use first user message as basis for title
        first_user_msg = next((msg for msg in messages if msg["role"] == "user"), None)
        if first_user_msg:
            content = first_user_msg["content"]
            # Truncate and clean up for title
            title = content[:50].strip()
            if len(content) > 50:
                title += "..."
            return title
        
        return "New Conversation"

# Global memory manager instance
memory_manager = MemoryManager()

def get_db_session():
    """Get async database session - contextual wrapper for recommendations engine"""
    return memory_manager.async_session