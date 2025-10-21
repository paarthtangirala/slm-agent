"""
Task Reminders & Follow-up System for SLM Personal Agent
Intelligently tracks action items, deadlines, and follow-ups from conversations
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import re
from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer, Float
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

from .ollama_client import call_ollama
from .database import memory_manager

logger = logging.getLogger(__name__)

# Use the same base as the main database
from .database import Base

class ReminderType(str, Enum):
    TASK = "task"
    DEADLINE = "deadline"
    FOLLOW_UP = "follow_up"
    MEETING = "meeting"
    CONTACT = "contact"

class ReminderStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"
    SNOOZED = "snoozed"

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class Reminder(Base):
    __tablename__ = "reminders"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    description = Column(Text)
    reminder_type = Column(String, nullable=False)  # ReminderType enum
    priority = Column(String, default="medium")  # Priority enum
    status = Column(String, default="pending")  # ReminderStatus enum
    
    # Timing
    due_date = Column(DateTime)
    reminder_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Context
    conversation_id = Column(String)  # Link to conversation that created this
    source_message = Column(Text)  # Original message that triggered this reminder
    context_data = Column(Text)  # JSON for additional context
    
    # AI Analysis
    confidence_score = Column(Float, default=0.0)  # AI confidence in this reminder
    auto_generated = Column(Boolean, default=True)  # Whether AI generated this
    
    # Recurrence
    recurring = Column(Boolean, default=False)
    recurrence_pattern = Column(String)  # daily, weekly, monthly, etc.
    next_occurrence = Column(DateTime)

@dataclass
class ReminderData:
    id: str
    title: str
    description: str
    reminder_type: ReminderType
    priority: Priority
    status: ReminderStatus
    due_date: Optional[datetime] = None
    reminder_date: Optional[datetime] = None
    conversation_id: Optional[str] = None
    source_message: Optional[str] = None
    context_data: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    auto_generated: bool = True
    recurring: bool = False
    recurrence_pattern: Optional[str] = None

class TaskReminderSystem:
    def __init__(self):
        # Use the same database as the main application
        self.engine = memory_manager.engine
        self.async_session = memory_manager.async_session
        self._initialized = False
        
    async def initialize(self):
        """Initialize the reminder system"""
        if self._initialized:
            return
            
        # Create reminder tables
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        self._initialized = True
        logger.info("Task Reminder System initialized")
    
    async def extract_reminders_from_conversation(self, conversation_id: str, message: str, response: str) -> List[ReminderData]:
        """Extract potential reminders from a conversation using AI"""
        try:
            full_text = f"User: {message}\nAssistant: {response}"
            
            system_prompt = """You are an intelligent task extraction assistant. Analyze the conversation and extract actionable items, deadlines, follow-ups, and reminders.

CRITICAL: You MUST respond with ONLY a valid JSON array. No explanations, no markdown, no extra text.

Format (copy this exactly):
[
  {
    "title": "Brief title",
    "description": "Detailed description", 
    "type": "task",
    "priority": "medium",
    "due_in_days": 3,
    "reminder_in_days": 1,
    "confidence": 0.8,
    "recurring": false,
    "recurrence_pattern": null
  }
]

Valid values:
- type: "task", "deadline", "follow_up", "meeting", "contact"
- priority: "low", "medium", "high", "urgent"
- due_in_days: number or null
- reminder_in_days: number or null
- confidence: number between 0.1 and 1.0
- recurring: true or false
- recurrence_pattern: "daily", "weekly", "monthly", "yearly" or null

Look for actionable items like:
- "I need to email John by Friday" 
- "Call client next week"
- "Schedule meeting"
- "Remind me to..."

If no actionable items found, return: []"""

            prompt = f"""Analyze this conversation for actionable items, tasks, deadlines, and reminders:

{full_text}

Return ONLY the JSON array:"""

            response_text = await call_ollama(prompt, system_prompt)
            
            # Clean and parse the response with robust error handling
            clean_response = response_text.strip()
            logger.info(f"AI raw response for reminders: {clean_response[:200]}...")
            
            # Remove markdown code blocks
            if clean_response.startswith("```"):
                lines = clean_response.split('\n')
                clean_response = '\n'.join(lines[1:-1]) if len(lines) > 2 else clean_response
            
            # Extract JSON array
            start_idx = clean_response.find('[')
            end_idx = clean_response.rfind(']')
            
            if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
                logger.warning("No valid JSON array found in AI response")
                return []
            
            json_str = clean_response[start_idx:end_idx+1]
            
            # Try to parse JSON with multiple fallback strategies
            reminders_json = None
            try:
                reminders_json = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Primary JSON parse failed: {e}")
                # Try to fix common issues like missing quotes, trailing commas, etc.
                try:
                    # Simple fallback: if it's clearly empty or malformed, return empty
                    if 'no actionable' in clean_response.lower() or 'no clear' in clean_response.lower():
                        logger.info("AI indicated no actionable items found")
                        return []
                    
                    # For now, if JSON parsing fails, create a basic reminder from obvious keywords
                    if any(keyword in response_text.lower() for keyword in ['need to', 'must', 'should', 'have to', 'remind me']):
                        logger.info("Detected actionable language, creating fallback reminder")
                        # Create a basic fallback reminder
                        return [ReminderData(
                            id=str(uuid.uuid4()),
                            title="Action item detected",
                            description=f"Extracted from: {message[:200]}",
                            reminder_type=ReminderType.TASK,
                            priority=Priority.MEDIUM,
                            status=ReminderStatus.PENDING,
                            conversation_id=conversation_id,
                            source_message=message[:500],
                            confidence_score=0.5,
                            auto_generated=True
                        )]
                except Exception:
                    pass
                
                logger.error(f"Failed to parse reminders JSON: {e}")
                return []
            
            logger.info(f"AI returned reminders JSON: {reminders_json}")
            
            if not isinstance(reminders_json, list):
                logger.warning(f"Expected list but got {type(reminders_json)}")
                return []
            
            # Convert to ReminderData objects
            reminders = []
            for item in reminders_json:
                if isinstance(item, dict) and item.get("confidence", 0) >= 0.1:  # Lower confidence threshold
                    try:
                        # Calculate dates
                        due_date = None
                        reminder_date = None
                        
                        if item.get("due_in_days") is not None:
                            due_date = datetime.utcnow() + timedelta(days=int(item["due_in_days"]))
                        
                        if item.get("reminder_in_days") is not None:
                            reminder_date = datetime.utcnow() + timedelta(days=int(item["reminder_in_days"]))
                        elif due_date:
                            # Default: remind 1 day before due date
                            reminder_date = due_date - timedelta(days=1)
                        
                        reminder = ReminderData(
                            id=str(uuid.uuid4()),
                            title=item.get("title", "").strip()[:200],  # Limit title length
                            description=item.get("description", "").strip()[:1000],  # Limit description
                            reminder_type=ReminderType(item.get("type", "task")),
                            priority=Priority(item.get("priority", "medium")),
                            status=ReminderStatus.PENDING,
                            due_date=due_date,
                            reminder_date=reminder_date,
                            conversation_id=conversation_id,
                            source_message=message[:500],  # Limit source message length
                            confidence_score=float(item.get("confidence", 0.5)),
                            auto_generated=True,
                            recurring=bool(item.get("recurring", False)),
                            recurrence_pattern=item.get("recurrence_pattern")
                        )
                        
                        if reminder.title:  # Only add if title is not empty
                            reminders.append(reminder)
                    
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Skipping invalid reminder item: {e}")
                        continue
            
            logger.info(f"Extracted {len(reminders)} reminders from conversation")
            return reminders
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in reminder extraction: {e}")
            return []
        except Exception as e:
            logger.error(f"Reminder extraction error: {e}")
            return []
    
    async def save_reminders(self, reminders: List[ReminderData]) -> List[str]:
        """Save reminders to database"""
        await self.initialize()
        
        saved_ids = []
        async with self.async_session() as session:
            for reminder_data in reminders:
                reminder = Reminder(
                    id=reminder_data.id,
                    title=reminder_data.title,
                    description=reminder_data.description,
                    reminder_type=reminder_data.reminder_type.value,
                    priority=reminder_data.priority.value,
                    status=reminder_data.status.value,
                    due_date=reminder_data.due_date,
                    reminder_date=reminder_data.reminder_date,
                    conversation_id=reminder_data.conversation_id,
                    source_message=reminder_data.source_message,
                    context_data=json.dumps(reminder_data.context_data) if reminder_data.context_data else None,
                    confidence_score=reminder_data.confidence_score,
                    auto_generated=reminder_data.auto_generated,
                    recurring=reminder_data.recurring,
                    recurrence_pattern=reminder_data.recurrence_pattern
                )
                
                session.add(reminder)
                saved_ids.append(reminder.id)
            
            await session.commit()
        
        return saved_ids
    
    async def get_active_reminders(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all active reminders"""
        await self.initialize()
        
        async with self.async_session() as session:
            from sqlalchemy import select, or_
            
            # Get pending, overdue, and snoozed reminders
            stmt = select(Reminder).where(
                Reminder.status.in_([ReminderStatus.PENDING.value, ReminderStatus.OVERDUE.value, ReminderStatus.SNOOZED.value])
            ).order_by(Reminder.due_date.asc(), Reminder.priority.desc()).limit(limit)
            
            result = await session.execute(stmt)
            reminders = result.scalars().all()
            
            return [self._reminder_to_dict(reminder) for reminder in reminders]
    
    async def get_due_reminders(self, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """Get reminders that are due within the specified hours"""
        await self.initialize()
        
        cutoff_time = datetime.utcnow() + timedelta(hours=hours_ahead)
        
        async with self.async_session() as session:
            from sqlalchemy import select, and_
            
            stmt = select(Reminder).where(
                and_(
                    Reminder.status == ReminderStatus.PENDING.value,
                    Reminder.reminder_date <= cutoff_time
                )
            ).order_by(Reminder.reminder_date.asc())
            
            result = await session.execute(stmt)
            reminders = result.scalars().all()
            
            return [self._reminder_to_dict(reminder) for reminder in reminders]
    
    async def update_reminder_status(self, reminder_id: str, status: ReminderStatus, notes: str = None) -> bool:
        """Update reminder status"""
        await self.initialize()
        
        async with self.async_session() as session:
            from sqlalchemy import select, update
            
            stmt = select(Reminder).where(Reminder.id == reminder_id)
            result = await session.execute(stmt)
            reminder = result.scalar_one_or_none()
            
            if reminder:
                reminder.status = status.value
                reminder.updated_at = datetime.utcnow()
                
                if status == ReminderStatus.COMPLETED:
                    reminder.completed_at = datetime.utcnow()
                
                if notes:
                    context_data = json.loads(reminder.context_data) if reminder.context_data else {}
                    context_data["notes"] = notes
                    reminder.context_data = json.dumps(context_data)
                
                await session.commit()
                return True
            
            return False
    
    async def snooze_reminder(self, reminder_id: str, snooze_hours: int = 24) -> bool:
        """Snooze a reminder for specified hours"""
        await self.initialize()
        
        async with self.async_session() as session:
            from sqlalchemy import select
            
            stmt = select(Reminder).where(Reminder.id == reminder_id)
            result = await session.execute(stmt)
            reminder = result.scalar_one_or_none()
            
            if reminder:
                new_reminder_date = datetime.utcnow() + timedelta(hours=snooze_hours)
                reminder.reminder_date = new_reminder_date
                reminder.status = ReminderStatus.SNOOZED.value
                reminder.updated_at = datetime.utcnow()
                
                await session.commit()
                return True
            
            return False
    
    async def check_overdue_reminders(self):
        """Check and mark overdue reminders"""
        await self.initialize()
        
        current_time = datetime.utcnow()
        
        async with self.async_session() as session:
            from sqlalchemy import select, update, and_
            
            # Update overdue reminders
            stmt = update(Reminder).where(
                and_(
                    Reminder.status == ReminderStatus.PENDING.value,
                    Reminder.due_date < current_time
                )
            ).values(status=ReminderStatus.OVERDUE.value)
            
            await session.execute(stmt)
            await session.commit()
    
    async def get_reminder_statistics(self) -> Dict[str, Any]:
        """Get statistics about reminders"""
        await self.initialize()
        
        async with self.async_session() as session:
            from sqlalchemy import select, func
            
            # Count by status
            status_counts = {}
            for status in ReminderStatus:
                stmt = select(func.count(Reminder.id)).where(Reminder.status == status.value)
                result = await session.execute(stmt)
                status_counts[status.value] = result.scalar()
            
            # Count by type
            type_counts = {}
            for reminder_type in ReminderType:
                stmt = select(func.count(Reminder.id)).where(Reminder.reminder_type == reminder_type.value)
                result = await session.execute(stmt)
                type_counts[reminder_type.value] = result.scalar()
            
            # Count by priority
            priority_counts = {}
            for priority in Priority:
                stmt = select(func.count(Reminder.id)).where(Reminder.priority == priority.value)
                result = await session.execute(stmt)
                priority_counts[priority.value] = result.scalar()
            
            return {
                "status_distribution": status_counts,
                "type_distribution": type_counts,
                "priority_distribution": priority_counts,
                "total_reminders": sum(status_counts.values())
            }
    
    async def process_conversation_for_reminders(self, conversation_id: str, message: str, response: str) -> List[str]:
        """Main method to process a conversation and save any reminders found"""
        try:
            # Extract reminders
            reminders = await self.extract_reminders_from_conversation(conversation_id, message, response)
            
            if reminders:
                # Save to database
                saved_ids = await self.save_reminders(reminders)
                logger.info(f"Saved {len(saved_ids)} reminders from conversation {conversation_id}")
                return saved_ids
            
            return []
        except Exception as e:
            logger.error(f"Error processing conversation for reminders: {e}")
            return []
    
    def _reminder_to_dict(self, reminder: Reminder) -> Dict[str, Any]:
        """Convert Reminder model to dictionary"""
        return {
            "id": reminder.id,
            "title": reminder.title,
            "description": reminder.description,
            "type": reminder.reminder_type,
            "priority": reminder.priority,
            "status": reminder.status,
            "due_date": reminder.due_date.isoformat() if reminder.due_date else None,
            "reminder_date": reminder.reminder_date.isoformat() if reminder.reminder_date else None,
            "created_at": reminder.created_at.isoformat(),
            "updated_at": reminder.updated_at.isoformat(),
            "completed_at": reminder.completed_at.isoformat() if reminder.completed_at else None,
            "conversation_id": reminder.conversation_id,
            "source_message": reminder.source_message,
            "context_data": json.loads(reminder.context_data) if reminder.context_data else {},
            "confidence_score": reminder.confidence_score,
            "auto_generated": reminder.auto_generated,
            "recurring": reminder.recurring,
            "recurrence_pattern": reminder.recurrence_pattern
        }

# Global reminder system instance
reminder_system = TaskReminderSystem()