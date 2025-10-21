"""
Enhanced Memory & Personalization Engine for SLM Personal Agent
Advanced user preference learning, behavior analysis, and personalized experience delivery
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import re
from collections import defaultdict, Counter
import math

from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer, JSON, Float
from sqlalchemy.ext.asyncio import AsyncSession

from .database import Base, memory_manager, Conversation, Message
from .ollama_client import call_ollama

logger = logging.getLogger(__name__)

class PreferenceCategory(str, Enum):
    COMMUNICATION = "communication"
    WORKFLOW = "workflow" 
    CONTENT = "content"
    INTERFACE = "interface"
    ASSISTANCE = "assistance"
    PRIVACY = "privacy"
    NOTIFICATION = "notification"

class LearningStrength(str, Enum):
    WEAK = "weak"          # 1-2 observations
    MODERATE = "moderate"   # 3-5 observations
    STRONG = "strong"      # 6-10 observations
    VERY_STRONG = "very_strong"  # 10+ observations

@dataclass
class UserPreference:
    id: str
    category: PreferenceCategory
    preference_key: str
    preference_value: Any
    confidence: float  # 0.0 to 1.0
    strength: LearningStrength
    evidence_count: int
    last_observed: datetime
    context: Dict[str, Any]

@dataclass
class PersonalizationInsight:
    insight_type: str
    title: str
    description: str
    confidence: float
    actionable: bool
    recommendation: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

class UserProfile(Base):
    __tablename__ = "user_profiles"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, default="default")  # For multi-user support
    profile_name = Column(String, default="Default Profile")
    
    # Core characteristics
    communication_style = Column(String)  # formal, casual, technical, creative
    preferred_detail_level = Column(String)  # brief, moderate, detailed, comprehensive
    learning_pace = Column(String)  # fast, moderate, slow, adaptive
    interaction_mode = Column(String)  # direct, conversational, guided, exploratory
    
    # Behavioral patterns
    active_hours = Column(JSON)  # Hours of day when most active
    session_duration_preference = Column(String)  # short, medium, long
    multitasking_tolerance = Column(String)  # low, medium, high
    interruption_sensitivity = Column(String)  # high, medium, low
    
    # Content preferences
    preferred_formats = Column(JSON)  # text, voice, visual, interactive
    topic_interests = Column(JSON)  # Array of interested topics
    complexity_preference = Column(String)  # simple, balanced, complex
    
    # Technical preferences
    preferred_tools = Column(JSON)  # Most used features/tools
    automation_comfort = Column(String)  # manual, assisted, automated
    data_sharing_comfort = Column(String)  # minimal, selective, open
    
    # Personalization settings
    adaptive_ui_enabled = Column(Boolean, default=True)
    proactive_suggestions = Column(Boolean, default=True)
    learning_enabled = Column(Boolean, default=True)
    
    # Statistics
    total_interactions = Column(Integer, default=0)
    total_time_spent = Column(Float, default=0.0)  # Hours
    favorite_modes = Column(JSON)  # Usage frequency by mode
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UserPreferenceRecord(Base):
    __tablename__ = "user_preferences"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, default="default")
    
    # Preference details
    category = Column(String, nullable=False)  # PreferenceCategory enum
    preference_key = Column(String, nullable=False)
    preference_value = Column(Text, nullable=False)  # JSON-encoded value
    
    # Learning metadata
    confidence = Column(Float, default=0.5)
    strength = Column(String, default=LearningStrength.WEAK.value)
    evidence_count = Column(Integer, default=1)
    
    # Context and history
    context_data = Column(JSON)
    learning_source = Column(String)  # explicit, implicit, inferred
    
    # Timestamps
    first_observed = Column(DateTime, default=datetime.utcnow)
    last_observed = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class PersonalizationSession(Base):
    __tablename__ = "personalization_sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, default="default")
    session_start = Column(DateTime, default=datetime.utcnow)
    session_end = Column(DateTime)
    
    # Session characteristics
    primary_mode = Column(String)
    modes_used = Column(JSON)  # List of modes used
    duration_minutes = Column(Float)
    
    # Interaction patterns
    message_count = Column(Integer, default=0)
    avg_response_time = Column(Float)  # Seconds
    task_completion_rate = Column(Float)  # Percentage
    
    # Observed behaviors
    interruptions = Column(Integer, default=0)
    mode_switches = Column(Integer, default=0)
    help_requests = Column(Integer, default=0)
    satisfaction_indicators = Column(JSON)
    
    # Learning outcomes
    new_preferences_learned = Column(Integer, default=0)
    preferences_reinforced = Column(Integer, default=0)
    insights_generated = Column(Integer, default=0)

class PersonalizationEngine:
    def __init__(self):
        self.engine = memory_manager.engine
        self.async_session = memory_manager.async_session
        self._initialized = False
        self._current_session = None
        
        # Learning algorithms configuration
        self.confidence_threshold = 0.7
        self.min_evidence_for_strong = 6
        self.preference_decay_days = 30
        
        # Pattern recognition settings
        self.communication_patterns = {
            'formal': ['please', 'thank you', 'could you', 'would you mind'],
            'casual': ['hey', 'thanks', 'cool', 'awesome', 'yeah'],
            'technical': ['function', 'algorithm', 'implementation', 'optimize', 'debug'],
            'creative': ['innovative', 'unique', 'creative', 'artistic', 'design']
        }
        
        self.detail_patterns = {
            'brief': ['summary', 'brief', 'quick', 'short', 'tldr'],
            'detailed': ['detailed', 'comprehensive', 'thorough', 'complete', 'in-depth'],
            'step-by-step': ['step by step', 'guide me', 'walk through', 'tutorial']
        }
        
    async def initialize(self):
        """Initialize the personalization engine"""
        if self._initialized:
            return
            
        # Create personalization tables
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Initialize default user profile if none exists
        await self._ensure_user_profile()
        
        self._initialized = True
        logger.info("Personalization Engine initialized")
    
    async def _ensure_user_profile(self, user_id: str = "default"):
        """Ensure a user profile exists"""
        async with self.async_session() as session:
            from sqlalchemy import select
            
            stmt = select(UserProfile).where(UserProfile.user_id == user_id)
            result = await session.execute(stmt)
            profile = result.scalar_one_or_none()
            
            if not profile:
                profile = UserProfile(
                    user_id=user_id,
                    profile_name="Default Profile",
                    communication_style="conversational",
                    preferred_detail_level="moderate",
                    learning_pace="adaptive",
                    interaction_mode="conversational",
                    active_hours=[9, 10, 11, 14, 15, 16, 19, 20],
                    session_duration_preference="medium",
                    multitasking_tolerance="medium",
                    interruption_sensitivity="medium",
                    preferred_formats=["text", "interactive"],
                    topic_interests=[],
                    complexity_preference="balanced",
                    preferred_tools=[],
                    automation_comfort="assisted",
                    data_sharing_comfort="selective",
                    favorite_modes={}
                )
                session.add(profile)
                await session.commit()
    
    async def start_session(self, user_id: str = "default", primary_mode: str = "chat") -> str:
        """Start a new personalization session"""
        await self.initialize()
        
        session_record = PersonalizationSession(
            user_id=user_id,
            primary_mode=primary_mode,
            modes_used=[primary_mode]
        )
        
        async with self.async_session() as session:
            session.add(session_record)
            await session.commit()
            self._current_session = session_record.id
        
        return session_record.id
    
    async def end_session(self, session_id: str):
        """End a personalization session and analyze learnings"""
        await self.initialize()
        
        async with self.async_session() as session:
            from sqlalchemy import select
            
            stmt = select(PersonalizationSession).where(PersonalizationSession.id == session_id)
            result = await session.execute(stmt)
            session_record = result.scalar_one_or_none()
            
            if session_record:
                session_record.session_end = datetime.utcnow()
                session_record.duration_minutes = (
                    session_record.session_end - session_record.session_start
                ).total_seconds() / 60
                
                await session.commit()
                
                # Analyze session for learning opportunities
                await self._analyze_session(session_record)
    
    async def learn_from_interaction(self, user_id: str, message: str, response: str, 
                                   mode: str, metadata: Dict[str, Any] = None):
        """Learn user preferences from an interaction"""
        await self.initialize()
        
        try:
            # Analyze communication style
            await self._learn_communication_style(user_id, message)
            
            # Analyze detail preference
            await self._learn_detail_preference(user_id, message, response)
            
            # Analyze content preferences
            await self._learn_content_preferences(user_id, message, mode)
            
            # Update mode usage
            await self._update_mode_usage(user_id, mode)
            
            # Extract topic interests
            await self._learn_topic_interests(user_id, message)
            
            # Update session if active
            if self._current_session:
                await self._update_current_session(message, response, mode)
                
        except Exception as e:
            logger.error(f"Learning from interaction failed: {e}")
    
    async def _learn_communication_style(self, user_id: str, message: str):
        """Learn user's communication style from message patterns"""
        message_lower = message.lower()
        
        style_scores = {}
        for style, patterns in self.communication_patterns.items():
            score = sum(1 for pattern in patterns if pattern in message_lower)
            if score > 0:
                style_scores[style] = score
        
        if style_scores:
            # Find dominant style
            dominant_style = max(style_scores, key=style_scores.get)
            confidence = style_scores[dominant_style] / len(message.split()) * 10
            confidence = min(confidence, 1.0)
            
            await self._update_preference(
                user_id=user_id,
                category=PreferenceCategory.COMMUNICATION,
                key="style",
                value=dominant_style,
                confidence=confidence,
                context={"message_length": len(message), "patterns_found": style_scores}
            )
    
    async def _learn_detail_preference(self, user_id: str, message: str, response: str):
        """Learn user's preferred level of detail"""
        message_lower = message.lower()
        
        # Check for explicit detail requests
        detail_requests = {}
        for level, patterns in self.detail_patterns.items():
            score = sum(1 for pattern in patterns if pattern in message_lower)
            if score > 0:
                detail_requests[level] = score
        
        # Analyze response satisfaction based on length vs request
        if detail_requests:
            requested_level = max(detail_requests, key=detail_requests.get)
            response_length = len(response.split())
            
            # Infer satisfaction based on response appropriateness
            confidence = 0.6  # Base confidence for explicit requests
            
            await self._update_preference(
                user_id=user_id,
                category=PreferenceCategory.CONTENT,
                key="detail_level",
                value=requested_level,
                confidence=confidence,
                context={
                    "request_type": "explicit",
                    "response_length": response_length,
                    "patterns": detail_requests
                }
            )
    
    async def _learn_content_preferences(self, user_id: str, message: str, mode: str):
        """Learn content and format preferences"""
        # Track format preferences based on mode usage
        await self._update_preference(
            user_id=user_id,
            category=PreferenceCategory.CONTENT,
            key="preferred_format",
            value=mode,
            confidence=0.3,  # Lower confidence for implicit preferences
            context={"usage_context": "mode_selection"}
        )
        
        # Analyze complexity preference based on vocabulary
        complex_words = len([word for word in message.split() if len(word) > 7])
        total_words = len(message.split())
        
        if total_words > 5:  # Only analyze meaningful messages
            complexity_ratio = complex_words / total_words
            
            if complexity_ratio > 0.3:
                complexity_pref = "complex"
                confidence = min(complexity_ratio, 0.8)
            elif complexity_ratio < 0.1:
                complexity_pref = "simple"
                confidence = min(1 - complexity_ratio, 0.8)
            else:
                complexity_pref = "balanced"
                confidence = 0.5
            
            await self._update_preference(
                user_id=user_id,
                category=PreferenceCategory.CONTENT,
                key="complexity_preference",
                value=complexity_pref,
                confidence=confidence,
                context={
                    "complexity_ratio": complexity_ratio,
                    "message_length": total_words
                }
            )
    
    async def _learn_topic_interests(self, user_id: str, message: str):
        """Extract and learn topic interests from messages"""
        try:
            # Use AI to extract topics
            system_prompt = "Extract 3-5 main topics or subjects from this message. Return only topic keywords separated by commas."
            prompt = f"Extract topics from: {message}"
            
            response = await call_ollama(prompt, system_prompt)
            topics = [topic.strip().lower() for topic in response.split(',') if topic.strip()]
            
            for topic in topics[:5]:  # Limit to 5 topics
                if len(topic) > 2 and len(topic) < 30:  # Reasonable topic length
                    await self._update_preference(
                        user_id=user_id,
                        category=PreferenceCategory.CONTENT,
                        key="topic_interest",
                        value=topic,
                        confidence=0.4,
                        context={"extraction_method": "ai", "source_message_length": len(message)}
                    )
                    
        except Exception as e:
            logger.debug(f"Topic extraction failed: {e}")
    
    async def _update_mode_usage(self, user_id: str, mode: str):
        """Update mode usage statistics"""
        async with self.async_session() as session:
            from sqlalchemy import select
            
            stmt = select(UserProfile).where(UserProfile.user_id == user_id)
            result = await session.execute(stmt)
            profile = result.scalar_one_or_none()
            
            if profile:
                favorite_modes = profile.favorite_modes or {}
                favorite_modes[mode] = favorite_modes.get(mode, 0) + 1
                profile.favorite_modes = favorite_modes
                profile.total_interactions += 1
                
                await session.commit()
    
    async def _update_preference(self, user_id: str, category: PreferenceCategory, 
                               key: str, value: Any, confidence: float, 
                               context: Dict[str, Any] = None):
        """Update or create a user preference with learning reinforcement"""
        await self.initialize()
        
        async with self.async_session() as session:
            from sqlalchemy import select
            
            # Look for existing preference
            stmt = select(UserPreferenceRecord).where(
                UserPreferenceRecord.user_id == user_id,
                UserPreferenceRecord.category == category.value,
                UserPreferenceRecord.preference_key == key
            )
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()
            
            if existing:
                # Update existing preference
                existing.evidence_count += 1
                existing.last_observed = datetime.utcnow()
                existing.context_data = context or {}
                
                # Reinforce confidence with evidence
                evidence_multiplier = min(existing.evidence_count / 10, 1.0)
                new_confidence = (existing.confidence + confidence * evidence_multiplier) / 2
                existing.confidence = min(new_confidence, 1.0)
                
                # Update strength based on evidence
                if existing.evidence_count >= self.min_evidence_for_strong:
                    existing.strength = LearningStrength.VERY_STRONG.value
                elif existing.evidence_count >= 4:
                    existing.strength = LearningStrength.STRONG.value
                elif existing.evidence_count >= 2:
                    existing.strength = LearningStrength.MODERATE.value
                
                # Update value if confidence is high enough
                if new_confidence > existing.confidence * 0.8:
                    existing.preference_value = json.dumps(value)
            
            else:
                # Create new preference
                preference = UserPreferenceRecord(
                    user_id=user_id,
                    category=category.value,
                    preference_key=key,
                    preference_value=json.dumps(value),
                    confidence=confidence,
                    strength=LearningStrength.WEAK.value,
                    evidence_count=1,
                    context_data=context or {},
                    learning_source="implicit"
                )
                session.add(preference)
            
            await session.commit()
    
    async def get_personalized_suggestions(self, user_id: str = "default", 
                                         context: str = "") -> List[PersonalizationInsight]:
        """Generate personalized suggestions based on learned preferences"""
        await self.initialize()
        
        insights = []
        
        try:
            # Get user profile and preferences
            profile = await self._get_user_profile(user_id)
            preferences = await self._get_user_preferences(user_id)
            
            # Generate insights based on patterns
            insights.extend(await self._generate_workflow_insights(profile, preferences))
            insights.extend(await self._generate_content_insights(profile, preferences))
            insights.extend(await self._generate_interface_insights(profile, preferences))
            insights.extend(await self._generate_usage_insights(profile, preferences))
            
            # Sort by confidence and actionability
            insights.sort(key=lambda x: (x.actionable, x.confidence), reverse=True)
            
            return insights[:10]  # Return top 10 insights
            
        except Exception as e:
            logger.error(f"Personalized suggestions generation failed: {e}")
            return []
    
    async def _generate_workflow_insights(self, profile: UserProfile, 
                                        preferences: List[UserPreferenceRecord]) -> List[PersonalizationInsight]:
        """Generate workflow-related insights"""
        insights = []
        
        # Analyze favorite modes
        if profile.favorite_modes:
            top_mode = max(profile.favorite_modes, key=profile.favorite_modes.get)
            usage_ratio = profile.favorite_modes[top_mode] / sum(profile.favorite_modes.values())
            
            if usage_ratio > 0.6:
                insights.append(PersonalizationInsight(
                    insight_type="workflow_specialization",
                    title=f"Optimize for {top_mode.title()} Mode",
                    description=f"You use {top_mode} mode {usage_ratio:.0%} of the time. Consider customizing it further.",
                    confidence=min(usage_ratio, 0.9),
                    actionable=True,
                    recommendation=f"Add shortcuts or templates for {top_mode} mode",
                    data={"preferred_mode": top_mode, "usage_ratio": usage_ratio}
                ))
        
        # Check for automation opportunities
        automation_pref = self._get_preference_value(preferences, "automation_comfort")
        if automation_pref in ["assisted", "automated"] and profile.total_interactions > 10:
            insights.append(PersonalizationInsight(
                insight_type="automation_opportunity",
                title="Automation Opportunities",
                description="Based on your comfort with automation, consider setting up workflows.",
                confidence=0.7,
                actionable=True,
                recommendation="Explore the workflow automation features",
                data={"automation_level": automation_pref}
            ))
        
        return insights
    
    async def _generate_content_insights(self, profile: UserProfile, 
                                       preferences: List[UserPreferenceRecord]) -> List[PersonalizationInsight]:
        """Generate content-related insights"""
        insights = []
        
        # Analyze detail preferences
        detail_pref = self._get_preference_value(preferences, "detail_level")
        if detail_pref:
            insights.append(PersonalizationInsight(
                insight_type="content_optimization",
                title=f"Optimize for {detail_pref.title()} Responses",
                description=f"You prefer {detail_pref} explanations. Responses can be adjusted accordingly.",
                confidence=0.8,
                actionable=True,
                recommendation=f"Set default response style to {detail_pref}",
                data={"detail_preference": detail_pref}
            ))
        
        # Topic recommendations
        topic_interests = [p for p in preferences if p.preference_key == "topic_interest"]
        if len(topic_interests) >= 3:
            top_topics = sorted(topic_interests, key=lambda x: x.evidence_count, reverse=True)[:3]
            insights.append(PersonalizationInsight(
                insight_type="content_recommendation",
                title="Personalized Content Suggestions",
                description=f"Based on your interests in {', '.join([json.loads(t.preference_value) for t in top_topics])}",
                confidence=0.6,
                actionable=False,
                data={"top_topics": [json.loads(t.preference_value) for t in top_topics]}
            ))
        
        return insights
    
    async def _generate_interface_insights(self, profile: UserProfile, 
                                         preferences: List[UserPreferenceRecord]) -> List[PersonalizationInsight]:
        """Generate interface-related insights"""
        insights = []
        
        # Analyze session patterns
        if profile.session_duration_preference:
            insights.append(PersonalizationInsight(
                insight_type="interface_optimization",
                title="Session Management",
                description=f"Your preferred session length is {profile.session_duration_preference}.",
                confidence=0.6,
                actionable=True,
                recommendation="Adjust notification and break reminders accordingly",
                data={"session_preference": profile.session_duration_preference}
            ))
        
        return insights
    
    async def _generate_usage_insights(self, profile: UserProfile, 
                                     preferences: List[UserPreferenceRecord]) -> List[PersonalizationInsight]:
        """Generate usage pattern insights"""
        insights = []
        
        # Activity patterns
        if profile.active_hours and len(profile.active_hours) > 0:
            peak_hours = profile.active_hours
            insights.append(PersonalizationInsight(
                insight_type="usage_pattern",
                title="Optimal Activity Times",
                description=f"You're most active during hours {min(peak_hours)}-{max(peak_hours)}.",
                confidence=0.7,
                actionable=False,
                data={"peak_hours": peak_hours}
            ))
        
        # Usage frequency
        if profile.total_interactions > 50:
            avg_daily = profile.total_interactions / 30  # Assume 30-day period
            if avg_daily > 5:
                insights.append(PersonalizationInsight(
                    insight_type="engagement_level",
                    title="High Engagement User",
                    description=f"You average {avg_daily:.1f} interactions per day - you're a power user!",
                    confidence=0.8,
                    actionable=True,
                    recommendation="Consider advanced features and integrations",
                    data={"daily_average": avg_daily}
                ))
        
        return insights
    
    def _get_preference_value(self, preferences: List[UserPreferenceRecord], key: str) -> Any:
        """Get the most confident preference value for a key"""
        matching_prefs = [p for p in preferences if p.preference_key == key]
        if not matching_prefs:
            return None
        
        best_pref = max(matching_prefs, key=lambda x: x.confidence)
        try:
            return json.loads(best_pref.preference_value)
        except:
            return best_pref.preference_value
    
    async def _get_user_profile(self, user_id: str) -> UserProfile:
        """Get user profile"""
        async with self.async_session() as session:
            from sqlalchemy import select
            
            stmt = select(UserProfile).where(UserProfile.user_id == user_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
    
    async def _get_user_preferences(self, user_id: str) -> List[UserPreferenceRecord]:
        """Get all user preferences"""
        async with self.async_session() as session:
            from sqlalchemy import select
            
            stmt = select(UserPreferenceRecord).where(UserPreferenceRecord.user_id == user_id)
            result = await session.execute(stmt)
            return result.scalars().all()
    
    async def get_personalization_dashboard(self, user_id: str = "default") -> Dict[str, Any]:
        """Get comprehensive personalization dashboard"""
        await self.initialize()
        
        profile = await self._get_user_profile(user_id)
        preferences = await self._get_user_preferences(user_id)
        insights = await self.get_personalized_suggestions(user_id)
        
        # Organize preferences by category
        preferences_by_category = defaultdict(list)
        for pref in preferences:
            if pref.confidence > 0.3:  # Only include confident preferences
                preferences_by_category[pref.category].append({
                    "key": pref.preference_key,
                    "value": json.loads(pref.preference_value) if pref.preference_value.startswith('"') or pref.preference_value.startswith('[') or pref.preference_value.startswith('{') else pref.preference_value,
                    "confidence": pref.confidence,
                    "strength": pref.strength,
                    "evidence_count": pref.evidence_count
                })
        
        return {
            "profile": {
                "name": profile.profile_name if profile else "Default Profile",
                "total_interactions": profile.total_interactions if profile else 0,
                "communication_style": profile.communication_style if profile else "conversational",
                "detail_preference": profile.preferred_detail_level if profile else "moderate",
                "favorite_modes": profile.favorite_modes if profile else {},
                "active_hours": profile.active_hours if profile else [],
                "learning_enabled": profile.learning_enabled if profile else True
            },
            "preferences": dict(preferences_by_category),
            "insights": [asdict(insight) for insight in insights],
            "statistics": {
                "total_preferences": len(preferences),
                "strong_preferences": len([p for p in preferences if p.strength in [LearningStrength.STRONG.value, LearningStrength.VERY_STRONG.value]]),
                "confidence_distribution": self._get_confidence_distribution(preferences),
                "learning_activity": len([p for p in preferences if p.last_observed >= datetime.utcnow() - timedelta(days=7)])
            }
        }
    
    def _get_confidence_distribution(self, preferences: List[UserPreferenceRecord]) -> Dict[str, int]:
        """Get distribution of preference confidence levels"""
        distribution = {"high": 0, "medium": 0, "low": 0}
        
        for pref in preferences:
            if pref.confidence >= 0.7:
                distribution["high"] += 1
            elif pref.confidence >= 0.4:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
        
        return distribution
    
    async def export_user_data(self, user_id: str = "default") -> Dict[str, Any]:
        """Export all user personalization data"""
        await self.initialize()
        
        dashboard = await self.get_personalization_dashboard(user_id)
        
        # Add session history
        async with self.async_session() as session:
            from sqlalchemy import select, desc
            
            sessions_stmt = select(PersonalizationSession).where(
                PersonalizationSession.user_id == user_id
            ).order_by(desc(PersonalizationSession.session_start)).limit(50)
            
            result = await session.execute(sessions_stmt)
            sessions = result.scalars().all()
        
        dashboard["session_history"] = [
            {
                "session_id": s.id,
                "start": s.session_start.isoformat(),
                "end": s.session_end.isoformat() if s.session_end else None,
                "duration_minutes": s.duration_minutes,
                "primary_mode": s.primary_mode,
                "modes_used": s.modes_used,
                "message_count": s.message_count
            }
            for s in sessions
        ]
        
        return dashboard

# Global personalization engine instance
personalization_engine = PersonalizationEngine()