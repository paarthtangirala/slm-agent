"""
Smart Recommendations Engine for SLM Personal Agent
Analyzes user patterns, conversation history, and knowledge graph to provide intelligent suggestions
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import re

from .ollama_client import call_ollama
from .knowledge_graph import knowledge_graph
from .database import get_db_session, Conversation, Message

logger = logging.getLogger(__name__)

@dataclass
class Recommendation:
    id: str
    type: str  # workflow, document, topic, action, question
    title: str
    description: str
    confidence: float
    reasoning: str
    action_data: Dict[str, Any] = None
    priority: int = 1  # 1=high, 2=medium, 3=low

@dataclass
class UserPattern:
    pattern_type: str
    frequency: int
    last_occurrence: datetime
    context: Dict[str, Any]

class SmartRecommendationEngine:
    def __init__(self):
        self.patterns_cache = {}
        self.recommendations_cache = {}
        self.cache_ttl = timedelta(hours=1)
        self.last_analysis = None
        
    async def get_recommendations(self, limit: int = 5) -> List[Recommendation]:
        """Get personalized recommendations for the user"""
        try:
            # Check cache first
            if self._is_cache_valid():
                cached_recs = self.recommendations_cache.get('recommendations', [])
                if cached_recs:
                    return cached_recs[:limit]
            
            # Generate fresh recommendations
            recommendations = await self._generate_recommendations()
            
            # Cache results
            self.recommendations_cache = {
                'recommendations': recommendations,
                'timestamp': datetime.utcnow()
            }
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Recommendation generation error: {e}")
            return self._get_fallback_recommendations()
    
    async def _generate_recommendations(self) -> List[Recommendation]:
        """Generate recommendations based on user patterns and data"""
        recommendations = []
        
        # Analyze user patterns
        patterns = await self._analyze_user_patterns()
        
        # Get conversation insights
        conv_insights = await self._analyze_recent_conversations()
        
        # Get knowledge graph insights
        kg_insights = await self._get_knowledge_graph_insights()
        
        # Generate different types of recommendations
        workflow_recs = await self._recommend_workflows(patterns, conv_insights)
        topic_recs = await self._recommend_topics(kg_insights, conv_insights)
        action_recs = await self._recommend_actions(patterns, conv_insights)
        question_recs = await self._recommend_questions(kg_insights, patterns)
        
        # Combine and rank recommendations
        all_recs = workflow_recs + topic_recs + action_recs + question_recs
        
        # Sort by confidence and priority
        all_recs.sort(key=lambda x: (x.priority, -x.confidence))
        
        return all_recs[:10]  # Return top 10
    
    async def _analyze_user_patterns(self) -> Dict[str, UserPattern]:
        """Analyze user behavior patterns from conversation history"""
        patterns = {}
        
        try:
            async with get_db_session()() as session:
                # Get recent conversations (last 30 days)
                recent_date = datetime.utcnow() - timedelta(days=30)
                
                from sqlalchemy import select, and_
                conversations_result = await session.execute(
                    select(Conversation).where(Conversation.updated_at >= recent_date)
                )
                conversations = conversations_result.scalars().all()
                
                # Analyze conversation patterns
                mode_usage = Counter()
                time_patterns = defaultdict(int)
                topic_patterns = Counter()
                
                for conv in conversations:
                    # Get messages for this conversation
                    messages_result = await session.execute(
                        select(Message).where(Message.conversation_id == conv.id)
                    )
                    messages = messages_result.scalars().all()
                    
                    for msg in messages:
                        if msg.role == 'user':
                            # Analyze mode usage
                            mode_usage[conv.mode or 'chat'] += 1
                            
                            # Analyze time patterns
                            hour = msg.created_at.hour
                            time_patterns[f"hour_{hour}"] += 1
                            
                            # Extract topics from message content
                            topics = self._extract_topics(msg.content)
                            topic_patterns.update(topics)
                
                # Convert to UserPattern objects
                for mode, count in mode_usage.most_common(5):
                    patterns[f"mode_{mode}"] = UserPattern(
                        pattern_type="mode_preference",
                        frequency=count,
                        last_occurrence=datetime.utcnow(),
                        context={"mode": mode, "usage_count": count}
                    )
                
                for topic, count in topic_patterns.most_common(10):
                    patterns[f"topic_{topic}"] = UserPattern(
                        pattern_type="topic_interest",
                        frequency=count,
                        last_occurrence=datetime.utcnow(),
                        context={"topic": topic, "mentions": count}
                    )
        
        except Exception as e:
            logger.error(f"Pattern analysis error: {e}")
        
        return patterns
    
    async def _analyze_recent_conversations(self) -> Dict[str, Any]:
        """Analyze recent conversation content for insights"""
        insights = {
            "recent_topics": [],
            "unresolved_questions": [],
            "frequent_requests": [],
            "collaboration_contexts": []
        }
        
        try:
            async with get_db_session()() as session:
                # Get last 10 conversations
                from sqlalchemy import select, desc
                recent_convs = await session.execute(
                    select(Conversation).order_by(desc(Conversation.updated_at)).limit(10)
                )
                conversations = recent_convs.scalars().all()
                
                for conv in conversations:
                    messages_result = await session.execute(
                        select(Message).where(Message.conversation_id == conv.id)
                    )
                    messages = messages_result.scalars().all()
                    
                    # Extract insights from conversation
                    conv_text = " ".join([msg.content for msg in messages if msg.role == 'user'])
                    
                    # Look for unresolved questions
                    questions = re.findall(r'[^.!?]*\?', conv_text)
                    insights["unresolved_questions"].extend(questions[:2])
                    
                    # Extract mentioned topics
                    topics = self._extract_topics(conv_text)
                    insights["recent_topics"].extend(topics[:3])
                    
                    # Look for collaboration indicators
                    collab_words = ['team', 'meeting', 'project', 'colleague', 'partner', 'client']
                    if any(word in conv_text.lower() for word in collab_words):
                        insights["collaboration_contexts"].append(conv.title or "Collaboration")
        
        except Exception as e:
            logger.error(f"Conversation analysis error: {e}")
        
        return insights
    
    async def _get_knowledge_graph_insights(self) -> Dict[str, Any]:
        """Get insights from the knowledge graph"""
        try:
            # Get knowledge graph statistics
            stats = await knowledge_graph.get_graph_stats()
            
            # Get recent insights
            insights = await knowledge_graph.generate_insights()
            
            return {
                "stats": stats,
                "insights": insights,
                "central_entities": stats.get("most_connected", [])[:3],
                "entity_types": stats.get("entity_types", {})
            }
        except Exception as e:
            logger.error(f"Knowledge graph insights error: {e}")
            return {}
    
    async def _recommend_workflows(self, patterns: Dict, insights: Dict) -> List[Recommendation]:
        """Recommend relevant workflows based on patterns"""
        recommendations = []
        
        try:
            # Import workflow engine
            from .workflows import workflow_engine
            
            if workflow_engine:
                templates_list = workflow_engine.get_workflow_templates()
                templates = {t['id']: t for t in templates_list}
                
                # Recommend workflows based on user patterns
                for template_name, template in templates.items():
                    confidence = 0.0
                    reasoning_parts = []
                    
                    # Check if user frequently uses related modes
                    if template_name == "research-email" and patterns.get("mode_email"):
                        confidence += 0.3
                        reasoning_parts.append("frequently uses email mode")
                    
                    if template_name == "doc-analysis" and patterns.get("mode_docs"):
                        confidence += 0.3
                        reasoning_parts.append("frequently analyzes documents")
                    
                    # Check for relevant topics in recent conversations
                    relevant_topics = ["research", "analysis", "meeting", "email", "document"]
                    for topic in relevant_topics:
                        if any(topic in str(pattern.context) for pattern in patterns.values()):
                            confidence += 0.2
                            reasoning_parts.append(f"recent interest in {topic}")
                    
                    if confidence > 0.3:
                        recommendations.append(Recommendation(
                            id=f"workflow_{template_name}",
                            type="workflow",
                            title=f"Try {template['name']} Workflow",
                            description=template["description"],
                            confidence=min(confidence, 1.0),
                            reasoning=f"Recommended because you {', '.join(reasoning_parts)}",
                            action_data={"template_name": template_name},
                            priority=1 if confidence > 0.6 else 2
                        ))
        
        except Exception as e:
            logger.error(f"Workflow recommendation error: {e}")
        
        return recommendations
    
    async def _recommend_topics(self, kg_insights: Dict, conv_insights: Dict) -> List[Recommendation]:
        """Recommend topics to explore based on knowledge graph"""
        recommendations = []
        
        try:
            # Recommend exploring central entities
            central_entities = kg_insights.get("central_entities", [])
            for entity in central_entities[:2]:
                recommendations.append(Recommendation(
                    id=f"topic_{entity['name']}",
                    type="topic",
                    title=f"Explore {entity['name']}",
                    description=f"This {entity['type']} appears frequently in your knowledge base with {entity['connections']} connections",
                    confidence=0.7,
                    reasoning=f"Central topic in your knowledge graph",
                    action_data={"entity_name": entity["name"]},
                    priority=2
                ))
            
            # Recommend related topics based on recent conversations
            recent_topics = conv_insights.get("recent_topics", [])
            for topic in recent_topics[:2]:
                recommendations.append(Recommendation(
                    id=f"related_{topic}",
                    type="topic", 
                    title=f"Deep dive into {topic}",
                    description=f"You've been discussing {topic} recently - explore related information",
                    confidence=0.6,
                    reasoning="Based on recent conversation topics",
                    action_data={"topic": topic},
                    priority=2
                ))
        
        except Exception as e:
            logger.error(f"Topic recommendation error: {e}")
        
        return recommendations
    
    async def _recommend_actions(self, patterns: Dict, insights: Dict) -> List[Recommendation]:
        """Recommend specific actions based on user behavior"""
        recommendations = []
        
        try:
            # Recommend uploading documents if user queries docs but has few
            if patterns.get("mode_docs") and not insights.get("recent_uploads"):
                recommendations.append(Recommendation(
                    id="action_upload_docs",
                    type="action",
                    title="Upload More Documents",
                    description="You frequently query documents but might benefit from uploading more files",
                    confidence=0.8,
                    reasoning="High document query usage with limited uploads",
                    action_data={"action": "upload_documents"},
                    priority=1
                ))
            
            # Recommend trying voice mode if user hasn't used it
            if not patterns.get("mode_voice"):
                recommendations.append(Recommendation(
                    id="action_try_voice",
                    type="action",
                    title="Try Voice Chat",
                    description="Experience hands-free interaction with voice commands and responses",
                    confidence=0.6,
                    reasoning="Haven't explored voice capabilities yet",
                    action_data={"action": "try_voice_mode"},
                    priority=2
                ))
            
            # Recommend knowledge graph exploration
            if len(insights.get("collaboration_contexts", [])) > 0:
                recommendations.append(Recommendation(
                    id="action_knowledge_graph",
                    type="action",
                    title="Explore Knowledge Connections", 
                    description="Visualize relationships between your projects, contacts, and topics",
                    confidence=0.7,
                    reasoning="Detected collaboration contexts in conversations",
                    action_data={"action": "explore_knowledge_graph"},
                    priority=2
                ))
        
        except Exception as e:
            logger.error(f"Action recommendation error: {e}")
        
        return recommendations
    
    async def _recommend_questions(self, kg_insights: Dict, patterns: Dict) -> List[Recommendation]:
        """Recommend follow-up questions based on user interests"""
        recommendations = []
        
        try:
            # Generate questions using AI based on user patterns
            if patterns:
                pattern_summary = []
                for pattern_name, pattern in list(patterns.items())[:5]:
                    pattern_summary.append(f"{pattern.pattern_type}: {pattern.context}")
                
                system_prompt = """You are a helpful assistant that suggests intelligent follow-up questions based on user patterns. Generate 2-3 relevant questions that would help the user explore their interests deeper or discover new insights."""
                
                prompt = f"""Based on these user patterns, suggest 2-3 thoughtful questions:
{json.dumps(pattern_summary, indent=2)}

Generate questions that are:
1. Specific and actionable
2. Help explore connections between topics
3. Encourage deeper learning
4. Are relevant to their interests

Format as a JSON array of strings."""
                
                response = await call_ollama(prompt, system_prompt)
                
                # Parse AI-generated questions
                try:
                    # Clean response and extract JSON
                    clean_response = response.strip()
                    if clean_response.startswith("```"):
                        lines = clean_response.split('\n')
                        clean_response = '\n'.join(lines[1:-1])
                    
                    start_idx = clean_response.find('[')
                    end_idx = clean_response.rfind(']')
                    if start_idx != -1 and end_idx != -1:
                        json_str = clean_response[start_idx:end_idx+1]
                        questions = json.loads(json_str)
                        
                        for i, question in enumerate(questions[:3]):
                            recommendations.append(Recommendation(
                                id=f"question_{i}",
                                type="question",
                                title="Explore This Question",
                                description=question,
                                confidence=0.6,
                                reasoning="AI-generated based on your interests",
                                action_data={"question": question},
                                priority=3
                            ))
                except json.JSONDecodeError:
                    logger.warning("Failed to parse AI-generated questions")
        
        except Exception as e:
            logger.error(f"Question recommendation error: {e}")
        
        return recommendations
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract relevant topics from text"""
        # Simple topic extraction - could be enhanced with NLP
        text_lower = text.lower()
        topics = []
        
        # Common topic keywords
        topic_keywords = {
            'ai', 'machine learning', 'data science', 'python', 'javascript', 
            'project', 'analysis', 'research', 'meeting', 'email', 'document',
            'code', 'programming', 'development', 'business', 'technology',
            'workflow', 'automation', 'report', 'presentation', 'collaboration'
        }
        
        for keyword in topic_keywords:
            if keyword in text_lower:
                topics.append(keyword)
        
        return list(set(topics))  # Remove duplicates
    
    def _is_cache_valid(self) -> bool:
        """Check if recommendations cache is still valid"""
        if not self.recommendations_cache:
            return False
        
        timestamp = self.recommendations_cache.get('timestamp')
        if not timestamp:
            return False
        
        return datetime.utcnow() - timestamp < self.cache_ttl
    
    def _get_fallback_recommendations(self) -> List[Recommendation]:
        """Get fallback recommendations when analysis fails"""
        return [
            Recommendation(
                id="fallback_upload",
                type="action",
                title="Upload Documents",
                description="Upload documents to enhance your AI assistant's knowledge",
                confidence=0.5,
                reasoning="General recommendation for new users",
                action_data={"action": "upload_documents"},
                priority=2
            ),
            Recommendation(
                id="fallback_voice",
                type="action", 
                title="Try Voice Chat",
                description="Experience hands-free interaction with voice commands",
                confidence=0.5,
                reasoning="General recommendation for exploring features",
                action_data={"action": "try_voice_mode"},
                priority=3
            ),
            Recommendation(
                id="fallback_workflow",
                type="workflow",
                title="Explore Automated Workflows",
                description="Try the built-in workflow templates for common tasks",
                confidence=0.5,
                reasoning="General recommendation for productivity",
                action_data={"action": "explore_workflows"},
                priority=2
            )
        ]

# Global recommendations engine instance
recommendations_engine = SmartRecommendationEngine()