"""
Personal Knowledge Graph for SLM Personal Agent
Builds and maintains a map of user data, relationships, and insights
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from sqlalchemy import Column, String, DateTime, Text, Float, ForeignKey
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
import logging
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

Base = declarative_base()

class Entity(Base):
    __tablename__ = "entities"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)  # person, concept, document, topic, etc.
    description = Column(Text)
    properties = Column(Text)  # JSON string
    embedding = Column(Text)  # Serialized numpy array
    confidence = Column(Float, default=0.5)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    outgoing_relations = relationship("Relation", foreign_keys="Relation.source_id", back_populates="source")
    incoming_relations = relationship("Relation", foreign_keys="Relation.target_id", back_populates="target")

class Relation(Base):
    __tablename__ = "relations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source_id = Column(String, ForeignKey("entities.id"), nullable=False)
    target_id = Column(String, ForeignKey("entities.id"), nullable=False)
    relation_type = Column(String, nullable=False)  # knows, works_on, related_to, etc.
    strength = Column(Float, default=0.5)
    context = Column(Text)  # Additional context about the relationship
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    source = relationship("Entity", foreign_keys=[source_id], back_populates="outgoing_relations")
    target = relationship("Entity", foreign_keys=[target_id], back_populates="incoming_relations")

class Insight(Base):
    __tablename__ = "insights"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    content = Column(Text, nullable=False)
    insight_type = Column(String, nullable=False)  # pattern, recommendation, summary
    entities = Column(Text)  # JSON list of related entity IDs
    confidence = Column(Float, default=0.5)
    source = Column(String)  # What generated this insight
    created_at = Column(DateTime, default=datetime.utcnow)

@dataclass
class EntityData:
    id: str
    name: str
    type: str
    description: str = ""
    properties: Dict[str, Any] = None
    confidence: float = 0.5

@dataclass
class RelationData:
    source_id: str
    target_id: str
    relation_type: str
    strength: float = 0.5
    context: str = ""

class PersonalKnowledgeGraph:
    def __init__(self, database_url: str = "sqlite+aiosqlite:///./knowledge_graph.db"):
        self.engine = create_async_engine(database_url)
        self.async_session = async_sessionmaker(self.engine, expire_on_commit=False)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.graph = nx.DiGraph()  # In-memory graph for fast operations
        self._initialized = False
    
    async def initialize(self):
        """Initialize the knowledge graph database"""
        if self._initialized:
            return
        
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Load existing graph into memory
        await self._load_graph_to_memory()
        self._initialized = True
        logger.info("Personal Knowledge Graph initialized")
    
    async def _load_graph_to_memory(self):
        """Load the graph from database to memory for fast operations"""
        async with self.async_session() as session:
            # Load entities
            from sqlalchemy import select
            entities_result = await session.execute(select(Entity))
            entities = entities_result.scalars().all()
            
            for entity in entities:
                self.graph.add_node(entity.id, 
                                  name=entity.name, 
                                  type=entity.type,
                                  description=entity.description,
                                  properties=json.loads(entity.properties) if entity.properties else {},
                                  confidence=entity.confidence)
            
            # Load relations
            relations_result = await session.execute(select(Relation))
            relations = relations_result.scalars().all()
            
            for relation in relations:
                self.graph.add_edge(relation.source_id, relation.target_id,
                                  relation_type=relation.relation_type,
                                  strength=relation.strength,
                                  context=relation.context)
    
    async def extract_entities_from_text(self, text: str, context: str = "") -> List[EntityData]:
        """Extract entities from text using AI"""
        # Use AI to extract entities
        system_prompt = """You are an entity extraction assistant. Extract important entities (people, concepts, topics, organizations, projects) from the text. 
        IMPORTANT: You must respond with ONLY a valid JSON array in this exact format:
        [{"name": "entity_name", "type": "person", "description": "brief description"}, {"name": "another_entity", "type": "concept", "description": "brief description"}]
        
        Valid types: person, concept, organization, project, topic, skill, location
        Focus on meaningful entities that could be useful for building a knowledge graph.
        If no entities found, return: []"""
        
        prompt = f"""Extract entities from this text and return ONLY the JSON array:

Context: {context}
Text: {text}

JSON array:"""
        
        try:
            # Import the call_ollama function
            from .main import call_ollama
            response = await call_ollama(prompt, system_prompt)
            
            # Clean the response - remove any markdown formatting
            clean_response = response.strip()
            if clean_response.startswith("```"):
                # Remove markdown code blocks
                lines = clean_response.split('\n')
                clean_response = '\n'.join(lines[1:-1]) if len(lines) > 2 else clean_response
            
            # Try to find JSON array in the response
            start_idx = clean_response.find('[')
            end_idx = clean_response.rfind(']')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = clean_response[start_idx:end_idx+1]
            else:
                json_str = clean_response
            
            # Parse JSON response
            entities_json = json.loads(json_str)
            
            if not isinstance(entities_json, list):
                logger.warning(f"Expected list but got {type(entities_json)}")
                return []
            
            entities = []
            for entity_data in entities_json:
                if isinstance(entity_data, dict) and "name" in entity_data:
                    entity = EntityData(
                        id=str(uuid.uuid4()),
                        name=entity_data.get("name", "").strip(),
                        type=entity_data.get("type", "concept").strip(),
                        description=entity_data.get("description", "").strip(),
                        confidence=0.7
                    )
                    if entity.name:  # Only add if name is not empty
                        entities.append(entity)
            
            logger.info(f"Extracted {len(entities)} entities from text")
            return entities
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in entity extraction: {e}. Response was: {response[:200]}")
            return []
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            return []
    
    async def extract_relations_from_text(self, text: str, entities: List[EntityData]) -> List[RelationData]:
        """Extract relationships between entities"""
        if len(entities) < 2:
            return []
        
        entity_names = [e.name for e in entities]
        
        system_prompt = """You are a relationship extraction assistant. Identify relationships between the given entities based on the text.
        Return a JSON list with this format:
        [{"source": "entity1", "target": "entity2", "relation": "works_on|knows|part_of|related_to|manages", "strength": 0.1-1.0, "context": "brief context"}]"""
        
        prompt = f"""Find relationships between these entities in the text:

Entities: {', '.join(entity_names)}
Text: {text}

Return only the JSON array:"""
        
        try:
            from .main import call_ollama
            response = await call_ollama(prompt, system_prompt)
            
            relations_json = json.loads(response.strip())
            
            relations = []
            entity_name_to_id = {e.name: e.id for e in entities}
            
            for rel_data in relations_json:
                source_name = rel_data.get("source", "")
                target_name = rel_data.get("target", "")
                
                if source_name in entity_name_to_id and target_name in entity_name_to_id:
                    relation = RelationData(
                        source_id=entity_name_to_id[source_name],
                        target_id=entity_name_to_id[target_name],
                        relation_type=rel_data.get("relation", "related_to"),
                        strength=float(rel_data.get("strength", 0.5)),
                        context=rel_data.get("context", "")
                    )
                    relations.append(relation)
            
            return relations
        except Exception as e:
            logger.error(f"Relation extraction error: {e}")
            return []
    
    async def add_entities(self, entities: List[EntityData]) -> List[str]:
        """Add entities to the knowledge graph"""
        await self.initialize()
        
        added_ids = []
        async with self.async_session() as session:
            for entity_data in entities:
                # Check if entity already exists
                from sqlalchemy import select
                existing = await session.execute(
                    select(Entity).where(Entity.name == entity_data.name, Entity.type == entity_data.type)
                )
                existing_entity = existing.scalar_one_or_none()
                
                if existing_entity:
                    # Update existing entity
                    existing_entity.description = entity_data.description
                    existing_entity.confidence = max(existing_entity.confidence, entity_data.confidence)
                    existing_entity.updated_at = datetime.utcnow()
                    added_ids.append(existing_entity.id)
                else:
                    # Create new entity
                    embedding = self.embedding_model.encode([entity_data.description or entity_data.name])
                    
                    entity = Entity(
                        id=entity_data.id,
                        name=entity_data.name,
                        type=entity_data.type,
                        description=entity_data.description,
                        properties=json.dumps(entity_data.properties or {}),
                        embedding=json.dumps(embedding[0].tolist()),
                        confidence=entity_data.confidence
                    )
                    session.add(entity)
                    
                    # Add to in-memory graph
                    self.graph.add_node(entity.id,
                                      name=entity.name,
                                      type=entity.type,
                                      description=entity.description,
                                      properties=entity_data.properties or {},
                                      confidence=entity.confidence)
                    
                    added_ids.append(entity.id)
            
            await session.commit()
        
        return added_ids
    
    async def add_relations(self, relations: List[RelationData]):
        """Add relationships to the knowledge graph"""
        await self.initialize()
        
        async with self.async_session() as session:
            for rel_data in relations:
                # Check if relation already exists
                from sqlalchemy import select
                existing = await session.execute(
                    select(Relation).where(
                        Relation.source_id == rel_data.source_id,
                        Relation.target_id == rel_data.target_id,
                        Relation.relation_type == rel_data.relation_type
                    )
                )
                existing_relation = existing.scalar_one_or_none()
                
                if existing_relation:
                    # Update strength if higher
                    existing_relation.strength = max(existing_relation.strength, rel_data.strength)
                else:
                    # Create new relation
                    relation = Relation(
                        source_id=rel_data.source_id,
                        target_id=rel_data.target_id,
                        relation_type=rel_data.relation_type,
                        strength=rel_data.strength,
                        context=rel_data.context
                    )
                    session.add(relation)
                    
                    # Add to in-memory graph
                    self.graph.add_edge(rel_data.source_id, rel_data.target_id,
                                      relation_type=rel_data.relation_type,
                                      strength=rel_data.strength,
                                      context=rel_data.context)
            
            await session.commit()
    
    async def process_conversation(self, conversation_id: str, message: str, response: str):
        """Process a conversation to extract knowledge"""
        full_text = f"User: {message}\nAssistant: {response}"
        
        # Extract entities
        entities = await self.extract_entities_from_text(full_text, f"Conversation {conversation_id}")
        
        if entities:
            # Add entities to graph
            entity_ids = await self.add_entities(entities)
            
            # Extract relations
            relations = await self.extract_relations_from_text(full_text, entities)
            if relations:
                await self.add_relations(relations)
            
            logger.info(f"Processed conversation {conversation_id}: {len(entities)} entities, {len(relations)} relations")
    
    async def process_document(self, document_name: str, content: str):
        """Process a document to extract knowledge"""
        # Extract entities
        entities = await self.extract_entities_from_text(content, f"Document: {document_name}")
        
        if entities:
            # Add document entity
            doc_entity = EntityData(
                id=str(uuid.uuid4()),
                name=document_name,
                type="document",
                description=f"Document: {document_name}",
                confidence=0.9
            )
            entities.append(doc_entity)
            
            # Add entities to graph
            entity_ids = await self.add_entities(entities)
            
            # Create relations from document to extracted entities
            doc_relations = []
            for entity in entities[:-1]:  # Exclude the document entity itself
                doc_relations.append(RelationData(
                    source_id=doc_entity.id,
                    target_id=entity.id,
                    relation_type="contains",
                    strength=0.7,
                    context=f"Entity found in document {document_name}"
                ))
            
            # Extract relations between entities
            content_relations = await self.extract_relations_from_text(content, entities[:-1])
            
            all_relations = doc_relations + content_relations
            if all_relations:
                await self.add_relations(all_relations)
            
            logger.info(f"Processed document {document_name}: {len(entities)} entities, {len(all_relations)} relations")
    
    async def find_related_entities(self, entity_name: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Find entities related to a given entity"""
        await self.initialize()
        
        # Find entity by name
        target_entity = None
        for node_id, data in self.graph.nodes(data=True):
            if data['name'].lower() == entity_name.lower():
                target_entity = node_id
                break
        
        if not target_entity:
            return []
        
        # Find connected entities
        related = []
        
        # Get direct neighbors
        for neighbor in self.graph.neighbors(target_entity):
            edge_data = self.graph.edges[target_entity, neighbor]
            node_data = self.graph.nodes[neighbor]
            
            related.append({
                "entity": node_data['name'],
                "type": node_data['type'],
                "description": node_data['description'],
                "relation": edge_data['relation_type'],
                "strength": edge_data['strength'],
                "confidence": node_data['confidence']
            })
        
        # Get entities that point to this one
        for predecessor in self.graph.predecessors(target_entity):
            edge_data = self.graph.edges[predecessor, target_entity]
            node_data = self.graph.nodes[predecessor]
            
            related.append({
                "entity": node_data['name'],
                "type": node_data['type'],
                "description": node_data['description'],
                "relation": f"inverse_{edge_data['relation_type']}",
                "strength": edge_data['strength'],
                "confidence": node_data['confidence']
            })
        
        # Sort by strength and confidence
        related.sort(key=lambda x: x['strength'] * x['confidence'], reverse=True)
        
        return related[:max_results]
    
    async def generate_insights(self) -> List[Dict[str, Any]]:
        """Generate insights from the knowledge graph"""
        await self.initialize()
        
        insights = []
        
        # Find central entities (high degree)
        centrality = nx.degree_centrality(self.graph)
        top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for entity_id, centrality_score in top_central:
            node_data = self.graph.nodes[entity_id]
            insights.append({
                "type": "central_entity",
                "content": f"{node_data['name']} appears to be a central topic in your knowledge base",
                "entity": node_data['name'],
                "score": centrality_score,
                "reasoning": f"Connected to {self.graph.degree(entity_id)} other entities"
            })
        
        # Find clusters
        try:
            if len(self.graph.nodes) > 3:
                # Convert to undirected for clustering
                undirected = self.graph.to_undirected()
                communities = nx.community.greedy_modularity_communities(undirected)
                
                for i, community in enumerate(communities):
                    if len(community) > 2:
                        community_names = [self.graph.nodes[node]['name'] for node in community]
                        insights.append({
                            "type": "topic_cluster",
                            "content": f"Found related topic cluster: {', '.join(community_names[:3])}{'...' if len(community_names) > 3 else ''}",
                            "entities": community_names,
                            "size": len(community)
                        })
        except Exception as e:
            logger.warning(f"Clustering analysis failed: {e}")
        
        return insights
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        await self.initialize()
        
        stats = {
            "total_entities": len(self.graph.nodes),
            "total_relations": len(self.graph.edges),
            "entity_types": {},
            "relation_types": {},
            "most_connected": []
        }
        
        # Count entity types
        for _, data in self.graph.nodes(data=True):
            entity_type = data['type']
            stats["entity_types"][entity_type] = stats["entity_types"].get(entity_type, 0) + 1
        
        # Count relation types
        for _, _, data in self.graph.edges(data=True):
            relation_type = data['relation_type']
            stats["relation_types"][relation_type] = stats["relation_types"].get(relation_type, 0) + 1
        
        # Most connected entities
        degrees = dict(self.graph.degree())
        most_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for entity_id, degree in most_connected:
            node_data = self.graph.nodes[entity_id]
            stats["most_connected"].append({
                "name": node_data['name'],
                "type": node_data['type'],
                "connections": degree
            })
        
        return stats

# Global knowledge graph instance
knowledge_graph = PersonalKnowledgeGraph()