"""
Graph data models for the knowledge graph implementation.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

import networkx as nx

@dataclass
class NodeMetadata:
    """Common metadata for all nodes."""
    id: str
    name: str
    type: str
    properties: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class Topic:
    """Represents a topic node in the knowledge graph."""
    metadata: NodeMetadata
    description: Optional[str] = None
    parent_topics: Set[str] = field(default_factory=set)
    subtopics: Set[str] = field(default_factory=set)

    @classmethod
    def create(cls, id: str, name: str, description: Optional[str] = None) -> 'Topic':
        """Factory method to create a Topic instance."""
        metadata = NodeMetadata(
            id=id,
            name=name,
            type="topic"
        )
        return cls(
            metadata=metadata,
            description=description
        )

@dataclass
class Post:
    """Represents a post node in the knowledge graph."""
    metadata: NodeMetadata
    content: str
    embedding: List[float]
    topics: Set[str] = field(default_factory=set)
    entities: Set[str] = field(default_factory=set)
    relevance_score: float = 0.0
    confidence_level: float = 0.0

    @classmethod
    def create(
        cls,
        id: str,
        name: str,
        content: str,
        embedding: List[float]
    ) -> 'Post':
        """Factory method to create a Post instance."""
        metadata = NodeMetadata(
            id=id,
            name=name,
            type="post"
        )
        return cls(
            metadata=metadata,
            content=content,
            embedding=embedding
        )

class KnowledgeGraph:
    """
    Knowledge graph implementation using NetworkX.
    Handles graph operations, relationships, and basic querying.
    """
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_node(self, node: Topic | Post) -> None:
        """Add a node to the graph with its properties."""
        node_data = {
            'type': node.metadata.type,
            'name': node.metadata.name,
            'properties': node.metadata.properties,
            'created_at': node.metadata.created_at,
            'updated_at': node.metadata.updated_at
        }

        if isinstance(node, Post):
            node_data.update({
                'content': node.content,
                'embedding': node.embedding,
                'topics': list(node.topics),  # Convert to list for serialization
                'entities': list(node.entities),
                'relevance_score': node.relevance_score,
                'confidence_level': node.confidence_level
            })
        elif isinstance(node, Topic):
            node_data.update({
                'description': node.description,
                'parent_topics': list(node.parent_topics),  # Convert to list for serialization
                'subtopics': list(node.subtopics)
            })

        self.graph.add_node(node.metadata.id, **node_data)

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Dict = None
    ) -> None:
        """Add a relationship between two nodes."""
        if properties is None:
            properties = {}

        self.graph.add_edge(
            source_id,
            target_id,
            type=relationship_type,
            properties=properties
        )

    def get_node(self, node_id: str) -> Optional[Dict]:
        """Retrieve a node and its properties by ID."""
        if node_id in self.graph:
            node_data = dict(self.graph.nodes[node_id])
            # Convert lists back to sets where appropriate
            if 'topics' in node_data:
                node_data['topics'] = set(node_data['topics'])
            if 'entities' in node_data:
                node_data['entities'] = set(node_data['entities'])
            if 'parent_topics' in node_data:
                node_data['parent_topics'] = set(node_data['parent_topics'])
            if 'subtopics' in node_data:
                node_data['subtopics'] = set(node_data['subtopics'])
            return {
                'id': node_id,
                **node_data
            }
        return None

    def get_relationships(self, node_id: str, relationship_type: Optional[str] = None) -> List[Dict]:
        """Get all relationships for a node, optionally filtered by type."""
        relationships = []

        for _, target, data in self.graph.edges(node_id, data=True):
            if relationship_type is None or data['type'] == relationship_type:
                relationships.append({
                    'source': node_id,
                    'target': target,
                    'type': data['type'],
                    'properties': data['properties']
                })

        return relationships

    def get_neighbors(self, node_id: str, relationship_type: Optional[str] = None) -> List[str]:
        """Get neighboring node IDs, optionally filtered by relationship type."""
        if relationship_type is None:
            return list(self.graph.neighbors(node_id))

        neighbors = []
        for _, target, data in self.graph.edges(node_id, data=True):
            if data['type'] == relationship_type:
                neighbors.append(target)

        return neighbors
