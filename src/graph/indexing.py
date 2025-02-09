"""
Indexing implementation for efficient graph querying.
"""
from typing import Dict, List, Optional, Set
import numpy as np
from collections import defaultdict

class GraphIndex:
    """
    Maintains indices for efficient graph querying:
    - Inverted text index
    - Vector similarity index
    - Relationship index
    """

    def __init__(self):
        # Text-based inverted index
        self.text_index = defaultdict(set)

        # Vector embeddings index
        self.vector_index = {}  # node_id -> embedding vector

        # Relationship index
        self.relationship_index = defaultdict(lambda: defaultdict(set))

        # Temporal index
        self.temporal_index = defaultdict(set)  # timestamp -> node_ids

    def index_text(self, node_id: str, text: str) -> None:
        """Index text content for a node."""
        # Simple tokenization (can be improved with better text processing)
        tokens = set(text.lower().split())

        for token in tokens:
            self.text_index[token].add(node_id)

    def index_vector(self, node_id: str, embedding: np.ndarray) -> None:
        """Index vector embedding for a node."""
        self.vector_index[node_id] = embedding

    def index_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str
    ) -> None:
        """Index a relationship between nodes."""
        self.relationship_index[relationship_type]['outgoing'][source_id].add(target_id)
        self.relationship_index[relationship_type]['incoming'][target_id].add(source_id)

    def index_temporal(self, node_id: str, timestamp: str) -> None:
        """Index a node by timestamp."""
        self.temporal_index[timestamp].add(node_id)

    def search_text(self, query_tokens: Set[str]) -> Set[str]:
        """Search for nodes matching text tokens."""
        if not query_tokens:
            return set()

        # Start with nodes matching first token
        first_token = next(iter(query_tokens))
        result_set = self.text_index[first_token].copy()

        # Intersect with nodes matching other tokens
        for token in query_tokens:
            result_set.intersection_update(self.text_index[token])

        return result_set

    def search_vectors(
        self,
        query_vector: np.ndarray,
        top_k: int = 10
    ) -> List[tuple[str, float]]:
        """Find most similar vectors using cosine similarity."""
        results = []

        for node_id, embedding in self.vector_index.items():
            similarity = np.dot(query_vector, embedding) / (
                np.linalg.norm(query_vector) * np.linalg.norm(embedding)
            )
            results.append((node_id, similarity))

        # Sort by similarity score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_related_nodes(
        self,
        node_id: str,
        relationship_type: str,
        direction: str = 'outgoing'
    ) -> Set[str]:
        """Get nodes related to the given node."""
        return self.relationship_index[relationship_type][direction][node_id]

    def search_temporal_range(
        self,
        start_time: str,
        end_time: str
    ) -> Set[str]:
        """Search for nodes within a time range."""
        results = set()

        for timestamp, node_ids in self.temporal_index.items():
            if start_time <= timestamp <= end_time:
                results.update(node_ids)

        return results

    def compound_search(
        self,
        text_tokens: Optional[Set[str]] = None,
        query_vector: Optional[np.ndarray] = None,
        relationship_filters: Optional[List[tuple[str, str, str]]] = None,
        temporal_range: Optional[tuple[str, str]] = None,
        top_k: int = 10
    ) -> List[tuple[str, float]]:
        """
        Perform a compound search using multiple criteria.
        Returns scored results that match all provided criteria.
        """
        candidate_nodes = set()
        scores = defaultdict(float)

        # Text search
        if text_tokens:
            text_matches = self.search_text(text_tokens)
            if not candidate_nodes:
                candidate_nodes = text_matches
            else:
                candidate_nodes.intersection_update(text_matches)

            for node_id in text_matches:
                scores[node_id] += 1.0

        # Vector similarity search
        if query_vector is not None:
            vector_matches = self.search_vectors(query_vector, top_k=len(self.vector_index))
            vector_nodes = {node_id for node_id, _ in vector_matches}

            if not candidate_nodes:
                candidate_nodes = vector_nodes
            else:
                candidate_nodes.intersection_update(vector_nodes)

            for node_id, similarity in vector_matches:
                scores[node_id] += similarity

        # Relationship filters
        if relationship_filters:
            for rel_type, source_id, direction in relationship_filters:
                related = self.get_related_nodes(source_id, rel_type, direction)
                if not candidate_nodes:
                    candidate_nodes = related
                else:
                    candidate_nodes.intersection_update(related)

        # Temporal range filter
        if temporal_range:
            start_time, end_time = temporal_range
            temporal_matches = self.search_temporal_range(start_time, end_time)
            if not candidate_nodes:
                candidate_nodes = temporal_matches
            else:
                candidate_nodes.intersection_update(temporal_matches)

        # Prepare final results
        results = [(node_id, scores[node_id]) for node_id in candidate_nodes]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
