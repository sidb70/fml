"""
Search implementation for the knowledge graph.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .models import KnowledgeGraph, Post

class GraphSearch:
    """
    Implements search functionality for the knowledge graph.
    Combines text similarity, graph traversal, and temporal relevance.
    """

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.graph = knowledge_graph
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def semantic_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Perform semantic search using text embeddings.
        Returns list of (post_id, similarity_score) tuples.
        """
        query_embedding = self.embedding_model.encode([query])[0]

        results = []
        for node_id, node_data in self.graph.graph.nodes(data=True):
            if node_data['type'] == 'post':
                post_embedding = node_data['properties'].get('embedding')
                if post_embedding is not None:
                    similarity = cosine_similarity(
                        [query_embedding],
                        [post_embedding]
                    )[0][0]
                    results.append((node_id, similarity))

        # Sort by similarity score and return top_k results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def path_based_search(
        self,
        start_node: str,
        relationship_types: List[str],
        max_depth: int = 3
    ) -> List[Dict]:
        """
        Search for related nodes by following specific relationship types.
        Returns list of paths with their relevance scores.
        """
        paths = []
        visited = set()

        def dfs(current_node: str, current_path: List[str], depth: int):
            if depth >= max_depth:
                return

            visited.add(current_node)

            for relationship in self.graph.get_relationships(current_node):
                rel_type = relationship['type']
                target = relationship['target']

                if rel_type in relationship_types and target not in visited:
                    new_path = current_path + [target]
                    path_score = self._calculate_path_score(new_path)

                    paths.append({
                        'path': new_path,
                        'score': path_score,
                        'relationships': relationship_types
                    })

                    dfs(target, new_path, depth + 1)

            visited.remove(current_node)

        dfs(start_node, [start_node], 0)
        paths.sort(key=lambda x: x['score'], reverse=True)
        return paths

    def multi_hop_search(
        self,
        query: str,
        hop_types: List[str],
        max_hops: int = 2
    ) -> List[Dict]:
        """
        Perform multi-hop search starting from semantically similar nodes.
        """
        # First find semantically similar posts
        initial_results = self.semantic_search(query, top_k=5)

        multi_hop_results = []
        for post_id, similarity in initial_results:
            # For each similar post, explore paths
            paths = self.path_based_search(
                post_id,
                hop_types,
                max_depth=max_hops
            )

            for path in paths:
                result = {
                    'start_node': post_id,
                    'initial_similarity': similarity,
                    'path': path['path'],
                    'combined_score': (similarity + path['score']) / 2
                }
                multi_hop_results.append(result)

        multi_hop_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return multi_hop_results

    def _calculate_path_score(self, path: List[str]) -> float:
        """
        Calculate relevance score for a path based on:
        - Node importance
        - Relationship strength
        - Path length penalty
        """
        if not path:
            return 0.0

        # Get node scores
        node_scores = []
        for node_id in path:
            node = self.graph.get_node(node_id)
            if node:
                # Use relevance_score if available, otherwise default to 0.5
                score = node.get('properties', {}).get('relevance_score', 0.5)
                node_scores.append(score)

        # Path length penalty (longer paths get lower scores)
        length_penalty = 1.0 / len(path)

        # Combine scores
        if node_scores:
            avg_node_score = sum(node_scores) / len(node_scores)
            return avg_node_score * length_penalty

        return 0.0
