"""
Test script for building and querying a knowledge graph using Wikipedia data.
"""
import logging
from typing import List, Dict, Set, Optional, Tuple
import wikipedia
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer
import re  # Added missing import

from src.graph.models import KnowledgeGraph, Post, Topic
from src.graph.optimized_indexing import OptimizedIndex, SearchConfig
from src.graph.visualization import create_knowledge_graph_visualization, analyze_clusters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WikiKnowledgeGraphBuilder:
    """Builds a knowledge graph from Wikipedia articles."""

    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.knowledge_graph = KnowledgeGraph()
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize optimized index with in-memory storage
        self.search_index = OptimizedIndex(
            SearchConfig(
                vector_dimension=384,  # all-MiniLM-L6-v2 dimension
                index_buffer_size=1000,
                cache_ttl=3600,
                max_workers=8,
                batch_size=100,
                initial_mmap_size=1024 * 1024  # 1MB initial size
            ),
            use_redis=False  # Disable Redis for now
        )

        self.processed_titles: Set[str] = set()
        self.failed_articles: Set[str] = set()

        # Store embeddings and relationships for visualization
        self.embeddings = []
        self.titles = []  # Added to store titles in order
        self.relationships = []

        # Common disambiguation patterns
        self.disambiguation_patterns = [
            r'\(disambiguation\)$',
            r' \(.*?\)$',  # Matches any parenthetical suffix
            r' disambiguation$',
            r'^List of ',
            r'^Category:'
        ]

    def _is_disambiguation_title(self, title: str) -> bool:
        """Check if a title appears to be a disambiguation page."""
        return any(re.search(pattern, title) for pattern in self.disambiguation_patterns)

    def _get_wiki_page(self, title: str) -> Tuple[Optional[str], Optional[str], List[str]]:
        """
        Get Wikipedia page content and links.
        Handles disambiguation pages by trying to find the most relevant article.
        """
        try:
            # First try direct title
            try:
                page = wikipedia.page(
                    title,
                    auto_suggest=False,
                    preload=True
                )
                if not self._is_disambiguation_title(page.title):
                    return page.title, page.content, page.links
            except wikipedia.DisambiguationError as e:
                # Handle disambiguation by searching
                logger.info(f"Disambiguating '{title}'...")
                search_results = wikipedia.search(title, results=5)

                # Try each search result until we find a non-disambiguation page
                for result in search_results:
                    if self._is_disambiguation_title(result):
                        continue

                    try:
                        page = wikipedia.page(
                            result,
                            auto_suggest=False,
                            preload=True
                        )
                        if not self._is_disambiguation_title(page.title):
                            logger.info(f"Selected '{page.title}' for '{title}'")
                            return page.title, page.content, page.links
                    except:
                        continue

                # If we get here, we couldn't find a good alternative
                logger.warning(f"Could not disambiguate '{title}'")
                return None, None, []
            except Exception as e:
                logger.warning(f"Failed to fetch '{title}': {str(e)}")
                return None, None, []

        except Exception as e:
            logger.warning(f"Failed to fetch '{title}': {str(e)}")
            self.failed_articles.add(title)
            return None, None, []

    def build_graph(
        self,
        seed_topics: List[str],
        max_articles: int = 100,
        max_depth: int = 2
    ) -> None:
        """
        Build knowledge graph starting from seed topics.

        Args:
            seed_topics: Initial Wikipedia topics to start from
            max_articles: Maximum number of articles to process
            max_depth: Maximum depth of topic exploration
        """
        topics_to_process = [(topic, 0) for topic in seed_topics]
        documents_for_indexing = []

        with tqdm(total=max_articles, desc="Building graph") as pbar:
            while topics_to_process and len(self.processed_titles) < max_articles:
                current_topic, depth = topics_to_process.pop(0)

                if current_topic in self.processed_titles:
                    continue

                # Get Wikipedia content
                title, content, links = self._get_wiki_page(current_topic)
                if not title or not content:
                    continue

                # Skip if we've already processed this title (after disambiguation)
                if title in self.processed_titles:
                    continue

                # Create topic node
                topic_id = f"topic_{len(self.processed_titles)}"
                topic = Topic.create(
                    id=topic_id,
                    name=title,
                    description=content[:200]  # Use first 200 chars as description
                )
                self.knowledge_graph.add_node(topic)

                # Generate embedding and create post node
                embedding = self.embedding_model.encode(
                    content[:10000],  # Limit content length for embedding
                    show_progress_bar=False
                )

                post_id = f"post_{len(self.processed_titles)}"
                post = Post.create(
                    id=post_id,
                    name=title,
                    content=content,
                    embedding=embedding.tolist()
                )
                self.knowledge_graph.add_node(post)

                # Store embedding and relationship for visualization
                self.embeddings.append(embedding)
                self.titles.append(title)  # Store title in order
                self.relationships.append((title, title, "has_content"))  # Self-reference for visualization

                # Add to documents for batch indexing
                documents_for_indexing.append(
                    (post_id, f"{title}\n{content[:1000]}")  # Index title and start of content
                )

                # Add relationship between topic and post
                self.knowledge_graph.add_relationship(
                    topic_id,
                    post_id,
                    "has_content"
                )

                # Process links for next level
                if depth < max_depth:
                    # Filter out disambiguation and special pages
                    filtered_links = [
                        link for link in links
                        if not self._is_disambiguation_title(link)
                        and link not in self.processed_titles
                        and link not in self.failed_articles
                    ]

                    # Add filtered links to processing queue
                    for link in filtered_links:
                        topics_to_process.append((link, depth + 1))
                        # Add relationship for visualization
                        self.relationships.append((title, link, "links_to"))

                self.processed_titles.add(title)
                pbar.update(1)

        # Convert embeddings to numpy array
        self.embeddings = np.array(self.embeddings)

        # Batch index documents
        logger.info("Indexing documents...")
        self.search_index.parallel_text_index(documents_for_indexing)

        # Index vectors
        if len(self.embeddings) > 0:
            logger.info("Indexing vectors...")
            self.search_index.batch_index_vectors(
                self.embeddings.astype(np.float32),
                [f"post_{i}" for i in range(len(self.embeddings))]
            )

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search the knowledge graph using the optimized index.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        query_embedding = query_embedding.astype(np.float32)  # Ensure float32

        # Perform compound search
        results = self.search_index.compound_search(
            query_vector=query_embedding,
            text_query=query,
            top_k=top_k
        )

        # Fetch full content for results
        detailed_results = []
        for post_id, score in results:
            node = self.knowledge_graph.get_node(post_id)
            if node:
                detailed_results.append({
                    'id': post_id,
                    'title': node.get('name', ''),
                    'score': float(score),
                    'content_preview': node.get('content', '')[:200] + '...'
                })

        return detailed_results

    def visualize(
        self,
        queries: Optional[List[str]] = None,
        min_cluster_size: int = 3
    ) -> None:
        """
        Create visualization of the knowledge graph with optional query points.

        Args:
            queries: Optional list of search queries to include
            min_cluster_size: Minimum size for topic clusters
        """
        if len(self.embeddings) == 0:
            logger.warning("No embeddings available for visualization")
            return

        # Get query embeddings if provided
        if queries:
            query_embeddings = [
                (query, self.embedding_model.encode(query).astype(np.float32))
                for query in queries
            ]
        else:
            query_embeddings = None

        # Create visualization
        create_knowledge_graph_visualization(
            embeddings=self.embeddings,
            titles=self.titles,  # Use stored titles in order
            relationships=self.relationships,
            queries=query_embeddings,
            min_cluster_size=min_cluster_size
        )

        # Analyze and log cluster statistics
        stats = analyze_clusters(
            embeddings=self.embeddings,
            titles=self.titles,  # Use stored titles in order
            min_cluster_size=min_cluster_size
        )

        logger.info("\nCluster Analysis:")
        logger.info(f"Number of clusters: {stats['num_clusters']}")
        logger.info(f"Noise points: {stats['noise_points']}")
        logger.info("\nCluster sizes:")
        for cluster_id, size in stats['cluster_sizes'].items():
            logger.info(f"Cluster {cluster_id}: {size} documents")

        logger.info("\nCluster contents:")
        for cluster_id, titles in stats['clusters'].items():
            logger.info(f"\nCluster {cluster_id}:")
            for title in titles:
                logger.info(f"  - {title}")

def main():
    """Main function to demonstrate graph building and searching."""
    # Initialize builder
    builder = WikiKnowledgeGraphBuilder()

    # Build graph from seed topics
    seed_topics = [
        "Artificial intelligence",
        "Machine learning",
        "Knowledge graph",
        "Natural language processing",
        "Neural network"
    ]

    builder.build_graph(
        seed_topics=seed_topics,
        max_articles=50,  # Limit for testing
        max_depth=2
    )

    # Define test queries
    test_queries = [
        "deep learning applications",
        "knowledge representation",
        "neural network architectures",
        "NLP transformers"
    ]

    # Perform searches
    logger.info("\nPerforming searches...")
    for query in test_queries:
        logger.info(f"\nSearching for: {query}")
        results = builder.search(query)

        for i, result in enumerate(results, 1):
            logger.info(f"\n{i}. {result['title']} (Score: {result['score']:.3f})")
            logger.info(f"Preview: {result['content_preview']}")

    # Create visualization with query points
    logger.info("\nCreating visualization...")
    builder.visualize(queries=test_queries)

if __name__ == "__main__":
    main()
