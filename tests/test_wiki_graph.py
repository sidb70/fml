"""
Tests for Wikipedia knowledge graph builder and search functionality.
"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from src.graph.models import KnowledgeGraph
from src.graph.optimized_indexing import OptimizedIndex, SearchConfig
from test import WikiKnowledgeGraphBuilder

class MockWikipediaPage:
    def __init__(self, title, content, links):
        self.title = title
        self.content = content
        self.links = links

class TestWikiKnowledgeGraph(unittest.TestCase):
    """Test cases for Wikipedia knowledge graph functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.builder = WikiKnowledgeGraphBuilder()

        # Mock data
        self.mock_pages = {
            "Artificial intelligence": MockWikipediaPage(
                "Artificial intelligence",
                "AI is the simulation of human intelligence by machines.",
                ["Machine learning", "Neural network"]
            ),
            "Machine learning": MockWikipediaPage(
                "Machine learning",
                "Machine learning is a subset of AI that focuses on data and algorithms.",
                ["Artificial intelligence", "Deep learning"]
            )
        }

        # Mock embedding
        self.mock_embedding = np.random.rand(384)  # all-MiniLM-L6-v2 dimension

    @patch('wikipedia.page')
    @patch('sentence_transformers.SentenceTransformer.encode')
    def test_graph_building(self, mock_encode, mock_wiki_page):
        """Test knowledge graph construction."""
        # Configure mocks
        mock_encode.return_value = self.mock_embedding
        mock_wiki_page.side_effect = lambda title, **kwargs: self.mock_pages[title]

        # Build graph
        self.builder.build_graph(
            seed_topics=["Artificial intelligence"],
            max_articles=2,
            max_depth=1
        )

        # Verify graph structure
        self.assertEqual(len(self.builder.processed_titles), 2)
        self.assertTrue(
            "Artificial intelligence" in self.builder.processed_titles
        )
        self.assertTrue(
            "Machine learning" in self.builder.processed_titles
        )

    @patch('wikipedia.page')
    @patch('sentence_transformers.SentenceTransformer.encode')
    def test_search_functionality(self, mock_encode, mock_wiki_page):
        """Test search capabilities."""
        # Configure mocks
        mock_encode.return_value = self.mock_embedding
        mock_wiki_page.side_effect = lambda title, **kwargs: self.mock_pages[title]

        # Build graph
        self.builder.build_graph(
            seed_topics=["Artificial intelligence"],
            max_articles=2,
            max_depth=1
        )

        # Test search
        results = self.builder.search("machine learning AI")

        self.assertTrue(len(results) > 0)
        self.assertTrue(all(
            isinstance(result, dict) and 'score' in result
            for result in results
        ))

    def test_optimized_index_integration(self):
        """Test integration with optimized indexing."""
        # Create test data
        test_docs = [
            ("doc1", "Test content about AI"),
            ("doc2", "Another test about machine learning")
        ]
        test_vectors = np.random.rand(2, 384)
        test_ids = ["doc1", "doc2"]

        # Test batch indexing
        self.builder.search_index.parallel_text_index(test_docs)
        self.builder.search_index.batch_index_vectors(test_vectors, test_ids)

        # Test compound search
        results = self.builder.search_index.compound_search(
            query_vector=np.random.rand(384),
            text_query="test AI",
            top_k=2
        )

        self.assertTrue(len(results) <= 2)
        self.assertTrue(all(
            isinstance(score, float) for _, score in results
        ))

    def test_error_handling(self):
        """Test error handling for failed Wikipedia requests."""
        with patch('wikipedia.page') as mock_wiki_page:
            # Simulate failed request
            mock_wiki_page.side_effect = Exception("Page not found")

            # Should handle error gracefully
            self.builder.build_graph(
                seed_topics=["NonexistentTopic"],
                max_articles=1,
                max_depth=1
            )

            self.assertTrue("NonexistentTopic" in self.builder.failed_articles)
            self.assertEqual(len(self.builder.processed_titles), 0)

if __name__ == '__main__':
    unittest.main()
