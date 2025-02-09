"""
Optimized indexing implementation for high-scale graph querying.
"""
from typing import Dict, List, Optional, Set, Any
import numpy as np
from collections import defaultdict
import faiss
import redis
from datetime import datetime
import msgpack
from concurrent.futures import ThreadPoolExecutor
import mmap
import os
from dataclasses import dataclass
from functools import lru_cache
import xxhash
import atexit

@dataclass
class SearchConfig:
    """Configuration for search optimization."""
    vector_dimension: int = 384  # Default for all-MiniLM-L6-v2
    index_buffer_size: int = 10000  # Number of vectors to buffer before bulk indexing
    cache_ttl: int = 3600  # Cache TTL in seconds
    max_workers: int = 8  # Thread pool size
    batch_size: int = 1000  # Batch size for processing
    initial_mmap_size: int = 1024 * 1024  # 1MB initial size for mmap files

class OptimizedIndex:
    """
    Highly optimized indexing system using:
    - FAISS for vector similarity
    - Redis for caching and real-time indexing
    - Memory-mapped files for large-scale text indexing
    - Parallel processing for batch operations
    """

    def __init__(
        self,
        config: SearchConfig,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        use_redis: bool = False  # Make Redis optional
    ):
        self.config = config

        # Initialize FAISS index based on dataset size
        self.vector_index = None  # Will be initialized during first batch_index_vectors call
        self.is_trained = False

        # Initialize Redis if enabled
        self.use_redis = use_redis
        if use_redis:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=True
                )
                self.redis_client.ping()  # Test connection
            except redis.ConnectionError:
                print("Warning: Redis connection failed, falling back to in-memory cache")
                self.use_redis = False

        # In-memory cache as fallback
        self.memory_cache = {}
        self.id_mapping = {}

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)

        # Register cleanup handler
        atexit.register(self.close)

        # Buffers for batch processing
        self.vector_buffer = []
        self.vector_id_buffer = []
        self.text_index = defaultdict(set)  # In-memory text index

        # Initialize memory-mapped files
        self._init_mmap_files()

    def _create_mmap_file(self, filename: str, initial_size: int) -> tuple:
        """Create a memory-mapped file with initial size."""
        # Create directory if it doesn't exist
        os.makedirs('index_data', exist_ok=True)
        filepath = os.path.join('index_data', filename)

        # Create or open the file
        f = open(filepath, 'wb+')

        # Write initial empty bytes
        f.write(b'\0' * initial_size)
        f.flush()

        # Create memory map
        mm = mmap.mmap(f.fileno(), initial_size, access=mmap.ACCESS_WRITE)

        return f, mm

    def _init_mmap_files(self):
        """Initialize memory-mapped files for text and relationship indices."""
        # Create memory-mapped files with initial size
        self.text_index_file, self.text_index_mmap = self._create_mmap_file(
            'text_index.mmap',
            self.config.initial_mmap_size
        )

        self.rel_index_file, self.rel_index_mmap = self._create_mmap_file(
            'rel_index.mmap',
            self.config.initial_mmap_size
        )

    def _initialize_vector_index(self, num_vectors: int):
        """Initialize the appropriate FAISS index based on dataset size."""
        d = self.config.vector_dimension

        if num_vectors < 1000:
            # For small datasets, use a simple flat index
            self.vector_index = faiss.IndexFlatL2(d)
            self.is_trained = True  # Flat index doesn't need training
        else:
            # For larger datasets, use IVF index
            nlist = min(int(np.sqrt(num_vectors)), num_vectors // 30)  # Ensure enough points per cluster
            quantizer = faiss.IndexFlatL2(d)
            self.vector_index = faiss.IndexIVFFlat(quantizer, d, nlist)
            self.is_trained = False

    def _get_cache_key(self, prefix: str, *args) -> str:
        """Generate cache key with xxhash for speed."""
        return f"{prefix}:{xxhash.xxh64_hexdigest(str(args))}"

    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result with fallback to in-memory cache."""
        if self.use_redis:
            cached = self.redis_client.get(cache_key)
            if cached:
                return msgpack.unpackb(cached)
        return self.memory_cache.get(cache_key)

    def _set_cached_result(self, cache_key: str, result: Any) -> None:
        """Set cached result with fallback to in-memory cache."""
        if self.use_redis:
            try:
                self.redis_client.setex(
                    cache_key,
                    self.config.cache_ttl,
                    msgpack.packb(result)
                )
            except:
                pass
        self.memory_cache[cache_key] = result

    def batch_index_vectors(
        self,
        vectors: np.ndarray,
        ids: List[str]
    ) -> None:
        """Batch index vectors using FAISS."""
        if self.vector_index is None:
            self._initialize_vector_index(len(vectors))

        if not self.is_trained and isinstance(self.vector_index, faiss.IndexIVFFlat):
            # Train the index with the vectors
            self.vector_index.train(vectors)
            self.is_trained = True

        # Convert string IDs to integers for FAISS
        int_ids = np.array([hash(id) & 0x7FFFFFFF for id in ids], dtype=np.int64)

        # Add vectors to the index
        if isinstance(self.vector_index, faiss.IndexIVFFlat):
            self.vector_index.add_with_ids(vectors, int_ids)
        else:
            # Flat index doesn't support add_with_ids
            self.vector_index.add(vectors)
            # Store mapping separately
            for i, (str_id, int_id) in enumerate(zip(ids, int_ids)):
                self.id_mapping[i] = str_id

    def parallel_text_index(
        self,
        documents: List[tuple[str, str]]
    ) -> None:
        """Index text documents in parallel."""
        def _index_chunk(chunk: List[tuple[str, str]]) -> Dict[str, Set[str]]:
            chunk_index = defaultdict(set)
            for doc_id, text in chunk:
                tokens = set(text.lower().split())
                for token in tokens:
                    chunk_index[token].add(doc_id)
            return chunk_index

        # Split documents into chunks
        chunks = [
            documents[i:i + self.config.batch_size]
            for i in range(0, len(documents), self.config.batch_size)
        ]

        # Process chunks in parallel
        futures = [
            self.executor.submit(_index_chunk, chunk)
            for chunk in chunks
        ]

        # Merge results
        for future in futures:
            chunk_index = future.result()
            self._merge_text_index(chunk_index)

    def _merge_text_index(self, chunk_index: Dict[str, Set[str]]) -> None:
        """Merge chunk index into main text index."""
        for token, doc_ids in chunk_index.items():
            self.text_index[token].update(doc_ids)

    def search_vectors(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        nprobe: int = 10
    ) -> List[tuple[str, float]]:
        """
        Efficient vector similarity search using FAISS.
        """
        if self.vector_index is None:
            return []

        cache_key = self._get_cache_key('vector_search', query_vector.tobytes(), top_k, nprobe)
        cached_result = self._get_cached_result(cache_key)

        if cached_result:
            return cached_result

        # Set number of clusters to probe for IVF index
        if isinstance(self.vector_index, faiss.IndexIVFFlat):
            self.vector_index.nprobe = nprobe

        # Reshape query vector if needed
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)

        # Perform search
        distances, indices = self.vector_index.search(query_vector, top_k)

        # Convert results back to string IDs
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx != -1:
                if isinstance(self.vector_index, faiss.IndexIVFFlat):
                    str_id = self.id_mapping.get(idx)
                else:
                    str_id = self.id_mapping.get(i)
                if str_id:
                    results.append((str_id, float(dist)))

        self._set_cached_result(cache_key, results)
        return results

    def compound_search(
        self,
        query_vector: Optional[np.ndarray] = None,
        text_query: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> List[tuple[str, float]]:
        """
        Optimized compound search implementation.
        """
        cache_key = self._get_cache_key(
            'compound_search',
            query_vector.tobytes() if query_vector is not None else None,
            text_query,
            str(filters),
            top_k
        )

        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result

        candidates = set()
        scores = defaultdict(float)

        # Vector similarity search
        if query_vector is not None:
            vector_results = self.search_vectors(
                query_vector,
                top_k=top_k * 2  # Wider initial set for better results
            )
            for doc_id, score in vector_results:
                candidates.add(doc_id)
                scores[doc_id] += score

        # Text search
        if text_query:
            text_results = self._text_search(text_query)
            if candidates:
                candidates.intersection_update(text_results)
            else:
                candidates = text_results

        # Apply filters
        if filters:
            filter_results = self._apply_filters(filters)
            if filter_results:
                candidates.intersection_update(filter_results)

        # Prepare final results
        results = [
            (doc_id, scores[doc_id])
            for doc_id in candidates
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        self._set_cached_result(cache_key, results)
        return results

    def _text_search(self, query: str) -> Set[str]:
        """Text search using in-memory index."""
        tokens = set(query.lower().split())
        results = None

        for token in tokens:
            token_results = self.text_index.get(token, set())

            if results is None:
                results = token_results
            else:
                results.intersection_update(token_results)

            if not results:  # Early termination if no matches
                break

        return results or set()

    def _apply_filters(self, filters: Dict[str, Any]) -> Set[str]:
        """Apply filters using in-memory sets."""
        # For now, return empty set as filters are not implemented
        return set()

    def close(self):
        """Clean up resources."""
        try:
            self.executor.shutdown(wait=True)
            self.text_index_mmap.close()
            self.text_index_file.close()
            self.rel_index_mmap.close()
            self.rel_index_file.close()
            if self.use_redis:
                self.redis_client.close()
        except:
            pass  # Ignore errors during cleanup
