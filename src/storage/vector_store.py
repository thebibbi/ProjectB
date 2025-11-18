"""
FAISS Vector Store

Manages face and ReID embeddings for similarity search.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from src.logger import log
from src.config import settings


class FAISSVectorStore:
    """
    FAISS-based vector storage for face and ReID embeddings

    Maintains two separate indexes:
    - Face embeddings (512-dim)
    - ReID embeddings (2048-dim)
    """

    def __init__(self, index_dir: Optional[Path] = None):
        """
        Initialize FAISS vector store

        Args:
            index_dir: Directory to store indexes
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss not installed. Run: pip install faiss-cpu"
            )

        self.index_dir = index_dir or settings.faiss_index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.face_dim = settings.faiss_face_dim
        self.reid_dim = settings.faiss_reid_dim

        # Initialize indexes
        self.face_index = faiss.IndexFlatIP(self.face_dim)  # Inner product (cosine similarity)
        self.reid_index = faiss.IndexFlatIP(self.reid_dim)

        # Metadata mappings: appearance_id <-> FAISS index
        self.face_id_map: Dict[str, int] = {}  # appearance_id -> faiss_idx
        self.reid_id_map: Dict[str, int] = {}

        # Reverse mapping: faiss_idx -> appearance_id
        self.face_idx_to_id: Dict[int, str] = {}
        self.reid_idx_to_id: Dict[int, str] = {}

        log.info("FAISS vector store initialized")
        log.info(f"  Face dimension: {self.face_dim}")
        log.info(f"  ReID dimension: {self.reid_dim}")
        log.info(f"  Index directory: {self.index_dir}")

        # Try to load existing indexes
        self.load()

    def add_face_embedding(
        self,
        embedding: np.ndarray,
        appearance_id: str
    ) -> int:
        """
        Add face embedding to index

        Args:
            embedding: Face embedding vector (512-dim)
            appearance_id: Unique appearance ID

        Returns:
            FAISS index of added embedding
        """
        # Normalize embedding for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        embedding = embedding.reshape(1, -1).astype('float32')

        # Add to FAISS index
        faiss_idx = self.face_index.ntotal
        self.face_index.add(embedding)

        # Update mappings
        self.face_id_map[appearance_id] = faiss_idx
        self.face_idx_to_id[faiss_idx] = appearance_id

        log.debug(f"Added face embedding for {appearance_id} at index {faiss_idx}")

        return faiss_idx

    def add_reid_embedding(
        self,
        embedding: np.ndarray,
        appearance_id: str
    ) -> int:
        """
        Add ReID embedding to index

        Args:
            embedding: ReID embedding vector (2048-dim)
            appearance_id: Unique appearance ID

        Returns:
            FAISS index of added embedding
        """
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        embedding = embedding.reshape(1, -1).astype('float32')

        # Add to FAISS index
        faiss_idx = self.reid_index.ntotal
        self.reid_index.add(embedding)

        # Update mappings
        self.reid_id_map[appearance_id] = faiss_idx
        self.reid_idx_to_id[faiss_idx] = appearance_id

        log.debug(f"Added ReID embedding for {appearance_id} at index {faiss_idx}")

        return faiss_idx

    def search_faces(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        min_similarity: float = 0.6
    ) -> List[Tuple[str, float]]:
        """
        Search for similar face embeddings

        Args:
            query_embedding: Query embedding (512-dim)
            k: Number of results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of (appearance_id, similarity_score) tuples
        """
        if self.face_index.ntotal == 0:
            return []

        # Normalize query
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')

        # Search
        similarities, indices = self.face_index.search(query_embedding, min(k, self.face_index.ntotal))

        # Filter and convert results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if sim >= min_similarity and idx in self.face_idx_to_id:
                appearance_id = self.face_idx_to_id[idx]
                results.append((appearance_id, float(sim)))

        return results

    def search_reid(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        min_similarity: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Search for similar ReID embeddings

        Args:
            query_embedding: Query embedding (2048-dim)
            k: Number of results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of (appearance_id, similarity_score) tuples
        """
        if self.reid_index.ntotal == 0:
            return []

        # Normalize query
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')

        # Search
        similarities, indices = self.reid_index.search(query_embedding, min(k, self.reid_index.ntotal))

        # Filter and convert results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if sim >= min_similarity and idx in self.reid_idx_to_id:
                appearance_id = self.reid_idx_to_id[idx]
                results.append((appearance_id, float(sim)))

        return results

    def save(self):
        """Save indexes and mappings to disk"""
        try:
            import faiss

            # Save FAISS indexes
            face_index_path = self.index_dir / "face_embeddings.index"
            reid_index_path = self.index_dir / "reid_embeddings.index"

            faiss.write_index(self.face_index, str(face_index_path))
            faiss.write_index(self.reid_index, str(reid_index_path))

            # Save mappings
            mappings_path = self.index_dir / "index_mappings.pkl"
            with open(mappings_path, 'wb') as f:
                pickle.dump({
                    'face_id_map': self.face_id_map,
                    'reid_id_map': self.reid_id_map,
                    'face_idx_to_id': self.face_idx_to_id,
                    'reid_idx_to_id': self.reid_idx_to_id
                }, f)

            log.info(f"Saved FAISS indexes to {self.index_dir}")
            log.info(f"  Face embeddings: {self.face_index.ntotal}")
            log.info(f"  ReID embeddings: {self.reid_index.ntotal}")

        except Exception as e:
            log.error(f"Failed to save indexes: {e}")

    def load(self):
        """Load indexes and mappings from disk"""
        try:
            import faiss

            face_index_path = self.index_dir / "face_embeddings.index"
            reid_index_path = self.index_dir / "reid_embeddings.index"
            mappings_path = self.index_dir / "index_mappings.pkl"

            if not all([face_index_path.exists(), reid_index_path.exists(), mappings_path.exists()]):
                log.info("No existing indexes found, starting fresh")
                return

            # Load FAISS indexes
            self.face_index = faiss.read_index(str(face_index_path))
            self.reid_index = faiss.read_index(str(reid_index_path))

            # Load mappings
            with open(mappings_path, 'rb') as f:
                mappings = pickle.load(f)
                self.face_id_map = mappings['face_id_map']
                self.reid_id_map = mappings['reid_id_map']
                self.face_idx_to_id = mappings['face_idx_to_id']
                self.reid_idx_to_id = mappings['reid_idx_to_id']

            log.success(f"Loaded FAISS indexes from {self.index_dir}")
            log.info(f"  Face embeddings: {self.face_index.ntotal}")
            log.info(f"  ReID embeddings: {self.reid_index.ntotal}")

        except Exception as e:
            log.warning(f"Failed to load indexes: {e}")
            log.info("Starting with fresh indexes")

    def clear(self):
        """Clear all indexes and mappings"""
        self.face_index.reset()
        self.reid_index.reset()
        self.face_id_map.clear()
        self.reid_id_map.clear()
        self.face_idx_to_id.clear()
        self.reid_idx_to_id.clear()

        log.info("Cleared all FAISS indexes")

    def get_stats(self) -> dict:
        """Get statistics about the vector store"""
        return {
            "face_embeddings": self.face_index.ntotal,
            "reid_embeddings": self.reid_index.ntotal,
            "total_appearances": len(set(list(self.face_id_map.keys()) + list(self.reid_id_map.keys())))
        }
