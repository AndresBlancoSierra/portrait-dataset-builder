"""Abstract vector backend interface with USearch, FAISS, and HNSWLib implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from portrait_dataset_builder.logging import get_logger

logger = get_logger("vector_backend")


class VectorBackend(ABC):
    """Abstract interface for vector storage and search."""

    @abstractmethod
    def add(self, vec_id: int, embedding: np.ndarray) -> None:
        """Add a vector with its ID."""

    @abstractmethod
    def search(self, query: np.ndarray, k: int = 100) -> list[tuple[int, float]]:
        """Search for k nearest neighbors. Returns list of (id, distance)."""

    @abstractmethod
    def remove(self, vec_id: int) -> None:
        """Remove a vector by ID."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist index to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load index from disk."""

    @abstractmethod
    def __len__(self) -> int:
        """Return number of vectors in index."""

    @abstractmethod
    def __contains__(self, vec_id: int) -> bool:
        """Check if ID exists in index."""


class USearchBackend(VectorBackend):
    """USearch backend (recommended - MIT license, <1MB wheel)."""

    def __init__(self, dim: int = 512, metric: str = "cosine") -> None:
        try:
            from usearch.index import Index

            self.index = Index(ndim=dim, metric=metric)
            self._dim = dim
            logger.info("USearch backend initialized (dim={}, metric={})", dim, metric)
        except ImportError:
            raise ImportError(
                "usearch not installed. Install with: pip install usearch"
            ) from None

    def add(self, vec_id: int, embedding: np.ndarray) -> None:
        self.index.add(vec_id, embedding.astype(np.float32))

    def search(self, query: np.ndarray, k: int = 100) -> list[tuple[int, float]]:
        results = self.index.search(query.astype(np.float32), k)
        if results is None or len(results) == 0:
            return []
        return list(zip(results.keys, results.distances, strict=False))

    def remove(self, vec_id: int) -> None:
        self.index.remove(vec_id)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.index.save(path)

    def load(self, path: str) -> None:
        if Path(path).exists():
            self.index.load(path)

    def __len__(self) -> int:
        return len(self.index)

    def __contains__(self, vec_id: int) -> bool:
        return self.index.contains(vec_id)


class FAISSBackend(VectorBackend):
    """FAISS backend (alternative - good GPU support)."""

    def __init__(self, dim: int = 512, metric: str = "cosine") -> None:
        try:
            import faiss

            if metric == "cosine":
                self.index = faiss.IndexFlatIP(dim)
            else:
                self.index = faiss.IndexFlatL2(dim)
            self._dim = dim
            self._id_map: dict[int, int] = {}
            self._reverse_map: dict[int, int] = {}
            self._next_id = 0
            logger.info("FAISS backend initialized (dim={}, metric={})", dim, metric)
        except ImportError:
            raise ImportError(
                "faiss not installed. Install with: pip install faiss-cpu"
            ) from None

    def add(self, vec_id: int, embedding: np.ndarray) -> None:
        import faiss

        vec = embedding.astype(np.float32).reshape(1, -1)
        if self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(vec)
        self.index.add(vec)
        internal_id = self.index.ntotal - 1
        self._id_map[vec_id] = internal_id
        self._reverse_map[internal_id] = vec_id

    def search(self, query: np.ndarray, k: int = 100) -> list[tuple[int, float]]:
        import faiss

        vec = query.astype(np.float32).reshape(1, -1)
        if self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(vec)
        k = min(k, self.index.ntotal)
        if k == 0:
            return []
        distances, indices = self.index.search(vec, k)
        results = []
        for dist, idx in zip(distances[0], indices[0], strict=False):
            if idx == -1:
                continue
            img_id = self._reverse_map.get(int(idx))
            if img_id is not None:
                results.append((img_id, float(dist)))
        return results

    def remove(self, vec_id: int) -> None:
        internal_id = self._id_map.pop(vec_id, None)
        if internal_id is not None:
            self._reverse_map.pop(internal_id, None)

    def save(self, path: str) -> None:
        import faiss

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, path)

    def load(self, path: str) -> None:
        import faiss

        if Path(path).exists():
            self.index = faiss.read_index(path)

    def __len__(self) -> int:
        return self.index.ntotal

    def __contains__(self, vec_id: int) -> bool:
        return vec_id in self._id_map


class HNSWLibBackend(VectorBackend):
    """HNSWLib backend (alternative - good balance of speed/quality)."""

    def __init__(self, dim: int = 512, metric: str = "cosine") -> None:
        try:
            import hnswlib

            space = "cosine" if metric == "cosine" else "l2"
            self.index = hnswlib.BFIndex(space=space, dim=dim)
            self._dim = dim
            self._id_map: dict[int, int] = {}
            self._reverse_map: dict[int, int] = {}
            self._next_id = 0
            logger.info("HNSWLib backend initialized (dim={}, metric={})", dim, metric)
        except ImportError:
            raise ImportError(
                "hnswlib not installed. Install with: pip install hnswlib"
            ) from None

    def add(self, vec_id: int, embedding: np.ndarray) -> None:
        vec = embedding.astype(np.float32)
        self.index.add_items(vec, self._next_id)
        self._id_map[vec_id] = self._next_id
        self._reverse_map[self._next_id] = vec_id
        self._next_id += 1

    def search(self, query: np.ndarray, k: int = 100) -> list[tuple[int, float]]:
        k = min(k, len(self.index))
        if k == 0:
            return []
        labels, distances = self.index.knn_query(query.astype(np.float32), k=k)
        results = []
        for label, dist in zip(labels[0], distances[0], strict=False):
            img_id = self._reverse_map.get(int(label))
            if img_id is not None:
                results.append((img_id, float(dist)))
        return results

    def remove(self, vec_id: int) -> None:
        internal_id = self._id_map.pop(vec_id, None)
        if internal_id is not None:
            self._reverse_map.pop(internal_id, None)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.index.save_index(path)

    def load(self, path: str) -> None:
        if Path(path).exists():
            self.index.load_index(path)

    def __len__(self) -> int:
        return len(self.index)

    def __contains__(self, vec_id: int) -> bool:
        return vec_id in self._id_map


def create_vector_backend(
    backend: str = "usearch", dim: int = 512, metric: str = "cosine"
) -> VectorBackend:
    """Factory function to create vector backend."""
    backends = {
        "usearch": USearchBackend,
        "faiss": FAISSBackend,
        "hnswlib": HNSWLibBackend,
    }
    if backend not in backends:
        raise ValueError(f"Unknown backend: {backend}. Available: {list(backends.keys())}")
    return backends[backend](dim=dim, metric=metric)

