"""Shared clustering utilities — Union-Find and similarity-based clustering."""

from __future__ import annotations

import numpy as np


class UnionFind:
    """Union-Find with path halving and union-by-rank."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def components(self) -> list[list[int]]:
        """Return all connected components as lists of indices."""
        comp_map: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in comp_map:
                comp_map[root] = []
            comp_map[root].append(i)
        return list(comp_map.values())


def build_cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Build pairwise cosine similarity matrix from L2-normalized embeddings.

    Args:
        embeddings: (N, D) array of embeddings (should be L2-normalized)

    Returns:
        (N, N) similarity matrix with values in [-1, 1]
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    normalized = embeddings / norms
    return normalized @ normalized.T


def cluster_by_similarity(
    similarity_matrix: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    """Cluster indices using Union-Find on a similarity matrix.

    Args:
        similarity_matrix: (N, N) pairwise similarity matrix
        threshold: minimum similarity to merge two items

    Returns:
        List of clusters, each cluster is a list of indices.
        Only clusters with size > 1 are returned.
    """
    n = similarity_matrix.shape[0]
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= threshold:
                uf.union(i, j)

    return uf.components()


def get_components(uf: UnionFind) -> list[list[int]]:
    """Get connected components from a Union-Find instance."""
    return uf.components()
