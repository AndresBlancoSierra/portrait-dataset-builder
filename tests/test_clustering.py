"""Test core clustering utilities."""

from __future__ import annotations

import numpy as np

from portrait_dataset_builder.core.clustering import (
    UnionFind,
    build_cosine_similarity_matrix,
    cluster_by_similarity,
)


class TestUnionFind:
    def test_disconnected(self) -> None:
        uf = UnionFind(5)
        assert len(uf.components()) == 5

    def test_single_union(self) -> None:
        uf = UnionFind(5)
        uf.union(0, 1)
        comps = uf.components()
        assert len(comps) == 4
        assert {0, 1} in [set(c) for c in comps]

    def test_chain(self) -> None:
        uf = UnionFind(4)
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(2, 3)
        comps = uf.components()
        assert len(comps) == 1
        assert set(comps[0]) == {0, 1, 2, 3}

    def test_no_cross(self) -> None:
        uf = UnionFind(6)
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(4, 5)
        comps = uf.components()
        assert len(comps) == 3
        for comp in comps:
            assert len(comp) == 2


class TestCosineSimilarityMatrix:
    def test_identical_embeddings(self) -> None:
        emb = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        mat = build_cosine_similarity_matrix(emb)
        assert mat[0, 1] == 1.0

    def test_orthogonal_embeddings(self) -> None:
        emb = np.array([[1.0, 0.0], [0.0, 1.0]])
        mat = build_cosine_similarity_matrix(emb)
        assert abs(mat[0, 1]) < 1e-6

    def test_opposite_embeddings(self) -> None:
        emb = np.array([[1.0, 0.0], [-1.0, 0.0]])
        mat = build_cosine_similarity_matrix(emb)
        assert mat[0, 1] == -1.0

    def test_diagonal_is_one(self) -> None:
        emb = np.random.randn(5, 10).astype(np.float32)
        mat = build_cosine_similarity_matrix(emb)
        for i in range(5):
            assert abs(mat[i, i] - 1.0) < 1e-6


class TestClusterBySimilarity:
    def test_high_threshold_no_multi_clusters(self) -> None:
        emb = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        mat = build_cosine_similarity_matrix(emb)
        clusters = cluster_by_similarity(mat, threshold=0.9)
        assert all(len(c) == 1 for c in clusters)

    def test_low_threshold_one_cluster(self) -> None:
        emb = np.array([
            [1.0, 0.0, 0.0],
            [0.95, 0.1, 0.0],
            [0.9, 0.05, 0.0],
        ], dtype=np.float32)
        mat = build_cosine_similarity_matrix(emb)
        clusters = cluster_by_similarity(mat, threshold=0.85)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_two_clusters(self) -> None:
        emb = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.99, 0.01],
        ], dtype=np.float32)
        mat = build_cosine_similarity_matrix(emb)
        clusters = cluster_by_similarity(mat, threshold=0.95)
        assert len(clusters) == 2
        for comp in clusters:
            assert len(comp) == 2
