"""
recommender.py
===============
Hybrid recommendation engine for LegalGraph. Combines three independent
signals into a single ranked score for each candidate CaseLaw:

    1. Collaborative Filtering (CF) score
       -- cosine similarity over an implicit user-case interaction matrix
          (simulated "who else looked up / cited this case" history).
    2. Vector Similarity (VS) score
       -- semantic similarity between the user's free-text query and each
          case's summary embedding (via vector_store.LegalVectorStore).
    3. Graph Centrality (GC) score
       -- degree centrality of each case node in the citation graph (a
          proxy for how authoritative / frequently-cited a precedent is).

Final score = weighted sum (weights configurable), producing a ranked list
of recommended cases along with a breakdown of each contributing signal so
the UI / GraphRAG explanation layer can show *why* something was ranked
highly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from graph_store import LegalGraphStore
from vector_store import LegalVectorStore


@dataclass
class RecommendationWeights:
    """Tunable weights for the hybrid scoring function. Must sum to 1.0 (not enforced, but recommended)."""
    collaborative: float = 0.30
    vector: float = 0.50
    graph_centrality: float = 0.20


@dataclass
class Recommendation:
    case_id: str
    title: str
    court: str
    year: int
    summary: str
    final_score: float
    cf_score: float
    vector_score: float
    centrality_score: float


class HybridRecommender:
    """
    Combines collaborative filtering, semantic vector similarity, and graph
    centrality into a single ranked recommendation list.
    """

    def __init__(self, graph_store: LegalGraphStore, vector_store: LegalVectorStore,
                 citation_history: pd.DataFrame,
                 weights: Optional[RecommendationWeights] = None) -> None:
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.citation_history = citation_history
        self.weights = weights or RecommendationWeights()

        self._user_item_matrix: Optional[pd.DataFrame] = None
        self._item_similarity: Optional[pd.DataFrame] = None
        self._build_cf_matrix()

    # ------------------------------------------------------------------ #
    # Collaborative filtering (item-item cosine similarity over implicit
    # user-case interactions -- a lightweight stand-in for full matrix
    # factorization, appropriate given the small synthetic dataset).
    # ------------------------------------------------------------------ #
    def _build_cf_matrix(self) -> None:
        """Pivot the long-format citation history into a user x case implicit-feedback matrix."""
        if self.citation_history.empty:
            self._user_item_matrix = pd.DataFrame()
            self._item_similarity = pd.DataFrame()
            return

        self._user_item_matrix = self.citation_history.pivot_table(
            index="user_id", columns="case_id", values="weight", fill_value=0
        )

        # Item-item cosine similarity: cases co-cited by the same users score higher.
        matrix = self._user_item_matrix.to_numpy(dtype=float)
        norms = np.linalg.norm(matrix, axis=0, keepdims=True)
        norms[norms == 0] = 1e-9  # avoid divide-by-zero for cases with no interactions
        normalized = matrix / norms
        sim_matrix = normalized.T @ normalized  # (n_cases x n_cases) cosine similarity

        self._item_similarity = pd.DataFrame(
            sim_matrix,
            index=self._user_item_matrix.columns,
            columns=self._user_item_matrix.columns,
        )

    def _cf_scores_for_seed_cases(self, seed_case_ids: List[str]) -> Dict[str, float]:
        """
        Given one or more "seed" cases a user has already engaged with,
        return CF similarity scores for all other cases (mean similarity to
        the seed set). If no seeds / no history, returns an empty dict
        (CF signal contributes 0 for all candidates).
        """
        if self._item_similarity is None or self._item_similarity.empty:
            return {}

        valid_seeds = [c for c in seed_case_ids if c in self._item_similarity.columns]
        if not valid_seeds:
            return {}

        sims = self._item_similarity[valid_seeds].mean(axis=1)
        return sims.to_dict()

    # ------------------------------------------------------------------ #
    # Graph centrality
    # ------------------------------------------------------------------ #
    def _centrality_scores(self) -> Dict[str, float]:
        """Degree centrality across the whole graph, restricted to CaseLaw nodes."""
        all_centrality = self.graph_store.degree_centrality()
        case_ids = {c["case_id"] for c in self.graph_store.nodes_by_type("CaseLaw")}
        return {node_id: score for node_id, score in all_centrality.items() if node_id in case_ids}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def recommend(self, query: str, top_k: int = 5,
                   seed_case_ids: Optional[List[str]] = None) -> List[Recommendation]:
        """
        Produce a ranked list of case recommendations for a free-text legal
        `query`, optionally personalized by a list of `seed_case_ids`
        representing cases the user has already viewed/cited (drives the CF
        component). If no seeds are given, CF contributes 0 and the ranking
        is driven by vector similarity + graph centrality only.
        """
        seed_case_ids = seed_case_ids or []

        # 1) Vector similarity: retrieve a generous candidate pool via semantic search
        vector_hits = self.vector_store.semantic_search(query, top_k=max(top_k * 3, 10))
        vector_scores = {hit["metadata"]["case_id"]: hit["similarity"]
                          for hit in vector_hits if "case_id" in hit["metadata"]}

        if not vector_scores:
            return []

        # 2) Collaborative filtering scores over the same candidate pool
        cf_scores = self._cf_scores_for_seed_cases(seed_case_ids)

        # 3) Graph centrality scores
        centrality_scores = self._centrality_scores()
        max_centrality = max(centrality_scores.values()) if centrality_scores else 1.0

        recommendations: List[Recommendation] = []
        for case_id, v_score in vector_scores.items():
            node = self.graph_store.get_node(case_id)
            if node is None:
                continue

            cf = cf_scores.get(case_id, 0.0)
            centrality_raw = centrality_scores.get(case_id, 0.0)
            centrality_norm = centrality_raw / max_centrality if max_centrality > 0 else 0.0

            final = (
                self.weights.collaborative * cf
                + self.weights.vector * v_score
                + self.weights.graph_centrality * centrality_norm
            )

            recommendations.append(Recommendation(
                case_id=case_id,
                title=node.get("title", case_id),
                court=node.get("court", ""),
                year=node.get("year", 0),
                summary=node.get("summary", ""),
                final_score=round(float(final), 4),
                cf_score=round(float(cf), 4),
                vector_score=round(float(v_score), 4),
                centrality_score=round(float(centrality_norm), 4),
            ))

        recommendations.sort(key=lambda r: r.final_score, reverse=True)
        return recommendations[:top_k]


if __name__ == "__main__":
    from data_generator import generate_all

    bundle = generate_all(persist=False)

    gs = LegalGraphStore()
    gs.load_from_bundle(bundle)

    # NOTE: requires network access to download the sentence-transformers
    # model on first run. Falls back gracefully with a clear error if
    # offline (see README for offline-mode instructions).
    try:
        vs = LegalVectorStore()
        vs.index_cases(bundle["cases"], reset=True)
        vs.index_statutes(bundle["statutes"])

        recommender = HybridRecommender(gs, vs, bundle["citation_history"])
        seed = [bundle["cases"][0]["case_id"]]
        results = recommender.recommend(
            "liability for negligence causing personal injury", top_k=5, seed_case_ids=seed
        )
        for r in results:
            print(f"[{r.final_score:.3f}] {r.title}  "
                  f"(cf={r.cf_score:.3f} vec={r.vector_score:.3f} centrality={r.centrality_score:.3f})")
    except Exception as e:  # pragma: no cover - demo convenience only
        print(f"Vector store unavailable in this environment ({e}). "
              f"Recommender logic itself is unit-testable independently of the embedding model.")
