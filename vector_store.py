"""
vector_store.py
================
Semantic vector search layer for LegalGraph.

Supports two pluggable backends:
    1. "lightweight" (Default, Cloud-Safe, Ultra-Low RAM: ~20MB):
       Uses scikit-learn TF-IDF + sublinear n-gram cosine similarity.
       Zero PyTorch, zero background threads, ideal for Streamlit Community Cloud (under 1GB limit).
    2. "chroma" (Heavy: ~700MB+ RAM):
       Uses ChromaDB + sentence-transformers (all-MiniLM-L6-v2) with PyTorch embeddings.
"""

from __future__ import annotations

import os
import gc
from typing import Any, Dict, List, Optional

import numpy as np


class LightweightVectorStore:
    """
    Ultra-low-memory vector search engine (~20MB RAM) using TF-IDF and
    n-gram cosine similarity. Designed to prevent memory crashes on resource-constrained
    environments like Streamlit Community Cloud without requiring 1GB+ PyTorch.
    """

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
            min_df=1,
        )
        self.docs: List[str] = []
        self.doc_ids: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.tfidf_matrix: Optional[Any] = None
        self._fitted = False

    def index_cases(self, cases: List[Dict[str, Any]], reset: bool = False) -> None:
        if reset:
            self.docs.clear()
            self.doc_ids.clear()
            self.metadatas.clear()
            self._fitted = False

        for c in cases:
            concepts_str = " ".join(c.get("concepts", []))
            doc_text = f"{c['title']}. {c['summary']} {concepts_str}"
            self.doc_ids.append(c["case_id"])
            self.docs.append(doc_text)
            self.metadatas.append({
                "case_id": c["case_id"],
                "title": c["title"],
                "court": c["court"],
                "year": c["year"],
                "entity_type": "CaseLaw",
            })

        self._fit()

    def index_statutes(self, statutes: List[Dict[str, Any]]) -> None:
        for s in statutes:
            doc_text = f"{s['code_section']} {s['title']}. {s['summary']}"
            self.doc_ids.append(f"stat_{s['statute_id']}")
            self.docs.append(doc_text)
            self.metadatas.append({
                "statute_id": s["statute_id"],
                "title": s["title"],
                "code_section": s.get("code_section", ""),
                "entity_type": "Statute",
            })

        self._fit()

    def _fit(self) -> None:
        if self.docs:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.docs)
            self._fitted = True

    def semantic_search(self, query: str, top_k: int = 5,
                          entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self._fitted or not self.docs or self.tfidf_matrix is None:
            return []

        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # Rank indices
        ranked_indices = np.argsort(sims)[::-1]

        results: List[Dict[str, Any]] = []
        for idx in ranked_indices:
            meta = self.metadatas[idx]
            if entity_type and meta.get("entity_type") != entity_type:
                continue

            results.append({
                "id": self.doc_ids[idx],
                "document": self.docs[idx],
                "metadata": meta,
                "similarity": float(sims[idx]),
            })
            if len(results) >= top_k:
                break

        return results

    def count(self) -> int:
        return len(self.docs)


class ChromaVectorStore:
    """
    ChromaDB + SentenceTransformer backend (heavier, requires PyTorch).
    Lazy-imports dependencies so PyTorch is not imported unless this backend is chosen.
    """

    def __init__(self, persist_dir: Optional[str] = None, collection_name: str = "case_summaries") -> None:
        import chromadb
        from chromadb.utils import embedding_functions

        self.persist_dir = persist_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def index_cases(self, cases: List[Dict[str, Any]], reset: bool = False) -> None:
        if reset:
            name = self.collection.name
            self.client.delete_collection(name)
            self.collection = self.client.get_or_create_collection(
                name=name, embedding_function=self.embedding_fn, metadata={"hnsw:space": "cosine"}
            )

        ids = [c["case_id"] for c in cases]
        documents = [f"{c['title']}. {c['summary']}" for c in cases]
        metadatas = [
            {
                "case_id": c["case_id"],
                "title": c["title"],
                "court": c["court"],
                "year": c["year"],
                "entity_type": "CaseLaw",
            }
            for c in cases
        ]
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def index_statutes(self, statutes: List[Dict[str, Any]]) -> None:
        ids = [f"stat_{s['statute_id']}" for s in statutes]
        documents = [f"{s['title']}. {s['summary']}" for s in statutes]
        metadatas = [
            {
                "statute_id": s["statute_id"],
                "title": s["title"],
                "entity_type": "Statute",
            }
            for s in statutes
        ]
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def semantic_search(self, query: str, top_k: int = 5,
                          entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        where_filter = {"entity_type": entity_type} if entity_type else None
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
        )

        output: List[Dict[str, Any]] = []
        if not results.get("ids") or not results["ids"][0]:
            return output

        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            output.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": 1.0 - distance,
            })
        return output

    def count(self) -> int:
        return self.collection.count()


class LegalVectorStore:
    """
    Unified vector store facade.
    Defaults to 'lightweight' for safe low-RAM operation (~20MB),
    preventing Streamlit Community Cloud resource limit crashes.
    """

    def __init__(self, backend: Optional[str] = None) -> None:
        requested = backend or os.environ.get("VECTOR_BACKEND", "lightweight").lower()

        if requested == "chroma":
            try:
                self._impl = ChromaVectorStore()
                self.backend_name = "ChromaDB (Sentence-Transformers)"
            except Exception as e:
                print(f"ChromaDB initialization failed ({e}). Falling back to Lightweight engine.")
                self._impl = LightweightVectorStore()
                self.backend_name = "Lightweight (Low-RAM Fallback)"
        else:
            self._impl = LightweightVectorStore()
            self.backend_name = "Lightweight (Low-RAM)"

    def index_cases(self, cases: List[Dict[str, Any]], reset: bool = False) -> None:
        self._impl.index_cases(cases, reset=reset)

    def index_statutes(self, statutes: List[Dict[str, Any]]) -> None:
        self._impl.index_statutes(statutes)

    def semantic_search(self, query: str, top_k: int = 5,
                          entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._impl.semantic_search(query, top_k=top_k, entity_type=entity_type)

    def count(self) -> int:
        return self._impl.count()


if __name__ == "__main__":
    from data_generator import generate_all

    bundle = generate_all(persist=False)
    store = LegalVectorStore()
    store.index_cases(bundle["cases"], reset=True)
    store.index_statutes(bundle["statutes"])
    hits = store.semantic_search("negligence personal injury", top_k=3)
    for h in hits:
        print(h["metadata"].get("title"), "sim:", round(h["similarity"], 3))
