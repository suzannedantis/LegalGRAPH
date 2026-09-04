"""
vector_store.py
================
Semantic vector search layer for LegalGraph, built on ChromaDB (local,
embedded, persistent -- no server / paid tier required) with embeddings from
`sentence-transformers` (all-MiniLM-L6-v2, a small 384-dim model that runs
comfortably on CPU).

Responsibilities:
    - Embed CaseLaw summaries (and Statute summaries) into a Chroma collection.
    - Provide semantic search: given a free-text query, return the
      top-k most similar cases/statutes by cosine similarity.

This module is the "R" (retrieval) half of GraphRAG's vector leg -- graph_rag.py
combines its output with graph traversal context before calling the LLM.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class LegalVectorStore:
    """
    Wraps a persistent ChromaDB client + collection, using a local
    SentenceTransformer embedding function so no external API calls (and no
    cost) are needed to build or query the index.
    """

    def __init__(self, persist_dir: str = PERSIST_DIR, collection_name: str = "case_summaries") -> None:
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        # PersistentClient writes the index to disk so it survives restarts
        # (no external vector DB server required).
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},  # cosine similarity for semantic search
        )

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #
    def index_cases(self, cases: List[Dict[str, Any]], reset: bool = False) -> None:
        """
        Embed and upsert each case's summary text into the collection.
        Document text = "<title>. <summary>" so the title's legal terms also
        contribute to the embedding. Metadata carries fields needed to hydrate
        the graph lookup later without re-querying the source JSON.
        """
        if reset:
            self._reset_collection()

        ids = [c["case_id"] for c in cases]
        documents = [f"{c['title']}. {c['summary']}" for c in cases]
        metadatas = [
            {
                "case_id": c["case_id"],
                "title": c["title"],
                "court": c["court"],
                "year": c["year"],
            }
            for c in cases
        ]

        # Chroma's `upsert` embeds `documents` via the collection's embedding_fn
        # automatically -- we never call sentence-transformers directly.
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def index_statutes(self, statutes: List[Dict[str, Any]]) -> None:
        """Embed and upsert statute summaries into the same collection (typed via metadata)."""
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

    def _reset_collection(self) -> None:
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name, embedding_function=self.embedding_fn, metadata={"hnsw:space": "cosine"}
        )

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def semantic_search(self, query: str, top_k: int = 5,
                         entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return the top_k most semantically similar indexed documents to `query`.

        Each result dict: {id, document, metadata, similarity}
        `similarity` is derived from Chroma's cosine distance (1 - distance),
        so higher == more similar, bounded roughly in [0, 1].
        """
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
        """Number of documents currently indexed."""
        return self.collection.count()


if __name__ == "__main__":
    from data_generator import generate_all

    bundle = generate_all(persist=False)
    store = LegalVectorStore()
    store.index_cases(bundle["cases"], reset=True)
    store.index_statutes(bundle["statutes"])

    print(f"Indexed {store.count()} documents.")

    query = "employer liable for injury caused by defective machinery"
    print(f"\nSemantic search for: '{query}'")
    for r in store.semantic_search(query, top_k=5):
        print(f"  [{r['similarity']:.3f}] {r['metadata'].get('title')}")
