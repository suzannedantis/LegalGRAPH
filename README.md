# LegalGraph: Judicial Precedent & Statute Recommendation Engine

A GraphRAG system that recommends legal precedents, court opinions, and
statutes for a free-text research query, and uses an LLM to explain the
legal reasoning path connecting the recommendations.

## Architecture

| Layer | Module | Tech |
|---|---|---|
| Mock data | `data_generator.py` | pure Python / pandas |
| Graph store | `graph_store.py` | NetworkX (see note below) |
| Vector store | `vector_store.py` | ChromaDB + `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Recommender | `recommender.py` | hybrid: item-item CF + vector similarity + graph centrality |
| GraphRAG | `graph_rag.py` | retrieval (vector + graph path) → LLM (Groq / Gemini / offline mock) |
| UI | `app.py` | Streamlit + Pyvis |

### Why NetworkX instead of Kùzu?
The spec allows either. This implementation uses NetworkX because it's pure
Python with no native binary to install/compile, which makes the project
trivially portable across environments. All graph access goes through the
`LegalGraphStore` class in `graph_store.py` — swapping in a real Kùzu (or
Neo4j) backend later only requires re-implementing that one class; nothing
in `recommender.py`, `graph_rag.py`, or `app.py` talks to the graph library
directly.

## Setup

```bash
pip install networkx chromadb sentence-transformers pyvis streamlit groq
# optional, if you prefer Gemini over Groq:
pip install google-genai
```

The first run of `vector_store.py` (or `app.py`) downloads the
`all-MiniLM-L6-v2` model (~90MB) from Hugging Face — this requires internet
access once; it's cached locally afterward.

### LLM API key (optional but recommended)
Get a **free** API key from one of:
- Groq: https://console.groq.com (recommended — fast, generous free tier)
- Google Gemini: https://ai.google.dev

```bash
export GROQ_API_KEY="your-key-here"
# or
export GEMINI_API_KEY="your-key-here"
```

If neither is set, `graph_rag.py` automatically falls back to an offline
`MockLLMClient` so the rest of the app still runs (recommendations, graph
visualization) — only the generated-explanation text is a placeholder.

## Running

Generate mock data (also runs automatically on first `app.py` load):
```bash
python3 data_generator.py
```

Run each module's own demo/smoke-test:
```bash
python3 graph_store.py
python3 vector_store.py
python3 recommender.py
python3 graph_rag.py
```

Launch the full app:
```bash
streamlit run app.py
```

## Pipeline walkthrough

1. **`data_generator.py`** creates 24 synthetic `CaseLaw` records, 8
   `Judge`s, 8 `Statute`s, and 10 `LegalConcept`s, wired together with
   `CITES` / `AFFIRMED_BY` / `OVERRULED_BY` / `HANDLED_BY` edges, plus a
   simulated implicit user-citation-history table (15 synthetic
   "researchers").
2. **`graph_store.py`** loads this into a `networkx.MultiDiGraph` and
   exposes Cypher-flavored traversal helpers: `find_paths`,
   `shortest_legal_path`, `multi_hop_citation_network`,
   `case_full_context`.
3. **`vector_store.py`** embeds every case/statute summary into a
   persistent local ChromaDB collection using `all-MiniLM-L6-v2`, exposing
   `semantic_search(query, top_k)`.
4. **`recommender.py`**'s `HybridRecommender` blends:
   - **Collaborative filtering**: item-item cosine similarity over the
     simulated user-case interaction matrix.
   - **Vector similarity**: semantic closeness of the query to case
     summaries.
   - **Graph centrality**: `networkx.degree_centrality` as an authority
     proxy.
   into one weighted `final_score` per case (weights adjustable live in the
   Streamlit sidebar).
5. **`graph_rag.py`**'s `GraphRAGEngine` takes the top recommendations,
   pulls each one's full graph context (cited statutes/cases, concepts,
   judge, who cites/affirms/overrules it), computes the shortest multi-hop
   citation path between the top two candidates, formats all of this into a
   structured context block, and sends it to the LLM with a system prompt
   instructing it to explain the legal reasoning in plain English.
6. **`app.py`** wires all of the above into a Streamlit UI: search box →
   ranked recommendation cards with score breakdowns → interactive Pyvis
   citation-network graph → GraphRAG-generated rationale with an expandable
   view of the exact multi-hop path and full LLM context.

## Notes on the demo dataset

`data_generator.py` uses `random.seed(42)` for reproducibility. Because
citation edges are randomly generated on a small case pool, not every pair
of cases will have a connecting path — `graph_rag.py` handles this
gracefully and reports "No direct multi-hop citation path was found" rather
than failing.
