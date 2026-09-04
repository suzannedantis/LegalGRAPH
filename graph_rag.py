"""
graph_rag.py
============
The GraphRAG orchestration layer for LegalGraph.

Pipeline (this is the core "GraphRAG" pattern of the project):

    1. RETRIEVE (vector leg)   -- semantic search over case summaries
       (vector_store.LegalVectorStore) to find the cases most relevant to
       the user's natural-language query.
    2. RETRIEVE (graph leg)    -- for the top semantic hits, pull each
       case's immediate graph context (cited statutes, cited/citing cases,
       legal concepts, handling judge) AND compute a multi-hop citation
       path between the top-2 recommended cases via
       graph_store.LegalGraphStore.find_paths / shortest_legal_path. This
       captures *structural* legal reasoning (e.g. "Case A cites Statute X,
       which underlies Case B, which was later affirmed by Case C") that
       pure vector similarity cannot express.
    3. AUGMENT                -- both retrieval results are serialized into
       a structured natural-language context block.
    4. GENERATE                -- the context block + user query are sent to
       an LLM (Groq's free-tier API by default; Gemini optionally) with a
       system prompt instructing it to act as a legal reasoning assistant
       and explain, in plain English, why the retrieved precedents/statutes
       are relevant and how they connect.

The LLM call is provider-abstracted behind `LLMClient` so swapping Groq for
Gemini (or another OpenAI-compatible free endpoint) only requires
implementing one method.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from graph_store import LegalGraphStore
from vector_store import LegalVectorStore


def _load_env_file() -> None:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.isfile(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k not in os.environ:
                        os.environ[k] = v

_load_env_file()


# ========================================================================= #
# LLM client abstraction
# ========================================================================= #
class LLMClient(ABC):
    """Minimal provider-agnostic interface so graph_rag.py doesn't hardcode a vendor SDK."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's plain-text completion for the given prompts."""
        raise NotImplementedError


class GroqLLMClient(LLMClient):
    """
    Uses Groq's free-tier hosted inference API (OpenAI-compatible, very low
    latency). Requires the `GROQ_API_KEY` environment variable.
    Default model: llama-3.1-8b-instant (fast, free-tier friendly).
    """

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        try:
            from groq import Groq
        except ImportError as e:  # pragma: no cover
            raise ImportError("Install the `groq` package: pip install groq") from e

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "No Groq API key found. Set the GROQ_API_KEY environment variable "
                "(free tier available at https://console.groq.com)."
            )
        self.client = Groq(api_key=key)
        self.model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        candidates = [self.model, "openai/gpt-oss-20b", "qwen/qwen3.6-27b", "groq/compound-mini"]
        last_err = None
        for mod in candidates:
            try:
                response = self.client.chat.completions.create(
                    model=mod,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=1800,
                )
                self.model = mod
                return response.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                continue
        raise last_err


class GeminiLLMClient(LLMClient):
    """
    Uses Google's Gemini API (free tier available) via the `google-genai` SDK.
    Requires the `GEMINI_API_KEY` environment variable.
    """

    def __init__(self, model: str = "gemini-1.5-flash", api_key: Optional[str] = None) -> None:
        try:
            from google import genai
        except ImportError as e:  # pragma: no cover
            raise ImportError("Install the `google-genai` package: pip install google-genai") from e

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "No Gemini API key found. Set the GEMINI_API_KEY environment variable "
                "(free tier available at https://ai.google.dev)."
            )
        self.client = genai.Client(api_key=key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=f"{system_prompt}\n\n{user_prompt}",
        )
        return response.text or ""


class MockLLMClient(LLMClient):
    """
    Offline fallback that deterministically summarizes the retrieved context
    without calling any external API. Useful for local dev / demos without
    an API key, and lets app.py degrade gracefully instead of crashing.
    """

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return (
            "[MOCK LLM — no API key configured]\n\n"
            "Based on the retrieved precedents and statutes below, the recommended "
            "case(s) are relevant because they address closely related legal concepts "
            "and share citation/graph relationships with each other. Configure "
            "GROQ_API_KEY or GEMINI_API_KEY to generate a real plain-English legal "
            "rationale.\n\n--- Retrieved context passed to the LLM ---\n" + user_prompt
        )


def get_default_llm_client() -> LLMClient:
    """
    Resolve an LLM client based on available environment variables, preferring
    Groq, then Gemini, then falling back to the offline MockLLMClient so the
    app never hard-crashes for lack of an API key.
    """
    if os.environ.get("GROQ_API_KEY"):
        return GroqLLMClient()
    if os.environ.get("GEMINI_API_KEY"):
        return GeminiLLMClient()
    return MockLLMClient()


# ========================================================================= #
# GraphRAG engine
# ========================================================================= #
SYSTEM_PROMPT = """You are an expert judicial research assistant. Explain case law relationships in clear, professional, and accessible plain English.

You will receive:
1. A legal research question.
2. Retrieved candidate precedents (court cases).
3. Graph context: cited statutes, handling judges, legal concepts, and citation connections between cases.

Format your response using structured Markdown:

### 📌 Summary of Legal Precedents
Provide a clear 2-3 sentence overview answering the researcher's question directly based on the retrieved law.

### 🏛️ Relevant Court Cases & Why They Matter
For each relevant case:
- **Case Name & Year**:
  - **Key Ruling**: What the court decided in 1-2 sentences.
  - **Why It Matters**: How this precedent applies to the user's issue.
  - **Statutory Link**: Mention any cited statutes.

### 🔗 How the Cases Connect
Explain how the retrieved cases connect (direct citation, affirmed on appeal, overruled, or shared statutory principles). If there is no direct citation chain, explain their shared doctrine.

### 💡 Key Takeaway
Provide 1-2 actionable insights or conclusions based on the current state of this precedent.

Guidelines:
- Write in clean, modern Markdown with bold terms and neat bullet points.
- Do NOT use raw HTML tags.
- Be concise, grounded strictly in the provided context, and frame this as legal research assistance."""


class GraphRAGEngine:
    """
    Orchestrates the retrieve (vector + graph) -> augment -> generate pipeline.
    """

    def __init__(self, graph_store: LegalGraphStore, vector_store: LegalVectorStore,
                 llm_client: Optional[LLMClient] = None) -> None:
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.llm_client = llm_client or get_default_llm_client()

    # ------------------------------------------------------------------ #
    # Retrieval helpers
    # ------------------------------------------------------------------ #
    def retrieve_semantic_candidates(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Vector leg: top-k semantically similar cases to the query."""
        return self.vector_store.semantic_search(query, top_k=top_k)

    def retrieve_graph_context(self, case_ids: List[str]) -> Dict[str, Any]:
        """
        Graph leg: for each candidate case, pull its immediate graph context
        (statutes/concepts/judges/citations), plus — if at least two
        candidates were retrieved — the multi-hop citation path connecting
        the top two, which is the key "reasoning path" the LLM explains.
        """
        contexts = {cid: self.graph_store.case_full_context(cid) for cid in case_ids}

        connecting_path = None
        if len(case_ids) >= 2:
            candidate_pairs = []
            for i in range(min(len(case_ids), 4)):
                for j in range(min(len(case_ids), 4)):
                    if i != j:
                        candidate_pairs.append((case_ids[i], case_ids[j]))
            for src, tgt in candidate_pairs:
                p = self.graph_store.shortest_legal_path(src, tgt)
                if p and len(p) > 1:
                    connecting_path = p
                    break

        return {"case_contexts": contexts, "connecting_path": connecting_path}

    # ------------------------------------------------------------------ #
    # Context formatting (augment step)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _format_case_context(case_id: str, ctx: Dict[str, Any]) -> str:
        if not ctx or "case" not in ctx:
            return f"- {case_id}: (no graph data found)"

        case = ctx["case"]
        lines = [f"### {case.get('title')} ({case.get('year')}, {case.get('court')})",
                 f"Summary: {case.get('summary')}"]

        if ctx.get("cited_statutes"):
            statutes = "; ".join(s.get("title", "") for s in ctx["cited_statutes"] if s)
            lines.append(f"Cites statutes: {statutes}")

        if ctx.get("concepts"):
            concepts = ", ".join(c.get("name", "") for c in ctx["concepts"] if c)
            lines.append(f"Invokes legal concepts: {concepts}")

        if ctx.get("judges"):
            judges = ", ".join(j.get("name", "") for j in ctx["judges"] if j)
            lines.append(f"Handled by: {judges}")

        if ctx.get("cited_cases"):
            cited = "; ".join(c.get("title", "") for c in ctx["cited_cases"] if c)
            lines.append(f"Cites prior cases: {cited}")

        if ctx.get("cited_by_cases"):
            citing = "; ".join(c.get("title", "") for c in ctx["cited_by_cases"] if c)
            lines.append(f"Cited by later cases: {citing}")

        if ctx.get("affirmed_by"):
            aff = "; ".join(c.get("title", "") for c in ctx["affirmed_by"] if c)
            lines.append(f"Affirmed by: {aff}")

        if ctx.get("overruled_by"):
            ovr = "; ".join(c.get("title", "") for c in ctx["overruled_by"] if c)
            lines.append(f"Overruled by: {ovr}")

        return "\n".join(lines)

    @staticmethod
    def _format_connecting_path(path: Optional[List[Dict[str, Any]]]) -> str:
        if not path:
            return "No direct multi-hop citation path was found between the top candidates."

        hop_strs = []
        for hop in path:
            node = hop["node"] or {}
            label = node.get("title") or node.get("name") or node.get("case_id", "?")
            rel = hop["rel_type"]
            if rel is None:
                hop_strs.append(label)
            else:
                hop_strs.append(f"--[{rel}]--> {label}")
        return "Path: " + " ".join(hop_strs)

    def build_context_block(self, query: str, semantic_hits: List[Dict[str, Any]],
                             graph_context: Dict[str, Any]) -> str:
        """Assemble the full natural-language context block sent to the LLM."""
        parts = [f"USER QUERY: {query}\n", "RETRIEVED CANDIDATE PRECEDENTS:\n"]

        case_ids = [hit["metadata"]["case_id"] for hit in semantic_hits if "case_id" in hit["metadata"]]
        for cid in case_ids:
            ctx = graph_context["case_contexts"].get(cid, {})
            parts.append(self._format_case_context(cid, ctx))
            parts.append("")

        parts.append("MULTI-HOP CITATION PATH BETWEEN TOP TWO CANDIDATES:")
        parts.append(self._format_connecting_path(graph_context.get("connecting_path")))

        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    # Full pipeline
    # ------------------------------------------------------------------ #
    def explain(self, query: str, case_ids: Optional[List[str]] = None,
                top_k: int = 3) -> Dict[str, Any]:
        """
        Run the full GraphRAG pipeline for a user query and return both the
        LLM's plain-English explanation and the structured context it was
        generated from (for UI display / auditability).

        If `case_ids` is provided (e.g. from recommender.py's ranked list),
        those are used directly instead of re-running semantic search, so
        the explanation is grounded in exactly what was recommended.
        """
        if case_ids:
            semantic_hits = [{"metadata": {"case_id": cid}, "similarity": None} for cid in case_ids[:top_k]]
        else:
            semantic_hits = self.retrieve_semantic_candidates(query, top_k=top_k)

        candidate_ids = [hit["metadata"]["case_id"] for hit in semantic_hits if "case_id" in hit["metadata"]]
        graph_context = self.retrieve_graph_context(candidate_ids)
        context_block = self.build_context_block(query, semantic_hits, graph_context)

        explanation = self.llm_client.generate(SYSTEM_PROMPT, context_block)

        return {
            "query": query,
            "candidate_case_ids": candidate_ids,
            "context_block": context_block,
            "connecting_path": graph_context.get("connecting_path"),
            "explanation": explanation,
        }

    def answer_followup(self, initial_query: str, context_block: str,
                        history: List[Dict[str, str]], question: str) -> str:
        """
        Answer a follow-up user question grounded in the retrieved legal context and chat history.
        """
        prompt = (
            f"INITIAL RESEARCH CONTEXT:\n{context_block}\n\n"
            f"CONVERSATION HISTORY:\n"
        )
        for msg in history[-4:]:
            prompt += f"{msg.get('role', 'user').upper()}: {msg.get('content', '')}\n"
        prompt += f"\nNEW FOLLOW-UP QUESTION FROM RESEARCHER: {question}\n\n"
        prompt += "Please answer clearly, concisely, and grounded in the legal precedents and statutes provided."

        system = (
            "You are a legal research assistant. Answer the researcher's follow-up question "
            "using the established case precedents, statutes, and citations. Be objective, accurate, "
            "and concise. Avoid fabricating citations not found in the context."
        )
        return self.llm_client.generate(system, prompt)

    def explain_case_pair(self, case_a_id: str, case_b_id: str) -> Dict[str, Any]:
        """
        Find and explain the legal relationship path between two arbitrary cases.
        """
        ctx_a = self.graph_store.case_full_context(case_a_id)
        ctx_b = self.graph_store.case_full_context(case_b_id)
        path = self.graph_store.shortest_legal_path(case_a_id, case_b_id)

        case_a = ctx_a.get("case", {})
        case_b = ctx_b.get("case", {})

        path_desc = self._format_connecting_path(path) if path else "No direct path found in the citation graph."

        prompt = (
            f"CASE A: {case_a.get('title')} ({case_a.get('year')}) - {case_a.get('court')}\n"
            f"Summary: {case_a.get('summary')}\n"
            f"Concepts: {', '.join(case_a.get('concepts', []))}\n\n"
            f"CASE B: {case_b.get('title')} ({case_b.get('year')}) - {case_b.get('court')}\n"
            f"Summary: {case_b.get('summary')}\n"
            f"Concepts: {', '.join(case_b.get('concepts', []))}\n\n"
            f"CONNECTING GRAPH PATH: {path_desc}\n\n"
            f"Explain in 2-3 concise paragraphs how these two cases relate conceptually and procedurally, "
            f"and what this means for legal precedent."
        )

        system = (
            "You are a judicial research assistant. Compare these two cases and explain "
            "their legal and precedential relationship in plain English."
        )
        explanation = self.llm_client.generate(system, prompt)
        return {
            "case_a": case_a,
            "case_b": case_b,
            "path": path,
            "path_desc": path_desc,
            "explanation": explanation,
        }


if __name__ == "__main__":
    from data_generator import generate_all

    bundle = generate_all(persist=False)
    gs = LegalGraphStore()
    gs.load_from_bundle(bundle)

    class FakeVectorStore:
        def semantic_search(self, query, top_k=5, entity_type=None):
            cases = bundle["cases"][:top_k]
            return [{"metadata": {"case_id": c["case_id"]}, "similarity": 0.9 - i * 0.05}
                    for i, c in enumerate(cases)]

    engine = GraphRAGEngine(gs, FakeVectorStore(), llm_client=MockLLMClient())
    result = engine.explain("liability for negligence causing personal injury", top_k=3)
    print(result["explanation"][:500])
    print("\n--- connecting path ---")
    print(engine._format_connecting_path(result["connecting_path"]))
