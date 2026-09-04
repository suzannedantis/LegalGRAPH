"""
app.py
======
User-Friendly Streamlit Frontend for LegalGraph: Judicial Precedent & Statute Engine.

Designed for clarity and intuitive use:
    - Guided 3-step visual workflow (Pick topic -> Search -> Read plain-English explanation).
    - 1-Click quick topic buttons that autofill AND search immediately.
    - AI Legal Summary front-and-center in plain English.
    - Simplified tabs with clear, non-technical labels.
    - Technical dials and weights neatly tucked into an "Advanced Settings" drawer.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from data_generator import generate_all
from graph_store import LegalGraphStore
from vector_store import LegalVectorStore
from recommender import HybridRecommender, RecommendationWeights
from graph_rag import GraphRAGEngine, get_default_llm_client, MockLLMClient
from pdf_generator import generate_legal_memo_pdf

# --------------------------------------------------------------------------- #
# Page config & User-Friendly Custom Styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="LegalGraph | Legal Precedent & Statute Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Background */
    .stApp {
        background-color: #0c111d;
        color: #f1f5f9;
    }

    /* Top Hero Header */
    .welcome-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 22px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    .welcome-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f59e0b, #fbbf24, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .welcome-sub {
        color: #94a3b8;
        font-size: 1.0rem;
        margin-top: 6px;
        line-height: 1.5;
    }

    /* 3-Step Guide Banner */
    .steps-row {
        display: flex;
        gap: 12px;
        margin-top: 18px;
        flex-wrap: wrap;
    }
    .step-pill {
        flex: 1;
        min-width: 200px;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 10px 14px;
        font-size: 0.85rem;
        color: #cbd5e1;
    }
    .step-pill strong {
        color: #f59e0b;
        display: block;
        font-size: 0.9rem;
        margin-bottom: 2px;
    }

    /* AI Answer Box */
    .ai-summary-box {
        background: #111a2e;
        border: 1px solid #1e293b;
        border-left: 5px solid #f59e0b;
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 20px;
        line-height: 1.7;
        font-size: 0.98rem;
    }
    .ai-summary-box h4 {
        margin-top: 0;
        color: #fbbf24;
        font-size: 1.15rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Case Cards */
    .case-card {
        background: #111a2e;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .case-card-title {
        font-size: 1.18rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .case-card-sub {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 10px;
    }

    /* Match Badge */
    .match-pill {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-block;
    }

    /* Tag Chips */
    .chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 4px;
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.25);
    }

    /* Stepper */
    .stepper-chain {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        background: #0f172a;
        padding: 14px 18px;
        border-radius: 12px;
        border: 1px solid #1e293b;
        margin: 14px 0 20px 0;
    }
    .chain-node {
        background: #1e293b;
        border: 1px solid #38bdf8;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 0.88rem;
        font-weight: 600;
        color: #f8fafc;
    }
    .chain-link {
        color: #f59e0b;
        font-size: 0.8rem;
        font-weight: 700;
        background: rgba(245, 158, 11, 0.12);
        padding: 4px 8px;
        border-radius: 6px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

NODE_COLORS = {
    "CaseLaw": "#38bdf8",      # Cyan
    "Statute": "#f43f5e",      # Rose
    "Judge": "#f59e0b",        # Amber
    "LegalConcept": "#10b981",  # Emerald
}
EDGE_COLORS = {
    "CITES": "#64748b",
    "AFFIRMED_BY": "#38bdf8",
    "OVERRULED_BY": "#ef4444",
    "HANDLED_BY": "#f59e0b",
    "INVOKES": "#10b981",
}


# --------------------------------------------------------------------------- #
# Cached resource initialization
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading legal database...")
def load_data_bundle() -> Dict[str, Any]:
    return generate_all(persist=True)


@st.cache_resource(show_spinner="Loading citation network...")
def build_graph_store(_bundle_marker: int) -> LegalGraphStore:
    bundle = load_data_bundle()
    store = LegalGraphStore()
    store.load_from_bundle(bundle)
    return store


@st.cache_resource(show_spinner="Preparing search engine...")
def build_vector_store(_bundle_marker: int, backend: str = "lightweight") -> LegalVectorStore:
    bundle = load_data_bundle()
    store = LegalVectorStore(backend=backend)
    store.index_cases(bundle["cases"], reset=False)
    store.index_statutes(bundle["statutes"])
    import gc
    gc.collect()
    return store


@st.cache_resource
def get_llm():
    return get_default_llm_client()


# --------------------------------------------------------------------------- #
# Pyvis visual graph generator
# --------------------------------------------------------------------------- #
def render_citation_network(
    graph_store: LegalGraphStore,
    seed_case_ids: List[str],
    max_hops: int = 2,
    allowed_node_types: Optional[Set[str]] = None,
    allowed_edge_types: Optional[Set[str]] = None,
) -> str:
    net = Network(height="540px", width="100%", directed=True, bgcolor="#0c111d", font_color="#e2e8f0")
    net.barnes_hut(gravity=-2600, central_gravity=0.25, spring_length=130, damping=0.9)

    seen_nodes: Set[str] = set()
    seen_edges: Set[tuple] = set()

    for case_id in seed_case_ids:
        subgraph = graph_store.multi_hop_citation_network(case_id, max_hops=max_hops)

        for node in subgraph["nodes"]:
            if not node:
                continue
            node_id = node.get("case_id") or node.get("statute_id") or node.get("judge_id") or node.get("concept_id")
            if not node_id or node_id in seen_nodes:
                continue

            node_type = node.get("type", "CaseLaw")
            if allowed_node_types and node_type not in allowed_node_types:
                continue

            seen_nodes.add(node_id)
            raw_title = node.get("title") or node.get("name") or node_id
            label = (raw_title[:30] + "...") if len(raw_title) > 30 else raw_title
            is_seed = node_id in seed_case_ids

            net.add_node(
                node_id,
                label=label,
                title=f"[{node_type}] {raw_title}",
                color=NODE_COLORS.get(node_type, "#94a3b8"),
                size=24 if is_seed else 14,
                borderWidth=3 if is_seed else 1,
                font={"color": "#f8fafc", "size": 12 if is_seed else 10, "face": "Plus Jakarta Sans"},
            )

        for edge in subgraph["edges"]:
            rel = edge["rel_type"]
            if allowed_edge_types and rel not in allowed_edge_types:
                continue
            src, tgt = edge["source"], edge["target"]
            edge_key = (src, tgt, rel)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            if src in seen_nodes and tgt in seen_nodes:
                net.add_edge(
                    src,
                    tgt,
                    label=rel,
                    color=EDGE_COLORS.get(rel, "#64748b"),
                    arrows="to",
                    font={"color": "#94a3b8", "size": 9, "align": "middle"},
                )

    tmp_dir = tempfile.gettempdir()
    out_path = os.path.join(tmp_dir, "legalgraph_rendered.html")
    net.write_html(out_path, open_browser=False, notebook=False)
    return out_path


def render_precedent_stepper(connecting_path: Optional[List[Dict[str, Any]]]) -> None:
    """Render a clean, informative step-by-step precedent chain connecting court cases."""
    if not connecting_path or len(connecting_path) <= 1:
        st.info(
            "ℹ️ **No direct citation chain between these specific cases.**\n\n"
            "In court records, these cases do not directly cite one another, but they are connected "
            "conceptually through shared statutory provisions and common legal doctrines."
        )
        return

    st.markdown("##### 🔗 How These Cases Connect in Court (Precedent Chain):")

    # Visual Flow Badges
    stepper_html = """
    <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 10px; background: #0f172a; padding: 18px 20px; border-radius: 12px; border: 1px solid #1e293b; margin-bottom: 16px;">
    """

    for i, hop in enumerate(connecting_path):
        node = hop.get("node", {}) if isinstance(hop, dict) and "node" in hop else hop
        rel = hop.get("rel_type") if isinstance(hop, dict) else None

        # Add connection badge if there's an incoming edge
        if rel:
            rel_upper = str(rel).upper()
            if "AFFIRM" in rel_upper:
                rel_badge = "<span style='background:rgba(56,189,248,0.18); color:#38bdf8; border:1px solid rgba(56,189,248,0.35); padding:5px 10px; border-radius:6px; font-weight:700; font-size:0.8rem;'>➔ 🟢 Affirmed On Appeal By ➔</span>"
            elif "OVERRULE" in rel_upper:
                rel_badge = "<span style='background:rgba(239,68,68,0.18); color:#ef4444; border:1px solid rgba(239,68,68,0.35); padding:5px 10px; border-radius:6px; font-weight:700; font-size:0.8rem;'>➔ 🔴 Later Overruled By ➔</span>"
            elif "CITE" in rel_upper:
                rel_badge = "<span style='background:rgba(245,158,11,0.18); color:#f59e0b; border:1px solid rgba(245,158,11,0.35); padding:5px 10px; border-radius:6px; font-weight:700; font-size:0.8rem;'>➔ 📜 Cites As Precedent ➔</span>"
            else:
                label = str(rel).replace("_", " ").title()
                rel_badge = f"<span style='background:rgba(148,163,184,0.18); color:#cbd5e1; border:1px solid rgba(148,163,184,0.35); padding:5px 10px; border-radius:6px; font-weight:700; font-size:0.8rem;'>➔ {label} ➔</span>"
            stepper_html += rel_badge

        # Node Badge
        title = node.get("title") or node.get("name") or node.get("code_section") or f"Case {i+1}"
        year = node.get("year")
        meta = f" <span style='color:#94a3b8; font-weight:400;'>({year})</span>" if year else ""
        stepper_html += f"<div style='background:#1e293b; border:1px solid #38bdf8; border-radius:8px; padding:8px 14px; color:#f8fafc; font-weight:600; font-size:0.9rem;'>🏛️ {title}{meta}</div>"

    stepper_html += "</div>"
    st.markdown(stepper_html, unsafe_allow_html=True)

    # Narrative Breakdown
    with st.expander("📝 Read Step-by-Step Chain Explanation", expanded=True):
        for i in range(len(connecting_path) - 1):
            curr_hop = connecting_path[i]
            next_hop = connecting_path[i + 1]
            c_node = curr_hop.get("node", {}) if isinstance(curr_hop, dict) and "node" in curr_hop else curr_hop
            n_node = next_hop.get("node", {}) if isinstance(next_hop, dict) and "node" in next_hop else next_hop
            rel = next_hop.get("rel_type", "CITES")

            c_title = c_node.get("title", "Case")
            c_year = f" ({c_node.get('year')})" if c_node.get("year") else ""
            c_court = f" *[{c_node.get('court')}]*" if c_node.get("court") else ""

            n_title = n_node.get("title", "Case")
            n_year = f" ({n_node.get('year')})" if n_node.get("year") else ""
            n_court = f" *[{n_node.get('court')}]*" if n_node.get("court") else ""

            rel_upper = str(rel).upper()
            if "AFFIRM" in rel_upper:
                st.markdown(f"**Step {i+1}:** 🏛️ **{c_title}**{c_year}{c_court} was **affirmed on appeal** by 🏛️ **{n_title}**{n_year}{n_court}.")
            elif "OVERRULE" in rel_upper:
                st.markdown(f"**Step {i+1}:** 🏛️ **{c_title}**{c_year}{c_court} was **later overruled** by 🏛️ **{n_title}**{n_year}{n_court}.")
            elif "CITE" in rel_upper:
                st.markdown(f"**Step {i+1}:** 🏛️ **{c_title}**{c_year}{c_court} is cited as **binding legal precedent** by 🏛️ **{n_title}**{n_year}{n_court}.")
            else:
                label_text = str(rel).replace('_', ' ').lower()
                st.markdown(f"**Step {i+1}:** 🏛️ **{c_title}**{c_year} links via **{label_text}** to 🏛️ **{n_title}**{n_year}.")


# --------------------------------------------------------------------------- #
# Simple Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(bundle: Dict[str, Any]) -> tuple[RecommendationWeights, Optional[str], str]:
    with st.sidebar:
        st.markdown("### ⚖️ **LegalGraph**")
        st.caption("Judicial Precedent & Statute Engine")

        # Telemetry
        llm_client = get_llm()
        if isinstance(llm_client, MockLLMClient):
            st.warning("⚠️ AI Mode: Offline (Mock LLM)")
        else:
            st.success("🟢 AI Assistant: **Groq Connected**")

        st.info("⚡ Search: **Cloud-Optimized (Low-RAM)**")

        st.markdown(
            """
            **How it works:**
            LegalGraph searches through judicial court decisions, links them with relevant statutes,
            and uses AI to explain how the precedents apply to your situation.
            """
        )

        st.markdown("---")

        # Advanced Settings (Collapsed by default so users aren't overwhelmed)
        with st.expander("⚙️ Advanced Tuning (Optional)"):
            engine_mode = st.selectbox(
                "Memory Profile / Engine",
                ["Lightweight (Low-RAM: ~20MB)", "ChromaDB + PyTorch (~700MB)"],
                index=0,
                help="Lightweight uses <20MB RAM, safe for Streamlit Cloud. ChromaDB uses ~700MB.",
            )
            backend = "lightweight" if "Lightweight" in engine_mode else "chroma"

            st.caption("Adjust how results are prioritized:")
            vec_w = st.slider("Text & Meaning Similarity", 0.0, 1.0, 0.50, 0.05,
                              help="How closely the case text matches your words")
            gc_w = st.slider("Court Precedent Authority", 0.0, 1.0, 0.25, 0.05,
                             help="Prioritize landmark cases heavily cited by other courts")
            cf_w = st.slider("Researcher Behavior Patterns", 0.0, 1.0, 0.25, 0.05,
                             help="Collaborative filtering based on past research queries")

            total = cf_w + vec_w + gc_w
            if total > 0:
                cf_w, vec_w, gc_w = cf_w / total, vec_w / total, gc_w / total

            user_ids = ["(no personalization)"] + sorted(bundle["citation_history"]["user_id"].unique().tolist())
            selected_user = st.selectbox(
                "Simulate Researcher Profile",
                user_ids,
                help="Tailors recommendations based on a simulated lawyer's citation habits",
            )
            user_choice = None if selected_user == "(no personalization)" else selected_user

        weights = RecommendationWeights(collaborative=cf_w, vector=vec_w, graph_centrality=gc_w)
        return weights, user_choice, backend


# --------------------------------------------------------------------------- #
# Main Application
# --------------------------------------------------------------------------- #
def main() -> None:
    bundle = load_data_bundle()
    weights, selected_user, backend = render_sidebar(bundle)

    graph_store = build_graph_store(len(bundle["cases"]))
    vector_store = build_vector_store(len(bundle["cases"]), backend=backend)
    llm_client = get_llm()
    rag_engine = GraphRAGEngine(graph_store, vector_store, llm_client=llm_client)

    # Session State
    if "current_query" not in st.session_state:
        st.session_state.current_query = "liability for negligence causing personal injury by robotic systems"
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "rag_result" not in st.session_state:
        st.session_state.rag_result = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "auto_run_search" not in st.session_state:
        st.session_state.auto_run_search = False

    # 1. Clean, Friendly Welcome & 3-Step Guide
    st.markdown(
        """
        <div class="welcome-card">
            <h1 class="welcome-title">LegalGraph · Legal Research Assistant</h1>
            <p class="welcome-sub">
                Instant legal precedent discovery and plain-English AI explanations grounded in real statutes and case law.
            </p>
            <div class="steps-row">
                <div class="step-pill">
                    <strong>1. Pick or Type a Topic</strong>
                    Click an example below or type your legal question.
                </div>
                <div class="step-pill">
                    <strong>2. Instant Graph Search</strong>
                    Finds relevant court cases, judges, and laws.
                </div>
                <div class="step-pill">
                    <strong>3. Clear Legal Explanation</strong>
                    Read the AI summary of how the precedents connect.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Quick Example Topic Buttons (1-Click Search)
    st.markdown("##### 💡 Try an Example Topic (1-Click):")
    p1, p2, p3, p4 = st.columns(4)

    trigger_search = False

    if p1.button("🤖 Defective Robots & AI", use_container_width=True, help="Manufacturer & employer liability for injuries"):
        st.session_state.current_query = "employer and manufacturer liability for defective robotic equipment causing personal injury"
        trigger_search = True
    if p2.button("💼 Corporate Director Misconduct", use_container_width=True, help="Breach of fiduciary duties during mergers"):
        st.session_state.current_query = "breach of fiduciary duty by corporate directors during merger and insider transactions"
        trigger_search = True
    if p3.button("🔒 Consumer Privacy & Data Breach", use_container_width=True, help="Unauthorized collection of biometric data"):
        st.session_state.current_query = "unauthorized collection and breach of personal consumer biometric data under privacy statutes"
        trigger_search = True
    if p4.button("🏥 Hospital & Medical Negligence", use_container_width=True, help="Failure of healthcare staff standard of care"):
        st.session_state.current_query = "hospital negligence and failure of medical staff to adhere to statutory standard of care"
        trigger_search = True

    # 3. Simple Search Input
    c_input, c_btn = st.columns([4, 1.2])
    with c_input:
        query = st.text_input(
            "Legal Question",
            value=st.session_state.current_query,
            placeholder="Type your legal research question in plain English...",
            label_visibility="collapsed",
        )
    with c_btn:
        search_clicked = st.button("🔍 Search Precedents", type="primary", use_container_width=True)

    # Run search if button clicked or if a topic was clicked or if first visit
    if search_clicked or trigger_search or (st.session_state.search_results is None and query.strip()):
        st.session_state.current_query = query
        seed_ids: List[str] = []
        if selected_user:
            hist = bundle["citation_history"]
            seed_ids = hist[hist["user_id"] == selected_user]["case_id"].tolist()

        recommender = HybridRecommender(graph_store, vector_store, bundle["citation_history"], weights=weights)
        with st.spinner("Finding relevant cases and analyzing precedent connections..."):
            recs = recommender.recommend(query, top_k=4, seed_case_ids=seed_ids)
            st.session_state.search_results = recs

            top_ids = [r.case_id for r in recs]
            rag_out = rag_engine.explain(query, case_ids=top_ids, top_k=min(3, len(top_ids)))
            st.session_state.rag_result = rag_out

    # 4. Results Navigation Tabs
    tab_results, tab_graph, tab_pair, tab_library = st.tabs([
        "📋 AI Legal Summary & Cases",
        "🕸️ Visual Connection Map",
        "🔗 Compare Any Two Cases",
        "📖 Law Library & Statutes",
    ])

    # ========================================================================= #
    # TAB 1: AI Legal Summary & Cases
    # ========================================================================= #
    with tab_results:
        recs = st.session_state.get("search_results")
        rag_data = st.session_state.get("rag_result")

        if not recs or not rag_data:
            st.info("👆 Type a question or click one of the example topics above to see results.")
        else:
            # 1. Plain-English AI Explanation (Top of page)
            st.markdown(
                """
                <div style="background: linear-gradient(90deg, rgba(245, 158, 11, 0.12) 0%, rgba(30, 41, 59, 0.5) 100%); border: 1px solid rgba(245, 158, 11, 0.35); border-left: 5px solid #f59e0b; border-radius: 12px; padding: 16px 20px; margin-bottom: 16px;">
                    <h3 style="margin: 0; color: #fbbf24; font-size: 1.25rem; font-weight: 800; display: flex; align-items: center; gap: 8px;">
                        <span>⚖️</span> AI Legal Analysis & Synthesis
                    </h3>
                    <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.85rem;">
                        Synthesized from judicial opinions, statutory grounds, and citation precedents
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(rag_data.get("explanation", ""))
            st.markdown("<hr style='border-color: #1e293b; margin: 20px 0;'>", unsafe_allow_html=True)

            # 2. Visual Precedent Chain (How cases connect)
            connecting_path = rag_data.get("connecting_path")
            render_precedent_stepper(connecting_path)
            st.markdown("<hr style='border-color: #1e293b; margin: 20px 0;'>", unsafe_allow_html=True)

            # 3. Recommended Cases Cards
            st.markdown("##### 🏛️ Relevant Court Decisions:")

            for rank, rec in enumerate(recs, start=1):
                ctx = graph_store.case_full_context(rec.case_id)
                case_info = ctx.get("case", {})
                judge = ctx.get("judge", {})
                concepts = case_info.get("concepts", [])
                cited_statutes = ctx.get("cited_statutes", [])
                cited_cases = ctx.get("cited_cases", [])
                affirming_cases = ctx.get("affirmed_by", [])
                overruling_cases = ctx.get("overruled_by", [])

                with st.container():
                    col_info, col_match = st.columns([4.2, 1.1])
                    with col_info:
                        st.markdown(
                            f"""
                            <div class="case-card-title">{rank}. {rec.title}</div>
                            <div class="case-card-sub">
                                <b>{rec.court}</b> · Decided: <b>{rec.year}</b> · Presiding Judge: <b>{judge.get('name', 'N/A')}</b>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.write(rec.summary)

                        # Concepts
                        chips = "".join([f"<span class='chip'>{c}</span>" for c in concepts])
                        st.markdown(chips, unsafe_allow_html=True)

                    with col_match:
                        pct = int(rec.final_score * 100)
                        st.markdown(
                            f"""
                            <div style="text-align: right; padding-top: 6px;">
                                <div class="match-pill">{pct}% Match</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Simple Background Accordion
                    with st.expander(f"➕ See laws cited & case history ({rec.case_id})"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**📜 Statutes Grounding this Case:**")
                            if cited_statutes:
                                for s in cited_statutes:
                                    st.markdown(f"- **{s.get('code_section')}**: {s.get('title')}")
                            else:
                                st.caption("No specific statute cited.")
                        with c2:
                            st.markdown("**⚖️ Subsequent Court Rulings:**")
                            if affirming_cases:
                                for ac in affirming_cases:
                                    st.markdown(f"- 🟢 **Affirmed on appeal by:** *{ac.get('title')}*")
                            if overruling_cases:
                                for oc in overruling_cases:
                                    st.markdown(f"- 🔴 **Later overruled by:** *{oc.get('title')}*")
                            if not affirming_cases and not overruling_cases:
                                st.caption("Active standing precedent (not overturned).")

                    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

            # 4. Friendly Follow-Up Chat Box
            st.markdown("---")
            st.markdown("##### 💬 Have a follow-up question?")
            st.caption("Ask anything about these cases, how statutes apply, or legal defenses.")

            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            user_q = st.chat_input("e.g., Does this liability rule apply to third-party suppliers?")
            if user_q:
                st.session_state.chat_history.append({"role": "user", "content": user_q})
                with st.chat_message("user"):
                    st.write(user_q)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing with Groq..."):
                        ans = rag_engine.answer_followup(
                            initial_query=rag_data.get("query"),
                            context_block=rag_data.get("context_block"),
                            history=st.session_state.chat_history,
                            question=user_q,
                        )
                        st.write(ans)
                        st.session_state.chat_history.append({"role": "assistant", "content": ans})

            # Download Memo Button
            memo_markdown = f"""# Legal Research Memorandum
Generated by LegalGraph Assistant
Query: {rag_data.get('query')}

## Legal Analysis
{rag_data.get('explanation')}

## Precedent Citation Path
{rag_engine._format_connecting_path(connecting_path)}

## Recommended Cases
"""
            for r in recs:
                memo_markdown += f"- {r.title} ({r.year}) - {r.court}\n  {r.summary}\n"

            # Export Legal Memorandum (PDF & Markdown)
            st.markdown("---")
            st.markdown("##### 📥 Export Official Research Memorandum")
            col_pdf, col_md = st.columns([1.6, 1.2])

            with col_pdf:
                with st.spinner("Generating formatted PDF memo..."):
                    pdf_bytes = generate_legal_memo_pdf(
                        query=rag_data.get("query", ""),
                        ai_explanation=rag_data.get("explanation", ""),
                        connecting_path=connecting_path,
                        recommendations=recs,
                        graph_store=graph_store,
                    )
                st.download_button(
                    label="📄 Download Official Memorandum (PDF)",
                    data=pdf_bytes,
                    file_name="LegalGraph_Research_Memorandum.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                )

            with col_md:
                st.download_button(
                    label="📝 Download Markdown (.md)",
                    data=memo_markdown,
                    file_name="LegalGraph_Research_Memorandum.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

    # ========================================================================= #
    # TAB 2: Visual Connection Map
    # ========================================================================= #
    with tab_graph:
        st.markdown("#### 🕸️ Visual Connection Map")
        st.caption("See how court cases, statutes, and judges are linked together in the legal network.")

        recs = st.session_state.get("search_results")
        if not recs:
            st.info("Run a search first to view the connection map.")
        else:
            seed_case_ids = [r.case_id for r in recs]

            # Simple Controls
            g1, g2 = st.columns([2, 3])
            with g1:
                depth = st.radio("Map Scope", ["Direct Connections (1 hop)", "Extended Network (2 hops)"], index=1, horizontal=True)
                hop_count = 1 if "1" in depth else 2
            with g2:
                st.markdown(
                    """
                    <div style="display: flex; gap: 10px; padding-top: 10px; flex-wrap: wrap;">
                        <span class="chip" style="background:#38bdf8; color:#0f172a;">● Court Cases</span>
                        <span class="chip" style="background:#f43f5e; color:#fff;">● Statutes</span>
                        <span class="chip" style="background:#f59e0b; color:#0f172a;">● Judges</span>
                        <span class="chip" style="background:#10b981; color:#0f172a;">● Concepts</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            html_file = render_citation_network(
                graph_store,
                seed_case_ids[:3],
                max_hops=hop_count,
            )
            with open(html_file, "r", encoding="utf-8") as f:
                html_data = f.read()
            components.html(html_data, height=560, scrolling=False)
            st.caption("💡 Tip: Click and drag any circle to rearrange the graph. Scroll with your mouse wheel to zoom in and out.")

    # ========================================================================= #
    # TAB 3: Compare Any Two Cases
    # ========================================================================= #
    with tab_pair:
        st.markdown("#### 🔗 Compare Any Two Cases")
        st.caption("Pick any two court cases from the library to see if and how they relate through past citations.")

        all_cases = sorted(bundle["cases"], key=lambda c: c["title"])
        case_options = {f"{c['title']} ({c['year']})": c["case_id"] for c in all_cases}

        c_a, c_b = st.columns(2)
        with c_a:
            case_a_label = st.selectbox("First Case", list(case_options.keys()), index=0)
            id_a = case_options[case_a_label]
        with c_b:
            case_b_label = st.selectbox("Second Case", list(case_options.keys()), index=min(1, len(case_options) - 1))
            id_b = case_options[case_b_label]

        if st.button("Compare These Two Cases", type="primary"):
            if id_a == id_b:
                st.warning("Please choose two different cases to compare.")
            else:
                with st.spinner("Finding relationship chain and generating legal comparison..."):
                    pair_res = rag_engine.explain_case_pair(id_a, id_b)

                    st.markdown(
                        """
                        <div style="background: linear-gradient(90deg, rgba(56, 189, 248, 0.12) 0%, rgba(30, 41, 59, 0.5) 100%); border: 1px solid rgba(56, 189, 248, 0.35); border-left: 5px solid #38bdf8; border-radius: 12px; padding: 16px 20px; margin: 16px 0 14px 0;">
                            <h3 style="margin: 0; color: #38bdf8; font-size: 1.25rem; font-weight: 800; display: flex; align-items: center; gap: 8px;">
                                <span>⚖️</span> Comparative Legal Analysis
                            </h3>
                            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.85rem;">
                                Doctrinal and precedential relationship between the selected court decisions
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(pair_res.get("explanation", ""))
                    render_precedent_stepper(pair_res.get("path"))

    # ========================================================================= #
    # TAB 4: Law Library & Statutes
    # ========================================================================= #
    with tab_library:
        st.markdown("#### 📖 Legal Database & Case Library")
        st.caption("Search and browse all court cases and statutes in the system.")

        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Court Cases", "Codified Statutes", "Presiding Judges"])

        with sub_tab1:
            search_case = st.text_input("Search cases by title or keyword", "", key="case_search_box")
            cases_df = pd.DataFrame(bundle["cases"])
            if search_case:
                cases_df = cases_df[cases_df["title"].str.contains(search_case, case=False) |
                                    cases_df["summary"].str.contains(search_case, case=False)]
            st.dataframe(
                cases_df[["case_id", "title", "court", "year", "summary"]],
                use_container_width=True,
                height=380,
            )

        with sub_tab2:
            statutes_df = pd.DataFrame(bundle["statutes"])
            st.dataframe(
                statutes_df[["statute_id", "code_section", "title", "summary"]],
                use_container_width=True,
                height=350,
            )

        with sub_tab3:
            judges_df = pd.DataFrame(bundle["judges"])
            st.dataframe(
                judges_df[["judge_id", "name", "court"]],
                use_container_width=True,
                height=350,
            )


if __name__ == "__main__":
    main()
