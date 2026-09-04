"""
graph_store.py
===============
Graph persistence + traversal layer for LegalGraph.

NOTE ON DB CHOICE
------------------
The spec allows Kùzu DB *or* NetworkX. This implementation uses NetworkX
(via a thin `LegalGraphStore` wrapper) because:
  1. It is pure-Python / zero native-binary install friction (Kùzu ships
     platform-specific compiled bindings that can be brittle in constrained
     or sandboxed environments).
  2. It is trivially embeddable and serializable (pickle / GraphML), which
     suits a demo/portfolio project.
  3. The wrapper below exposes a small, Cypher-flavoured query surface
     (`find_paths`, `neighbors`, `shortest_legal_path`, ...), so swapping the
     backend for a real Kùzu instance later only requires re-implementing
     this class -- none of recommender.py / graph_rag.py / app.py need to
     change, since they only talk to `LegalGraphStore`'s public methods.

GRAPH SCHEMA
------------
Nodes (each tagged with a `type` attribute so we can filter/traverse by
label, mimicking a labeled property graph):
    CaseLaw       -> case_id, title, court, year, summary
    Judge         -> judge_id, name, court
    Statute       -> statute_id, title, code_section, summary
    LegalConcept  -> concept_id, name, description

Edges (directed, each tagged with a `rel_type` attribute):
    CITES          : CaseLaw -> CaseLaw | CaseLaw -> Statute
    AFFIRMED_BY    : CaseLaw -> CaseLaw   (lower court case affirmed on appeal)
    OVERRULED_BY   : CaseLaw -> CaseLaw   (precedent later overturned)
    HANDLED_BY     : CaseLaw -> Judge
    INVOKES        : CaseLaw -> LegalConcept  (derived from case.concepts)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class LegalGraphStore:
    """
    Wraps a `networkx.MultiDiGraph` and exposes graph-construction and
    Cypher-like traversal helper methods used by the recommender and
    GraphRAG modules.
    """

    def __init__(self) -> None:
        # MultiDiGraph: directed, allows multiple edge types between the
        # same pair of nodes (e.g. a case could in theory both CITE and
        # later OVERRULED_BY the same other case).
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def load_from_bundle(self, bundle: Dict[str, Any]) -> None:
        """
        Populate the graph from an in-memory data bundle as produced by
        `data_generator.generate_all()`.
        """
        for j in bundle["judges"]:
            self.graph.add_node(j["judge_id"], type="Judge", **j)

        for s in bundle["statutes"]:
            self.graph.add_node(s["statute_id"], type="Statute", **s)

        for c in bundle["concepts"]:
            self.graph.add_node(c["concept_id"], type="LegalConcept", **c)

        for case in bundle["cases"]:
            self.graph.add_node(case["case_id"], type="CaseLaw", **case)
            # Derive INVOKES edges from the case's embedded concept list
            for concept_id in case.get("concepts", []):
                if self.graph.has_node(concept_id):
                    self.graph.add_edge(case["case_id"], concept_id, rel_type="INVOKES")

        for rel_type, edges in bundle["relationships"].items():
            for e in edges:
                if self.graph.has_node(e["source"]) and self.graph.has_node(e["target"]):
                    self.graph.add_edge(e["source"], e["target"], rel_type=rel_type)

    def load_from_disk(self, data_dir: str = DATA_DIR) -> None:
        """Load previously persisted JSON files (from data_generator.py) and build the graph."""
        def _read(name: str) -> Any:
            path = os.path.join(data_dir, f"{name}.json")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Expected {path} — run data_generator.py first to produce mock data."
                )
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        bundle = {
            "judges": _read("judges"),
            "statutes": _read("statutes"),
            "concepts": _read("concepts"),
            "cases": _read("cases"),
            "relationships": _read("relationships"),
        }
        self.load_from_bundle(bundle)

    # ------------------------------------------------------------------ #
    # Basic lookups
    # ------------------------------------------------------------------ #
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Return a node's attribute dict, or None if it does not exist."""
        if node_id in self.graph.nodes:
            return dict(self.graph.nodes[node_id])
        return None

    def nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        """Return all node attribute dicts matching a given `type` label."""
        return [dict(attrs) for _, attrs in self.graph.nodes(data=True) if attrs.get("type") == node_type]

    def neighbors(self, node_id: str, rel_type: Optional[str] = None,
                  direction: str = "out") -> List[Tuple[str, str]]:
        """
        Return (neighbor_id, rel_type) tuples adjacent to `node_id`.
        `direction`: "out" (outgoing edges), "in" (incoming), or "both".
        Optionally filter to a single `rel_type` (e.g. "CITES").
        """
        results: List[Tuple[str, str]] = []
        if direction in ("out", "both"):
            for _, target, data in self.graph.out_edges(node_id, data=True):
                if rel_type is None or data.get("rel_type") == rel_type:
                    results.append((target, data.get("rel_type", "")))
        if direction in ("in", "both"):
            for source, _, data in self.graph.in_edges(node_id, data=True):
                if rel_type is None or data.get("rel_type") == rel_type:
                    results.append((source, data.get("rel_type", "")))
        return results

    def degree_centrality(self) -> Dict[str, float]:
        """
        Compute degree centrality across the whole graph. Used by the
        recommender as a proxy for a precedent's overall "authority" /
        how frequently it sits on citation paths.
        """
        return nx.degree_centrality(self.graph)

    # ------------------------------------------------------------------ #
    # Cypher-like multi-hop traversal
    # ------------------------------------------------------------------ #
    def find_paths(self, source_id: str, target_id: str, max_hops: int = 4) -> List[List[Dict[str, Any]]]:
        """
        Find all simple directed paths between `source_id` and `target_id`
        up to `max_hops` edges, mirroring a Cypher query such as:

            MATCH p = (a:CaseLaw {case_id: $source})-[*1..4]->(b {id: $target})
            RETURN p

        Returns a list of paths; each path is a list of "hop" dicts:
            {"node": <node_attrs>, "rel_type": <edge label into this node>}
        The first hop's rel_type is None (it's the path's starting node).
        """
        if source_id not in self.graph or target_id not in self.graph:
            return []

        paths: List[List[Dict[str, Any]]] = []
        try:
            for node_path in nx.all_simple_paths(self.graph, source_id, target_id, cutoff=max_hops):
                hops: List[Dict[str, Any]] = [{"node": self.get_node(node_path[0]), "rel_type": None}]
                for i in range(len(node_path) - 1):
                    edge_data = self.graph.get_edge_data(node_path[i], node_path[i + 1])
                    # MultiDiGraph edge_data is keyed by edge-key; take the first matching rel_type
                    rel_type = next(iter(edge_data.values()))["rel_type"] if edge_data else None
                    hops.append({"node": self.get_node(node_path[i + 1]), "rel_type": rel_type})
                paths.append(hops)
        except nx.NodeNotFound:
            return []
        return paths

    def shortest_legal_path(self, source_id: str, target_id: str) -> Optional[List[Dict[str, Any]]]:
        """Convenience wrapper returning the single shortest path (fewest hops), or None."""
        paths = self.find_paths(source_id, target_id, max_hops=10)
        if not paths:
            return None
        return min(paths, key=len)

    def multi_hop_citation_network(self, case_id: str, max_hops: int = 2) -> Dict[str, Any]:
        """
        Explore outward from a case up to `max_hops` following CITES /
        AFFIRMED_BY / OVERRULED_BY edges (ignoring HANDLED_BY / INVOKES so
        the subgraph stays focused on precedent relationships), mirroring:

            MATCH (c:CaseLaw {case_id: $id})-[:CITES|AFFIRMED_BY|OVERRULED_BY*1..2]->(related)
            RETURN related

        Returns {"nodes": [...], "edges": [...]} suitable for pyvis rendering.
        """
        if case_id not in self.graph:
            return {"nodes": [], "edges": []}

        precedent_rels = {"CITES", "AFFIRMED_BY", "OVERRULED_BY"}
        visited_nodes = {case_id}
        edges_out: List[Dict[str, str]] = []
        frontier = [case_id]

        for _ in range(max_hops):
            next_frontier = []
            for node in frontier:
                for target, rel in self.neighbors(node, direction="both"):
                    if rel not in precedent_rels:
                        continue
                    edges_out.append({"source": node, "target": target, "rel_type": rel})
                    if target not in visited_nodes:
                        visited_nodes.add(target)
                        next_frontier.append(target)
            frontier = next_frontier

        nodes_out = [self.get_node(n) for n in visited_nodes]
        return {"nodes": nodes_out, "edges": edges_out}

    def case_full_context(self, case_id: str) -> Dict[str, Any]:
        """
        Gather a single case's full immediate graph context: statutes it
        cites, concepts it invokes, the judge who handled it, and any
        cases it cites or is cited by. Used as structured context for the
        GraphRAG prompt.
        """
        node = self.get_node(case_id)
        if node is None:
            return {}

        cited_statutes = [self.get_node(n) for n, r in self.neighbors(case_id, "CITES", "out")
                           if self.get_node(n) and self.get_node(n).get("type") == "Statute"]
        cited_cases = [self.get_node(n) for n, r in self.neighbors(case_id, "CITES", "out")
                       if self.get_node(n) and self.get_node(n).get("type") == "CaseLaw"]
        cited_by_cases = [self.get_node(n) for n, r in self.neighbors(case_id, "CITES", "in")
                          if self.get_node(n) and self.get_node(n).get("type") == "CaseLaw"]
        concepts = [self.get_node(n) for n, r in self.neighbors(case_id, "INVOKES", "out")]
        judges = [self.get_node(n) for n, r in self.neighbors(case_id, "HANDLED_BY", "out")]
        affirmed_by = [self.get_node(n) for n, r in self.neighbors(case_id, "AFFIRMED_BY", "out")]
        overruled_by = [self.get_node(n) for n, r in self.neighbors(case_id, "OVERRULED_BY", "out")]

        return {
            "case": node,
            "cited_statutes": cited_statutes,
            "cited_cases": cited_cases,
            "cited_by_cases": cited_by_cases,
            "concepts": concepts,
            "judges": judges,
            "affirmed_by": affirmed_by,
            "overruled_by": overruled_by,
        }


if __name__ == "__main__":
    from data_generator import generate_all

    bundle = generate_all(persist=False)
    store = LegalGraphStore()
    store.load_from_bundle(bundle)

    print(f"Graph built: {store.graph.number_of_nodes()} nodes, {store.graph.number_of_edges()} edges")

    cases = store.nodes_by_type("CaseLaw")
    a, b = cases[0]["case_id"], cases[-1]["case_id"]
    path = store.shortest_legal_path(a, b)
    if path:
        print(f"\nSample shortest path {a} -> {b} ({len(path) - 1} hops):")
        for hop in path:
            label = hop["node"].get("title") or hop["node"].get("name") or hop["node"].get("case_id")
            print(f"  [{hop['rel_type']}] -> {label}")
    else:
        print(f"\nNo path found between {a} and {b} (expected for a sparse random demo graph).")

    print(f"\nDegree centrality top-3:")
    dc = store.degree_centrality()
    for node_id, score in sorted(dc.items(), key=lambda x: -x[1])[:3]:
        print(f"  {node_id}: {score:.4f}")
