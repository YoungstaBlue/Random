"""Relational Tree Graphs & Charts (Module 4).

Processes evidence, facts, and party relationships into structured
node/edge graph data for the Base 44 app's graph renderer.
"""
from __future__ import annotations

from ..schemas import CategorizedPayload, GraphData, GraphEdge, GraphNode


def build_relational_graph(case_id: str, payload: CategorizedPayload) -> GraphData:
    nodes = [GraphNode(id=f"case-{case_id}", label=f"Case {case_id}", group="case")]
    edges = []

    def _add(prefix: str, group: str, items):
        for i, item in enumerate(items):
            nid = f"{prefix}-{i}"
            nodes.append(GraphNode(id=nid, label=_truncate(item), group=group))
            edges.append(GraphEdge(source=f"case-{case_id}", target=nid, relation=group))

    _add("evi", "evidence", payload.evidence)
    _add("stat", "statutory_authority", payload.statutory_authority)
    _add("prec", "precedent", payload.precedent)
    _add("proc", "procedural_rule", payload.procedural_rules)
    return GraphData(nodes=nodes, edges=edges)


def _truncate(text: str, length: int = 60) -> str:
    text = " ".join(text.split())
    return text[:length] + ("…" if len(text) > length else "")
