"""Decision Tree Engine (Module 4).

Generates structural node-and-branch data mapping procedural choices, legal
strategies, and probable outcomes. Emits a :class:`TreeNode` (JSON-serializable)
ready for the Base 44 app's tree renderer.
"""
from __future__ import annotations

from ..schemas import CategorizedPayload, TreeNode


def build_decision_tree(case_id: str, payload: CategorizedPayload) -> TreeNode:
    """Map procedural rules -> strategic options -> probable outcomes."""
    strategy_branches = []
    for i, rule in enumerate(payload.procedural_rules or ["No procedural rules parsed"]):
        strategy_branches.append(
            TreeNode(
                id=f"rule-{i}",
                label=rule,
                kind="procedural_choice",
                children=[
                    TreeNode(id=f"rule-{i}-comply", label="Comply / file", kind="strategy",
                             children=[TreeNode(id=f"rule-{i}-comply-out",
                                                label="Motion accepted", kind="outcome")]),
                    TreeNode(id=f"rule-{i}-challenge", label="Challenge / object", kind="strategy",
                             children=[TreeNode(id=f"rule-{i}-challenge-out",
                                                label="Ruling pending", kind="outcome")]),
                ],
            )
        )
    return TreeNode(
        id=f"case-{case_id}",
        label=f"Case {case_id}: procedural decision map",
        kind="root",
        children=strategy_branches,
    )
