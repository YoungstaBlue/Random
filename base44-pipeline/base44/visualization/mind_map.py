"""Mind Mapping Integration (Module 4).

Clusters legal doctrines, sub-issues, and precedent connections into a radial
:class:`TreeNode` for the Base 44 app's brainstorming layer.
"""
from __future__ import annotations

from ..schemas import CategorizedPayload, TreeNode


def build_mind_map(case_id: str, payload: CategorizedPayload) -> TreeNode:
    clusters = {
        "Evidence": payload.evidence,
        "Statutory Authority": payload.statutory_authority,
        "Precedent": payload.precedent,
        "Procedural Rules": payload.procedural_rules,
    }
    branches = []
    for ci, (label, items) in enumerate(clusters.items()):
        leaves = [
            TreeNode(id=f"mm-{ci}-{li}", label=_truncate(item), kind="sub_issue")
            for li, item in enumerate(items)
        ]
        branches.append(TreeNode(id=f"mm-{ci}", label=label, kind="doctrine", children=leaves))
    return TreeNode(id=f"mindmap-{case_id}", label=f"Case {case_id} doctrines",
                    kind="root", children=branches)


def _truncate(text: str, length: int = 50) -> str:
    text = " ".join(text.split())
    return text[:length] + ("…" if len(text) > length else "")
