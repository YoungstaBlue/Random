"""Typed JSON payload contracts exchanged between every base44 module.

Modules communicate ONLY through these pydantic models (Block A requirement #3:
strict modularity). Sparse matrices and index handles are runtime artifacts and
deliberately kept out of these JSON contracts; they are passed by reference in
the vectorize/index/query layer instead.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Block A — ingestion / vectorization / retrieval
# --------------------------------------------------------------------------- #
class RawDocument(BaseModel):
    doc_id: str
    text: str
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CleanedDocument(BaseModel):
    doc_id: str
    text: str                       # cleaned, lemmatized string ready for TF-IDF
    tokens: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchHit(BaseModel):
    doc_id: str
    score: float                    # cosine similarity in [0, 1]
    bucket: Optional[int] = None    # LSH bucket the candidate came from
    snippet: Optional[str] = None


class QueryResult(BaseModel):
    query: str
    hits: List[SearchHit] = Field(default_factory=list)
    candidates_scanned: int = 0     # docs actually compared (<< N, proves ANN)
    total_docs: int = 0
    backend: Dict[str, str] = Field(default_factory=dict)  # which libs were used


# --------------------------------------------------------------------------- #
# Block B — legal domain payloads
# --------------------------------------------------------------------------- #
class Citation(BaseModel):
    raw: str
    normalized: Optional[str] = None
    jurisdiction: Optional[str] = None      # federal / state / territory (e.g. Puerto Rico)
    verified: bool = False
    notes: Optional[str] = None


class PayloadKind(str, Enum):
    EVIDENCE = "evidence"
    STATUTORY_AUTHORITY = "statutory_authority"
    PRECEDENT = "precedent"
    PROCEDURAL_RULE = "procedural_rule"


class CategorizedPayload(BaseModel):
    """Module 6: incoming case data parsed into the four legal buckets."""
    evidence: List[str] = Field(default_factory=list)
    statutory_authority: List[str] = Field(default_factory=list)
    precedent: List[str] = Field(default_factory=list)
    procedural_rules: List[str] = Field(default_factory=list)


class PriorityTask(BaseModel):
    task_id: str
    kind: PayloadKind
    summary: str
    deadline: Optional[date] = None
    priority_score: float = 0.0


# ---- Module 4: visualization payloads ----
class TreeNode(BaseModel):
    id: str
    label: str
    kind: Optional[str] = None
    children: List["TreeNode"] = Field(default_factory=list)


class GraphNode(BaseModel):
    id: str
    label: str
    group: Optional[str] = None


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: Optional[str] = None


class GraphData(BaseModel):
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)


class VisualizationBundle(BaseModel):
    decision_tree: TreeNode
    relational_graph: GraphData
    mind_map: TreeNode


# ---- Module 5: formatting payloads ----
class CourtFormatConfig(BaseModel):
    court_name: str = "IN THE UNITED STATES DISTRICT COURT"
    district: Optional[str] = None
    case_number: Optional[str] = None
    plaintiff: Optional[str] = None
    defendant: Optional[str] = None
    document_title: str = "MOTION"
    attorney_name: Optional[str] = None
    attorney_bar: Optional[str] = None


class FormattedDocument(BaseModel):
    caption: str
    body: str
    signature_block: str
    certificate_of_service: str
    full_text: str


# ---- Module 7: verification payloads ----
class ComplianceReport(BaseModel):
    passed: bool
    issues: List[str] = Field(default_factory=list)     # blocking
    warnings: List[str] = Field(default_factory=list)   # non-blocking
    citations_checked: List[Citation] = Field(default_factory=list)


# ---- Module 5: version control ----
class RevisionRecord(BaseModel):
    version: int
    timestamp: datetime
    author: str
    summary: str
    content_hash: str


# --------------------------------------------------------------------------- #
# Orchestrator result (Module 8 metrics attached)
# --------------------------------------------------------------------------- #
class StageMetric(BaseModel):
    stage: str
    duration_ms: float
    detail: Dict[str, Any] = Field(default_factory=dict)


class PipelineResult(BaseModel):
    case_id: str
    categorized: CategorizedPayload
    priorities: List[PriorityTask] = Field(default_factory=list)
    query_result: Optional[QueryResult] = None
    visualization: Optional[VisualizationBundle] = None
    formatted_document: Optional[FormattedDocument] = None
    compliance: Optional[ComplianceReport] = None
    revision: Optional[RevisionRecord] = None
    metrics: List[StageMetric] = Field(default_factory=list)


TreeNode.model_rebuild()
