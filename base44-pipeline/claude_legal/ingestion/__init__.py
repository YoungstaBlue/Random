"""Block A · Component 1 — Data Ingestion & Preprocessing."""
from .loaders import load_documents, load_text
from .preprocess import Preprocessor
from .security import SecurityException, SemanticFirewall

__all__ = ["Preprocessor", "load_documents", "load_text",
           "SemanticFirewall", "SecurityException"]
