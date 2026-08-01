"""base44 — Unified Legal Analysis & Vector-Matrix Pipeline.

Block A (the high-performance vector-matrix / e-discovery engine) is nested
inside Block B (the 8-module legal-analysis architecture) as its semantic-search
core. See ARCHITECTURE.md and MASTER_PROMPT.md for the full design.
"""

from .config import Settings, get_settings

__all__ = ["Settings", "get_settings", "__version__"]
__version__ = "0.1.0"
