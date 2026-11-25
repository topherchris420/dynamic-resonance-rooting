"""External evidence: a fixed public dataset and a preregistered comparison.

The core test suite shows that DRR implements its definitions. This package
runs one domain comparison and records whether that comparison met its
preregistered decision rule.
"""

from .protocol import classify_claim
from .qbo import load_preregistration, load_qbo_series, run_structural_change_benchmark

__all__ = [
    "classify_claim",
    "load_preregistration",
    "load_qbo_series",
    "run_structural_change_benchmark",
]
