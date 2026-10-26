"""Graph-aware random-intercept models for longitudinal connectome data."""
from .cells import build_cells_from_gordon, cell_labels_from_gordon, CellStructure
from .panel import Panel, stack_waves
from .models import GraphAwareRandomInterceptEM, GraphAwareBlockVEM
from .ols import cell_wise_ols_with_se
from .inference import z_to_pvals, adjust_pvalues, testing_summary_from_z

__all__ = [
    "build_cells_from_gordon", "cell_labels_from_gordon", "CellStructure",
    "Panel", "stack_waves",
    "GraphAwareRandomInterceptEM", "GraphAwareBlockVEM",
    "cell_wise_ols_with_se",
    "z_to_pvals", "adjust_pvalues", "testing_summary_from_z",
]
