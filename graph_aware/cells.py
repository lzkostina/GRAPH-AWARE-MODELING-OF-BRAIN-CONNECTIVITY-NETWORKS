"""Map edges of a parcellated connectome to system-pair cells.

Edges are the strict upper triangle of an R x R matrix, flattened in
``np.triu_indices(R, k=1)`` order. With K systems there are K(K+1)/2 cells
(e.g. Gordon 333 + subcortical/cerebellar: R=418, K=15, C=120).
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class CellStructure:
    cells: np.ndarray             # (C, 2) system pairs (a <= b), labels 1..K
    cell_id_of_edge: np.ndarray   # (E,) cell index 0..C-1
    edges_per_cell: List[np.ndarray]
    tri_idx: Tuple[np.ndarray, np.ndarray]   # ROI indices of each edge
    roi_to_system: np.ndarray     # (R,) contiguous labels 1..K

    @property
    def C(self):
        return len(self.edges_per_cell)

    @property
    def E(self):
        return len(self.cell_id_of_edge)


def build_cells(K: int, base: int = 1) -> np.ndarray:
    labels = np.arange(base, base + K, dtype=int)
    C1, C2 = np.meshgrid(labels, labels, indexing="ij")
    mask = C1 <= C2
    return np.column_stack((C1[mask], C2[mask]))


def pair_to_cell_id(a0: np.ndarray, b0: np.ndarray, K: int) -> np.ndarray:
    """0-based (a0 <= b0) -> cell id, matching build_cells order."""
    return (a0 * K - (a0 * (a0 - 1)) // 2 + (b0 - a0)).astype(int)


def edges_per_cell_from_ids(cell_id_of_edge: np.ndarray, C: int) -> List[np.ndarray]:
    order = np.argsort(cell_id_of_edge, kind="mergesort")
    counts = np.bincount(cell_id_of_edge, minlength=C)
    return np.split(order, np.cumsum(counts)[:-1])


def map_edges_to_cells(system_labels) -> CellStructure:
    """system_labels: (R,) arbitrary labels, one per ROI."""
    raw = np.asarray(system_labels)
    uniq = np.unique(raw)
    K = uniq.size
    lab0 = np.searchsorted(uniq, raw)          # contiguous 0..K-1
    R = lab0.size

    iu_i, iu_j = np.triu_indices(R, k=1)
    a, b = lab0[iu_i], lab0[iu_j]
    cid = pair_to_cell_id(np.minimum(a, b), np.maximum(a, b), K)
    cells = build_cells(K, base=1)
    epc = edges_per_cell_from_ids(cid, cells.shape[0])

    allc = np.concatenate(epc)
    if not np.array_equal(np.sort(allc), np.arange(len(cid))):
        raise RuntimeError("edges_per_cell does not partition the edges")
    return CellStructure(cells, cid, epc, (iu_i, iu_j), lab0 + 1)


def build_cells_from_gordon(regions: pd.DataFrame, n_edges: int,
                            network_col: str = "NetworkNumber") -> CellStructure:
    if network_col not in regions.columns:
        raise KeyError(f"regions must contain column '{network_col}'")
    R = len(regions)
    if R * (R - 1) // 2 != n_edges:
        raise ValueError(
            f"Connectomes have E={n_edges} edges but parcellation has R={R} ROIs "
            f"(R(R-1)/2={R * (R - 1) // 2}). Different ROI set or flattening?")
    return map_edges_to_cells(regions[network_col].to_numpy())


def cell_labels_from_gordon(regions: pd.DataFrame, cs: CellStructure,
                            network_col="NetworkNumber", name_col="Community"):
    """List of (system_a, system_b) names, one per cell."""
    names = (regions[[network_col, name_col]].drop_duplicates()
             .sort_values(network_col)[name_col].tolist())
    return [(names[a - 1], names[b - 1]) for a, b in cs.cells]
