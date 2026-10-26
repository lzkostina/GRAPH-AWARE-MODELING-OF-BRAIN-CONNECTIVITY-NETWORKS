"""Synthetic longitudinal connectomes drawn from the block-V model.

Lets anyone run the full pipeline and tests without access to real data.
"""
import numpy as np
import pandas as pd

from .cells import map_edges_to_cells


def simulate_panel(
    n_subjects=300,
    n_waves=3,
    n_rois=24,
    n_systems=4,
    p_missing_wave=0.15,
    wave_gap_months=24.0,
    d=0.02,
    v=0.05,
    within_cell_corr=0.3,
    age_effect_sd=0.002,
    n_null_cells=None,
    n_sites=4,
    site_effect_sd=0.02,
    seed=0,
):
    """
    Returns (waves, regions, truth).

    waves   : list of {"features": DataFrame, "connectomes": ndarray}, the same
              input format `stack_waves` expects for real data.
    regions : DataFrame with NetworkNumber / Community per ROI.
    truth   : dict with true alpha (C, 2), d, site effects, CellStructure.

    Each subject is observed at wave 0 and, independently with probability
    1 - p_missing_wave, at each later wave, so the panel is unbalanced.
    Residuals are equicorrelated within each cell (correlation
    `within_cell_corr`), independent across cells.
    """
    rng = np.random.default_rng(seed)
    if n_rois < 2 * n_systems:
        raise ValueError("need at least 2 ROIs per system")
    # every system gets >= 2 ROIs, so no cell is empty
    extra = rng.integers(0, n_systems, size=n_rois - 2 * n_systems)
    sizes = 2 + np.bincount(extra, minlength=n_systems)
    systems = np.repeat(np.arange(1, n_systems + 1), sizes)
    regions = pd.DataFrame({
        "ParcelID": np.arange(1, n_rois + 1),
        "NetworkNumber": systems,
        "Community": [f"Sys{s}" for s in systems],
    })
    cs = map_edges_to_cells(systems)
    E, C = cs.E, cs.C

    alpha = np.zeros((C, 2))
    alpha[:, 0] = rng.normal(0.2, 0.1, C)
    alpha[:, 1] = rng.normal(0.0, age_effect_sd, C)
    if n_null_cells is None:
        n_null_cells = C // 2
    null_cells = rng.choice(C, size=n_null_cells, replace=False)
    alpha[null_cells, 1] = 0.0

    eta = rng.normal(0, 0.05, E)
    for eidx in cs.edges_per_cell:
        eta[eidx] -= eta[eidx].mean()

    site = rng.integers(0, n_sites, n_subjects)
    site_eff = np.r_[0.0, rng.normal(0, site_effect_sd, n_sites - 1)]
    sex = rng.integers(1, 3, n_subjects)
    age0 = rng.normal(119.0, 7.5, n_subjects)                   # months
    b = rng.normal(0, np.sqrt(d), n_subjects)

    rho = within_cell_corr
    chol = [np.linalg.cholesky(v * ((1 - rho) * np.eye(len(e)) + rho)) for e in cs.edges_per_cell]

    waves = []
    ids = np.array([f"SUBJ{i:05d}" for i in range(n_subjects)])
    for t in range(n_waves):
        present = np.ones(n_subjects, bool) if t == 0 else rng.random(n_subjects) > p_missing_wave
        idx = np.flatnonzero(present)
        n = len(idx)
        age = age0[idx] + t * wave_gap_months + rng.normal(0, 2.0, n)
        age_c = age - 131.0                                       # approx. pooled mean
        Y = np.empty((n, E))
        for c, eidx in enumerate(cs.edges_per_cell):
            mu = alpha[c, 0] + alpha[c, 1] * age_c
            Y[:, eidx] = (mu[:, None] + eta[eidx][None, :]
                          + rng.standard_normal((n, len(eidx))) @ chol[c].T)
        Y += (b[idx] + site_eff[site[idx]])[:, None]
        feats = pd.DataFrame({
            "Subject": ids[idx],
            "age": age,
            "sex": sex[idx],
            "site_id_l": [f"site{s + 1:02d}" for s in site[idx]],
            "meanFD": np.abs(rng.normal(0.1, 0.03, n)),
        })
        waves.append({"features": feats, "connectomes": Y})

    truth = {"alpha": alpha, "eta": eta, "d": d, "site_effects": site_eff,
             "null_cells": np.sort(null_cells), "cells": cs, "age_center": 131.0}
    return waves, regions, truth