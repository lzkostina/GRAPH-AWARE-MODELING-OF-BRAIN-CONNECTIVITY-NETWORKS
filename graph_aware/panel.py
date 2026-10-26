"""Stack any number of waves into the long format the models use.

Input contract, per wave:
  * a features DataFrame with a subject-ID column (default "Subject"),
  * a connectome array (n_rows, E) whose row i belongs to features.iloc[i].

Output: one row per (subject, wave) observation, wave-major order,
with integer subject_id 0..N-1. Subjects may have different numbers of
waves (unbalanced panels are fine for the models).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class Panel:
    Y: np.ndarray                 # (N_obs, E)
    subject_id: np.ndarray        # (N_obs,) int, 0..N_subjects-1
    wave: np.ndarray              # (N_obs,) int, index into wave_names
    covariates: pd.DataFrame      # (N_obs, ...) includes subject and wave columns
    subjects: np.ndarray          # (N_subjects,) original subject labels
    wave_names: List[str]
    report: Dict = field(default_factory=dict)

    @property
    def N_obs(self):
        return self.Y.shape[0]

    @property
    def N_subjects(self):
        return len(self.subjects)

    def matrix(self, cols: Sequence[str]) -> np.ndarray:
        """Numeric covariate matrix (N_obs, len(cols)); fails on NaN."""
        cols = list(cols)
        missing = [c for c in cols if c not in self.covariates.columns]
        if missing:
            raise KeyError(f"Covariates not found: {missing}")
        M = self.covariates[cols].to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(M).all():
            bad = [c for c in cols if not np.isfinite(self.covariates[c].to_numpy(float)).all()]
            raise ValueError(f"Non-finite values in {bad}; impute before fitting.")
        return M

    def centered(self, col: str = "age") -> np.ndarray:
        """Column centered at its pooled mean over all observations."""
        x = self.matrix([col])[:, 0]
        return x - x.mean()

    def dummies(self, col: str, drop_first: bool = True):
        """One-hot encode a categorical column -> (matrix, names)."""
        d = pd.get_dummies(self.covariates[col].astype(str), prefix=col,
                           drop_first=drop_first)
        return d.to_numpy(dtype=np.float64), list(d.columns)


def stack_waves(
    waves: Sequence[Dict],
    subject_col: str = "Subject",
    wave_names: Optional[Sequence[str]] = None,
    min_waves: int = 2,
    time_invariant: Sequence[str] = (),
    dtype=np.float64,
    verbose: bool = True,
) -> Panel:
    """
    waves : list of {"features": DataFrame, "connectomes": ndarray}, in time order.
    min_waves : keep subjects observed in at least this many waves.
    time_invariant : columns forced constant within subject, set to the value
        at the subject's first observed wave (e.g. site, sex). Changes are counted
        in Panel.report["harmonized"].
    """
    T = len(waves)
    wave_names = list(wave_names) if wave_names is not None else [f"wave{t}" for t in range(T)]
    if len(wave_names) != T:
        raise ValueError("wave_names must have one entry per wave")

    E = None
    frames = []
    for t, w in enumerate(waves):
        f, Y = w["features"], np.asarray(w["connectomes"])
        if Y.ndim != 2:
            raise ValueError(f"[{wave_names[t]}] connectomes must be 2D (n, E), got {Y.shape}")
        if len(f) != Y.shape[0]:
            raise ValueError(f"[{wave_names[t]}] {len(f)} feature rows vs {Y.shape[0]} connectome rows")
        if subject_col not in f.columns:
            raise KeyError(f"[{wave_names[t]}] missing subject column '{subject_col}'")
        s = f[subject_col].astype(str)
        if not s.is_unique:
            raise ValueError(f"[{wave_names[t]}] duplicate subjects")
        if E is None:
            E = Y.shape[1]
        elif Y.shape[1] != E:
            raise ValueError(f"[{wave_names[t]}] E={Y.shape[1]} differs from E={E}")
        if not np.isfinite(Y).all():
            raise ValueError(f"[{wave_names[t]}] non-finite connectome values; clean first")
        g = f.copy()
        g[subject_col] = s.values
        g["_wave"] = t
        g["_row"] = np.arange(len(f))
        frames.append(g)

    long = pd.concat(frames, ignore_index=True)
    n_waves = long.groupby(subject_col)["_wave"].nunique()
    keep = n_waves.index[n_waves >= min_waves]
    long = long[long[subject_col].isin(keep)]

    # Harmonize time-invariant covariates to the first observed wave
    harmonized = {}
    long = long.sort_values(["_wave", subject_col], kind="mergesort")
    for col in time_invariant:
        if col not in long.columns:
            raise KeyError(f"time-invariant column '{col}' not found")
        first = long.groupby(subject_col)[col].first()
        new = long[subject_col].map(first)
        harmonized[col] = int((new.astype(str).values != long[col].astype(str).values).sum())
        long[col] = new.values

    subjects = np.sort(long[subject_col].unique())
    subject_id = np.searchsorted(subjects, long[subject_col].values)
    wave = long["_wave"].to_numpy()

    Y = np.empty((len(long), E), dtype=dtype)
    pos = 0
    for t, w in enumerate(waves):
        rows = long.loc[long["_wave"] == t, "_row"].to_numpy()
        Y[pos:pos + len(rows)] = np.asarray(w["connectomes"])[rows]
        pos += len(rows)

    cov = long.drop(columns=["_row"]).rename(columns={"_wave": "wave"}).reset_index(drop=True)
    counts = np.bincount(wave, minlength=T)
    report = {
        "n_subjects": len(subjects),
        "n_obs": len(long),
        "obs_per_wave": dict(zip(wave_names, counts.tolist())),
        "waves_per_subject": np.bincount(np.bincount(subject_id)).tolist(),
        "harmonized": harmonized,
    }
    if verbose:
        print(f"Panel: {report['n_subjects']} subjects, {report['n_obs']} observations, E={E}")
        print(f"  observations per wave: {report['obs_per_wave']}")
        for col, n in harmonized.items():
            if n:
                print(f"  '{col}' harmonized to first wave for {n} observations")
    return Panel(Y, subject_id.astype(np.int64), wave, cov, subjects, wave_names, report)
