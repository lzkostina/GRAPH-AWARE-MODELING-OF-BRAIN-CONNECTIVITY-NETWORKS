"""Regenerate the small synthetic dataset in examples/data/.

    python examples/make_example_data.py

20 subjects, 3 waves (some subjects miss a later wave), 16 ROIs in 4 systems
(120 edges, 10 cells). Kept small so every cell block is smaller than the
number of observations; with tiny N, large blocks make block-V EM chase a
near-singular covariance and it does not converge. Same file layout as real data: per wave a features CSV
and a connectome .npy whose row i matches CSV row i, plus a parcellation CSV.

True effects: age (in half the cells) and site. All other covariates are noise.
Entirely simulated; contains no real subjects.
"""
from pathlib import Path

import numpy as np

from graph_aware.simulate import simulate_panel

OUT = Path(__file__).parent / "data"
WAVES = ["baseline", "year2", "year4"]


def main(seed=2026):
    OUT.mkdir(exist_ok=True)
    waves, regions, truth = simulate_panel(
        n_subjects=20, n_waves=3, n_rois=16, n_systems=4, p_missing_wave=0.15,
        n_sites=3, age_effect_sd=0.004, seed=seed)
    regions.to_csv(OUT / "parcellation.csv", index=False)

    rng = np.random.default_rng(seed)
    ids = waves[0]["features"]["Subject"].to_numpy()
    # subject-level traits shared across waves
    hisp = dict(zip(ids, rng.integers(0, 2, len(ids))))
    race = dict(zip(ids, rng.integers(0, 4, len(ids))))        # 0 = reference
    g0 = dict(zip(ids, rng.normal(0, 1, len(ids))))
    ses = dict(zip(ids, rng.normal(0, 1, len(ids))))

    for name, w in zip(WAVES, waves):
        f = w["features"]
        s = f["Subject"]
        f["sleep_hours"] = np.round(rng.normal(9.5, 0.8, len(f)) - 0.02 * (f["age"] - 119), 2)
        f["g"] = np.round(s.map(g0) + rng.normal(0, 0.3, len(f)), 3)
        f["hisp"] = s.map(hisp)
        for k in (1, 2, 3):
            f[f"race_{k}"] = (s.map(race) == k).astype(int)
        f["ses"] = np.round(s.map(ses), 3)
        f["age"] = f["age"].round(1)
        f["meanFD"] = f["meanFD"].round(4)
        f.to_csv(OUT / f"features_{name}.csv", index=False)
        np.save(OUT / f"connectomes_{name}.npy", w["connectomes"].astype(np.float64))
        print(f"{name}: {len(f)} subjects, connectomes {w['connectomes'].shape}")


if __name__ == "__main__":
    main()