"""End-to-end run on synthetic 3-wave data. No real data needed.

    python examples/synthetic_demo.py
"""
import numpy as np
import graph_aware as G
from graph_aware.simulate import simulate_panel

waves, regions, truth = simulate_panel(n_subjects=500, n_waves=3, p_missing_wave=0.2, seed=0)
P = G.stack_waves(waves, wave_names=["baseline", "year2", "year4"],
                  time_invariant=["site_id_l", "sex"])
cs = G.build_cells_from_gordon(regions, P.Y.shape[1])
labels = G.cell_labels_from_gordon(regions, cs)

sites, site_names = P.dummies("site_id_l")
model = G.GraphAwareBlockVEM(
    Y=P.Y, age=P.centered("age"), subject_id=P.subject_id,
    cell_id_of_edge=cs.cell_id_of_edge, edges_per_cell=cs.edges_per_cell,
    global_covariates=sites, global_covariate_names=site_names,
).fit()
diag = model.diagnostics(focus="age", cell_labels=labels)

print(f"\ncorr(true, estimated age slopes) = "
      f"{np.corrcoef(truth['alpha'][:, 1], diag['focus_alpha'])[0, 1]:.3f}")
null = truth["null_cells"]
print(f"false positives among {len(null)} null cells: {diag['focus_sig_adj'][null].sum()}")
