# graphaware

Graph-aware random-intercept models for longitudinal functional connectivity,
extending Kim, Kessler & Levina (2023, *AOAS*) to repeated measurements.
Works with **any number of waves**. Panels can be unbalanced, so subjects may
potentially miss waves. 

For observation *n* of subject *m*, and edge *e* in system-pair cell *c*:

```
y_ne = x_n' α_c + η_e + z_n' γ + b_m + ε_ne,    b_m ~ N(0, d),   ε_n ~ N(0, V)
```

| model | V |
|---|---|
| `GraphAwareRandomInterceptEM` | diagonal |
| `GraphAwareBlockVEM` | block-diagonal, one dense block per cell |

`α_c` holds the cell-level effects (intercept, age, extra covariates). `γ` holds
covariates with one shared effect on every edge (e.g. site). Both models are
fit by EM. Inference is per cell, using subject-clustered sandwich SEs with
BH/BY/Holm correction.

## Install

```bash
pip install -e ".[dev]"
pytest                          # ~3 s, synthetic data only
python examples/synthetic_demo.py
```

## Running on your data

1. Prepare one features CSV and one connectome `.npy` per wave. In the `.npy`,
   row *i* must belong to CSV row *i*, with shape `(n, E)` and
   `E = R(R-1)/2`, flattened in `np.triu_indices(R, k=1)` order. You also need
   a parcellation CSV with one row per ROI, in connectome order, that has
   `NetworkNumber` and `Community` columns.
2. Run `cp config.example.yaml config.yaml` and edit the paths. To add a wave,
   add an entry under `waves:`.
3. `python scripts/fit_model.py config.yaml`

The outputs, `results/cells_<covariate>.csv` and `results/fit_summary.json`,
hold cell-level summaries only.

From Python:

```python
import graphaware as G
P  = G.stack_waves([{"features": f0, "connectomes": Y0},
                    {"features": f1, "connectomes": Y1},
                    {"features": f2, "connectomes": Y2}],
                   wave_names=["baseline", "year2", "year4"],
                   time_invariant=["site_id_l", "sex"])
cs = G.build_cells_from_gordon(regions, P.Y.shape[1])
sites, names = P.dummies("site_id_l")
m  = G.GraphAwareBlockVEM(Y=P.Y, age=P.centered("age"), subject_id=P.subject_id,
                          cell_id_of_edge=cs.cell_id_of_edge, edges_per_cell=cs.edges_per_cell,
                          global_covariates=sites - sites.mean(0), global_covariate_names=names).fit()
res = m.diagnostics(focus="age", cell_labels=G.cell_labels_from_gordon(regions, cs))
```

`stack_waves` checks that rows are aligned and subjects are unique within each
wave, and that every wave has the same number of edges. For each subject it
sets time-invariant columns to their value at the first observed wave (for
example, site for subjects who moved), and it drops subjects seen in fewer
than `min_waves` waves.

**Memory.** `Y` is float64 with shape `(N_obs, E)`. With 2,400 subjects × 3
waves × 87,153 edges that is about 5 GB, and the M-step makes temporary copies
of the same size, so plan for roughly 3×.

## Layout

```
graphaware/
  cells.py     parcellation → system-pair cells
  panel.py     stack any number of waves into long format
  models.py    diagonal-V and block-V EM, LL, predictive LL, diagnostics
  ols.py       pooled cell-wise OLS, naive + clustered SE
  inference.py p-values, BH/BY/Holm
  simulate.py  synthetic panels drawn from the block-V model
scripts/fit_model.py   config-driven run
tests/                 synthetic tests (no real data)
```

## Notes and known issues

- **Initialization of `d`.** The original notebook took per-subject OLS
  intercepts. When a subject has fewer observations than columns of `X` (two
  waves plus any extra covariate), every subject falls back to the same value,
  so `d` starts at 1e-8. The default is now a moment estimator that works for
  any T and covariate set. `d_init="subject_ols"` reproduces the old behaviour.
- **Block-V standard errors** reuse the diagonal weights `1/v_e` in the
  sandwich, while the block-V point estimates use `V_c^{-1} 1` weights. This
  matches the notebook results, but it is worth a look.
- **Shrinkage of large blocks** triggers when `E_c > N_obs`, where `N_obs`
  counts observations, not subjects. With more waves, fewer blocks get shrunk.
- **Convergence** uses relative LL change < `tol`. The LL is of order 1e8, so
  the default `1e-6` can stop after a few iterations.
- Random intercept only. With three or more waves, random age slopes become
  identifiable but are not implemented.
