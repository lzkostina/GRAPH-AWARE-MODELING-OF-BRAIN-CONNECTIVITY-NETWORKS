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

## Quick start

```bash
# 1. Get the code
git clone github.com:lzkostina/GRAPH-AWARE-MODELING-OF-BRAIN-CONNECTIVITY-NETWORKS.git
cd graphaware-connectivity

# 2. Create an isolated environment (recommended)
python -m venv .venv
source .venv/bin/activate

# 3. Install the package 
pip install -e ".[dev]"

# 4. Optional: check the install (~3 s)
pytest -q   
```

## Quick example (bundled synthetic data)
```bash
# Run the example. Must be run from the repo root.
python scripts/fit_model.py examples/example_config.yaml
```
examples/data/ holds a tiny fully simulated panel: 20 subjects, 3 waves (some subjects miss a wave), 
16 ROIs in 4 systems (120 edges, 10 cells). Its files have exactly the layout real data needs.
Results go to examples/results/. examples/make_example_data.py regenerates the data.

Only age (in about half the cells) and site have true effects. With about 20 subject clusters, the sandwich SEs are 
badly anticonservative, so pure-noise subject-level covariates such as g can come out "significant". 
That's expected at this size: this example shows the mechanics, not the inference.
## Running on your data

1. Prepare one features CSV and one connectome `.npy` per wave. In the `.npy`,
   row *i* must belong to CSV row *i*, with shape `(n, E)` and
   `E = R(R-1)/2`, flattened in `np.triu_indices(R, k=1)` order. You also need
   a parcellation CSV with one row per ROI, in connectome order, that has
   `NetworkNumber` and `Community` columns.
2. Run `cp config.example.yaml config.yaml` and edit the paths. To add a wave,
   add an entry under `waves:`.
3. `python scripts/fit_model.py config.yaml`

Interactions go under model.interactions as pairs, either [age, g] or "age:g". 
Each term must be age or a column listed in extra_covariates. Products are 
built from the centered columns, so the age main effect is the effect at the 
mean of the other covariate. List the interaction itself under 
inference.focus ("age:g"), or use focus: all. BH correction is applied 
over cells, separately for each focus covariate.
The outputs, `results/cells_<covariate>.csv` and `results/fit_summary.json`,
hold cell-level summaries only.

From Python:

```python
import scripts as G

P = G.stack_waves([{"features": f0, "connectomes": Y0},
                   {"features": f1, "connectomes": Y1},
                   {"features": f2, "connectomes": Y2}],
                  wave_names=["baseline", "year2", "year4"],
                  time_invariant=["site_id_l", "sex"])
cs = G.build_cells_from_gordon(regions, P.Y.shape[1])
sites, names = P.dummies("site_id_l")
m = G.GraphAwareBlockVEM(Y=P.Y, age=P.centered("age"), subject_id=P.subject_id,
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
graph_aware/
  cells.py     parcellation → system-pair cells
  panel.py     stack any number of waves into long format
  models.py    diagonal-V and block-V EM, LL, predictive LL, diagnostics
  ols.py       pooled cell-wise OLS, naive + clustered SE
  inference.py p-values, BH/BY/Holm
  simulate.py  synthetic panels drawn from the block-V model
scripts/fit_model.py   config-driven run
tests/                 synthetic tests (no real data)
```

## Notes 

- **Shrinkage of large blocks** triggers when `E_c > N_obs`, where `N_obs`
  counts observations, not subjects. With more waves, fewer blocks get shrunk.
- **Convergence** uses relative LL change < `tol`. The LL is of order 1e8, so
  the default `1e-6` can stop after a few iterations.
- Random intercept only. With three or more waves, random age slopes become
  identifiable but are not implemented.
