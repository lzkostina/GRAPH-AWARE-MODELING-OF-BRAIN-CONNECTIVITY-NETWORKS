import numpy as np
import pandas as pd
import pytest

import graph_aware as G
from graph_aware.simulate import simulate_panel


@pytest.fixture(scope="module")
def sim():
    waves, regions, truth = simulate_panel(n_subjects=400, n_waves=3, p_missing_wave=0.2, seed=7)
    cs = G.build_cells_from_gordon(regions, waves[0]["connectomes"].shape[1])
    P = G.stack_waves(waves, time_invariant=["site_id_l", "sex"], verbose=False)
    return waves, regions, truth, cs, P


def _fit(cls, P, cs, **kw):
    sites, names = P.dummies("site_id_l")
    return cls(Y=P.Y, age=P.centered("age"), subject_id=P.subject_id,
               cell_id_of_edge=cs.cell_id_of_edge, edges_per_cell=cs.edges_per_cell,
               global_covariates=sites, global_covariate_names=names,
               verbose=False, **kw).fit()


def test_cells_partition(sim):
    _, regions, _, cs, _ = sim
    K = regions.NetworkNumber.nunique()
    assert cs.C == K * (K + 1) // 2
    assert np.array_equal(np.sort(np.concatenate(cs.edges_per_cell)), np.arange(cs.E))
    labels = G.cell_labels_from_gordon(regions, cs)
    assert len(labels) == cs.C


def test_stack_waves_alignment(sim):
    waves, _, _, _, P = sim
    # every stacked row equals the source row for that subject and wave
    for n in np.random.default_rng(0).choice(P.N_obs, 50, replace=False):
        t, s = P.wave[n], P.subjects[P.subject_id[n]]
        f = waves[t]["features"]
        i = np.flatnonzero(f["Subject"].values == s)[0]
        assert np.array_equal(P.Y[n], waves[t]["connectomes"][i])
    assert P.report["waves_per_subject"][1] == 0          # min_waves=2 enforced


def test_stack_waves_harmonizes_time_invariant():
    waves, _, _ = simulate_panel(n_subjects=50, n_waves=2, p_missing_wave=0, n_rois=8, n_systems=2)
    waves[1]["features"].loc[0, "site_id_l"] = "site99"   # a "mover"
    P = G.stack_waves(waves, time_invariant=["site_id_l"], verbose=False)
    assert P.report["harmonized"]["site_id_l"] == 1
    assert (P.covariates.groupby("Subject")["site_id_l"].nunique() == 1).all()


def test_stack_waves_rejects_misaligned():
    waves, _, _ = simulate_panel(n_subjects=30, n_waves=2, n_rois=8, n_systems=2)
    waves[1]["connectomes"] = waves[1]["connectomes"][:-1]
    with pytest.raises(ValueError):
        G.stack_waves(waves, verbose=False)


@pytest.mark.parametrize("cls", [G.GraphAwareRandomInterceptEM, G.GraphAwareBlockVEM])
def test_em_monotone(sim, cls):
    _, _, _, cs, P = sim
    m = _fit(cls, P, cs, max_iter=25, tol=0)
    ll = np.array([v for _, v in m.log_likelihood_history])
    assert np.all(np.diff(ll) > -1e-6 * np.abs(ll[1:]))


@pytest.mark.parametrize("cls", [G.GraphAwareRandomInterceptEM, G.GraphAwareBlockVEM])
def test_recovers_age_slopes(sim, cls):
    _, _, truth, cs, P = sim
    m = _fit(cls, P, cs)
    est, true = m.alpha[:, 1], truth["alpha"][:, 1]
    assert np.corrcoef(est, true)[0, 1] > 0.95
    assert abs(m.d - truth["d"]) / truth["d"] < 0.25
    diag = m.diagnostics(focus="age")
    nonnull = np.setdiff1d(np.arange(cs.C), truth["null_cells"])
    assert diag["focus_sig_adj"][nonnull].mean() > 0.5


def test_blockv_beats_diag_out_of_sample():
    waves, regions, _ = simulate_panel(n_subjects=600, n_waves=2, p_missing_wave=0, seed=11)
    cs = G.build_cells_from_gordon(regions, waves[0]["connectomes"].shape[1])
    P = G.stack_waves(waves, verbose=False)
    train = P.subject_id < 400
    age = P.centered()
    kw = dict(cell_id_of_edge=cs.cell_id_of_edge, edges_per_cell=cs.edges_per_cell, verbose=False)
    lls = []
    for cls in (G.GraphAwareRandomInterceptEM, G.GraphAwareBlockVEM):
        m = cls(Y=P.Y[train], age=age[train], subject_id=P.subject_id[train], **kw).fit()
        lls.append(m.predictive_log_likelihood(P.Y[~train], age[~train], P.subject_id[~train]))
    assert np.isfinite(lls).all() and lls[1] > lls[0]


def test_ols_clustered_wider_than_naive(sim):
    _, _, _, cs, P = sim
    X = np.column_stack([np.ones(P.N_obs), P.centered()])
    out = G.cell_wise_ols_with_se(P.Y, X, P.subject_id, cs.cell_id_of_edge)
    assert np.median(out["se_clust"][:, 1] / out["se_naive"][:, 1]) > 1.5


def test_bh_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.stats.multitest")
    p = np.random.default_rng(0).uniform(size=200) ** 2
    assert np.allclose(G.adjust_pvalues(p, "BH"), sm.multipletests(p, method="fdr_bh")[1])


def test_d_init_with_time_invariant_covariate():
    """With sex in X, per-subject OLS is unidentified; 'moments' init must not collapse."""
    waves, regions, truth = simulate_panel(n_subjects=300, n_waves=3, seed=2)
    P = G.stack_waves(waves, time_invariant=["sex"], verbose=False)
    cs = G.build_cells_from_gordon(regions, P.Y.shape[1])
    sex = P.matrix(["sex"]); sex -= sex.mean()
    m = G.GraphAwareBlockVEM(Y=P.Y, age=P.centered(), subject_id=P.subject_id,
                             cell_id_of_edge=cs.cell_id_of_edge, edges_per_cell=cs.edges_per_cell,
                             extra_covariates=sex, covariate_names=["sex"], verbose=False).fit()
    assert abs(m.d - truth["d"]) / truth["d"] < 0.25
