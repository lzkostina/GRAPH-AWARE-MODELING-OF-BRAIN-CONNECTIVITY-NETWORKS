"""Graph-aware random-intercept models, fit by EM.

Model, for observation n (subject m(n), any wave) and edge e in cell c:

    y_{n,e} = x_n' alpha_c + eta_e + z_n' gamma + b_{m(n)} + eps_{n,e}
    b_m ~ N(0, d),   eps_{n,.} ~ N(0, V)

* alpha_c : cell-level fixed effects (intercept, age, extra covariates)
* eta_e   : edge-specific intercept deviation, V^{-1}-weighted mean zero per cell
* gamma   : "global" covariates with one shared effect on all edges (e.g. site)
* b_m     : subject random intercept, shared across waves and edges
* V       : diagonal (GraphAwareRandomInterceptEM) or block-diagonal with one
            dense block per cell (GraphAwareBlockVEM)

Data are long format: one row of Y per observation, `subject_id` links rows to
subjects. Nothing assumes two waves; subjects can have different numbers of
observations.
"""
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .inference import adjust_pvalues, z_to_pvals


def _safe_inv(A):
    try:
        L, low = cho_factor(A)
        return cho_solve((L, low), np.eye(A.shape[0]))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


def _as_2d(x):
    if x is None:
        return None
    x = np.asarray(x, dtype=np.float64)
    return x[:, None] if x.ndim == 1 else x


def _fmt_time(s):
    s = int(round(s))
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


class GraphAwareRandomInterceptEM:
    """Random intercept + diagonal residual covariance."""

    def __init__(
        self,
        Y,
        age,
        subject_id,
        cell_id_of_edge,
        edges_per_cell,
        extra_covariates=None,
        global_covariates=None,
        covariate_names=None,
        global_covariate_names=None,
        interaction_specs=None,
        max_iter=50,
        tol=1e-6,
        edge_batch=512,
        d_init="moments",
        verbose=True,
    ):
        self.Y = np.asarray(Y, dtype=np.float64)
        self.age = np.asarray(age, dtype=np.float64).reshape(-1)
        self.cell_id_of_edge = np.asarray(cell_id_of_edge)
        self.edges_per_cell = [np.asarray(e, dtype=np.int64) for e in edges_per_cell]
        self.verbose = bool(verbose)
        self.tol = tol
        self.max_iter = max_iter
        self.B = edge_batch
        self.d_init = d_init

        # Subjects: remap any labels to contiguous 0..M-1
        self.subject_labels, self.subject_id = np.unique(np.asarray(subject_id), return_inverse=True)
        self.N_subjects = len(self.subject_labels)

        self.Z_extra = _as_2d(extra_covariates)
        self.n_extra = 0 if self.Z_extra is None else self.Z_extra.shape[1]
        self.base_extra_covariate_names = (
            list(covariate_names) if covariate_names is not None
            else [f"x{j}" for j in range(self.n_extra)])
        if len(self.base_extra_covariate_names) != self.n_extra:
            raise ValueError("len(covariate_names) must match number of extra covariates.")

        self.Z_global = _as_2d(global_covariates)
        self.has_global = self.Z_global is not None
        self.n_global = 0 if self.Z_global is None else self.Z_global.shape[1]
        self.global_covariate_names = (
            list(global_covariate_names) if global_covariate_names is not None
            else [f"g{j}" for j in range(self.n_global)])
        if len(self.global_covariate_names) != self.n_global:
            raise ValueError("len(global_covariate_names) must match number of global covariates.")

        self.interaction_specs = list(interaction_specs or [])

        self.X, self.Zg, self.covariate_names_full = self._build_design_matrix(
            self.age, self.Z_extra, self.Z_global)
        self.p = self.X.shape[1]
        self.N_obs, self.E = self.Y.shape
        self.C = len(self.edges_per_cell)

        if len(self.age) != self.N_obs or len(self.subject_id) != self.N_obs:
            raise ValueError("Y, age and subject_id must have the same number of rows.")
        self._validate_cell_structure()

        order = np.argsort(self.subject_id, kind="mergesort")
        counts = np.bincount(self.subject_id, minlength=self.N_subjects)
        self.obs_per_subject = np.split(order, np.cumsum(counts)[:-1])
        self.T_subj = counts.astype(np.float64)

        self.XtX_inv = np.linalg.pinv(self.X.T @ self.X)
        self._XtX_inv_Xt = self.XtX_inv @ self.X.T

        self.alpha = self.eta = self.d = self.V_diag = None
        self.b_mean = self.b_var = self.gamma = None
        self.log_likelihood_history = []

    # ── setup helpers ────────────────────────────────────────────────

    def _validate_cell_structure(self):
        nonempty = [e for e in self.edges_per_cell if len(e) > 0]
        if not nonempty:
            raise ValueError("All cells are empty.")
        all_edges = np.concatenate(nonempty)
        if all_edges.min() < 0 or all_edges.max() >= self.E:
            raise ValueError("edges_per_cell has edge indices out of range")
        if not np.array_equal(np.sort(all_edges), np.arange(self.E)):
            raise ValueError("edges_per_cell must partition 0..E-1 exactly")
        rec = np.empty(self.E, dtype=np.int64)
        for c, eidx in enumerate(self.edges_per_cell):
            rec[eidx] = c
        if not np.array_equal(rec, self.cell_id_of_edge):
            raise ValueError("cell_id_of_edge inconsistent with edges_per_cell")

    def _edge_batches(self, eidx):
        for start in range(0, len(eidx), self.B):
            yield eidx[start:start + self.B]

    def _resolve_focus_col(self, focus):
        if isinstance(focus, str):
            if focus not in self.covariate_names_full:
                raise ValueError(f"Unknown focus='{focus}'. Available: {self.covariate_names_full}")
            return self.covariate_names_full.index(focus)
        if isinstance(focus, (int, np.integer)) and 0 <= focus < self.p:
            return int(focus)
        raise ValueError("focus must be a covariate name or a valid column index")

    def _build_design_matrix(self, age, extra_covariates=None, global_covariates=None):
        age = np.asarray(age, dtype=np.float64).reshape(-1)
        N = len(age)
        parts, names = [np.ones((N, 1)), age[:, None]], ["intercept", "age"]

        Zextra = _as_2d(extra_covariates)
        if self.n_extra > 0:
            if Zextra is None or Zextra.shape != (N, self.n_extra):
                raise ValueError(f"extra_covariates must have shape {(N, self.n_extra)}")
            parts.append(Zextra)
            names.extend(self.base_extra_covariate_names)

        if self.interaction_specs:
            lookup = {"intercept": np.ones(N), "age": age}
            if Zextra is not None:
                for j, nm in enumerate(self.base_extra_covariate_names):
                    lookup[nm] = Zextra[:, j]
            for a, b in self.interaction_specs:
                if a not in lookup or b not in lookup:
                    raise ValueError(f"Unknown interaction term ({a}, {b}).")
                parts.append((lookup[a] * lookup[b])[:, None])
                names.append(f"{a}:{b}")

        Zg = None
        if self.has_global:
            Zg = _as_2d(global_covariates)
            if Zg is None or Zg.shape != (N, self.n_global):
                raise ValueError(f"global_covariates must have shape {(N, self.n_global)}")

        return np.column_stack(parts), Zg, names

    def _Y_minus_global(self, Y=None, Zg=None):
        Y = self.Y if Y is None else Y
        Zg = self.Z_global if Zg is None else Zg
        if Zg is not None and self.gamma is not None:
            return Y - (Zg @ self.gamma)[:, None]
        return Y

    # ── initialization ───────────────────────────────────────────────

    def initialize(self):
        if self.verbose:
            print("Initializing...")

        cell_means = np.empty((self.N_obs, self.C))
        for c, eidx in enumerate(self.edges_per_cell):
            cell_means[:, c] = self.Y[:, eidx].mean(axis=1) if len(eidx) else 0.0
        self.alpha = (self._XtX_inv_Xt @ cell_means).T            # (C, p)

        self.eta = np.zeros(self.E)
        for c, eidx in enumerate(self.edges_per_cell):
            if len(eidx):
                self.eta[eidx] = (self.Y[:, eidx] - (self.X @ self.alpha[c])[:, None]).mean(axis=0)
        self._enforce_eta_constraint()

        self.gamma = np.zeros(self.n_global) if self.has_global else None

        self.V_diag = np.empty(self.E)
        for c, eidx in enumerate(self.edges_per_cell):
            if len(eidx):
                r = self.Y[:, eidx] - (self.X @ self.alpha[c])[:, None] - self.eta[eidx][None, :]
                self.V_diag[eidx] = np.maximum(r.var(axis=0), 1e-8)

        self.d = self._initial_d(cell_means.mean(axis=1))

        self.b_mean = np.zeros(self.N_subjects)
        self.b_var = np.zeros(self.N_subjects)

        if self.verbose:
            print(f"  alpha range : [{self.alpha.min():.4f}, {self.alpha.max():.4f}]")
            print(f"  V range     : [{self.V_diag.min():.4f}, {self.V_diag.max():.4f}]")
            print(f"  d init      : {self.d:.6f}")

    def _initial_d(self, gm):
        """
        Starting value for the random-intercept variance from the grand cell
        mean gm_n of each observation.

        "moments" (default): residualize gm on X, then
            d0 = Var(subject mean residual) - mean(sigma2_within / T_m),
        floored at 10% of the between-subject variance. Works for any T and
        any covariates, including time-invariant ones.

        "subject_ols": the notebook version (per-subject OLS intercepts). When
        a subject has fewer observations than columns of X (e.g. T=2 with any
        extra covariate) it falls back to a common value for every subject and
        d0 collapses to 1e-8; EM then has to escape from ~0. Kept only to
        reproduce earlier results.
        """
        if self.d_init == "subject_ols":
            global_int = (self._XtX_inv_Xt @ gm)[0]
            beta_subj = np.empty(self.N_subjects)
            for m, oidx in enumerate(self.obs_per_subject):
                X_m = self.X[oidx]
                if len(oidx) >= self.p and np.linalg.matrix_rank(X_m) == self.p:
                    beta_subj[m] = (_safe_inv(X_m.T @ X_m) @ (X_m.T @ gm[oidx]))[0]
                else:
                    beta_subj[m] = global_int
            return max(float(np.var(beta_subj, ddof=1)) if self.N_subjects >= 2 else 1e-3, 1e-8)

        if self.d_init != "moments":
            raise ValueError("d_init must be 'moments' or 'subject_ols'")
        r = gm - self.X @ (self._XtX_inv_Xt @ gm)
        rbar = np.bincount(self.subject_id, weights=r, minlength=self.N_subjects) / self.T_subj
        between = float(np.var(rbar, ddof=1))
        multi = self.T_subj > 1
        if multi.any():
            dev2 = (r - rbar[self.subject_id]) ** 2
            ss = np.bincount(self.subject_id, weights=dev2, minlength=self.N_subjects)
            sigma2_w = ss[multi].sum() / (self.T_subj[multi] - 1).sum()
            d0 = between - np.mean(sigma2_w / self.T_subj)
        else:
            d0 = between
        return max(d0, 0.1 * between, 1e-8)

    def _enforce_eta_constraint(self):
        for c, eidx in enumerate(self.edges_per_cell):
            if len(eidx) == 0:
                continue
            if self.V_diag is None:
                shift = self.eta[eidx].mean()
            else:
                w = 1.0 / self.V_diag[eidx]
                shift = np.sum(w * self.eta[eidx]) / np.sum(w)
            self.eta[eidx] -= shift
            self.alpha[c, 0] += shift

    # ── precision hooks (overridden by block-V) ───────────────────────

    def _total_w(self):
        """1' V^{-1} 1 over all edges."""
        return (1.0 / self.V_diag).sum()

    def _resid_dot_inv1(self, Y_adj, X):
        """sum_e r_{n,e} [V^{-1} 1]_e for every observation n."""
        s = np.zeros(Y_adj.shape[0])
        inv_v = 1.0 / self.V_diag
        for c, eidx in enumerate(self.edges_per_cell):
            if len(eidx):
                r = Y_adj[:, eidx] - (X @ self.alpha[c])[:, None] - self.eta[eidx][None, :]
                s += r @ inv_v[eidx]
        return s

    # ── E-step ───────────────────────────────────────────────────────

    def e_step(self):
        s_obs = self._resid_dot_inv1(self._Y_minus_global(), self.X)
        S = np.bincount(self.subject_id, weights=s_obs, minlength=self.N_subjects)
        lam = 1.0 / self.d + self.T_subj * self._total_w()
        self.b_var = 1.0 / lam
        self.b_mean = self.b_var * S

    # ── M-step ───────────────────────────────────────────────────────

    def _update_d(self):
        self.d = max((self.b_var.sum() + np.sum(self.b_mean ** 2)) / self.N_subjects, 1e-8)

    def _update_gamma(self):
        """Pooled GLS update for shared covariates (same effect on every edge)."""
        if not self.has_global:
            return
        gam = self.b_mean[self.subject_id]
        s = self._resid_dot_inv1(self.Y - gam[:, None], self.X)
        Z = self.Z_global
        self.gamma = np.linalg.pinv(self._total_w() * (Z.T @ Z)) @ (Z.T @ s)

    def _cell_weights(self, c, eidx):
        """(V^{-1} 1 restricted to the cell, its sum)."""
        w = 1.0 / self.V_diag[eidx]
        return w, w.sum()

    def _update_alpha_eta(self):
        """
        Conditional WLS M-step given posterior mean random intercepts:
        slopes from edge-centered, precision-weighted cell averages; then
        cell intercept = weighted mean of edge intercepts, eta = remainder.
        """
        Y_adj = self._Y_minus_global(self.Y - self.b_mean[self.subject_id][:, None])
        alpha_new = np.zeros((self.C, self.p))
        eta_new = np.zeros(self.E)

        has_slopes = self.p > 1
        if has_slopes:
            X1 = self.X[:, 1:]
            X1_mean = X1.mean(axis=0)
            X1c = X1 - X1_mean
            XtX1_inv = np.linalg.pinv(X1c.T @ X1c)

        for c, eidx in enumerate(self.edges_per_cell):
            if len(eidx) == 0:
                continue
            w, w_sum = self._cell_weights(c, eidx)
            Y_c = Y_adj[:, eidx]
            Y_c_mean = Y_c.mean(axis=0)
            if has_slopes:
                ybar_c = (Y_c - Y_c_mean) @ w / w_sum
                slopes = XtX1_inv @ (X1c.T @ ybar_c)
                edge_int = Y_c_mean - X1_mean @ slopes
            else:
                slopes = np.zeros(0)
                edge_int = Y_c_mean.copy()
            a0 = float(w @ edge_int) / w_sum
            alpha_new[c, 0] = a0
            alpha_new[c, 1:] = slopes
            eta_new[eidx] = edge_int - a0

        self.alpha, self.eta = alpha_new, eta_new

    def _update_V(self):
        """v_e = mean_n E[(y - mu - b)^2 | y] = mean_n [rhat^2 + b_var]."""
        gam = self.b_mean[self.subject_id]
        bvar = self.b_var[self.subject_id]
        Y_adj = self._Y_minus_global()
        V_new = np.empty(self.E)
        for c, eidx in enumerate(self.edges_per_cell):
            for batch in self._edge_batches(eidx):
                r = Y_adj[:, batch] - (self.X @ self.alpha[c])[:, None] \
                    - self.eta[batch][None, :] - gam[:, None]
                V_new[batch] = np.mean(r ** 2 + bvar[:, None], axis=0)
        self.V_diag = np.maximum(V_new, 1e-8)

    # ── log-likelihood ───────────────────────────────────────────────

    def _logdet_V(self):
        return np.log(self.V_diag).sum()

    def _quad_and_lin(self, Y_adj, X):
        """Per observation: r' V^{-1} r and r' V^{-1} 1."""
        inv_v = 1.0 / self.V_diag
        quad = np.zeros(Y_adj.shape[0]); lin = np.zeros(Y_adj.shape[0])
        for c, eidx in enumerate(self.edges_per_cell):
            if len(eidx):
                r = Y_adj[:, eidx] - (X @ self.alpha[c])[:, None] - self.eta[eidx][None, :]
                quad += (r ** 2) @ inv_v[eidx]
                lin += r @ inv_v[eidx]
        return quad, lin

    def _marginal_ll(self, Y_adj, X, subj_idx, M, return_by_subject=False):
        """Exact marginal LL using the rank-one structure of d 11' + I ⊗ V."""
        if not np.isfinite(self.d) or self.d <= 0:
            return -np.inf
        quad, lin = self._quad_and_lin(Y_adj, X)
        q = np.bincount(subj_idx, weights=quad, minlength=M)
        b = np.bincount(subj_idx, weights=lin, minlength=M)
        T = np.bincount(subj_idx, minlength=M).astype(float)
        kappa = 1.0 + self.d * T * self._total_w()
        if np.any(kappa <= 0) or not np.all(np.isfinite(kappa)):
            return -np.inf
        ll = -0.5 * (T * self.E * np.log(2 * np.pi) + T * self._logdet_V()
                     + np.log(kappa) + q - (self.d / kappa) * b ** 2)
        return ll if return_by_subject else float(ll.sum())

    def compute_log_likelihood(self):
        if any(x is None for x in (self.alpha, self.eta, self.d, self.V_diag)):
            raise ValueError("Model parameters are not initialized.")
        return self._marginal_ll(self._Y_minus_global(), self.X, self.subject_id, self.N_subjects)

    def predictive_log_likelihood(self, Y, age, subject_id, extra_covariates=None,
                                  global_covariates=None, return_by_subject=False):
        """Marginal LL of new data (b integrated out) under fitted parameters."""
        Y = np.asarray(Y, dtype=np.float64)
        X, Zg, _ = self._build_design_matrix(age, extra_covariates, global_covariates)
        if Y.shape[0] != X.shape[0] or len(subject_id) != X.shape[0]:
            raise ValueError("Y, age and subject_id must have the same number of rows.")
        _, subj_idx = np.unique(np.asarray(subject_id), return_inverse=True)
        return self._marginal_ll(self._Y_minus_global(Y, Zg), X, subj_idx,
                                 subj_idx.max() + 1, return_by_subject)

    # ── fit ──────────────────────────────────────────────────────────

    def _em_iteration(self):
        self.e_step()
        self._update_d()
        self._update_gamma()
        self._update_alpha_eta()
        self._update_V()

    def fit(self, compute_ll_every=1):
        if self.alpha is None:
            self.initialize()
        self.log_likelihood_history = []
        prev_ll = -np.inf
        t0 = time.time()

        for it in range(self.max_iter):
            self._em_iteration()
            if it % compute_ll_every:
                continue
            ll = self.compute_log_likelihood()
            self.log_likelihood_history.append((it, ll))
            rel = np.inf if prev_ll == -np.inf else abs(ll - prev_ll) / max(abs(ll), 1.0)

            if self.verbose:
                el = time.time() - t0
                eta_s = el / (it + 1) * (self.max_iter - it - 1)
                print(f"  iter {it + 1:3d}/{self.max_iter}  LL={ll:.4f}  rel Δ={rel:.2e}  "
                      f"elapsed {_fmt_time(el)}, ETA ≤ {_fmt_time(eta_s)}")

            if it > 0 and ll < prev_ll - 1e-6:
                if self.verbose:
                    print(f"  Warning: LL decreased at iter {it + 1}: {prev_ll:.4f} -> {ll:.4f}")
            elif it > 0 and rel < self.tol:
                if self.verbose:
                    print(f"  Converged at iter {it + 1}")
                break
            prev_ll = ll
        else:
            if self.verbose:
                print(f"  Reached max_iter={self.max_iter} without meeting tol={self.tol:g}")
        return self

    # ── inference ────────────────────────────────────────────────────

    def _cell_se(self, se_type):
        """SEs of alpha (C, p), using the diagonal edge weights 1/v_e."""
        alpha_se = np.empty((self.C, self.p))
        gam = self.b_mean[self.subject_id]
        Y_adj = self._Y_minus_global()
        for c, eidx in enumerate(self.edges_per_cell):
            if len(eidx) == 0:
                alpha_se[c] = np.inf
                continue
            w = 1.0 / self.V_diag[eidx]
            A_inv = self.XtX_inv / w.sum()
            if se_type == "model":
                var = A_inv
            else:
                r = Y_adj[:, eidx] - (self.X @ self.alpha[c])[:, None] \
                    - self.eta[eidx][None, :] - gam[:, None]
                u = np.zeros((self.N_subjects, self.p))
                np.add.at(u, self.subject_id, self.X * (r @ w)[:, None])
                var = A_inv @ (u.T @ u) @ A_inv
            alpha_se[c] = np.sqrt(np.maximum(np.diag(var), 0.0))
        return alpha_se

    def diagnostics(self, focus="age", cell_labels=None, z_threshold=1.96,
                    adjust_method="BH", alpha=0.05, se_type="clustered", top_k=5):
        """
        Per-cell estimates, SEs, z, raw and multiplicity-adjusted p-values for
        one focus covariate. se_type: "clustered" (subject sandwich) or "model".
        """
        if self.alpha is None:
            raise RuntimeError("Call fit() first")
        if se_type not in ("model", "clustered"):
            raise ValueError("se_type must be 'model' or 'clustered'")

        j = self._resolve_focus_col(focus)
        name = self.covariate_names_full[j]
        n_edges = np.array([len(e) for e in self.edges_per_cell])
        avg_v = np.array([self.V_diag[e].mean() if len(e) else np.nan for e in self.edges_per_cell])

        alpha_se = self._cell_se(se_type)
        alpha_z = self.alpha / np.maximum(alpha_se, 1e-15)
        z = alpha_z[:, j]
        p_raw = z_to_pvals(z)
        p_adj = adjust_pvalues(p_raw, method=adjust_method)
        sig_raw, sig_adj = np.abs(z) > z_threshold, p_adj < alpha

        if self.verbose:
            print(f"\n{'=' * 60}\n{type(self).__name__}: {self.C} cells, {self.E} edges, "
                  f"{self.N_subjects} subjects, {self.N_obs} observations\n{'=' * 60}")
            print(f"  SE type = {se_type!r};  d = {self.d:.6f};  focus = {name!r}")
            print(f"  |z| > {z_threshold:.2f}: {sig_raw.sum()} / {self.C};  "
                  f"{adjust_method} at {alpha:.2f}: {sig_adj.sum()} / {self.C}")
            for idx in np.argsort(np.abs(z))[::-1][:top_k]:
                lbl = cell_labels[idx] if cell_labels is not None else idx
                print(f"  cell {idx:3d} {str(lbl):>35s}: {name}={self.alpha[idx, j]:+.6f} "
                      f"SE={alpha_se[idx, j]:.6f} z={z[idx]:+.2f} p_adj={p_adj[idx]:.3e} "
                      f"({n_edges[idx]} edges)")

        return {
            "alpha": self.alpha, "eta": self.eta, "d": self.d, "V_diag": self.V_diag,
            "alpha_se": alpha_se, "alpha_z": alpha_z, "n_edges": n_edges, "avg_v": avg_v,
            "focus_col": j, "focus_name": name,
            "focus_alpha": self.alpha[:, j], "focus_se": alpha_se[:, j], "focus_z": z,
            "focus_p_raw": p_raw, "focus_p_adj": p_adj,
            "focus_sig_raw": sig_raw, "focus_sig_adj": sig_adj,
            "covariate_names": self.covariate_names_full,
            "adjust_method": adjust_method, "alpha_level": alpha, "se_type": se_type,
            "gamma": self.gamma, "global_covariate_names": self.global_covariate_names,
        }


class GraphAwareBlockVEM(GraphAwareRandomInterceptEM):
    """
    Random intercept + block-diagonal V with one dense (E_c x E_c) block per cell.

    Blocks larger than the number of observations are shrunk toward their
    diagonal mean in the M-step (see `_update_V`).
    """

    def __init__(self, *args, shrink_if_needed=True, **kwargs):
        # Accept and ignore legacy arguments from the notebook API
        for legacy in ("edge_to_roi", "roi_to_network", "cells"):
            kwargs.pop(legacy, None)
        super().__init__(*args, **kwargs)
        self.shrink_if_needed = shrink_if_needed
        self.V_blocks = None

    def _precompute_block_quantities(self):
        self._Vc_inv, self._Vc_inv1, self._Vc_w, self._Vc_logdet = [], [], [], []
        for c, eidx in enumerate(self.edges_per_cell):
            if len(eidx) == 0:
                self._Vc_inv.append(None); self._Vc_inv1.append(None)
                self._Vc_w.append(0.0); self._Vc_logdet.append(0.0)
                continue
            Vb = self.V_blocks[c]
            n = len(eidx)
            try:
                L, low = cho_factor(Vb)
                inv = cho_solve((L, low), np.eye(n))
                ld = 2.0 * np.log(np.diag(L)).sum()
            except np.linalg.LinAlgError:
                Vr = Vb + 1e-8 * np.eye(n)
                inv = np.linalg.inv(Vr)
                ld = np.linalg.slogdet(Vr)[1]
            inv1 = inv @ np.ones(n)
            self._Vc_inv.append(inv); self._Vc_inv1.append(inv1)
            self._Vc_w.append(float(inv1.sum())); self._Vc_logdet.append(float(ld))

    def initialize(self):
        super().initialize()
        self.V_blocks = [np.diag(self.V_diag[e]) if len(e) else None for e in self.edges_per_cell]
        self._precompute_block_quantities()

    def _total_w(self):
        return float(sum(self._Vc_w))

    def _logdet_V(self):
        return float(sum(self._Vc_logdet))

    def _resid_dot_inv1(self, Y_adj, X):
        s = np.zeros(Y_adj.shape[0])
        for c, eidx in enumerate(self.edges_per_cell):
            if len(eidx):
                r = Y_adj[:, eidx] - (X @ self.alpha[c])[:, None] - self.eta[eidx][None, :]
                s += r @ self._Vc_inv1[c]
        return s

    def _cell_weights(self, c, eidx):
        return self._Vc_inv1[c], self._Vc_w[c]

    def _quad_and_lin(self, Y_adj, X):
        quad = np.zeros(Y_adj.shape[0]); lin = np.zeros(Y_adj.shape[0])
        for c, eidx in enumerate(self.edges_per_cell):
            if len(eidx):
                r = Y_adj[:, eidx] - (X @ self.alpha[c])[:, None] - self.eta[eidx][None, :]
                quad += ((r @ self._Vc_inv[c]) * r).sum(axis=1)
                lin += r @ self._Vc_inv1[c]
        return quad, lin

    def _update_V(self):
        """
        V_c = (1/N_obs) sum_n [rhat_n rhat_n' + b_var_n 11'], then, if
        E_c > N_obs, shrink toward its diagonal mean with
        rho = min(0.9, E_c/N_obs - 1). A 1e-8 ridge is always added.
        """
        gam = self.b_mean[self.subject_id]
        bvar_mean = self.b_var[self.subject_id].mean()
        Y_adj = self._Y_minus_global()
        V_diag_new = np.empty(self.E)
        blocks = []
        for c, eidx in enumerate(self.edges_per_cell):
            n = len(eidx)
            if n == 0:
                blocks.append(None)
                continue
            r = Y_adj[:, eidx] - (self.X @ self.alpha[c])[:, None] \
                - self.eta[eidx][None, :] - gam[:, None]
            Vb = (r.T @ r) / self.N_obs + bvar_mean
            if self.shrink_if_needed and n > self.N_obs:
                rho = min(0.9, n / self.N_obs - 1.0)
                Vb = (1.0 - rho) * Vb + rho * float(np.mean(np.diag(Vb))) * np.eye(n)
            Vb += 1e-8 * np.eye(n)
            blocks.append(Vb)
            V_diag_new[eidx] = np.diag(Vb)
        self.V_blocks = blocks
        self.V_diag = np.maximum(V_diag_new, 1e-8)
        self._precompute_block_quantities()

    def diagnostics(self, *args, se_type="clustered", **kwargs):
        # Matches the notebook version: block-V point estimates, subject-clustered
        # sandwich SEs built with diagonal weights 1/v_e (see README, "Known issues").
        if se_type != "clustered":
            raise ValueError("GraphAwareBlockVEM supports se_type='clustered' only")
        return super().diagnostics(*args, se_type="clustered", **kwargs)

    def block_summary(self):
        sizes = np.array([len(e) for e in self.edges_per_cell])
        print(f"{self.C} blocks; sizes min={sizes.min()} median={int(np.median(sizes))} "
              f"max={sizes.max()}; {int((sizes > self.N_obs).sum())} blocks with E_c > N_obs (shrunk)")
