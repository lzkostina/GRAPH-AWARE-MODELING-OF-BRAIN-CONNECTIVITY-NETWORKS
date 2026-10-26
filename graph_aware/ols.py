"""Pooled cell-wise OLS with naive and subject-clustered SEs."""
import numpy as np


def cell_wise_ols_with_se(Y, X, subject_id, cell_id_of_edge):
    """
    Y : (N_obs, E) stacked long-format connectomes (any number of waves)
    X : (N_obs, p) design, e.g. [1, centered age]
    subject_id : (N_obs,) integer 0..N_subjects-1
    cell_id_of_edge : (E,) integer 0..C-1

    Returns dict with 'beta', 'se_naive', 'se_clust', each (C, p).
    """
    N_obs, E = Y.shape
    p = X.shape[1]
    N_subjects = int(subject_id.max()) + 1
    C = int(cell_id_of_edge.max()) + 1

    beta = np.zeros((C, p)); se_naive = np.zeros((C, p)); se_clust = np.zeros((C, p))
    XtX_inv = np.linalg.inv(X.T @ X)

    for c in range(C):
        mask = cell_id_of_edge == c
        E_c = int(mask.sum())
        if E_c == 0:
            continue
        Y_c = Y[:, mask]
        A_inv = XtX_inv / E_c
        beta[c] = A_inv @ (X.T @ Y_c.sum(axis=1))
        resid = Y_c - (X @ beta[c])[:, None]

        sigma2 = float(np.sum(resid ** 2)) / (N_obs * E_c - p)
        se_naive[c] = np.sqrt(np.diag(sigma2 * A_inv))

        scores = X * resid.sum(axis=1, keepdims=True)
        u = np.zeros((N_subjects, p))
        np.add.at(u, subject_id, scores)
        se_clust[c] = np.sqrt(np.diag(A_inv @ (u.T @ u) @ A_inv))

    return {"beta": beta, "se_naive": se_naive, "se_clust": se_clust}
