"""p-values and multiple-testing correction."""
import numpy as np
from scipy.stats import norm


def z_to_pvals(zvals, two_sided=True):
    zvals = np.asarray(zvals, dtype=float)
    return 2.0 * norm.sf(np.abs(zvals)) if two_sided else norm.sf(zvals)


def adjust_pvalues(pvals, method="BH"):
    """Adjusted p-values. method: 'BH', 'BY' (arbitrary dependence) or 'holm'."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    if m == 0:
        return np.array([], dtype=float)
    if np.any(~np.isfinite(pvals)) or np.any((pvals < 0) | (pvals > 1)):
        raise ValueError("pvals must be finite and in [0, 1].")

    order = np.argsort(pvals)
    p_sorted = pvals[order]
    ranks = np.arange(1, m + 1, dtype=float)
    method = method.lower()

    if method in ("bh", "by"):
        c_m = np.sum(1.0 / ranks) if method == "by" else 1.0
        adj = p_sorted * m * c_m / ranks
        adj = np.minimum.accumulate(adj[::-1])[::-1]
    elif method == "holm":
        adj = np.maximum.accumulate(p_sorted * (m - np.arange(m)))
    else:
        raise ValueError("method must be one of {'BH', 'BY', 'holm'}")

    out = np.empty(m)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out


def testing_summary_from_z(zvals, alpha=0.05, method="BH", two_sided=True):
    zvals = np.asarray(zvals, dtype=float)
    p_raw = z_to_pvals(zvals, two_sided=two_sided)
    p_adj = adjust_pvalues(p_raw, method=method)
    return {"z": zvals, "p_raw": p_raw, "p_adj": p_adj,
            "sig_raw": p_raw < alpha, "sig_adj": p_adj < alpha}
