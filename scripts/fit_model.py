"""Fit the graph-aware model on any number of waves, driven by a YAML config.

    python scripts/fit_model.py config.yaml

Writes only cell-level summaries (no subject-level data) to output_dir.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import graph_aware as G


def load_waves(cfg):
    waves, names = [], []
    for w in cfg["waves"]:
        f = pd.read_csv(w["features"])
        Y = np.load(w["connectomes"], mmap_mode="r")
        waves.append({"features": f, "connectomes": Y})
        names.append(w["name"])
    return waves, names


def build_inputs(P, mcfg):
    age = P.centered(mcfg.get("age_col", "age"))
    extra_cols = mcfg.get("extra_covariates", []) or []
    extra = P.matrix(extra_cols) if extra_cols else None
    if extra is not None and mcfg.get("center_extra", True):
        extra = extra - extra.mean(axis=0)

    g_parts, g_names = [], []
    for col in mcfg.get("global_categorical", []) or []:
        M, nm = P.dummies(col)
        g_parts.append(M); g_names += nm
    num = mcfg.get("global_numeric", []) or []
    if num:
        g_parts.append(P.matrix(num)); g_names += list(num)
    Zg = np.column_stack(g_parts) if g_parts else None
    if Zg is not None:
        Zg = Zg - Zg.mean(axis=0)      # keeps the cell intercepts interpretable
    return age, extra, extra_cols, Zg, g_names


def parse_interactions(mcfg, extra_cols):
    """Accept [[a, b], ...] or ["a:b", ...]; terms must be 'age' or extra covariates."""
    allowed = ["age"] + list(extra_cols)
    specs = []
    for item in mcfg.get("interactions", []) or []:
        pair = item.split(":") if isinstance(item, str) else list(item)
        if len(pair) != 2:
            raise ValueError(f"interaction {item!r}: give exactly two terms")
        a, b = (str(x).strip() for x in pair)
        a = "age" if a == mcfg.get("age_col", "age") else a
        b = "age" if b == mcfg.get("age_col", "age") else b
        bad = [x for x in (a, b) if x not in allowed]
        if bad:
            raise ValueError(f"interaction {item!r}: {bad} not in age or extra_covariates "
                             f"(global covariates cannot enter interactions)")
        if a == b:
            raise ValueError(f"interaction {item!r}: use a squared column instead")
        if (a, b) in specs or (b, a) in specs:
            raise ValueError(f"interaction {item!r} listed twice")
        specs.append((a, b))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(cfg.get("output_dir", "results")); out.mkdir(parents=True, exist_ok=True)

    waves, names = load_waves(cfg)
    P = G.stack_waves(waves, subject_col=cfg.get("subject_col", "Subject"), wave_names=names,
                      min_waves=cfg.get("min_waves", 2),
                      time_invariant=cfg.get("time_invariant", []))

    regions = pd.read_csv(cfg["parcellation"])
    ncol, mcol = cfg.get("network_col", "NetworkNumber"), cfg.get("name_col", "Community")
    cs = G.build_cells_from_gordon(regions, P.Y.shape[1], network_col=ncol)
    labels = G.cell_labels_from_gordon(regions, cs, ncol, mcol)

    mcfg = cfg.get("model", {})
    age, extra, extra_cols, Zg, g_names = build_inputs(P, mcfg)
    interactions = parse_interactions(mcfg, extra_cols)
    cls = {"blockv": G.GraphAwareBlockVEM, "diag": G.GraphAwareRandomInterceptEM}[mcfg.get("type", "blockv")]
    model = cls(Y=P.Y, age=age, subject_id=P.subject_id,
                cell_id_of_edge=cs.cell_id_of_edge, edges_per_cell=cs.edges_per_cell,
                extra_covariates=extra, covariate_names=extra_cols or None,
                global_covariates=Zg, global_covariate_names=g_names or None,
                interaction_specs=interactions,
                max_iter=mcfg.get("max_iter", 50), tol=float(mcfg.get("tol", 1e-6)))
    model.fit()

    icfg = cfg.get("inference", {})
    focus_list = icfg.get("focus", ["age"])
    if focus_list == "all":
        focus_list = model.covariate_names_full[1:]
    for focus in focus_list:
        if isinstance(focus, (list, tuple)):          # [age, g] -> "age:g"
            focus = ":".join(focus)
        d = model.diagnostics(focus=focus, cell_labels=labels,
                              adjust_method=icfg.get("adjust_method", "BH"),
                              alpha=icfg.get("alpha", 0.05))
        pd.DataFrame({
            "cell": np.arange(cs.C),
            "system_a": [a for a, _ in labels], "system_b": [b for _, b in labels],
            "n_edges": d["n_edges"], "estimate": d["focus_alpha"], "se": d["focus_se"],
            "z": d["focus_z"], "p": d["focus_p_raw"], "p_adj": d["focus_p_adj"],
            "significant": d["focus_sig_adj"],
        }).to_csv(out / f"cells_{focus.replace(':', '_x_')}.csv", index=False)

    summary = {"model": cls.__name__, "panel": P.report, "d": model.d,
               "covariates": model.covariate_names_full,
               "global_covariates": dict(zip(g_names, (model.gamma.tolist() if model.gamma is not None else []))),
               "log_likelihood": model.log_likelihood_history[-1][1],
               "n_iter": len(model.log_likelihood_history)}
    json.dump(summary, open(out / "fit_summary.json", "w"), indent=2, default=int)
    print(f"Results written to {out}/")


if __name__ == "__main__":
    main()
