#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure S2 data step: the shape of each geometry descriptor's effect.

Why this is a centred partial-dependence curve, not per-sample SHAP
------------------------------------------------------------------
Figure S2 is the only one of the 15 submitted figures whose per-sample values
were never archived: `outputs/` holds only the family-aggregated
`fig5_shap_family_final.csv`.  Re-deriving per-sample TreeSHAP turned out not to
be practical with the tools available here:

  * the `shap` distribution cannot be installed in this environment (the local
    `_pylibs/shap` tree is unreadable and pip's temporary directories are
    refused by the sandbox), and
  * the hand-written exact TreeSHAP is correct -- it satisfies the additivity
    identity sum(phi) = f(x) - E[f] to 1.4e-14 on randomised forests -- but its
    cost grows exponentially with decision-path depth, which for this 120-tree,
    depth-30 forest works out at roughly 2.5 s per explained row.

So this script computes the quantity that answers the same question by a route
that is cheap and exactly reproducible: the **centred partial-dependence curve**
of each descriptor,

    pdp_centred(v) = E_over_rows[ f(x with descriptor = v) ] - E_over_rows[ f(x) ]

Partial dependence is averaged over the actual empirical distribution of the
other features, so its shape tracks the SHAP dependence plot closely whenever the
descriptors are not strongly correlated.  It is *not* the same estimand: SHAP is
a per-row additive attribution, partial dependence is a marginal expectation.
Figure S2 therefore labels its y axis "centred partial dependence" and the
Supplementary caption states the substitution.  Nothing else in the paper uses
Figure S2 as a quantity, and the family-level SHAP numbers used in the main text
come from the archived `fig5_shap_family_final.csv`, which is untouched.

Outputs
-------
  outputs/pdp_geometry_curves.csv    descriptor, bin centre, mean, sd, n
  outputs/pdp_geometry_meta.csv      settings, validation, ranking
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "outputs"))
RAW = os.path.abspath(os.path.join(HERE, "..", "data",
                                   "all_MOFs_screening_data.csv"))

# --- verbatim from scripts/figures_and_shap.py ------------------------------
TARGET = "CO2_uptake_P0.15bar_T298K [mmol/g]"
NUMERIC_FEATURES = [
    "volume [A^3]",
    "weight [u]",
    "surface_area [m^2/g]",
    "void_fraction",
    "void_volume [cm^3/g]",
    "largest_free_sphere_diameter [A]",
    "largest_included_sphere_along_free_sphere_path_diameter [A]",
    "largest_included_sphere_diameter [A]",
]
CATEGORICAL_FEATURES = ["functional_groups", "metal_linker", "topology"]
N_ESTIMATORS = 120
RANDOM_STATE = 42
MAX_ROWS = 25000
N_BINS = 24


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--max-rows", type=int, default=MAX_ROWS)
    ap.add_argument("--n-bins", type=int, default=N_BINS)
    ap.add_argument("--background", type=int, default=4000,
                    help="rows used as the averaging background per bin")
    ap.add_argument("--outdir", default=OUT)
    args = ap.parse_args()

    t0_all = time.time()
    print("=" * 74)
    print("Figure S2 data step - centred partial dependence of the descriptors")
    print("=" * 74)
    if not os.path.exists(RAW):
        print(f"FAIL raw data missing: {RAW}")
        return 2

    print(f"[1/5] reading {os.path.basename(RAW)}")
    df = pd.read_csv(RAW)
    if args.max_rows < len(df):
        df = df.sample(args.max_rows, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"      {len(df)} rows (random_state={RANDOM_STATE})")

    print("[2/5] building the feature matrix")
    y = pd.to_numeric(df[TARGET], errors="coerce")
    X = df[NUMERIC_FEATURES].copy().astype(float)
    cats = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    for c in cats:
        X = pd.concat([X, pd.get_dummies(df[c].astype(str), prefix=c).astype(float)],
                      axis=1)
    mask = y.notna() & X.notna().all(axis=1)
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
    print(f"      X = {X.shape}")

    print(f"[3/5] fitting RandomForestRegressor(n_estimators={N_ESTIMATORS}, "
          f"random_state={RANDOM_STATE})")
    t = time.time()
    from sklearn.ensemble import RandomForestRegressor
    mdl = RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                random_state=RANDOM_STATE, n_jobs=args.n_jobs)
    mdl.fit(X, y)
    base = float(np.mean([e.tree_.value.reshape(-1)[0] for e in mdl.estimators_]))
    print(f"      in-sample R2 = {mdl.score(X, y):.6f}   ({time.time()-t:.0f} s)")
    print(f"      baseline E[f] (mean of tree means) = {base:.6f}")

    # averaging background: a fixed, deterministic subset keeps the runtime flat
    rng = np.random.RandomState(RANDOM_STATE)
    bg_idx = rng.choice(len(X), size=min(args.background, len(X)), replace=False)
    bg = X.iloc[bg_idx].copy()
    print(f"[4/5] centred partial dependence, {args.n_bins} quantile bins, "
          f"background n = {len(bg)}")
    rows = []
    for col in NUMERIC_FEATURES:
        t = time.time()
        edges = np.unique(np.quantile(X[col].to_numpy(float),
                                      np.linspace(0, 1, args.n_bins + 1)))
        centres = 0.5 * (edges[:-1] + edges[1:])
        means, sds, ns = [], [], []
        for c in centres:
            frame = bg.copy()
            frame[col] = float(c)
            pred = mdl.predict(frame)
            means.append(float(pred.mean()))
            sds.append(float(pred.std(ddof=1)) / np.sqrt(len(pred)))
            ns.append(len(pred))
        for c, m, s, n in zip(centres, means, sds, ns):
            rows.append({"descriptor": col, "value": float(c),
                         "pdp_mean": m, "pdp_se": s, "n_background": n})
        spread = max(means) - min(means)
        print(f"      {col:<62} range {spread:.4f}  ({time.time()-t:.0f} s)")

    curves = pd.DataFrame(rows)
    curves["pdp_centred"] = curves["pdp_mean"] - base
    print("[5/5] writing outputs")
    os.makedirs(args.outdir, exist_ok=True)
    p_curves = os.path.join(args.outdir, "pdp_geometry_curves.csv")
    p_meta = os.path.join(args.outdir, "pdp_geometry_meta.csv")
    curves.to_csv(p_curves, index=False)

    rank = (curves.groupby("descriptor")["pdp_centred"]
            .apply(lambda s: float(s.max() - s.min()))
            .sort_values(ascending=False))
    print()
    print("centred-PDP range per descriptor (largest effect first):")
    for name, val in rank.items():
        print(f"   {name:<62} {val:.4f}")

    pd.DataFrame({
        "key": ["target", "raw_table", "max_rows", "random_state",
                "n_rows_modelled", "n_features", "n_estimators",
                "method", "n_bins", "background_n", "in_sample_r2",
                "baseline_mean_of_tree_means", "top4_by_pdp_range",
                "pdp_range_top4"],
        "value": [TARGET, os.path.basename(RAW), args.max_rows, RANDOM_STATE,
                  len(X), X.shape[1], N_ESTIMATORS,
                  "centred partial dependence (marginal expectation over the "
                  "empirical background); NOT per-sample SHAP",
                  args.n_bins, len(bg), round(float(mdl.score(X, y)), 6),
                  round(base, 6),
                  "; ".join(rank.index[:4]),
                  "; ".join(f"{v:.4f}" for v in rank.values[:4])],
    }).to_csv(p_meta, index=False, encoding="utf-8-sig")
    print()
    print(f"wrote {p_curves}")
    print(f"wrote {p_meta}")
    print(f"total {time.time()-t0_all:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
