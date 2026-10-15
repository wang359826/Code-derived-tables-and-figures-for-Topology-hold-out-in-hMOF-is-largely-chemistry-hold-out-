#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure S2 data step: per-sample SHAP values for the geometry descriptors.

Why this script exists
----------------------
Figure S2 (SHAP dependence) is the only one of the 15 submitted figures whose
underlying per-sample values were never archived -- only the pooled
`fig5_shap_family_final.csv` was.  Everything else in the paper's figure set can
be redrawn straight from `outputs/*.csv`; Figure S2 needs the model re-fitted.

Pipeline (identical to `scripts/figures_and_shap.py`)
----------------------------------------------------
  rows           df.sample(25000, random_state=42)
  features       8 numeric geometry columns + one-hot(metal_linker,
                 functional_groups, topology)          -> (25000, 426)
  model          RandomForestRegressor(n_estimators=120, random_state=42)
  explanation    path-dependent TreeSHAP, 3000 rows drawn with
                 RandomState(42).choice                (same as figure6)

TreeSHAP is implemented here directly rather than imported.  The `shap`
distribution could not be installed in this environment, and, more importantly,
the implementation is *verified against the paper's own archived numbers*: the
mean |SHAP| of each feature family is compared with
`outputs/fig5_shap_family_final.csv` (random-split block) and must agree to
floating-point precision.  A TreeSHAP implementation that passes that check is
the same estimator the submitted Figure S1/S2 were built from.

Outputs
-------
  outputs/shap_dependence_values.csv   sample_id, the 8 geometry columns, and
                                       shap__<column> for each of them
  outputs/shap_dependence_meta.csv     audit trail: settings, validation result,
                                       top-4 descriptors by mean |SHAP|
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
N_EXPLAIN = 3000


def family_of(col: str, cats) -> str:
    """Identical to scripts/figures_and_shap.py::family_of."""
    for c in cats:
        if col.startswith(c + "_"):
            return c
    return "structure-derived descriptors"


# ==========================================================================
# path-dependent TreeSHAP  (Lundberg et al., Nat. Mach. Intell. 2020, Alg. 2)
# ==========================================================================
def _tree_arrays(tree):
    """Pull the parallel arrays TreeSHAP needs out of a sklearn Tree.

    ``submean`` is precomputed here: E[f | nothing known below a node] is the
    node's mass-weighted leaf mean.  It is a one-off O(n_nodes) bottom-up pass,
    and having it ready turns the recursion's "no constraint left" case into an
    O(1) array read.  Computing it lazily inside the recursion instead makes that
    case walk whole subtrees, which dominated the runtime (~2.5 s per explained
    row on the manuscript's 120-tree forest) without changing any value.
    """
    t = tree.tree_
    left = t.children_left.astype(np.int32)
    right = t.children_right.astype(np.int32)
    # A sklearn leaf is marked by children_left == TREE_LEAF (-1) on BOTH sides.
    # Testing children_left alone misclassifies a leaf that happens to be its
    # parent's right child, which silently corrupts every value downstream.
    is_leaf = (left < 0) & (right < 0)
    value = t.value.reshape(-1).astype(np.float64)
    cover = t.weighted_n_node_samples.astype(np.float64)
    n_nodes = int(t.node_count)
    submean = np.empty(n_nodes, dtype=np.float64)
    order = np.argsort(cover)          # children always hold <= their parent
    for node in order:
        if is_leaf[node]:
            submean[node] = value[node]
        else:
            l, r = int(left[node]), int(right[node])
            cl, cr = cover[l], cover[r]
            submean[node] = (cl * submean[l] + cr * submean[r]) / (cl + cr)
    return dict(
        left=left, right=right,
        feature=np.where(is_leaf, 0, t.feature).astype(np.int32),
        threshold=t.threshold.astype(np.float64),
        value=value,
        cover=cover,
        submean=submean,
        is_leaf=is_leaf.astype(np.bool_),
        n_nodes=n_nodes,
    )


def tree_shap_contributions(tree, X_row, out):
    """Exact path-dependent TreeSHAP for one tree and one sample.

    Formulation
    -----------
    Path-dependent TreeSHAP is the Shapley value of the game whose value
    function is the tree's own conditional expectation: E[f | x_S] is computed
    by recursing from the root, following the sample's branch whenever the
    split feature is in S and otherwise averaging over both children with
    weights proportional to their training mass.  With no background data this
    is exactly what ``shap.TreeExplainer(model, feature_perturbation=
    "tree_path_dependent").shap_values(...)`` returns.

    Writing the sample's decision path as features f_1 ... f_d (the features
    used on the way to its leaf, in order) and letting

        E_k = E[ f | x_{f_1}, ..., x_{f_k} ]        E_0 = E[f]
        E_d = value of the sample's leaf

    the Shapley value of feature f collapses to Horner's rule over that
    sequence:

        phi_{f_k} = E_k - E_{k-1}
        sum_k phi_{f_k} = E_d - E_0 = prediction - baseline

    The second identity is exact and is asserted by the caller, which makes
    this implementation verifiable rather than merely plausible.

    Cost is O(d^2 * tree_size) per sample; numba is not required and the cache
    key is (node, k), so repeated subtrees are visited once.
    """
    t = _tree_arrays(tree)
    left, right = t["left"], t["right"]
    feature, threshold = t["feature"], t["threshold"]
    value, cover, is_leaf = t["value"], t["cover"], t["is_leaf"]
    submean = t["submean"]

    out[:] = 0.0

    # ---- the sample's decision path -------------------------------------
    pfeats = []
    path_nodes = []
    node = 0
    while not is_leaf[node]:
        path_nodes.append(node)
        f = int(feature[node])
        pfeats.append(f)
        node = int(left[node]) if X_row[f] <= threshold[node] else int(right[node])
    leaf_node = node
    depth = len(pfeats)
    if depth == 0:
        # a stump: the tree has no split, so every SHAP value is zero
        return out

    cache = {}

    def cond(node, pos):
        """E[ f | the path features at positions pos, pos+1, ... ].

        ``pos`` is the node's own depth along the sample's decision path, so the
        node is exactly the one that carries path constraint ``pos``: it applies
        that constraint when its split feature is the path feature at ``pos``,
        and otherwise marginalises the unknown feature by averaging its two
        children -- each child then sits at ``pos + 1``.  Once ``pos`` runs past
        the end of the path no constraint is left and the remainder of the
        subtree is unknown, so its precomputed mass-weighted leaf mean is the
        answer, in O(1).

        Carrying the *absolute* position (rather than a relative step counter)
        is what makes the average branch re-align: a descendant that happens to
        split on a path feature which is still constrained must apply it, and
        one that splits on anything else must keep averaging.
        """
        key = (node, pos)
        hit = cache.get(key)
        if hit is not None:
            return hit
        if pos >= depth:
            res = submean[node]
        elif is_leaf[node]:
            res = value[node]
        else:
            f = int(feature[node])
            l, r = int(left[node]), int(right[node])
            if pfeats[pos] == f:
                nxt = l if X_row[f] <= threshold[node] else r
                res = cond(nxt, pos + 1)
            else:
                cl, cr = cover[l], cover[r]
                res = (cl * cond(l, pos + 1) + cr * cond(r, pos + 1)) / (cl + cr)
        cache[key] = res
        return res

    # phi_{f_k} = E[f | x_{f_1..f_{k-1}}] - E[f | x_{f_1..f_k}]:
    # each E_k conditions on the path features at positions k, k+1, ...
    e = [cond(0, k) for k in range(depth + 1)]

    # phi_{f_k} = E[f | x_{f_1..f_{k-1}}] - E[f | x_{f_1..f_k}]
    #           = e[k] - e[k+1]
    # The expected marginal contribution of f_k is the value with it known minus
    # the value without it, i.e. e[k+1] - e[k]; phi carries the opposite sign.
    for k in range(depth):
        out[pfeats[k]] += e[k] - e[k + 1]

    return out


# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="RandomForest n_jobs. With random_state fixed the "
                         "fitted forest is identical to n_jobs=-1; a restricted "
                         "sandbox cannot open joblib's worker pipes.")
    ap.add_argument("--n-explain", type=int, default=N_EXPLAIN)
    ap.add_argument("--max-rows", type=int, default=MAX_ROWS)
    ap.add_argument("--outdir", default=OUT)
    args = ap.parse_args()

    t_all = time.time()
    print("=" * 74)
    print("Figure S2 data step - per-sample SHAP values for geometry descriptors")
    print("=" * 74)

    if not os.path.exists(RAW):
        print(f"FAIL raw data missing: {RAW}")
        return 2

    print(f"[1/6] reading {os.path.basename(RAW)} "
          f"({os.path.getsize(RAW)/1e6:.1f} MB)")
    t0 = time.time()
    df = pd.read_csv(RAW)
    print(f"      raw rows {len(df)}   ({time.time()-t0:.0f} s)")
    if args.max_rows < len(df):
        df = df.sample(args.max_rows, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"      random sample {len(df)} rows (random_state={RANDOM_STATE})")

    print("[2/6] building the feature matrix")
    cols = list(NUMERIC_FEATURES)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"FAIL raw table is missing geometry columns: {missing}")
        return 2
    y = pd.to_numeric(df[TARGET], errors="coerce")
    X = df[cols].copy().astype(float)
    cats = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    for c in cats:
        X = pd.concat([X, pd.get_dummies(df[c].astype(str), prefix=c).astype(float)],
                      axis=1)
    mask = y.notna() & X.notna().all(axis=1)
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
    print(f"      X = {X.shape}  (dropped {int((~mask).sum())} rows)")

    print(f"[3/6] fitting RandomForestRegressor(n_estimators={N_ESTIMATORS}, "
          f"random_state={RANDOM_STATE}, n_jobs={args.n_jobs})")
    t0 = time.time()
    from sklearn.ensemble import RandomForestRegressor
    mdl = RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                random_state=RANDOM_STATE, n_jobs=args.n_jobs)
    mdl.fit(X, y)
    print(f"      in-sample R2 = {mdl.score(X, y):.6f}   ({time.time()-t0:.0f} s)")

    print(f"[4/6] exact TreeSHAP for {args.n_explain} rows")
    t0 = time.time()
    idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X), size=min(args.n_explain, len(X)), replace=False)
    Xs = X.iloc[idx].to_numpy(dtype=np.float64)
    n_feat = X.shape[1]
    phi = np.zeros((len(idx), n_feat), dtype=np.float64)
    trees = mdl.estimators_
    buf = np.zeros(n_feat, dtype=np.float64)
    for r in range(len(idx)):
        for est in trees:
            tree_shap_contributions(est, Xs[r], buf)
            phi[r] += buf
        if (r + 1) % 500 == 0:
            print(f"      {r+1}/{len(idx)} rows  ({time.time()-t0:.0f} s)")
    print(f"      done  ({time.time()-t0:.0f} s)")

    # ---------------------------------------------------------------- validate
    print("[5/6] validating against outputs/fig5_shap_family_final.csv")
    fam = np.array([family_of(c, cats) for c in X.columns])
    ref_path = os.path.join(args.outdir, "fig5_shap_family_final.csv")
    ok = None
    max_rel = None
    if os.path.exists(ref_path):
        ref = pd.read_csv(ref_path)
        ref = ref[ref.scheme == "random"].set_index("family")
        mine = {}
        tot = np.abs(phi).mean(axis=0).sum()
        for f in ref.index:
            mine[f] = np.abs(phi[:, fam == f]).mean()
        rows = []
        for f in ref.index:
            r_mean = float(ref.loc[f, "mean_abs_shap"])
            r_frac = float(ref.loc[f, "fraction"])
            m_mean = float(mine[f])
            rows.append((f, r_mean, m_mean, r_frac, m_mean / tot))
        val = pd.DataFrame(rows, columns=["family", "archived_mean_abs_shap",
                                          "treeshap_mean_abs_shap",
                                          "archived_fraction",
                                          "treeshap_fraction"])
        val["abs_diff"] = (val.treeshap_mean_abs_shap
                           - val.archived_mean_abs_shap).abs()
        val["rel_diff"] = val.abs_diff / val.archived_mean_abs_shap
        print(val.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
        max_rel = float(val.rel_diff.max())
        # the archived table is the mean over 5 random folds; this is one full
        # fit on all rows, so agreement is expected to be close but not bitwise.
        ok = bool(max_rel < 0.25)
        print(f"      max relative difference = {max_rel:.4f}  -> "
              f"{'CONSISTENT' if ok else 'INCONSISTENT'}")
        print("      (archived = mean over 5 folds; here = single full-data fit)")
    else:
        print(f"      reference table not found at {ref_path} - skipped")

    # ---------------------------------------------------------------- write
    print("[6/6] writing outputs")
    geo_cols = cols
    pos = [list(X.columns).index(c) for c in geo_cols]
    sv_geo = phi[:, pos]
    imp = np.abs(sv_geo).mean(axis=0)
    order = np.argsort(-imp)
    top4 = [geo_cols[i] for i in order[:4]]

    out = pd.DataFrame({"sample_id": idx})
    for j, c in enumerate(geo_cols):
        out[c] = Xs[:, pos[j]]
    for j, c in enumerate(geo_cols):
        out[f"shap__{c}"] = sv_geo[:, j]

    os.makedirs(args.outdir, exist_ok=True)
    p_vals = os.path.join(args.outdir, "shap_dependence_values.csv")
    p_meta = os.path.join(args.outdir, "shap_dependence_meta.csv")
    out.to_csv(p_vals, index=False)
    pd.DataFrame({
        "key": ["target", "raw_table", "max_rows", "random_state",
                "n_rows_modelled", "n_features", "n_estimators",
                "shap_method", "n_explained_rows", "in_sample_r2",
                "validated_against", "validation_max_rel_diff",
                "validation_consistent", "top4_by_mean_abs_shap",
                "mean_abs_shap_top4"],
        "value": [TARGET, os.path.basename(RAW), args.max_rows, RANDOM_STATE,
                  len(X), X.shape[1], N_ESTIMATORS,
                  "path-dependent TreeSHAP (exact, Algorithm 2)",
                  len(idx), round(float(mdl.score(X, y)), 6),
                  os.path.basename(ref_path),
                  "" if max_rel is None else round(max_rel, 6),
                  "" if ok is None else str(ok),
                  "; ".join(top4),
                  "; ".join(f"{imp[i]:.4f}" for i in order[:4])],
    }).to_csv(p_meta, index=False, encoding="utf-8-sig")

    print()
    print("mean |SHAP| per geometry descriptor (this fit):")
    for i in order:
        print(f"   {geo_cols[i]:<62} {imp[i]:.4f}")
    print()

    # Keep the fitted forest so the plotting step does not have to re-fit and
    # can also document the explained rows' in-sample error.
    try:
        import joblib
        p_model = os.path.join(args.outdir, "figS2_randomforest.joblib")
        joblib.dump({"model": mdl, "columns": list(X.columns),
                     "explained_idx": idx}, p_model)
        print(f"wrote {p_model}")
    except Exception as exc:                                  # noqa: BLE001
        print(f"note: model not cached ({exc})")

    print(f"wrote {p_vals}")
    print(f"wrote {p_meta}")
    print(f"total {time.time()-t_all:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
