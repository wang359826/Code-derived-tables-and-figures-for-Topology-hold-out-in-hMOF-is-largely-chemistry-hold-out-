#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
confound_diagnostic.py -- a drop-in diagnostic for grouped splits in MOF-ML
===========================================================================

WHAT THIS IS FOR
----------------
If you are about to report a *grouped* (topology / scaffold / metal-cluster)
cross-validation result, this script tells you how much of that grouping is
already implied by chemical composition. If the split variable and "chemistry"
are largely the same variable, a topology hold-out is not a test of topological
transferability -- it is (mostly) a test of transfer to unseen chemistry.

You run it on *your* dataset, before training anything. It takes seconds to
minutes and needs no model.

THE THREE NUMBERS, AND WHAT EACH ONE ANSWERS
--------------------------------------------
1. `frac_single_topology` -- share of distinct chemical compositions that occur
   in exactly one group. Answers: "how much of my chemistry is private to one
   group?" High means the group label is largely a relabelling of chemistry.

2. `cond_frac` = H(group | chemistry) / H(group) -- entropy of the group label
   remaining after conditioning on chemistry, normalised. Answers: "if I knew
   the chemistry, how much would I still not know about the group?" 0 means the
   group is fully determined by chemistry; 1 means chemistry tells you nothing.

   This is more informative than (1): (1) is a count of private compositions,
   (2) is an information-theoretic residual. A dataset can have a low (1) but a
   still-informative group label, or vice versa.

3. `n_shared_pairs` / max Jaccard -- how many group *pairs* share any chemistry
   at all. Answers: "does any group retain an independent handle?" If exactly
   one pair shares chemistry, that pair is your only option for a controlled
   contrast (see `--pair-report`).

ALWAYS REPORT n NEXT TO (1) AND (2). Both are sample-size dependent: sampling
fewer rows mechanically raises exclusivity, because a composition needs two
samples to appear in two groups. On the companion hMOF dataset the same
statistic reads 98.9% at n = 25,000 and 93.2% at n = 324,426. A number without
n is not comparable to anything, including a rerun of itself.

USAGE
-----
  # generic: tell it your columns
  python confound_diagnostic.py --csv my_data.csv \
      --group-col topology --chem-cols metal_linker,linker1,linker2,func_groups

  # two granularities in one pass (required whenever your chemistry definition
  # is a choice rather than a given -- see NOTE below)
  python confound_diagnostic.py --csv my_data.csv --group-col topology \
      --coarse-cols metal_linker,linker1,linker2 \
      --fine-cols   metal_linker,linker1,linker2,func_groups

  # hMOF preset (columns are known)
  python confound_diagnostic.py --preset hmof --max-rows 50000

  # also print which group pair, if any, could support a controlled contrast
  python confound_diagnostic.py --preset hmof --max-rows 50000 --pair-report

NOTE ON GRANULARITY
-------------------
"Chemical composition" is not one thing. The same database gives 74.7% and
93.2% depending on whether functional groups are included. Report the definition
you used; better, report both. Do NOT align two datasets by randomly compressing
chemistry to a common granularity -- random compression can only make a
definition coarser, never finer, so it distorts large libraries more than small
ones and manufactures a difference instead of removing one.

OUTPUT
------
  <outdir>/confound_diagnostic.csv    one row per (grouping, granularity)
  <outdir>/confound_diagnostic_pairs.csv   top group pairs by shared chemistry
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")
DATA = os.path.join(HERE, "..", "data")

# Placeholder for a missing chemistry value; see composition_key().
NO_VALUE = "<NA>"

# hMOF column names (see DATA.md)
PRESETS = {
    "hmof": {
        "csv": os.path.join(DATA, "all_MOFs_screening_data.csv"),
        "group_col": "topology",
        "coarse": ["metal_linker", "organic_linker1", "organic_linker2"],
        "fine": ["metal_linker", "organic_linker1", "organic_linker2",
                 "functional_groups"],
    },
}


def _entropy(p) -> float:
    """Shannon entropy in bits of a probability vector."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def composition_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Join the chemistry columns into one hashable key per row.

    Fails loudly if any column is missing -- a silent `linker_pair_`-style
    prefix mismatch once produced an empty frame here and the whole check was
    skipped without an error.

    Missing values are filled with a placeholder BEFORE the type cast. hMOF's
    `functional_groups` column contains NaN, and `astype(str)` leaves those as
    floats, which makes `"|".join` raise `TypeError: expected str instance,
    float found`. Two rows that both have "no functional group" are the same
    chemistry, so an explicit placeholder is the correct treatment -- dropping
    them would silently change the answer.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"chemistry columns not found: {missing}. "
            f"Available: {list(df.columns)[:40]}"
        )
    sub = df[cols].astype(object).where(pd.notna(df[cols]), NO_VALUE)
    sub = sub.astype(str)

    keys = sub.agg("|".join, axis=1)
    # A key made entirely of placeholders is not a chemistry, it is missing data.
    all_blank = (sub == NO_VALUE).all(axis=1)
    if all_blank.any():
        frac = float(all_blank.mean())
        print(f"  WARNING: {int(all_blank.sum()):,} rows ({frac:.3%}) have no "
              "chemistry value in any column; they are grouped under a single "
              "'missing' key. Check whether that is intended.")
    return keys


def diagnose(chem: pd.Series, group: pd.Series, label: str,
             granularity: str) -> dict:
    """Compute the three diagnostics for one (key, group) pair."""
    if len(chem) == 0:
        raise ValueError(f"[{label}/{granularity}] empty input -- refusing to "
                         "report a statistic computed on zero rows")

    d = pd.DataFrame({"ck": np.asarray(chem, dtype=object),
                      "grp": np.asarray(group, dtype=object)})
    d = d.dropna()
    n = len(d)
    if n == 0:
        raise ValueError(f"[{label}/{granularity}] all rows dropped as NA")

    pair = d.groupby(["ck", "grp"], observed=True).size().reset_index(name="n_ct")
    n_groups_per_chem = pair.groupby("ck", observed=True)["grp"].nunique()

    # (1) share of compositions living in a single group
    frac_single = float((n_groups_per_chem == 1).mean())

    # (2) conditional entropy ratio
    p_grp = d["grp"].value_counts(normalize=True).to_numpy()
    H_grp = _entropy(p_grp)
    n_c = pair.groupby("ck", observed=True)["n_ct"].transform("sum")
    H_grp_given_chem = float(
        (pair["n_ct"] / n * (-np.log2(pair["n_ct"] / n_c))).sum())
    cond_frac = float(H_grp_given_chem / H_grp) if H_grp > 0 else np.nan

    return {
        "label": label,
        "granularity": granularity,
        "n_samples": n,
        "n_groups": int(d["grp"].nunique()),
        "n_chemistries": int(d["ck"].nunique()),
        "samples_per_chemistry": round(n / max(1, d["ck"].nunique()), 3),
        "frac_single_topology": round(frac_single, 6),
        "H_group": round(H_grp, 4),
        "H_group_given_chem": round(H_grp_given_chem, 4),
        "cond_frac": round(cond_frac, 6),
    }


def group_pairs(chem: pd.Series, group: pd.Series, top_n: int = 20) -> pd.DataFrame:
    """Jaccard overlap of the chemistry sets of the largest groups."""
    d = pd.DataFrame({"ck": np.asarray(chem, dtype=object),
                      "grp": np.asarray(group, dtype=object)}).dropna()
    big = d["grp"].value_counts().head(top_n).index.tolist()
    sets = {g: set(d.loc[d["grp"] == g, "ck"]) for g in big}
    rows = []
    for a, b in itertools.combinations(big, 2):
        inter = len(sets[a] & sets[b])
        union = len(sets[a] | sets[b])
        rows.append({"group_a": a, "group_b": b, "shared_chemistries": inter,
                     "jaccard": round(inter / union, 6) if union else 0.0})
    out = pd.DataFrame(rows)
    return out.sort_values("shared_chemistries", ascending=False).reset_index(drop=True)


def verdict(row: dict, n_shared_pairs: int, total_pairs: int) -> str:
    """One-line plain-language reading of the numbers."""
    f, c = row["frac_single_topology"], row["cond_frac"]
    if c >= 0.5:
        base = ("grouping is largely independent of chemistry -- a grouped "
                "split is plausibly testing what its name says")
    elif c >= 0.2:
        base = ("grouping is substantially confounded with chemistry -- report "
                "both the split effect and this diagnostic together")
    else:
        base = ("grouping is largely a relabelling of chemistry -- a grouped "
                "split here does NOT isolate group transferability")
    if n_shared_pairs == 0:
        tail = ("; no pair of groups shares any chemistry, so no controlled "
                "contrast is possible on this dataset")
    elif n_shared_pairs == 1:
        tail = (f"; exactly 1 of {total_pairs} group pairs shares chemistry -- "
                "that pair is the only place a controlled contrast can be run")
    else:
        tail = f"; {n_shared_pairs} of {total_pairs} group pairs share chemistry"
    return base + tail


def main():
    ap = argparse.ArgumentParser(
        description="Diagnose how much a grouped split is implied by chemistry.")
    ap.add_argument("--csv", help="input CSV")
    ap.add_argument("--preset", choices=sorted(PRESETS),
                    help="use a known column layout instead of --csv")
    ap.add_argument("--group-col", help="column holding the grouping variable")
    ap.add_argument("--chem-cols", help="comma-separated chemistry columns")
    ap.add_argument("--coarse-cols", help="comma-separated, coarser definition")
    ap.add_argument("--fine-cols", help="comma-separated, finer definition")
    ap.add_argument("--label", default=None, help="name shown in the output")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="random subsample (seeded); omit for the full file")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pair-report", action="store_true",
                    help="also list group pairs by shared chemistry")
    ap.add_argument("--top-n-groups", type=int, default=20,
                    help="how many of the largest groups to compare pairwise")
    ap.add_argument("--outdir", default=OUT)
    args = ap.parse_args()

    # ---- resolve inputs -------------------------------------------------
    if args.preset:
        p = PRESETS[args.preset]
        csv = args.csv or p["csv"]
        group_col = args.group_col or p["group_col"]
        coarse = args.coarse_cols.split(",") if args.coarse_cols else p.get("coarse")
        fine = args.fine_cols.split(",") if args.fine_cols else p.get("fine")
        label = args.label or args.preset
    else:
        if not args.csv or not args.group_col:
            ap.error("--csv and --group-col are required unless --preset is used")
        csv = args.csv
        group_col = args.group_col
        coarse = args.coarse_cols.split(",") if args.coarse_cols else None
        fine = args.fine_cols.split(",") if args.fine_cols else None
        if args.chem_cols and not (coarse or fine):
            coarse = args.chem_cols.split(",")
        label = args.label or os.path.splitext(os.path.basename(csv))[0]

    if not os.path.exists(csv):
        raise SystemExit(
            f"input not found: {csv}\n"
            "Download the dataset first (see DATA.md) or pass --csv.")

    # ---- load -----------------------------------------------------------
    print(f"reading {csv}")
    df = pd.read_csv(csv, low_memory=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")
    if args.max_rows and args.max_rows < len(df):
        # NEVER use head(): these files are sorted by group, so the first rows
        # contain only a handful of groups and every statistic is wrong.
        df = df.sample(n=args.max_rows, random_state=args.seed)
        print(f"  sampled {len(df):,} rows (random_state={args.seed})")

    if group_col not in df.columns:
        raise SystemExit(f"grouping column '{group_col}' not in the file")

    # ---- run ------------------------------------------------------------
    granularities = []
    if coarse:
        granularities.append(("coarse", coarse))
    if fine:
        granularities.append(("fine", fine))
    if not granularities:
        raise SystemExit("no chemistry columns given (--chem-cols / "
                         "--coarse-cols / --fine-cols)")

    group = df[group_col]
    rows, pair_frames = [], []
    for gran, cols in granularities:
        chem = composition_key(df, cols)
        row = diagnose(chem, group, label, gran)
        rows.append(row)
        print(f"\n[{gran}] " + ", ".join(cols))
        print(f"  n = {row['n_samples']:,}   groups = {row['n_groups']}   "
              f"chemistries = {row['n_chemistries']:,}   "
              f"n/M = {row['samples_per_chemistry']}")
        print(f"  single-group chemistry : {row['frac_single_topology']:.4f}")
        print(f"  H(group|chem)/H(group) : {row['cond_frac']:.4f}")

        if args.pair_report:
            pf = group_pairs(chem, group, args.top_n_groups)
            pf.insert(0, "granularity", gran)
            pair_frames.append(pf)

    # ---- pair summary + verdict ----------------------------------------
    n_shared = 0
    total_pairs = 0
    if pair_frames:
        allp = pd.concat(pair_frames, ignore_index=True)
        coarse_pairs = allp[allp["granularity"] == "coarse"]
        if len(coarse_pairs):
            top = coarse_pairs.iloc[0]
            print(f"\n  most-overlapping pair ({top['group_a']} vs "
                  f"{top['group_b']}): {int(top['shared_chemistries'])} shared "
                  f"compositions, Jaccard {top['jaccard']:.3f}")
        first = pair_frames[0]
        total_pairs = len(first)
        n_shared = int((first["shared_chemistries"] > 0).sum())

    print()
    for row in rows:
        print(f"[{row['granularity']}] {verdict(row, n_shared, total_pairs)}")

    # ---- write ----------------------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)
    out = pd.DataFrame(rows)
    out["n_shared_pairs"] = n_shared
    out["n_pairs_compared"] = total_pairs
    p1 = os.path.join(args.outdir, "confound_diagnostic.csv")
    out.to_csv(p1, index=False)
    print(f"\nsaved: {p1}")
    if pair_frames:
        p2 = os.path.join(args.outdir, "confound_diagnostic_pairs.csv")
        pd.concat(pair_frames, ignore_index=True).to_csv(p2, index=False)
        print(f"saved: {p2}")


if __name__ == "__main__":
    main()
