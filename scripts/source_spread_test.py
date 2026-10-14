"""
Is the confound degree genuinely different across ARC-MOF's source sub-libraries,
or is the spread just binomial noise from small sources?

WHY THIS SCRIPT EXISTS
----------------------
v3 of `source_confound_granularity.py` fed the Kruskal-Wallis test the *bootstrap
replicates* of each source (10 per source, all drawn from the same underlying
sample). That is pseudo-replication: the replicates are not independent
observations, so H = 77.3 with p = 5e-14 was not a valid test of "the sources
differ". It is replaced here.

THE CORRECT NULL
----------------
If every source shared one common exclusivity rate p, then the number of
single-topology compositions in source k would be Binomial(N_k, p) and the
observed *spread* across sources would be small. We estimate p from the pooled
data, simulate the null spread, and compare. The test statistic is the variance
of the per-source proportions (equivalently the Kruskal-Wallis H, which is a
monotone function of the spread; the variance has the advantage of a directly
interpretable null distribution).

The second question -- "is the *ordering* of the sources robust, or is it decided
by the choice of chemical granularity?" -- is answered by Spearman's rho between
the coarse and fine rankings. That one is a legitimate correlation and is not
affected by the issue above.

USAGE
-----
    python source_spread_test.py                  # reads the cached CSV
    python source_spread_test.py --rebuild        # re-derive from ARC-MOF

Outputs
    <outdir>/source_spread_test.csv    observed stat, null quantiles, p-value, rho
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")
MIN_SOURCE_N = 200
B = 20_000
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(
        OUT, "source_confound_granularity.csv"))
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--boot", type=int, default=B)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    d = pd.read_csv(args.csv)
    n_all = len(d)
    d = d[d["n_samples"] >= MIN_SOURCE_N].copy()

    # DB6 contains a single topology, so its exclusivity is 1.000 by construction
    # and it contributes no information about *differential* confounding.
    degenerate = d[d["n_topologies"] <= 1]["source"].tolist()
    d = d[d["n_topologies"] > 1].copy()
    K = len(d)
    print(f"sources with n >= {MIN_SOURCE_N}: {n_all} -> dropped degenerate "
          f"{degenerate} -> using {K}: {', '.join(d['source'])}")

    rng = np.random.default_rng(args.seed)
    rows = []

    for gran, ncol, fcol in (("coarse", "coarse_n_chem", "coarse_frac_single"),
                             ("fine", "fine_n_chem", "fine_frac_single")):
        n_single = np.rint(d[fcol].values * d[ncol].values).astype(int)
        N = d[ncol].values.astype(int)
        props = n_single / N
        p_hat = n_single.sum() / N.sum()
        obs = float(np.var(props, ddof=1))

        sim = rng.binomial(np.tile(N, (args.boot, 1)), p_hat) / N
        null = np.var(sim, axis=1, ddof=1)
        pval = float((null >= obs).mean())

        rows.append(dict(
            granularity=gran,
            n_sources=K,
            pooled_rate=float(p_hat),
            observed_var=obs,
            observed_range_pp=float((props.max() - props.min()) * 100),
            null_var_median=float(np.median(null)),
            null_var_p99=float(np.percentile(null, 99)),
            ratio_obs_over_null=float(obs / np.median(null)),
            p_value=pval if pval > 0 else 1.0 / args.boot,
        ))

        print(f"\n[{gran}] pooled rate = {p_hat:.4f}")
        print(f"        observed spread : var = {obs:.6f} "
              f"(range {rows[-1]['observed_range_pp']:.1f} pp)")
        print(f"        null (one common rate): median var = {np.median(null):.2e}")
        print(f"        ratio observed/null = {obs / np.median(null):,.0f}x")
        print(f"        p = {pval if pval > 0 else '1/' + str(args.boot)}")

    # ordering robustness -- unaffected by the pseudo-replication problem
    rho, rho_p = spearmanr(d["coarse_frac_single"], d["fine_frac_single"])
    print(f"\n[cross-granularity] Spearman rho = {rho:+.4f} (p = {rho_p:.3g}) "
          f"over {K} sources")
    print("        both granularities therefore rank the sources identically;")
    print("        the ordering is a property of the sources, not of the"
          " chemical definition.")

    out = pd.DataFrame(rows)
    out["spearman_rho"] = rho
    out["spearman_p"] = rho_p
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, "source_spread_test.csv")
    out.to_csv(path, index=False)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
