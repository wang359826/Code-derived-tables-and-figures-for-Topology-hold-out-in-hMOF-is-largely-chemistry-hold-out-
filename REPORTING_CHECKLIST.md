# Grouped-split reporting checklist

> Companion to `scripts/confound_diagnostic.py`. Written for anyone reporting a
> topology / scaffold / metal-cluster / linker hold-out result in MOF or porous-material
> machine learning.
>
> **The one-line reason this exists:** a grouped split is only a test of transferability
> *across the grouping variable* if that variable is separable from chemical composition.
> In enumerated MOF databases it largely is not. So a topology hold-out can be,
> in the majority, a chemistry hold-out wearing a different label — **in what it removes.
> Whether the resulting penalty is a chemistry penalty is a separate question that the
> counts cannot answer**, and in the companion study the controlled contrast prices the
> chemistry-attributable share at 34.9% of one measured degradation (§3.4), while the
> coarse composition-key split on a shared support set accounts for only 21.1% of the
> topology penalty (§3.1). Report the identity of what a split removes as a structural
> fact; report what the penalty is made of only from a controlled contrast, and say which
> one you have (rule B2).

---

## One-page summary

The 22 rules, one line each. Everything below this table is the same list with the
evidence attached; print this page and paste the disclosure block at the end.

| # | Rule | Why it matters here |
|---|---|---|
| **A1** | State n beside every exclusivity / entropy figure | both are sample-size dependent: 98.9% at n = 25,000 vs 93.2% at n = 324,426, same data |
| **A2** | State the granularity at which "chemistry" is defined | 74.7% (metal + 2 linkers) vs 93.2% (+ functional groups) |
| **A3** | Report pooled / weighted / macro / median; headline on macro or median | pooled alone reads +0.381 where macro reads −0.541 |
| **A4** | Check target variance on both sides before subtracting R² across splits | a 7.7× variance gap reversed the direction of the claim |
| **A5** | Log-transform ratio targets before modelling | selectivity R² swings +0.42 … −0.06 without it |
| **A6** | Report the number of *valid* folds | a zero-variance fold silently turns `macro` into NaN (10–11 folds here, not 11) |
| **B1** | Disclose the three diagnostic numbers in the paper, not only in the SI | single-group fraction, conditional-entropy ratio, count of pairs sharing chemistry |
| **B2** | Do not call a grouped split "a test of transfer across the group" unless separable | call it what it is: a test of transfer to unseen chemistry |
| **B3** | Report a ranking metric on a *shared support set* | top-1% hit rate fell 50.8% → 13.2% while R² fell 0.7066 → 0.3660 on the same 25,000-sample support set |
| **B4** | Never align two databases by random compression to a common granularity | compression only coarsens, so it manufactures the difference it claims to remove |
| **B5** | Align cross-database comparisons on r = n/M, not on raw n | two libraries can differ 100× in M at similar n; matching n matches nothing |
| **B6** | Compare groupwise metrics only at a shared support set | a different support set is a different denominator, and those cannot be subtracted |
| **B7** | Stratify the diagnostic by source sub-library | needs no cross-database comparison; range here 86.1% – 100.0% |
| **C1** | Fix the test set and change only the training set before any causal claim | the only design in this study that supports a causal reading |
| **C2** | Add a training-size-matched arm | separates *which* samples went from *how many*: 0.103 vs 0.107 vs 0.181 |
| **C3** | State how many independent contrasts you actually have | one pair is not a population |
| **C4** | A pair sharing zero chemistry is not a chemistry intervention | it reduces sample count and nothing else |
| **D1** | Never `head()` a sorted CSV | enumerated files are sorted by topology, so the first N rows are a handful of topologies |
| **D2** | Do not feed bootstrap replicates of one sample to a between-group test | resamples are not independent observations; the p-value is pseudo-replicated |
| **D3** | Distinguish "not detected" from "zero" | median −0.011 at p = 0.28 is not proof of absence |
| **D4** | Assert on row counts and column matches before computing | an empty frame once skipped a whole check silently |
| **D5** | Verify downloads by HTTP status *and* MD5 *and* file tail | two files were truncated while `curl` still exited 0 |

---

## Before you train: run the diagnostic

```bash
python scripts/confound_diagnostic.py --csv your_data.csv \
    --group-col topology \
    --coarse-cols metal_linker,linker1,linker2 \
    --fine-cols   metal_linker,linker1,linker2,functional_groups \
    --pair-report
```

Takes seconds to minutes, needs no model, and tells you whether the rest of this
checklist is mandatory or merely good practice.

---

## Checklist

### A. Always (regardless of dataset)

- [ ] **A1. State n alongside every exclusivity / entropy figure.**
      Both are sample-size dependent. The companion hMOF dataset reads **98.9% at
      n = 25,000** and **93.2% at n = 324,426** for the same statistic on the same
      data. A number without n is not comparable to anything — including a rerun
      of itself.
      *Why: a composition needs two samples before it can appear in two groups, so
      sampling mechanically raises exclusivity.*

- [ ] **A2. State the granularity at which "chemistry" is defined.**
      The same database gives **74.7%** (metal + two linkers) and **93.2%**
      (+ functional groups). "Confounding is X%" is meaningless without this.
      Report the definition, or report both.

- [ ] **A3. Report all four aggregation rules** — pooled, weighted, macro, median —
      and use **macro or median** for the headline. `pooled` concatenates every
      fold's predictions and is dominated by the largest group: with the largest
      group at 63% of the data, pooled = **+0.381** while macro = **−0.541**, and
      dropping that one group moves the remaining folds to **−0.678**. The sign
      reverses.
      *Why this is not pedantry: reporting pooled alone turns a negative result
      into a stably positive one.*

- [ ] **A4. Check target variance on each side before comparing R² across splits.**
      R² normalises by the test set's own variance, so it is not comparable between
      two test sets with different spreads. A worked failure: two test sets
      differing **7.7×** in target variance gave R² = −0.448 on the hold-out side
      while its **MAE was actually smaller** than the random split's (0.262 vs
      0.396) — the direction of the claim was reversed. Prefer MAE ÷ sd(y_test) when
      variances differ.

- [ ] **A5. Log-transform ratio targets before modelling.** CO₂/N₂ selectivity has a
      median of ~8 and a maximum of ~31,600; without the log transform R² swings
      between +0.42 and −0.06 depending on the random draw.

- [ ] **A6. Report the number of *valid* folds.** When a test fold has zero target
      variance R² is undefined. Silently keeping NaN turns `macro` into NaN; silently
      dropping it hides that the fold count changed. Affects 3 of 5 seeds in the
      companion study, so the effective fold count was 10–11, not 11.

### B. When the diagnostic says the grouping is confounded

- [ ] **B1. Disclose the three diagnostic numbers in the paper**, not just in the
      supplement: single-group chemistry fraction, H(group | chemistry) / H(group),
      and the number of group pairs sharing any chemistry. Without them a
      "leakage magnitude" figure cannot be compared across studies or databases.

- [ ] **B2. Do not describe a grouped split as an isolated test of transferability across the
      grouping variable** unless the diagnostic shows the variable is separable.
      If chemistry is mixed into the split, describe the result as a mixed stress test and
      quantify any controlled chemistry contribution separately; do not claim that the
      protocol measures chemistry extrapolation alone.

- [ ] **B3. Report a ranking metric that assumes nothing about target variance**,
      on a *shared support set*: overlap@k, regret@k, or mean true value of the
      selected top-k. In the companion study, switching to these metrics turned a
      comfortable R² story into a top-1% hit rate falling **50.8% → 13.2%** and
      regret growing **18.5% → 52.2%**. Same model, same data, same split — only
      the currency changed.

- [ ] **B4. Never align two datasets by randomly compressing chemistry to a common
      granularity.** Random compression can only make a definition *coarser*, never
      finer, so the larger library is distorted harder and the apparent difference
      is manufactured rather than removed (we measured 406% inflation). Use two
      *real* definitions and check whether the conclusion survives both.

- [ ] **B5. Align cross-database comparisons on r = n/M (samples per distinct
      composition), not on raw n.** Two databases can differ by two orders of
      magnitude in M while having similar n; matching on n matches nothing.

- [ ] **B6. Compare groupwise metrics at a shared support set, and say so.**
      Changing the support set changes the denominator, and metrics on different
      subsets cannot be subtracted.

- [ ] **B7. If the database is assembled from sub-libraries, stratify by source.**
      Compute the diagnostic per source. In the companion study the range across
      sources was **86.1% – 100.0%**, with the ordering identical at both real
      chemical granularities (ρ = 1.000). This check needs no cross-database
      comparison and therefore escapes the granularity-comparability problem.
      *Test the spread against a null of one common rate — do not feed bootstrap
      replicates to a rank test (see D2).*

### C. If you want to attribute the effect to chemistry rather than group identity

- [ ] **C1. Build a controlled contrast: hold the test set fixed and change only
      the training set.** Remove the chemically matched group from training and
      compare. This is the only design in the companion study that supports a causal
      reading, and it is available only where two groups actually share chemistry.

- [ ] **C2. Add a training-size-matched arm.** Removing a group also removes its
      samples, so a naive before/after cannot separate *which* samples went from
      *how many* went. Draw the same number of samples at random from the full pool
      and retrain, several draws: if the size-matched arm matches the full pool and
      both beat the group-removed arm, the effect is the identity of the samples.
      In the companion study: **0.1030 ± 0.0031** (size-matched) vs **0.1065 ±
      0.0032** (full pool) vs **0.1811 ± 0.0136** (group removed) — the first two
      agree to 3.3%, the third is 70% worse.

- [ ] **C3. Report how many independent contrasts you actually have.** The companion
      study rests on **one** isoreticular pair (pcu ↔ nbo, 408 shared compositions);
      **53 of the 55 pairs share nothing at all**, and the only other non-zero pair
      (rht ↔ tbo, 2 shared compositions) is too thin to train and evaluate on. One
      pair is not a population. Say so rather than implying generality.

- [ ] **C4. If a group pair shares *zero* chemistry, removing it is not a chemistry
      intervention.** It reduces sample count and nothing else; do not read a
      degradation there as a chemical effect.

### D. Statistics that are easy to get wrong

- [ ] **D1. Never `head()` a sorted file.** Enumerated MOF CSVs are sorted by
      topology, so the first N rows contain a handful of topologies and every
      statistic is wrong. Use a seeded random sample.

- [ ] **D2. Do not feed bootstrap replicates of one sample into a between-group
      test.** Resamples of the same rows are not independent observations; a
      Kruskal–Wallis run on them is pseudo-replicated and its p-value is invalid.
      Either test *between groups* using one value per group, or build the correct
      null (e.g. one common rate; simulate the spread under it). The companion study
      withdrew an H = 77.3, p = 5×10⁻¹⁴ figure for exactly this reason; the
      corrected test found the spread **3,554× larger** than the null median
      (p < 1/20,000), so the conclusion survived — but the original p-value did not.

- [ ] **D3. Distinguish "not detected" from "zero".** A median effect of −0.011
      with p = 0.28 at n = 25,000 means *not detected at this sample size*, not
      *proven absent*. State which.

- [ ] **D4. Fail loudly on empty frames.** A prefix-based column selection once
      returned an empty frame here and the entire check was silently skipped. Assert
      on row counts and on column matches before computing anything.

- [ ] **D5. Verify downloads before trusting them.** Check the HTTP status *and*
      the MD5 *and* the file tail: two files were truncated mid-download while
      `curl` still exited 0.

---

## Minimal disclosure block

If you want one paragraph to paste into a methods section:

> A grouped split is only informative about transfer across the grouping variable if
> that variable is separable from chemical composition. On our data (n = N, chemistry
> defined as *…*), **X%** of distinct chemical compositions occur in a single
> `<group>`, and conditioning on chemistry leaves **Y%** of the entropy of the
> `<group>` label. We therefore report grouped-split results with this diagnostic
> attached, alongside all four aggregation rules (pooled / weighted / macro /
> median), and use ranking metrics on a shared support set as the primary
> practitioner-facing measure.

---

## What this checklist cannot tell you

- It does **not** establish that the grouping variable is irrelevant. A residual of
  **7.8%** of the entropy is small but nonzero, and that residual is what makes a
  controlled contrast possible at all.
- It does **not** transfer to databases of *experimental* structures (CoRE MOF,
  QMOF), which were not built by enumeration. The diagnostic is cheap to run there;
  until someone does, treat the conclusion as scoped to enumerated libraries.
- It is **not** a novelty claim. Several 2026 studies report the split effect and its
  target-dependence; this checklist addresses the premise underneath them.
