# Reproduction manual

Everything in this repository exists to let a reader regenerate the results assigned
to each pipeline level, in one command per level. Level 5 is documented with its
commands and archived outputs but has not been run end to end in the current
environment. Nothing here depends on a GPU, on a private data source, or on a specific
editor.

Companion documents: `README.md` (claims → sections → scripts, English overview),
`REPORTING_CHECKLIST.md` (portable reporting rules, and why each one exists),
`DATA.md` (dataset URLs, MD5 checksums, two data traps), `LICENSE` (MIT, code only).

> This manual corresponds to the **v5** version of the manuscript
> (*topology hold-out is chemistry hold-out*; earlier prints of this manual said
> "v4", which was the version then current). An earlier v3 direction —
> "quantifying the leakage caused by random splitting" — was abandoned on
> 2026-10-09 after it turned out to be already covered by several 2026 studies.
> Nothing in this repository reproduces v3.

---

## 0. The shortest path

```bash
pip install -r requirements.txt            # 5 packages, CPU only
cd 03_reproduction_package        # 是包根，不是 scripts/
python scripts/run_all.py --level 1                # ~7.5 min  core evidence: Tables 8-11, 17; Figures 2-3
```

Add `--dry-run` to any level to print its commands without executing them.
Each step writes a log to `outputs/logs/`; a failing step stops the run and
prints the last 15 lines of its log, so downstream steps never consume partial output.

---

## 1. Environment

| Item | Value |
|---|---|
| Python | 3.13 (developed on Windows 11; every script is cross-platform) |
| Packages | `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib` (`shap` only for level 5) |
| Hardware | laptop CPU, ≥ 8 GB RAM, **no GPU required** |
| Deep learning | **Not needed.** The deep-learning arm was withdrawn (§3.7); `torch`, `torch_geometric` and `pymatgen` are deliberately absent from `requirements.txt`. |

```bash
pip install -r requirements.txt
# if the default index is unreachable, pick any working mirror, e.g.
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

All random seeds are fixed (`random_state = 42` unless a script is explicitly
given more seeds).

---

## 2. Data

See `DATA.md` for URLs, sizes, MD5 checksums and download commands. Minimum per level:

| Level | Required files |
|---|---|
| 1, 2, 3, 5 | `data/all_MOFs_screening_data.csv` (hMOF, 148 MB, 324,426 rows) and `data/top_MOFs_screening_data.csv` (elite subset, 5.8 MB, 8,206 rows) |
| 4 | additionally `data/arcmof/{all_topology_lists,mc-clusters,flig-clusters,func-clusters}.csv` (73 MB in total) |

Both datasets are public and free, and this work generates **no new data**.

> Verify every download by MD5 before use. During the first round two files were
> truncated silently while `curl` still exited 0 (§"Known pitfalls", item 7).

---

## 3. The five levels

| Level | What it reproduces | Wall time | Manuscript pieces |
|---|---|---|---|
| **1** | confound structure (full database), **per-held-out-topology chemistry seen/unseen decomposition**, controlled isoreticular contrast with 5 seeds and a size-matched arm, elite subset | **~7.5 min** *(7.3 min measured, plus 14 s for the full-database step; see below)* | §3.2, §3.4, §3.12; Tables 8–11, 17; Figures 2–3 |
| **2** | baseline comparison (both targets), four aggregation rules, 5-seed stability, mitigation strategies | **65.4 min** *(measured 2026-10-13; slowest step 34.7 min)* | §3.1, §3.5, §3.9; Tables 7, 12–14, S3; Figures 1, S5, S7 |
| **3** | screening utility on a shared support set; composition-matched control arm; 10 targets in a 2×2 design; **identity-block availability per split (§4.2)** | **78.5 min** *(measured 2026-10-13; the composition-matched arm trains 120 forests on one core, 45.1 min)* | §3.4, §3.11, §3.13, §3.14, §4.2; Tables 11 (composition-matched panel), 16, 18, 19, 25, 26; Figures 6–7 |
| **4** | replication on ARC-MOF, leave-one-source-out bound, per-source stratification with a valid spread test, sample-size dependence | **~30 min** *(29.9 min measured for an earlier 6-step form; see below)* | §3.15; Tables 20–24; Figures 8, S4, S6 |
| **5** | mechanism analyses and SHAP (optional), **plus the one-run recomputation of the three refuted explanations (§3.6)** | several hours *(not benchmarked)* | §3.6, §3.8, §3.10; Table 15; SI Tables S1–S2, S4–S5; Figures 4–5, S1–S3 |

**Scripts added in revision v6** (each is registered in `run_all.py` at the level
above, and each supports a specific claim added in this round):

| Script | Output | Claim it supports |
|---|---|---|
| `chemistry_seen.py` | `chemistry_seen.csv`, `chemistry_seen_by_topology.csv`, `topology_pair_overlap.csv` | §3.2 Table 9: the per-held-out-topology seen/unseen decomposition, i.e. "7 of 11 topologies are pure chemistry hold-outs"; also the 61.3% / 44.2% levels |
| `chemistry_seen.py --csv data/arcmof/arcmof_confound.csv --topology-col topology --tag arcmof` | `chemistry_seen_arcmof.csv`, `chemistry_seen_by_topology_arcmof.csv`, `topology_pair_overlap_arcmof.csv` | §3.15 ARC-MOF: the same decomposition on the second database, after `prep_arcmof.py` assembles its four metadata files |
| `prep_arcmof.py` | `data/arcmof/arcmof_confound.csv` | §3.15: merges ARC-MOF topology and cluster files into the chemistry columns the confound statistics expect |
| `mof_leakage_benchmark.py --loto --tag chemistry` | `leakage_results_chemistry.csv`, `loto_per_chemistry.csv`, `group_stats_chemistry.csv` | §3.1 / §3.5: the composition hold-out arm that directly tests whether topology hold-out and chemistry hold-out produce the same outcome (R² 0.6738 vs 0.3786; 1,590 chemistry groups) |
| `screening_utility.py --strategies …,chemistry --tag chemistry` | `screening_utility_chemistry.csv` | §3.11 Table 16 supplement: screening utility under composition hold-out (overlap@1% 42.4% vs topology 13.2%) |
| `null_model_controls.py` | `null_elite_permutation_row.csv`, `null_elite_permutation_composition.csv`, `null_topology_cardinality.csv` | §3.12 / §3.15: row-level vs composition-level permutation null; the row-level unit is the correct test (observed 100.0% vs null 21.4%), and the composition-level result is reported as an auditable negative |
| `partner_coverage_arm_e.py` | `arm_e_partner_coverage.csv` | §3.4 arm (e) / §4.5: removes only the partner-topology rows sharing a composition with the test set, so matched-partner coverage goes to zero while most of arm (a)'s compositions stay — 34.9% of the nbo degradation, and a sign reversal on the pcu side |
| `mof_leakage_benchmark.py --model {gb,ridge}` | `leakage_results_{gb,ridge}100k.csv` | §3.1: the random-vs-topology gap of Table 7 refitted under gradient boosting and ridge; the gap is 0.374 / 0.374 / 0.180 (RF / GB / ridge), not an artefact of one model family |

> `cardinality_correction.py --perm N` controls how many times the 587-column
> one-hot is refit under a shuffled group label (each refit is a full 5-fold RF on
> 25,000 × 587, roughly a minute). The shipped default is `--perm 3`; `--perm 0`
> skips that column entirely and still produces the ω² result, which is the part
> the manuscript quotes.

Times above are honest about their own history, because the two timed levels were
run by an earlier version of `run_all.py` than the one shipped here:

- **Level 1** was timed before `full_db_confound.py` joined the level — the step
  that computes the headline 93.2% on all 324,426 structures. That step has since
  been timed separately at **14 s**, hence 7.3 → ~7.5 min.
- **Level 4** was timed when the level still contained the withdrawn
  `granularity_control.py` and did **not** contain `arcmof_overlap.py` (12 s) or
  `sample_size_granularity.py` (39 s); both are now part of the level and both have
  been timed separately. Dropping one script and adding two that together cost
  51 s keeps the total at ~30 min.

Level 5 still quotes an estimate because it has never been run end to end; we would
rather publish an estimate than a number we have not produced. **Levels 2 and 3 are now
measured** (2026-10-13), and both came out well above the earlier estimates:

| Level | Earlier estimate | **Measured** | Slowest step |
|---|---|---|---|
| 2 | ~45 min | **65.4 min** | the 100,000-sample baseline with leave-one-topology-out: 34.7 min |
| 3 | ~60 min | **78.5 min** | the composition-matched control arm (120 forests, one core): 45.1 min |

Measured step by step (2026-10-13, Windows 11, 20-core desktop, 16 GB RAM):
level 2 = 34.7 + 17.1 + 0.03 + 0.02 + 8.5 + 5.1 min; level 3 = 2.3 + 21.8 + 0.05 +
0.5 + 45.1 + 5.5 + 2.4 + 0.9 min. Levels 1 (7.3 min) and 4 (29.9 min) were measured on
2026-10-10 on a laptop (8 GB RAM), i.e. on slower hardware — treat all four numbers as
"the order of magnitude on an ordinary machine", not as a benchmark.

**Verification actually performed (not merely intended).**

| Date | What | Result |
|---|---|---|
| 2026-10-10 | level 4, **as that level was defined at the time** (6 steps, including the withdrawn `granularity_control.py` and without `arcmof_overlap.py` / `sample_size_granularity.py`), re-run from scratch | all steps OK, **29.9 min**; `cmp` against the previous run showed **8/8 files byte-identical** (`confound_arcmof.csv`, `jaccard_arcmof.csv`, `exclusivity_across_databases.csv`, `source_confound_granularity.csv`, and the figures) |
| 2026-10-10 | the three steps missing from the timed runs above — `full_db_confound.py --tag full` (14 s), `arcmof_overlap.py` (12 s), `sample_size_granularity.py` (39 s) — re-run into a scratch directory | all exit 0, total 65 s. `cmp` against the archived files: **5/5 byte-identical** — `confound_full.csv` and `jaccard_full.csv` (which carry the headline 93.2% and the Jaccard table of §3.2), `arcmof_overlap.csv`, `confound_arcmof_excl_derived.csv`, `sample_size_granularity.csv` |
| 2026-10-10 | level 2, the second-target arm alone (`... --target selectivity --log-target --tag logsel100k`), re-run into a scratch directory | exit 0, 17 min while other jobs competed for the CPU. All four rows reproduced: random R² `0.8356804520124543` → topology R² `0.7211398375867859`, RMSE identical, MAE agreeing to ~1e-17. The only difference from the archived file is the strategy **label** ("未见拓扑" → "GroupKFold"), which changed after that file was written. `plot_main_figure.py` then reproduced exactly the two numbers quoted in §3.1: uptake 0.7529 → 0.3786 and selectivity 0.8357 → 0.7211. |

---

## 3b. Outside the pipeline: the standalone diagnostic and the checklist

Two files in this repository are deliberately **not** steps in any level, because
their purpose is to be pointed at *someone else's* data:

| File | What it is | How it is checked |
|---|---|---|
| `scripts/confound_diagnostic.py` | model-free diagnostic for **any** CSV with a grouping column and a chemistry definition. Reports, at two granularities, the single-group chemistry fraction, H(group \| chemistry) / H(group), n/M, and the group-pair Jaccard table; then prints a plain-language verdict | On the `hmof` preset at n = 50,000 it reproduces the n = 50,000 view **exactly** — the view the full-database statistics supersede: coarse 0.7533 / 0.1782, fine 0.9797 / 0.0233, pcu–nbo 387 shared / J = 0.358. On synthetic data with a *known* answer it returns "largely independent" (cond_frac 0.957) for chemistry drawn independently of the group, and "relabelling of chemistry" (cond_frac 0.000) for chemistry that deterministically fixes the group |
| `REPORTING_CHECKLIST.md` | the write-up of what to do with the diagnostic's answer, plus 22 reporting rules this study learned by breaking them | every number it cites is cross-checked against the archived outputs in `outputs/` (§3.2, §3.4, §3.5, §3.11, §3.13, §3.14); no figure appears in the checklist that is not also in a CSV here |
| `scripts/null_model_controls.py` | the two null-model controls of §3.12 / §3.15 (SI §S3): a **sample-level** topology permutation for the elite subset (b = 10,000) and a within-source permutation for every ARC-MOF source sub-library (b = 1,000) | reproduces §3.12's elite null (mean **21.4%**, sd 1.0%, against the observed 100.0%) and every source-level null of §3.15 Result 3; writes `null_elite_permutation.csv` and `null_topology_cardinality.csv` in `outputs/` |

```bash
python scripts/confound_diagnostic.py --csv your_data.csv \
    --group-col topology \
    --coarse-cols metal_linker,linker1,linker2 \
    --fine-cols   metal_linker,linker1,linker2,functional_groups \
    --pair-report
# or, to reproduce this paper's numbers directly:
python scripts/confound_diagnostic.py --preset hmof --max-rows 50000 --pair-report
```

It writes `confound_diagnostic.csv` and `confound_diagnostic_pairs.csv` next to
its output directory. It needs none of this paper's data when given a foreign CSV.

> ⚠️ **Those two filenames are the retired n = 50,000 convention.** The two files
> shipped with the earlier package were moved to `outputs/_deprecated_n50000/`
> (with a README explaining why) because their values (387 shared pcu–nbo
> compositions, J = 0.358, fine single-topology 97.97%) contradict the
> full-database figures the manuscript now reports (408, J = 0.371, 93.17%). The
> script is unchanged and still writes those filenames when you run it; only the
> archived copies were quarantined, so that a reader grepping `outputs/` for
> "pcu–nbo" finds one number rather than two.

The null-model controls need this paper's own data (they are not a foreign-data
tool) and are **not steps in any level**. Run them from `scripts/` with the
downloaded data:

```bash
python scripts/null_model_controls.py --outdir outputs            --granularity coarse -b 10000 --b-card 1000
python scripts/null_model_controls.py --outdir outputs/_fine      --granularity fine   -b 10000 --b-card 1000
# coarse on the common support set (mc/flig/func all present) — same口径 as the §3.15 table:
python scripts/null_model_controls.py --outdir outputs/common_support --granularity coarse -b 10000 --b-card 1000 --require-cols mc,flig,func
```

The **permutation unit is the sample (row)**: topology labels are reshuffled with
each topology's sample count preserved. Permuting at the *composition* level would
leave the single-topology fraction equal to its observed value by construction, so
the two units answer different questions; the manuscript states which one is used.

---

## 3c. What a fresh run reproduces — and the differences you should expect

On **2026-10-13** levels 2 and 3 were re-run from scratch against the archived
`outputs/`, on **newer library versions than the ones the archive was built with**
(pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, shap 0.52). Every step exited 0. The
result, file by file:

| Result | Status of a fresh run |
|---|---|
| `group_stats_final.csv`, `mitigation_results_mit25k.csv`, `screening_utility.csv`, `chemistry_seen_grouping.csv`, `leakage_results_{gb,ridge}100k.csv` | **byte-identical** |
| `leakage_results_final.csv`, `leakage_results_logsel100k.csv`, `loto_per_topology_final.csv`, `multi_seed_uptake25k_fix.csv`, `arm_e_partner_coverage.csv` | agree to **1×10⁻¹⁶ – 9×10⁻¹⁶** (last-digit floating-point noise), not byte-identical |
| `multitarget_confound.csv` | all 20 shared columns agree; the largest deviation is **5.6×10⁻⁷** in one auxiliary column (`topology_residual_R2`, from the 60-estimator residual ensemble). The fresh file also has **5 extra columns** (`group_col`, `R2_grp_*`, `split_effect_grp`, `identity_effect_grp`): the archived copy predates the `--group-col` option added for Table 25 |
| `composition_matched_control.csv` | numbers identical; 50 **label** strings differ, `diversity_matched_seedN` → `composition_matched_seedN` — the rename of arm (d), applied to the code but not to the archived file |
| `metrics_audit.csv` | **expected row-count difference**: the archived file carries two rows of the **withdrawn CGCNN arm** (§3.7); a fresh run writes only the five rows Table 12 uses |

So: **every number reported in the manuscript is reproduced, but byte-identity is a
property of the original environment, not of the code.** A reviewer on newer libraries
should expect agreement of order 10⁻¹⁵, with the three structural differences listed
above. Byte-for-byte identity was verified for level 4 (2026-10-10, 8/8 files).

---

## 3d. Two archive defects found and fixed on 2026-10-14

Both were found by re-deriving the manuscript's §3.12 numbers from the shipped `data/`
rather than from the archive. **In both cases the manuscript was right and the stored
output was wrong.** Nothing in the text was changed for these two; the scripts and the
stored outputs were.

### (a) `elite_vs_full.py` matched rows on the index (`MOFname` fix)

The script chose its matching path with

```python
if "filename" in full_qual.columns and "filename" in elite.columns:
```

Neither hMOF file has a `filename` column — the key is `MOFname` — so the condition was
always false and the `else` branch ran: `full_qual.index.isin(elite.index)`. Both files
carry a default `RangeIndex` starting at 0, so "retained" became *the first 8,206 rows of
the file in file order* and "lost" the remaining 1,382. Consequences, all of which
contradicted the manuscript:

| quantity | buggy archive | correct (and what the manuscript says) |
|---|---|---|
| pcu lost / sra lost / acs lost | 684 / 292 / 266 | **265 / 825 / 213** |
| bcu / fof / tbo lost | 94 / 42 / 0 | **54 / 24 / 1** |
| excluded-row mean uptake | 2.0460 | **2.6852** |
| retained-row mean uptake | 2.6880 (actually the first 8,206 rows' mean) | **2.5804** |
| `max_uptake_*` | listed acs/bcu/fof/pts — the topologies the elite file **keeps** | **nbo 1.954 / etb 1.949 / rht 0.776 / tbo 2.273**, the four §3.12 argues about |
| multi-topology compositions | 410 (the full-database total) | **131** (the elite subset's), with 410 kept under a separate key |

Fixed by keying on `MOFname`, by adding hard assertions (key present, no duplicate keys,
`retained + lost == qualifying`, neither side empty) so the positional fallback can no
longer run silently, and by splitting the two multi-topology counts into separate keys.
The step is now registered in level 1 of `run_all.py`.

### (b) `partner_coverage_arm_e.py` recorded arm (f)'s `n_train` as 50,000

Arm (f) builds a boolean mask `m2` of length `len(tv)` (= 50,000) and passed *that* to
`fit_eval`, whose helper computes `n_train` as `len(idx)`. So every arm-(f) row recorded
`n_train = 50000`, the pool size, where the manuscript and SI say 10,656 (the five-seed
mean of arm (b)'s training size, which arm (f) matches by construction). Passing the
index array `pick` instead fixes it — `X.iloc[pick]` and `topo.iloc[pick]` select the same
rows — and **changes the order of rows fed to the forest**, so arm (f)'s MAE moved in the
second decimal. The regenerated values, now in §3.4's arm-(e) table and in SI Table S8
supplement, are:

| arm (f) | before | after (5 seeds / seed 42) |
|---|---|---|
| nbo MAE | 0.1256 ± 0.0091 | **0.1250 ± 0.0073** / 0.1211 |
| pcu MAE | 0.2295 ± 0.0045 | **0.2304 ± 0.0045** / 0.2322 |
| nbo `n_train` (seed 42) | 50,000 (wrong) | **10,647** |
| pcu `n_train` (seed 42) | 50,000 (wrong) | **10,647** |

The script now asserts `n_train(f) == n_train(b)` for every (topology, seed), and takes
`--arms {a,b,e,f}` and `--n-jobs N` (default 1, because joblib's process pool cannot
create the named pipes it needs under a confined sandbox, and `n_jobs` does not change a
fit at fixed `random_state`). Re-running one arm alone then merging is a supported path:
`--arms f` writes `arm_e_partner_coverage_f.csv`.

### (c) `elite_subset_confound.py` and the elite fine missing-value convention

Table 17 declares its fine row as "6,214 compositions over the 8,145 annotated rows; the
61 unannotated rows are dropped". The script used the shared `confound_report`, which
does `fillna("NA")`, giving 6,275 over 8,206. The archive now carries **both**: `elite_fine`
(the declared drop convention, 6,214 / 8,145) and `elite_fine_fillna` (6,275 / 8,206). The
full-database fine count of Table 8 (202,481) uses the *fill* convention, and both the
manuscript and this file now say so, because the two fine figures are not interchangeable.

### (d) The checksum manifest is now generated

`outputs/MD5SUMS.txt` was hand-maintained and had drifted: it claimed to cover every file
while listing 65 of 78 (missing, among others, `leakage_results_chemistry.csv` and
`group_stats_chemistry.csv`, which Table 7 names as its source, and
`null_elite_permutation_row.csv`, which SI §S3 names), and its entry for
`arm_e_partner_coverage.csv` carried a hash and a byte count that no longer matched the
file. `scripts/make_md5sums.py` now regenerates it from the archive, `--check` verifies
it, and `run_all.py` calls it at the end of every level. Current state: **78 files, all
matching**.

### (e) Verification of the §3.12 defect in the other direction — two false alarms

Two further "archive vs text" disagreements were checked and resolved **in favour of the
text**, so nothing was changed:

- §3.12's "excluded rows' mean uptake is higher than the retained rows'" (2.69 against
  2.58) is **correct**; the buggy 2.046 figure came from defect (a) above.
- Table 22's ARC-MOF fine row differs from `sample_size_granularity.csv` in **one digit
  only**: the archive gives 99.7489% at n = 200,000, which rounds to 99.7%, and the table
  printed 99.8%. That single cell was corrected in the manuscript; the other three cells
  (99.9 / 99.9 / 99.8) were already right.

---

## 4. Manuscript item → output file → producer

> **Figure files were renamed on 2026-10-11** so that `figN_*` matches the manuscript's
> Figure N (and `figS*_*` the SI's Figure SN). The plotting scripts write these final
> names directly. `outputs/` ships only the two SI PNGs (`figS6_source_confound.png`,
> `figS7_sg_curve.png`); the other thirteen live in `01_manuscript/figures/` (embedded
> in the PDFs) and are regenerated into `outputs/` by the levels below.
> `FIGURE_MANIFEST.md` is the authoritative caption-number ↔ file mapping.

### Figures

| Figure | Content | Produced by | File |
|---|---|---|---|
| Figure 1 | leakage magnitude for two targets, same database and model | level 2, `plot_main_figure.py`, then `fig3_fig4_missing.py` redraws the submitted version from the same two CSVs | `fig1_leakage_two_targets.png` |
| Figure 2 | chemistry–topology confound structure and pairwise Jaccard | level 1, `fig_confound_and_control.py` | `fig2_confound_structure.png` |
| Figure 3 | controlled isoreticular contrast (pcu ↔ nbo), 5 seeds | level 1, `fig_confound_and_control.py` | `fig3_isoreticular_control.png` |
| Figure 4 | test-set R² vs nearest-neighbour distance to the training set | level 5, `leakage_mechanism.py` | `fig4_nn_distance.png` |
| Figure 5 | predicted vs observed, random split and topology hold-out | level 5, `figures_and_shap.py` | `fig5_pred_vs_actual.png` |
| Figure 6 | split effect vs descriptor-only R², 10 targets | level 3, `fig_multitarget.py` | `fig6_penalty_vs_geometry.png` |
| Figure 7 | the four cells of the 2×2 design | level 3, `fig_multitarget.py` | `fig7_multitarget_bars.png` |
| Figure 8 | replication in a second database (three panels) | level 4, `fig_cross_database.py` | `fig8_cross_database.png` |

**All 15 figures come out of levels 1–5.** The SI ones: `figS1_shap_family.png` (level 5,
`figures_and_shap.py`, redrawn from CSV by `plot_shap_family.py`),
`figS2_shap_dependence.png` (level 5, `figures_and_shap.py`), `figS3_within_vs_cross.png`
(level 5, `plot_within_cross.py`), `figS4_aggregation_stability.png` (level 4,
`fig_cross_database.py`, from the level-2 multi-seed CSV), `figS5_aggregation_rules.png`
and the submitted `fig1_leakage_two_targets.png` (level 2, `fig3_fig4_missing.py`, last
two steps of the level — plotting only, they recompute no number),
`figS6_source_confound.png` (level 4, `fig_source_confound.py`), `figS7_sg_curve.png`
(level 2, `fig_sg_curve.py`, from the archived `s_g_curve.csv`).

The one script deliberately left **outside** every level is `s_g_curve.py`, the *data*
step behind Figure S7 and SI Table S9: it re-derives those numbers from the raw 148 MB
hMOF file with its own 100,000-sample run, so it costs about as much as the first step of
level 2 but is needed only if you want to re-derive Table S9 rather than re-plot it. Run
it and then `fig_sg_curve.py` from `scripts/` if you change that analysis.
`FIGURE_MANIFEST.md` maps every figure to its SI section.

Not numbered in the manuscript (produced alongside the above):
`fig9_screening_utility.png` (§3.11 — valid but uncaptioned, see `FIGURE_MANIFEST.md`),
`leakage_comparison_final.png` and `loto_per_metal_final.csv` (§3.1 — intermediate views
of the baseline run).

### Tables

> 本表按**当前主稿编号**重建（2026-10-13）：编号与题注以主稿 §7 的 26 行清单为唯一权威。
> Table 1–6 是描述性表（文献、数据、定义、设计、描述符、聚合口径），没有结果文件；
> 已撤回的两项（CGCNN 对照、随机压缩粒度对齐）不再占表号，分别在 SI Table S2 与 §4.3
> 交代。SI 的表格为 Table S1–S9、分节 §S1–S15，见 `02_supplementary/SI.md`。

| Table | Content | Produced by | Archived file |
|---|---|---|---|
| 1 | three 2026 studies closest to this work | — (literature) | — |
| 2 | data sources, sizes and modelling samples | — (from `DATA.md`) | — |
| 3 | the two definitions of chemical composition | — (definition) | — |
| 4 | controlled-contrast design (four arms) | — (design) | — |
| 5 | the eight structure-derived descriptors | — (feature list) | — |
| 6 | the four aggregation rules | — (definition) | — |
| 7 | random split vs three grouped splits (100,000 samples, 426 features) | level 2, `mof_leakage_benchmark.py --loto --tag final` | `leakage_results_final.csv`, `group_stats_final.csv` |
| 8 | confound on the full hMOF database (324,426) | level 1, `full_db_confound.py --tag full` | `confound_full.csv`, `jaccard_full.csv` |
| 9 | chemistry seen/unseen per held-out topology | level 1, `chemistry_seen.py` | `chemistry_seen_by_topology.csv` (`chemistry_seen.csv` = totals) |
| 10 | chemical overlap of the 55 topology pairs | level 1, `chemistry_seen.py` | `topology_pair_overlap.csv` |
| 11 | controlled contrast, five seeds, incl. the size-matched arm | level 1, `confound_diagnosis.py --seeds 5` (+ `--size-match`) | `isoreticular_control_seeds.csv`, `isoreticular_control.csv`, `isoreticular_control_sizematched.csv` |
| 12 | LOTO / LOMO R² under four aggregation rules | level 2, `metrics_audit.py` | `metrics_audit.csv` |
| 13 | per-topology leave-one-out detail | level 2 (`--loto`) | `loto_per_topology_final.csv` |
| 14 | aggregation-rule stability across five seeds (25,000) | level 2, `multi_seed_repeat.py --tag uptake25k_fix` | `multi_seed_uptake25k_fix.csv` |
| 15 | omitted-variable analysis (η², ω², permutation null) | level 5, `omitted_variable.py`, `cardinality_correction.py` | `omitted_variable_final.csv`, `cardinality_correction.csv` |
| 16 | screening utility on a shared support set | level 3, `screening_utility.py` | `screening_utility.csv` |
| 17 | confound on the release's elite file | level 1, `elite_subset_confound.py` | `elite_subset_confound.csv`, `elite_subset_jaccard.csv` |
| 18 | the 2×2 design: median R² per cell, ten targets | level 3, `multitarget_confound.py` | `multitarget_confound.csv` |
| 19 | the three effects of the 2×2 design | level 3, `derive_multitarget_stats.py` | `multitarget_derived_stats.csv` |
| 20 | leave-one-source-out sensitivity of the ARC-MOF confound | level 4, `arcmof_overlap.py` | `arcmof_overlap.csv`, `confound_arcmof_excl_derived.csv` |
| 21 | cross-database confound statistics | level 4, `arcmof_confound.py` | `confound_arcmof.csv`, `jaccard_arcmof.csv` |
| 22 | the two databases at a glance; the same-n comparison | level 4, `sample_size_granularity.py` | `sample_size_granularity.csv` |
| 23 | per-source confounding across ARC-MOF sub-libraries | level 4, `source_confound_granularity.py` | `source_confound_granularity.csv` |
| 24 | cross-source spread vs a one-common-rate null | level 4, `source_spread_test.py` | `source_spread_test.csv` |
| 25 | identity effect under three grouping variables | level 3, `multitarget_confound.py --group-col {metal,linker_pair}` | `multitarget_confound.csv`, `multitarget_confound_metal.csv`, `multitarget_confound_linker_pair.csv` |
| 26 | availability of each identity block under each split | level 3, `chemistry_seen_grouping.py` | `chemistry_seen_grouping.csv` |

SI tables: S1 ← `fig7_nn_curve_final.csv`（level 5, `leakage_mechanism.py`）；
S2 ← 已撤回的 CGCNN 对照（无存档数值，只有诊断结论）；S3 ← `mitigation_results_mit25k.csv`
（level 2）；S4 / S5 ← `within_vs_cross_final.csv`（level 5, `omitted_variable.py
--within-vs-cross`）；S6 ← `multitarget_confound.csv`（level 3）；
S7 ← `confound_full.csv`（level 1；1,622 / 202,481 两档组成计数）；
S8 ← `isoreticular_control.csv`（level 1；seed 42 行）；S9 ← `s_g_curve.csv`
（standalone `s_g_curve.py`）。

Provenance-only files（在包内但不被正文引用）：`chem_topo_confound.csv`、
`chem_topo_jaccard.csv`（Table 8 之前的 n = 50,000 视角，已被全库视角取代；
Figure 2 的 pcu–nbo Jaccard 取自后者），以及 `leakage_results_logsel100k.csv`
（Figure 1 的第二目标）。`confound_by_source_arcmof.csv` **不随包发布**：每个来源
库当初各用自己的拓扑列，跨来源不可比，请用 `source_confound_granularity.csv`。

---

## 5. Full command list (equivalent to the levels, for step-by-step checking)

```bash
cd 03_reproduction_package        # 是包根，不是 scripts/
# ---- level 1: core confound evidence (§3.2 / §3.4 / §3.12)
python scripts/confound_diagnosis.py --max-rows 50000 --size-match   # 50,000 view + controlled contrast + size-matched arm
python scripts/full_db_confound.py --tag full                        # full-database view (the headline numbers)
python scripts/confound_diagnosis.py --max-rows 50000 --seeds 5      # 5-seed version quoted in the abstract
python scripts/elite_subset_confound.py
python scripts/fig_confound_and_control.py

# ---- level 2: results, aggregation rules, seeds (§3.1 / §3.5 / §3.9)
python scripts/mof_leakage_benchmark.py --max-rows 100000 --loto --tag final
python scripts/mof_leakage_benchmark.py --max-rows 100000 --target selectivity --log-target --tag logsel100k
python scripts/plot_main_figure.py --outdir ../outputs
python scripts/metrics_audit.py
python scripts/multi_seed_repeat.py --max-rows 25000 --seeds 5 --skip-group --tag uptake25k_fix
python scripts/mof_leakage_benchmark.py --max-rows 25000 --mitigation --tag mit25k

# ---- level 3: practical consequences (§3.11 / §3.13 / §3.14 / §4.2)
python scripts/screening_utility.py --max-rows 25000
python scripts/multitarget_confound.py --max-rows 25000 --top-linkers 200 \
       --resid-estimators 60 --outdir ../outputs
python scripts/fig_multitarget.py

# ---- grouping sensitivity of the identity effect (§4.2; ~25 min each)
# topology is the default and writes multitarget_confound.csv; the other two
# write multitarget_confound_<group>.csv and leave the main table untouched.
python scripts/multitarget_confound.py --max-rows 25000 --top-linkers 200 \
       --resid-estimators 60 --outdir ../outputs --group-col metal
python scripts/multitarget_confound.py --max-rows 25000 --top-linkers 200 \
       --resid-estimators 60 --outdir ../outputs --group-col linker_pair

# ---- level 4: cross-database replication (§3.15; needs the ARC-MOF files)
python scripts/arcmof_confound.py --top-n-topo 40 --out-tag arcmof
python scripts/exclusivity_across_databases.py
python scripts/source_confound_granularity.py --min-source-n 200 --boot 10   # --boot, NOT --seeds
python scripts/source_spread_test.py                                          # valid spread test (see note)
python scripts/arcmof_overlap.py
python scripts/sample_size_granularity.py
python scripts/fig_cross_database.py
python scripts/fig_source_confound.py

# ---- null-model controls (§3.12 / §3.15 Result 3; standalone, not part of a level)
python scripts/null_model_controls.py --outdir outputs                --granularity coarse -b 10000 --b-card 1000
python scripts/null_model_controls.py --outdir outputs/_fine          --granularity fine   -b 10000 --b-card 1000
python scripts/null_model_controls.py --outdir outputs/common_support --granularity coarse -b 10000 --b-card 1000 --require-cols mc,flig,func

# ---- level 5: mechanism analyses + SHAP (slow, optional)
python scripts/leakage_mechanism.py --max-rows 25000 --family --tag final
python scripts/omitted_variable.py --max-rows 25000 --within-vs-cross --tag final
python scripts/plot_within_cross.py
python scripts/figures_and_shap.py --max-rows 25000 --shap-samples 200 --tag final
python scripts/plot_shap_family.py --csv ../outputs/fig5_shap_family_final.csv \
       --out ../outputs/figS1_shap_family_final.png
```

> Why `--max-rows 50000` and not 25,000 in level 1: several reference numbers in
> §3.2 are quoted at n = 50,000, and the n = 25,000 figure (98.9%) is a different
> number that must not be mixed with them. The headline number of the paper is
> computed on the **full** database by `full_db_confound.py`, precisely because
> the statistic depends on n.

---

## 6. Known pitfalls (each one already handled in the code — do not revert them)

1. **Never use `head()` on either CSV.** Both hMOF files are sorted by topology;
   the first rows contain only a handful of topologies. Sampling goes through
   `df.sample(..., random_state=42)` everywhere. (The CIF bundle is worse: its
   first 35,000 entries are all pcu.)
2. **Take logs of ratio targets.** CO₂/N₂ selectivity has a median of 8.2 and a
   maximum of 31,634; without the log transform R² swings between +0.42 and
   −0.06 depending on the draw.
3. **Always report four aggregation rules.** `pooled` (concatenate every fold's
   predictions, then compute one R²) is dominated by the largest group: with pcu
   holding 63% of hMOF, pooled gives +0.381 while macro gives −0.541, and
   excluding pcu the remaining folds average −0.678 — the **sign reverses**.
4. **Check target variance before comparing R² across splits.** Our own CGCNN
   experiment failed on exactly this: the two test sets differed by a factor of
   7.7 in target variance, R² looked catastrophic (−0.448) while MAE was in fact
   *better* on the held-out side (0.262 vs 0.396). That arm has been withdrawn.
5. **Changing the support set changes the denominator.** Metrics on different
   subsets cannot be subtracted. §3.11 therefore compares all four splitting
   strategies on out-of-fold predictions for the *same* 25,000 samples, against
   the same oracle.
6. **Fail loudly on empty frames.** `build_features()` one-hot encodes only
   `functional_groups`, `metal_linker` and `topology` — it does **not** emit
   columns with a `linker_pair_` prefix, so a prefix-based selection silently
   returned an empty frame and the check that used it was skipped without error.
   Likewise, ARC-MOF's `all_topology_lists.csv` has its **first two column names
   swapped** (`Name` holds CIF filenames, `filename` holds topology labels), which
   made every merge empty. Both now assert.
7. **Verify MD5 after downloading.** Two files were truncated during a parallel
   download while `curl` still exited 0. Materials Cloud has also served HTTP 503
   for every record, returning a small HTML error page that would otherwise be
   appended straight into a data file. Check the HTTP status and the file tail.
8. **Do not align chemical granularity by random compression.** Random coarsening
   can only make a chemical definition *coarser*, never finer, so the larger
   library is compressed harder and the apparent spread explodes (we measured
   406%). Versions 1 and 2 of `source_confound_granularity.py` failed this way
   and are withdrawn. Version 3 uses two **real** chemical definitions plus
   bootstrap resampling of the *samples* — never of the chemistry.
9. **Align on r = n / M (samples per distinct composition), not on absolute n.**
   The two databases differ by two orders of magnitude in M; aligning on raw
   sample count aligns nothing.
10. **Report the number of valid folds.** When a test fold has zero target
    variance, R² is undefined; silently keeping NaN turns `macro` into NaN. This
    affects 3 of our 5 seeds, so the effective fold count is 10–11, not always 11.
11. **Never test a source-level difference using bootstrap draws as replicates.**
    An earlier version passed the 10 bootstrap replicates *per source* to
    Kruskal–Wallis. Those draws are resamples of one sample, not independent
    observations, so the resulting p-value is pseudo-replicated and has been
    withdrawn. `source_spread_test.py` simulates the correct null instead (one
    common rate; the statistic is the variance of the per-source proportions, and
    `scipy.stats.kruskal` on the *sources* would give the same answer on this data).
    Rule of thumb: if your replicates come from the same underlying sample, they are
    not a sample of sources.
12. **A missing value is not the string `"nan"` — pandas 3.0 changed this.**
    Under pandas < 3, `df[cols].astype(str).agg("|".join, axis=1)` silently turned a
    missing entry into the literal `"nan"`, so rows with no `functional_groups`
    became their own "chemical composition". Under pandas 3.0 the same expression
    keeps `NaN` and raises `TypeError: sequence item N: expected str instance,
    float found`. Build composition keys only after an explicit
    `dropna(subset=cols)` (this is what `null_model_controls.py` does). Concretely:
    61 of the 8,206 elite hMOF rows carry no `functional_groups`, which moves the
    fine-grained composition count M from **6,214** (missing rows dropped) to
    **6,275** (missing counted as a composition); the manuscript quotes the
    dropped-value figure, 6,214.

---

## 7. Analyses deliberately excluded

These scripts remain in the repository for provenance. They are **not** run by
`run_all.py` and their numbers are **not** used anywhere in the manuscript.

| Script | Why it is excluded |
|---|---|
| `cgcnn_mof.py` | Evidence withdrawn on 2026-10-09: the two test sets differed 7.7× in target variance and the held-out side had *smaller* MAE, so the original conclusion pointed the wrong way (§3.7, withdrawn Table 4). |
| `granularity_control.py` | The alignment method this paper itself rules out (§4.3 granularity declaration / withdrawn Table 13). `outputs/granularity_control.csv` is kept only to trace the history. |
| `chemistry_topology_factorial.py` | Design 2 is void: 387 of the 388 "multi-topology" compositions span only pcu ↔ nbo, and nbo's target variance is 1/29 of pcu's, so any comparison is meaningless (§3.3). |
| `source_confound_granularity.py`, v1 and v2 | v1 filtered topologies with a source-size-dependent threshold (DB0 down to 5 topologies, DB6 to 1, where the statistic is trivially 100%); v2 used the withdrawn random-compression alignment. The current file is v3. Do not revert. |
| `data/fetch_db2.sh`, `data/fix_part6.sh` | Download helpers for a Materials Cloud archive that returned 503 for every record; not needed. |

---

## 8. What we deliberately do *not* claim

Stated here so that a reader cannot misread the pipeline as evidence for
something larger than it is:

- ARC-MOF is **not** independent of hMOF. It spans all public MOF databases and
  shares authors with hMOF, and its v6 notes mention removing structures from the
  Boyd–Woo database. Level 4 therefore does **not** claim an independent
  replication; `arcmof_overlap.py` bounds the effect of any overlap by removing
  one source sub-library at a time (98.6–99.8%).
- Cross-database magnitudes are **not comparable** unless the target, the
  granularity of "chemistry" and the aggregation rule are declared together.
- The controlled contrast rests on **one** isoreticular pair (pcu ↔ nbo).
- One set of structure-derived scalar descriptors is used; there is no deep-learning
  arm in the current results (withdrawn, §3.7). The random-versus-topology gap was
  refitted under gradient boosting and ridge (§3.1), so it is not an artefact of the
  model family — but model capacity itself remains untested.
