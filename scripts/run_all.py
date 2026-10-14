#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
One-command reproduction of the results assigned to each manuscript pipeline level.

    Loss-of-Generality note
    -----------------------
    This pipeline reproduces exactly the results reported in the manuscript.
    Two analyses are deliberately NOT run here, because their evidence was
    withdrawn by the authors themselves:

        * the CGCNN deep-learning arm   (scripts/cgcnn_mof.py)
        * the random-compression granularity alignment (scripts/granularity_control.py)

    See REPRODUCE.md, section "Analyses deliberately excluded", for the reason
    and for the file each one used to produce.

Manuscript claim
----------------
    In the two enumerated MOF databases studied here (hMOF and ARC-MOF), holding
    out a topology removes substantial chemistry overlap: the two variables are
    structurally entangled, so the field's gold-standard "topology hold-out"
    protocol should not be read as an isolated test of topological transferability.
    (The scope is the two databases studied — the title says so, and §4.4 states
    what a generalisation beyond them would require.)

Levels (timings: L1/L4 measured 2026-10-10 on a laptop CPU, 8 GB RAM;
L2/L3 measured 2026-10-13 on a 20-core desktop, 16 GB RAM; both Windows 11)
---------------------------------------------------------------------------
    python run_all.py --level 1            # 7.3 min   core confound evidence (Tables 8-11, 17; Figures 2-3)
    python run_all.py --level 2            # 65.4 min  main results, aggregation audit, seeds (Tables 7, 12-14, S3; Figures 1, S5, S7)
    python run_all.py --level 3            # 78.5 min  practical consequences & 10 targets (Tables 16, 18, 19, 25, 26; Figures 6-7)
    python run_all.py --level 4            # 29.9 min  cross-database replication (Tables 20-24; Figures 8, S4, S6)
    python run_all.py --level 5            # hours*    mechanism analyses + SHAP (Table 15; SI Tables S1-S2, S4-S5; Figures 4-5, S1-S3)
    python run_all.py --level 1 --dry-run  # print the commands only
                                           # * not benchmarked; see the breakdown below

    Measured breakdown (only for levels actually benchmarked):
        L1 = 7.3 min  : confound_diagnosis 2.1 / seeds 4.8 / elite 0.4 / figures 0.05
                        (measured 2026-10-10, laptop, 8 GB RAM)
        L2 = 65.4 min : 100k+LOTO 34.7 / log-selectivity 100k 17.1 / plot 0.03
                        / metrics_audit 0.02 / multi-seed 8.5 / mitigation 5.1
                        (measured 2026-10-13, 20-core desktop, 16 GB RAM)
        L3 = 78.5 min : screening 2.3 / 2x2 ten targets 21.8 / figures 0.05
                        / grouping 0.5 / composition-matched 45.1 / arm (e) 5.5
                        / gradient-boosting 2.4 / ridge 0.9
                        (measured 2026-10-13, same 20-core desktop)
        L4 = 29.9 min : arcmof_confound 17.0 / exclusivity 1.7 / source_confound 10.4
                        / overlap 0.4 / sample_size 0.4 / figures 0.1
                        (measured 2026-10-10, laptop, 8 GB RAM)
        L5 : NOT benchmarked end-to-end. Its dominant cost is cardinality_correction.py
             --perm 3, for which the manuscript itself records ~34 min per full pass
             (note to Table 15), i.e. ~1.7 h for the three permutations alone, on top
             of the SHAP run; expect several hours.

    Verified reproducibility: on 2026-10-10, level 4 was re-run from scratch and
    every output was byte-identical to the previous run (cmp, 8/8 files). On 2026-10-13
    levels 2 and 3 were re-run from scratch on newer library versions (pandas 3.0.6,
    numpy 2.5.3, scikit-learn 1.9.1) and reproduced every reported number, but NOT
    byte-for-byte: agreement is 1e-16 to 1e-15 on most files, 5.6e-7 in one auxiliary
    column, with three expected structural differences. See REPRODUCE.md section 3c.

Data requirements
-----------------
    L1, L2, L3, L5 : data/all_MOFs_screening_data.csv     (hMOF, 324,426 rows)
                     data/top_MOFs_screening_data.csv     (elite subset, 8,206 rows)
    L4             : data/arcmof/{all_topology_lists,mc-clusters,flig-clusters,
                                  func-clusters}.csv      (ARC-MOF v6 metadata)
    See DATA.md for URLs, MD5 checksums and two known data traps.

Environment
-----------
    pip install -r requirements.txt      # 5 packages, CPU only, no GPU, no torch
                                         # (+ shap for level 5, see requirements.txt)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
# 2026-10-14 修：子进程的工作目录必须是 **包根**，不是 scripts/。
# 本包里有十几个脚本的默认路径是 CWD 相对的（`--csv data/...`、`--outdir outputs`），
# 它们指的是 <package>/data 与 <package>/outputs；而少数脚本（null_model_controls.py）
# 用 HERE 相对路径，两者都只在 CWD = 包根时同时成立。
# 原版 `cwd=HERE` 会让 `python run_all.py --level 1` 在第一个
# `confound_diagnosis.py --csv data/...` 步骤上直接 FileNotFoundError。
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
OUTDIR = os.path.join(HERE, "..", "outputs")
LOGDIR = os.path.join(OUTDIR, "logs")

PY = sys.executable

# ------------------------------------------------------------------ task table
LEVELS: dict[int, list[tuple[str, list[str]]]] = {
    # ======================= L1: core confound evidence =======================
    1: [
        # Runs two things in one pass so that the ~2 min feature build is paid once:
        #   - the confound statistics of sec. 3.2 (single-topology chemistry, H(T|C), Jaccard)
        #   - the controlled isoreticular contrast of sec. 3.4 (same test set, training set only changes)
        # --size-match adds the training-size-matched arm that answers "is it the *number*
        # of removed samples or their *identity*?" (reviewer point R1). Do not drop it.
        # NOTE: 50,000 is required here, not 25,000: several reference numbers in sec. 3.2
        # are quoted at n = 50,000. The headline number of the paper, however, comes from
        # the full database and is produced by the next step.
        ("[sec 3.2 + 3.4] chemistry-topology confound at n = 50,000 "
         "+ controlled isoreticular contrast (pcu <-> nbo) + size-matched arm "
         "-> chem_topo_confound.csv, chem_topo_jaccard.csv, "
         "isoreticular_control.csv, isoreticular_control_sizematched.csv",
         ["confound_diagnosis.py", "--max-rows", "50000", "--size-match"]),

        # The headline numbers of the abstract and Table 7 are computed on the FULL
        # database (324,426 structures), because the statistic is sample-size dependent.
        ("[sec 3.2, Table 8] confound statistics on the full database (no sampling) "
         "-> confound_full.csv, jaccard_full.csv",
         ["full_db_confound.py", "--tag", "full"]),

        # sec 3.2, Table 9: answers the question the title actually poses — of the
        # samples a topology hold-out removes, how many have chemistry that appears
        # nowhere in the remaining training set? Pure counting, no model. Also produces
        # the per-held-out-topology decomposition that carries the "7 of 11 topologies
        # are pure chemistry hold-outs" claim.
        ("[sec 3.2, Tables 9-10] per-held-out-topology chemistry seen/unseen decomposition "
         "-> chemistry_seen.csv, chemistry_seen_by_topology.csv, topology_pair_overlap.csv",
         ["chemistry_seen.py"]),

        # Five independent seeds (sampling + split + model all change). Source of
        # "MAE 0.1065 +- 0.0032 -> 0.1811 +- 0.0136 (+70.0%)" in the abstract.
        ("[sec 3.4, Table 11] controlled contrast with 5 independent seeds "
         "-> isoreticular_control_seeds.csv",
         ["confound_diagnosis.py", "--max-rows", "50000", "--seeds", "5"]),

        # Pre-empts the rebuttal "this only happens database-wide". It does not: on the
        # elite subset the confound is total (H(T|C) = 0.000 bit).
        ("[sec 3.12, Table 17] is the confound diluted where screening actually selects? "
         "-> elite_subset_confound.csv, elite_subset_jaccard.csv",
         ["elite_subset_confound.py"]),

        # sec 3.12 attributions: qualifying / retained / lost per topology, the two mean
        # uptakes, the per-topology maxima used in the threshold argument, and the 131
        # multi-topology compositions of the elite subset. Registered 2026-10-14: Table 17's
        # caption points at this output, and the script had been matching rows on the index
        # instead of on the MOFname key, so the archived per-topology losses and the two
        # means disagreed with the text. The script now asserts on the key and on the
        # retained + lost == qualifying identity, and fails loudly rather than silently
        # falling back to positional matching.
        ("[sec 3.12, Table 17] elite vs full-database attribution "
         "-> elite_vs_full.csv, elite_vs_full_by_topology.csv",
         ["elite_vs_full.py"]),

        ("[figures] Figure 2 (confound structure), Figure 3 (controlled contrast)",
         ["fig_confound_and_control.py"]),
    ],

    # ================ L2: main results, aggregation rules, seeds ===============
    2: [
        # Baseline: random < linker-pair < metal < topology. Also produces the per-group
        # leave-one-out tables (Tables 12, 13) and a side-by-side comparison panel.
        ("[sec 3.1, Table 7] main result at n = 100,000: four splitting strategies "
         "+ leave-one-topology-out / leave-one-metal-out per group "
         "-> leakage_results_final.csv, leakage_comparison_final.png, "
         "loto_per_topology_final.csv, loto_per_metal_final.csv, group_stats_final.csv",
         ["mof_leakage_benchmark.py", "--max-rows", "100000",
          "--loto", "--tag", "final"]),

        # Second target on the same database and model, needed for Figure 1 and for the
        # "magnitude is a property of the target" argument.
        ("[sec 3.1] same experiment on log10(CO2/N2 selectivity), n = 100,000 "
         "-> leakage_results_logsel100k.csv",
         ["mof_leakage_benchmark.py", "--max-rows", "100000",
          "--target", "selectivity", "--log-target", "--tag", "logsel100k"]),

        ("[figure] Figure 1 (two targets, same database and model)",
         ["plot_main_figure.py", "--outdir", "outputs"]),

        # The aggregation audit of Table 12: pooled / weighted / macro / median disagree
        # in sign on the very same predictions.
        ("[sec 3.5, Table 12] four aggregation rules for the same leave-one-topology-out run "
         "-> metrics_audit.csv",
         ["metrics_audit.py"]),

        # Five seeds where even the sampling changes: shows that pooled LOOKS stable
        # (4.2% dispersion) while macro is not (91.5%) — stability that is an artefact
        # of the largest group.
        ("[sec 3.5, Table 14] five-seed stability of each aggregation rule "
         "-> multi_seed_uptake25k_fix.csv",
         ["multi_seed_repeat.py", "--max-rows", "25000", "--seeds", "5",
          "--skip-group", "--tag", "uptake25k_fix"]),

        # SI Table S3: removing identity roughly does NOT close the gap; target encoding widens it.
        ("[sec 3.9, SI Table S3] mitigation strategies under a fixed protocol "
         "-> mitigation_results_mit25k.csv",
         ["mof_leakage_benchmark.py", "--max-rows", "25000",
          "--mitigation", "--tag", "mit25k"]),

        # Figure 1 (the submitted version) and Figure S5 are redrawn from the two CSVs
        # produced above. This step recomputes NOTHING — it only plots archived numbers —
        # and takes seconds. It is registered in this level (it used to be a standalone
        # script) so that every figure in the paper comes out of `run_all.py`.
        # NOTE: plot_main_figure.py above draws Figure 1 from the same two CSVs; this
        # last step overwrites it with the version that was submitted.
        ("[figures] Figure 1 (final redraw) and Figure S5 (aggregation rules)",
         ["fig3_fig4_missing.py"]),

        # Figure S7 is drawn from the archived s_g_curve.csv and takes seconds. Its DATA
        # step, s_g_curve.py, is deliberately left out of every level: it re-derives
        # Table S9 from the raw 148 MB hMOF file with its own 100,000-sample run, i.e. it
        # costs about as much as the first step of this level. Its numbers are archived,
        # so the figure never needs it — see REPRODUCE.md section 4.
        ("[figure] Figure S7 (S_g curve, redrawn from the archived CSV)",
         ["fig_sg_curve.py"]),

        # R1-M1 resolution: composition hold-out arm (metal+2 linkers, 1,622 groups).
        # This is the only experiment that can directly test whether topology hold-out
        # and chemistry hold-out give the same numbers at the outcome level.
        ("[sec 3.1, Table 7 supplement] composition hold-out arm "
         "(GroupKFold + LOTO on metal+2 linkers) "
         "-> leakage_results_chemistry.csv, loto_per_chemistry.csv",
         ["mof_leakage_benchmark.py", "--max-rows", "100000",
          "--loto", "--tag", "chemistry"]),
    ],

    # ================= L3: practical consequences and 10 targets ===============
    3: [
        # R2 traps (target variance differs per split), so ranking metrics are computed on
        # out-of-fold predictions for the SAME 25,000 samples: one support set, one oracle.
        ("[sec 3.11, Table 16] screening utility on a shared support set "
         "(overlap@k, enrichment, regret@k) "
         "-> screening_utility.csv, fig9_screening_utility.png",
         ["screening_utility.py", "--max-rows", "25000"]),

        # R1-M1 resolution: chemistry arm of the screening-utility panel, same 25,000
        # samples, same model, only the split changes. Directly comparable to Table 16.
        # The script auto-detects the chemistry group from build_features(); --tag
        # produces a separate CSV so Table 16 and this supplement do not overwrite each other.
        ("[Table 16 supplement] screening utility under composition hold-out "
         "(same 25,000 samples, chemistry as grouping variable) "
         "-> screening_utility_chemistry.csv",
         ["screening_utility.py", "--max-rows", "25000",
          "--strategies", "random,topology,linker_pair,metal,chemistry", "--tag", "chemistry"]),

        # Tables 18 / 25 / sec 3.13-3.14: the 2x2 design that separates the split effect
        # from the identity-feature effect. All four cells land in one CSV; the two
        # grouping-variable variants feed the identity-effect table (Table 25).
        ("[sec 3.13 + 3.14, Tables 18 + 25] ten targets in a 2x2 design "
         "(split effect vs identity effect) "
         "-> multitarget_confound.csv",
         ["multitarget_confound.py", "--max-rows", "25000",
          "--top-linkers", "200", "--resid-estimators", "60",
          "--outdir", "outputs"]),

        ("[figures] Figure 6 (split effect vs descriptor-only R2), "
         "Figure 7 (four cells of the 2x2)",
         ["fig_multitarget.py"]),

        ("[sec 3.13, Table 19] derive the three 2x2 effects and their statistics "
         "-> multitarget_derived_stats.csv",
         ["derive_multitarget_stats.py"]),

        # Table 26 / sec 4.2: the mechanism behind the identity-feature effect, measured
        # instead of inferred. Counts, per split, how much of each identity block is
        # actually available to a test sample (i.e. its category value occurs in the
        # training fold). No model refitting.
        ("[sec 4.2, Table 26] availability of each identity block under each split "
         "-> chemistry_seen_grouping.csv",
         ["chemistry_seen_grouping.py", "--max-rows", "25000"]),

        # sec 3.4 / sec 4.4: closes the "training chemical diversity" alternative
        # explanation that the size-matched arm leaves open. Stratifies the draw from
        # arm (a) so that its topology profile matches arm (b) exactly. The resulting
        # four-arm comparison is reported in sec 3.4; reproducing the partner-removed
        # composition profile confines arm (d) to arm (b)'s own row set, so it is an
        # internal-consistency check (see the manuscript note under the table).
        ("[sec 3.4 + 4.4] composition-matched control arm (5 seeds) "
         "-> composition_matched_control.csv, composition_matched_summary.csv",
         ["diversity_control.py"]),

        # sec 3.4 / sec 4.5: arm (e). Removes ONLY the partner-topology rows whose
        # composition occurs in the test set, so matched-partner coverage goes to zero
        # while most of arm (a)'s rows and compositions stay. This is the one design
        # inside hMOF that moves "composition diversity" and "partner coverage" in
        # opposite directions.
        ("[sec 3.4 + 4.5] arm (e) partner-coverage control (5 seeds) "
         "-> arm_e_partner_coverage.csv",
         ["partner_coverage_arm_e.py"]),

        # sec 3.1: model-family control for the Table 7 gap. Same 100,000 samples,
        # same 426-column matrix, same four splits; only the estimator changes.
        ("[sec 3.1] model-family control (gradient boosting, ridge) "
         "-> leakage_results_{gb,ridge}100k.csv",
         ["mof_leakage_benchmark.py", "--max-rows", "100000",
          "--model", "gb", "--tag", "gb100k"]),
        ("[sec 3.1] model-family control (ridge) "
         "-> leakage_results_ridge100k.csv",
         ["mof_leakage_benchmark.py", "--max-rows", "100000",
          "--model", "ridge", "--tag", "ridge100k"]),
    ],

    # ===================== L4: replication on a second database =================
    4: [
        # WARNING: this script also writes confound_by_source_arcmof.csv. That table is
        # NOT used: each source library originally picked its own topology column, so the
        # values are not comparable across sources. Use source_confound_granularity.csv
        # (Crystalnet topology for every source) instead — produced further down.
        ("[sec 3.15, Table 21] same diagnostic on ARC-MOF "
         "-> confound_arcmof.csv, jaccard_arcmof.csv",
         ["arcmof_confound.py", "--top-n-topo", "40", "--out-tag", "arcmof"]),

        ("[sec 3.15] single-topology chemistry, both databases, both granularities "
         "-> exclusivity_across_databases.csv",
         ["exclusivity_across_databases.py"]),

        # The parameter is --boot (number of bootstrap resamples), NOT --seeds.
        # v1 and v2 of this script are withdrawn: v1 filtered topologies per source with a
        # size-dependent threshold (not comparable), v2 aligned granularities by random
        # compression, which is the method this paper itself rules out. v3 uses two real
        # chemical definitions and bootstrap resampling of the SAMPLES only.
        ("[sec 3.15, Table 23] per-source sub-libraries at two real chemical granularities "
         "with bootstrap error bars "
         "-> source_confound_granularity.csv",
         ["source_confound_granularity.py", "--min-source-n", "200",
          "--boot", "10"]),

        # The source-to-source spread must be tested against a NULL of one common rate.
        # Feeding Kruskal-Wallis the 10 bootstrap draws per source is pseudo-replication
        # (the draws are not independent observations) and produced an invalid p-value;
        # that number is withdrawn. This script builds the correct null instead.
        ("[sec 3.15, Table 24] is the source-to-source spread real, or binomial noise? "
         "-> source_spread_test.csv",
         ["source_spread_test.py"]),

        # ARC-MOF is NOT independent of hMOF (it spans all public MOF databases and shares
        # authors with it). Rather than asserting independence, we bound the effect of any
        # overlap by removing one source sub-library at a time.
        ("[sec 3.15, Table 20] hMOF/ARC-MOF overlap test and "
         "leave-one-source-out bound "
         "-> arcmof_overlap.csv, confound_arcmof_excl_derived.csv",
         ["arcmof_overlap.py"]),

        # Replaces the withdrawn random-compression alignment: compare at matched n and
        # real granularity, with no re-compression at all. Feeds the same-n comparison
        # of Table 22.
        ("[sec 3.2 + 3.15, Table 22] sample-size dependence and same-n cross-database view "
         "-> sample_size_granularity.csv",
         ["sample_size_granularity.py"]),

        ("[figures] Figure 8 (cross-database), Figure S4 (aggregation stability)",
         ["fig_cross_database.py"]),

        ("[figure] Figure S6 (per-source stratification)",
         ["fig_source_confound.py"]),
    ],

    # ============== L5: mechanism analyses and SHAP (optional, slow) ===========
    5: [
        # H1 (geometric near-duplication) — the distance explains only a small part of the gap.
        ("[sec 3.6, SI Table S1] test-set R2 binned by nearest-neighbour distance + "
         "family overlap -> fig4_nn_distance_final.png, fig7_nn_curve_final.csv",
         ["leakage_mechanism.py", "--max-rows", "25000",
          "--family", "--tag", "final"]),

        # sec 3.6: the three refuted explanations put on ONE sampling and ONE model, so
        # that the shares of dR2 they account for can be quoted side by side.
        ("[sec 3.6] nearest-neighbour distance / building-block family / SHAP attribution "
         "recomputed on one common run -> refuted_explanations_summary.csv",
         ["refuted_explanations.py", "--max-rows", "25000"]),

        # Omitted-variable analysis (shows the missing variable is linker chemistry, not
        # topology) plus within- vs cross-topology R2.
        ("[sec 3.8 + 3.10, Table 15] omitted-variable analysis and within- vs cross-topology "
         "-> omitted_variable_final.csv, within_vs_cross_final.csv",
         ["omitted_variable.py", "--max-rows", "25000",
          "--within-vs-cross", "--tag", "final"]),

        # Prices the cardinality caveat of the same table: omega-squared for the four
        # eta-squared values, plus a label-permutation null for eta-squared and for the
        # out-of-fold R2 of the 587-group linker pair.
        ("[sec 3.8, Table 15] cardinality correction (omega-squared + permutation null) "
         "-> cardinality_correction.csv",
         ["cardinality_correction.py", "--max-rows", "25000", "--perm", "3"]),

        ("[figure] Figure S3 (within- vs cross-topology R2 per topology)",
         ["plot_within_cross.py"]),

        # Also produces Figure S1 (SHAP family) and Figure S2 (SHAP dependence).
        # Slow and memory-hungry; lower --shap-samples if needed.
        ("[figures] Figure 5 (predicted vs observed), Figure S1 + S2 (SHAP) "
         "-> fig5_pred_vs_actual_final.png, figS1_shap_family_final.png, "
         "figS2_shap_dependence_final.png, fig5_shap_family_final.csv",
         ["figures_and_shap.py", "--max-rows", "25000",
          "--shap-samples", "200", "--tag", "final"]),

        # Redraws the SHAP family panel from the stored CSV in seconds, so that changing
        # the figure does not require repeating the SHAP computation.
        ("[figure] Figure S1 (SHAP attribution by feature family, redrawn from CSV)",
         ["plot_shap_family.py",
          "--csv", "outputs/fig5_shap_family_final.csv",
          "--out", "outputs/figS1_shap_family_final.png"]),
    ],
}

NOTE = {
    2: "Measured 65.4 min (2026-10-13, 20-core desktop; see REPRODUCE.md).",
    3: "Measured 78.5 min (2026-10-13, 20-core desktop; see REPRODUCE.md).",
    4: "Requires the ARC-MOF metadata files listed in DATA.md. Measured 29.9 min "
       "(2026-10-10, laptop, 8 GB RAM).",
    5: "Several hours; the SHAP step is the slow one and needs a fair amount of memory. "
       "NOT benchmarked end-to-end.",
}


def run_one(desc: str, args: list[str], dry: bool, idx: int, total: int):
    """Run one step, stream it to a log file, and report success + wall time."""
    # 2026-10-14 修（第二处）：步骤表里写的是**裸脚本名**（"leakage_mechanism.py"），
    # 这只有 cwd = scripts/ 时才解析得到。既然子进程的 cwd 必须是包根（见 ROOT 注释），
    # 就必须把脚本解析成 HERE 下的绝对路径，否则每个步骤都会以
    # "can't open file '<root>/leakage_mechanism.py'" 失败。
    # 这样两条约定才同时成立：脚本文件按 HERE 定位，脚本内部的相对路径按包根定位。
    script = args[0] if os.path.isabs(args[0]) else os.path.join(HERE, args[0])
    cmd = [PY, script] + list(args[1:])
    tag = f"L{idx:02d}"
    print(f"\n{'=' * 74}")
    print(f"[{idx}/{total}] {desc}")
    print(f"{'=' * 74}")
    print("  " + " ".join(cmd))
    if dry:
        return 0.0, True

    t0 = time.time()
    os.makedirs(LOGDIR, exist_ok=True)
    log = os.path.join(LOGDIR, f"{tag}_{args[0].replace('.py', '')}.log")
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=fh,
                              stderr=subprocess.STDOUT)
    dt = time.time() - t0
    ok = proc.returncode == 0
    print(f"  -> {'OK' if ok else 'FAILED'} in {dt:.0f} s   log: {log}")
    if not ok:
        with open(log, encoding="utf-8", errors="replace") as fh:
            tail = "".join(fh.readlines()[-15:])
        print("  tail of log:\n" + tail)
    return dt, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=1, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without executing them")
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tasks = LEVELS[args.level]

    print("=" * 74)
    print(f"Reproduction pipeline — level {args.level} ({len(tasks)} steps)")
    print(f"interpreter: {PY}")
    if args.level in NOTE:
        print("note: " + NOTE[args.level])
    print("=" * 74)

    total_t, all_ok = 0.0, True
    for i, (desc, cmd_args) in enumerate(tasks, 1):
        dt, ok = run_one(desc, cmd_args, args.dry_run, i, len(tasks))
        total_t += dt
        all_ok &= ok
        if not ok:
            print("\nStopped after a failure; fix it before continuing so that no "
                  "downstream step consumes partial output.")
            break

    print(f"\n{'=' * 74}")
    if args.dry_run:
        print("dry run finished (nothing was executed)")
    else:
        # Regenerate the checksum manifest from what was actually written, so that
        # outputs/MD5SUMS.txt is a function of the archive rather than a claim about it
        # (added 2026-10-14: the hand-maintained manifest had drifted to covering 65 of
        # the 78 files, and one listed entry carried a stale hash).
        print("\n[manifest] regenerating outputs/MD5SUMS.txt from the archive")
        subprocess.run([PY, os.path.join(HERE, "make_md5sums.py")], cwd=ROOT)
        print(f"finished: {'all steps OK' if all_ok else 'FAILURES present'}"
              f", total {total_t / 60:.1f} min")
        print(f"outputs in: {os.path.abspath(args.outdir)}")
    print("=" * 74)


if __name__ == "__main__":
    main()
