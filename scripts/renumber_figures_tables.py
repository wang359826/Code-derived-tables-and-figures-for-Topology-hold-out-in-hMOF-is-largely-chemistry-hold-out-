#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
renumber_figures_tables.py —— 一次性完成两项结构性修订（投稿硬伤）

【历史脚本，勿再运行】本脚本是 2026-10 图序修订的第一轮工具，只对当时的
v5 工作目录生效（ROOT 指向的目录已不存在，运行会直接退出）。后续又经过
多轮修订（正文图收敛为 1–8、SI 图 S1–S7），当前发货图名以 build_pdf.py
的 FIG_MAP / FIG_MAP_SI 与 FIGURE_MANIFEST.md 为准。保留本脚本仅作修订
历史记录。

B1 图序：14 张图的首次引用顺序为 4 1 2 3 6 8 9 5 7 10 11 12 14 13，
        按"图号必须随首次引用递增"重编号：
            old -> new: 1->2 2->3 3->4 4->1 5->8 6->5 7->9 8->6 9->7
                        10->10 11->11 12->12 13->14 14->13
        同时：§6 题注块按新号排序；正文所有 "Figures? N" 引用同步替换；
        多处引用（Figures N and M）替换后按升序排列；图件文件同步改名。

B2 表号：正文 §1–§6 共 33 张 markdown 表格此前完全没有编号（0 处 "Table N"）。
        按出现顺序编号 1..33，在每张表前插入 "**Table N.** <题注>"，
        并把 §7 从"17 条遗留清单"改写为这 33 张表的完整清单（含来源 CSV）。

只改 01_manuscript/Manuscript_v5_EN.md 与 figures/ 下的文件名；中文对照稿随后由
脚本从英文稿正文重建。
"""
import os
import re
import shutil
import sys

ROOT = r"C:\Users\汪星人\Desktop\MOF_topholdout_投稿包_v5修订"
MD = os.path.join(ROOT, "01_manuscript", "Manuscript_v5_EN.md")
FIGDIR = os.path.join(ROOT, "01_manuscript", "figures")

FIG_MAP = {1: 2, 2: 3, 3: 4, 4: 1, 5: 8, 6: 5, 7: 9, 8: 6, 9: 7,
           10: 10, 11: 11, 12: 12, 13: 14, 14: 13}

# 新文件名：保持描述性后缀，只换数字前缀
FIG_FILES = {  # old filename -> description suffix
    "fig1_confound_structure.png": "confound_structure",
    "fig2_isoreticular_control.png": "isoreticular_control",
    "fig3_aggregation_rules.png": "aggregation_rules",
    "fig4_leakage_two_targets.png": "leakage_two_targets",
    "fig5_pred_vs_actual.png": "pred_vs_actual",
    "fig6_nn_distance.png": "nn_distance",
    "fig7_within_vs_cross.png": "within_vs_cross",
    "fig8_shap_family.png": "shap_family",
    "fig9_shap_dependence.png": "shap_dependence",
    "fig10_penalty_vs_geometry.png": "penalty_vs_geometry",
    "fig11_multitarget_bars.png": "multitarget_bars",
    "fig12_cross_database.png": "cross_database",
    "fig13_aggregation_stability.png": "aggregation_stability",
    "fig14_source_confound.png": "source_confound",
}

TABLE_CAPTIONS = {
1: "The three 2026 studies closest to this work: data, protocol and headline result.",
2: "Data sources, sizes and modelling samples.",
3: "The two definitions of chemical composition.",
4: "Controlled-contrast design: the two arms share a test set and differ only in the training set.",
5: "The eight structure-derived scalar descriptors and their nature.",
6: "The four aggregation rules for grouped splits.",
7: "Random split versus three grouped splits (100,000 samples, 426 features).",
8: "Chemistry–topology confounding on the full hMOF database (324,426 structures), under both weightings.",
9: "Chemical overlap of topology pairs at the coarse granularity (full database, 55 pairs).",
10: "Compositions that occur in more than one topology (full database).",
11: "Controlled contrast at seed 42: fixed test set, changing training set.",
12: "Controlled contrast over five seeds (mean ± sd).",
13: "Leave-one-topology-out and leave-one-metal-out R² under four aggregation rules (100,000 samples).",
14: "Per-topology leave-one-out detail (100,000 samples).",
15: "R² by nearest-neighbour distance bin (25,000 samples).",
16: "Why the deep-learning comparison is not interpretable: two test sets 7.7× apart in target variance.",
17: "Omitted-variable analysis: η² and residual R² for topology, metal centre and linker pair (25,000 samples).",
18: "Mitigation strategies compared under a fixed protocol (25,000 samples).",
19: "Within- versus cross-topology R² per topology (25,000 samples, eight descriptors).",
20: "Macro and median aggregation for the within/cross comparison (25,000 samples).",
21: "Screening utility on a shared support set: overlap@k, enrichment and regret@k (25,000 samples, k = 1%).",
22: "Confounding on the elite subset (n = 8,206) against a random 25,000-sample reference.",
23: "Ten-target panel: split effect and descriptor-only R² under both splits (25,000 samples).",
24: "The 2×2 design: median R² per cell over ten targets (25,000 samples).",
25: "The three effects of the 2×2 design.",
26: "Leave-one-source-out sensitivity of the ARC-MOF confound (fine granularity).",
27: "Cross-database confounding statistics, hMOF versus ARC-MOF.",
28: "Distinct compositions and samples per composition in the two databases.",
29: "Same-n, same-granularity-type comparison of the two databases.",
30: "Per-source confounding across ARC-MOF sub-libraries (n ≥ 200).",
31: "The cross-source spread against a one-common-rate parametric-bootstrap null.",
32: "Aggregation-rule stability across five seeds (25,000 samples).",
33: "The identity effect under three grouping variables (ten targets, 25,000 samples).",
}


def renumber_token(nums):
    """把一处引用里的旧号列表映射为新号并升序排列。"""
    new = sorted(FIG_MAP[int(n)] for n in nums)
    return new


def renumber_figure_refs(text):
    """替换正文里所有 'Figures? <nums>' 形态（含 and/,/- 连接与尾随字母）。"""
    # 单号 / 多号（N、N and M、N, M、N–M、N-M），保留尾随字母（如 12a）
    pat = re.compile(r"\b(Figures?)\s+(\d+(?:\s*(?:,|and|–|-)\s*\d+)*)([a-z]?)")

    def repl(m):
        head, numstr, tail = m.group(1), m.group(2), m.group(3)
        nums = [int(x) for x in re.findall(r"\d+", numstr)]
        new = renumber_token(nums)
        if len(new) == 1:
            out = str(new[0])
        elif len(nums) == 2 and " and " in numstr:
            out = f"{new[0]} and {new[1]}"
        else:
            sep = "–" if "–" in numstr else ", "
            out = sep.join(str(n) for n in new)
        return f"{head} {out}{tail}"

    return pat.sub(repl, text)


def main():
    if not os.path.exists(MD):
        print("【跳过】这是 2026-10 图序修订的一次性历史脚本，其工作目录已不存在：")
        print("       ", ROOT)
        print("        当前图名以 build_pdf.py 的 FIG_MAP / FIG_MAP_SI 为准，无需重跑。")
        return
    with open(MD, encoding="utf-8", newline="") as fh:
        raw = fh.read()
    eol = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    lines = text.split("\n")

    # ---------------- B1：§6 题注块重编号并排序 ----------------
    i6 = next(i for i, l in enumerate(lines) if l.startswith("## 6. Figure captions"))
    i7 = next(i for i, l in enumerate(lines) if l.startswith("## 7. Table captions"))
    head_block, cap_block, tail_block = lines[:i6 + 1], lines[i6 + 1:i7], lines[i7:]

    # 切成 14 个题注块
    starts = [k for k, l in enumerate(cap_block) if re.match(r"^\*\*Figure \d+\.\*\*", l)]
    assert len(starts) == 14, f"题注块数不是 14：{len(starts)}"
    blocks = []
    for idx, s in enumerate(starts):
        e = starts[idx + 1] if idx + 1 < len(starts) else len(cap_block)
        blocks.append(cap_block[s:e])
    new_blocks = {}
    for b in blocks:
        old = int(re.match(r"^\*\*Figure (\d+)\.\*\*", b[0]).group(1))
        body = renumber_figure_refs("\n".join(b))
        new_blocks[FIG_MAP[old]] = body
    ordered = [new_blocks[n] for n in range(1, 15)]
    print("B1 §6 题注：重编号并排序为 1..14")

    # ---------------- B1：正文引用替换（题注块之外）----------------
    body_text = "\n".join(head_block + ["@@CAPTIONS@@"]) + "\n@@TAIL@@" + "\n".join(tail_block)
    # 先替换题注块之外的正文（head + tail），题注块已单独处理
    head_text = "\n".join(head_block)
    tail_text = "\n".join(tail_block)
    head_text = renumber_figure_refs(head_text)
    tail_text = renumber_figure_refs(tail_text)
    text = head_text + "\n" + "\n".join(ordered) + "\n" + tail_text

    # ---------------- B2：正文表格编号 ----------------
    lines = text.split("\n")
    i7 = next(i for i, l in enumerate(lines) if l.startswith("## 7. Table captions"))
    out_lines, tno, inserted = [], 0, []
    i = 0
    while i < len(lines):
        l = lines[i]
        if (i < i7 and l.startswith("|") and i + 1 < len(lines)
                and re.match(r"^\|[\s\-:|]+\|$", lines[i + 1])):
            tno += 1
            # 找来源：在表格附近 ±22 行内搜 outputs/*.csv
            lo, hi = max(0, i - 22), min(len(lines), i + 40)
            srcs = []
            for ln in lines[lo:hi]:
                for m in re.finditer(r"outputs/[A-Za-z0-9_./-]+\.csv", ln):
                    if m.group(0) not in srcs:
                        srcs.append(m.group(0))
            cap = TABLE_CAPTIONS.get(tno, "")
            if out_lines and out_lines[-1].strip():
                out_lines.append("")
            out_lines.append(f"**Table {tno}.** {cap}")
            inserted.append({"n": tno, "caption": cap, "sources": srcs})
        out_lines.append(l)
        i += 1
    print(f"B2 表格：编号并插入题注 {tno} 张")
    assert tno == 33, f"表格数不是 33：{tno}"

    # ---------------- B2：重写 §7 ----------------
    i7 = next(i for i, l in enumerate(out_lines) if l.startswith("## 7. Table captions"))
    j8 = next(i for i, l in enumerate(out_lines) if l.startswith("## 8. References"))
    new7 = ["## 7. Table captions", "",
            "Tables are numbered in the order of their first citation in the text. Each entry gives",
            "the caption and the archived result file(s) it is computed from; all files are in",
            "`outputs/` of the reproduction package (see `MD5SUMS.txt`).", "",
            "| Table | Caption | Source |", "|---|---|---|"]
    for t in inserted:
        src = ", ".join(f"`{s}`" for s in t["sources"][:4]) or "—"
        new7.append(f"| **Table {t['n']}** | {t['caption']} | {src} |")
    new7 += ["", "> The random-compression granularity alignment discussed in §4.3 (granularity declaration) was withdrawn and",
             "> has no table; the CGCNN cross-topology arm is reported only through its diagnostic (Table 16)."]
    out_lines = out_lines[:i7] + new7 + [""] + out_lines[j8:]

    text = "\n".join(out_lines)

    with open(MD, "w", encoding="utf-8", newline="") as fh:
        fh.write(text.replace("\n", eol) if eol == "\r\n" else text)
    print("已写回", MD)

    # ---------------- 图件文件改名 ----------------
    moves = []
    for old, suffix in FIG_FILES.items():
        oldnum = int(re.match(r"fig(\d+)_", old).group(1))
        moves.append((old, f"fig{FIG_MAP[oldnum]}_{suffix}.png"))
    for old, _ in moves:  # 两阶段，避免任何冲突
        p = os.path.join(FIGDIR, old)
        if os.path.exists(p):
            os.rename(p, p + ".tmp")
    done = 0
    for old, new in moves:
        src, dst = os.path.join(FIGDIR, old + ".tmp"), os.path.join(FIGDIR, new)
        if os.path.exists(src):
            os.rename(src, dst)
            done += 1
    print(f"图件改名：{done} 个文件")


if __name__ == "__main__":
    main()
