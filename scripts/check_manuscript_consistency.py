#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_manuscript_consistency.py —— 投稿包一致性体检（改稿后必跑）

检查项（对应交接说明第 4 节「硬性不变量」）：
  1. 两稿正文自 `## 1. Introduction` 起是否逐字一致
  2. 摘要正文词数是否 ≤ 250
  3. §6 图题注是否为 1..N 连续，且正文首次引用顺序是否严格递增
  4. 正文表题注是否为 1..N 连续，且 §7 清单行数是否与之一致
  5. 图件文件名 figN_* 是否与 §6 题注一一对应
  6. 是否有残留临时标记（@@、TODO(AUTHOR) 等）
  7. 各节词数与总词数

用法：python check_manuscript_consistency.py            # 打印体检报告
      python check_manuscript_consistency.py --out report.txt
"""
import argparse
import collections
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MANU = os.path.join(ROOT, "01_manuscript")
FIGDIR = os.path.join(MANU, "figures")
ANCHOR = "## 1. Introduction"


def _first_existing(*paths):
    """Return the first path that exists, or None.

    The working layout keeps the v5 file names (Manuscript_v5_EN.md,
    SI_extended_results.md); the submission snapshot ships Manuscript.md / SI.md.
    Falling back keeps this checker runnable in both layouts.
    """
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


EN = _first_existing(os.path.join(MANU, "Manuscript_v5_EN.md"),
                     os.path.join(MANU, "Manuscript.md"))
ZH = _first_existing(os.path.join(MANU, "Manuscript_v5_zh_head_EN_body.md"))
SI = _first_existing(os.path.join(MANU, "SI_extended_results.md"),
                     os.path.join(MANU, "SI.md"),
                     os.path.join(ROOT, "02_supplementary", "SI.md"),
                     os.path.join(ROOT, "02_supplementary",
                                  "SI_extended_results.md"))


def rd(p):
    with open(p, encoding="utf-8") as fh:
        return fh.read().replace("\r\n", "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    buf = []

    def w(s=""):
        buf.append(s)

    if EN is None:
        sys.exit(f"找不到稿件源文件：{MANU}/Manuscript_v5_EN.md 或 Manuscript.md 均不存在")
    en = rd(EN)
    zh = rd(ZH) if ZH else None
    ok = True

    w("=" * 72)
    w("投稿包一致性体检")
    w("=" * 72)

    # 1) 两稿正文一致性（中文对照稿不在本布局时跳过，不算失败）
    if zh is None:
        w("\n[1] 两稿正文一致      : 跳过（中文对照稿不在本布局）")
    else:
        b1, b2 = en[en.index(ANCHOR):], zh[zh.index(ANCHOR):]
        same = b1 == b2
        ok &= same
        w(f"\n[1] 两稿正文一致      : {'OK' if same else '❌ 不一致'}"
          f"  (EN {len(b1):,} 字符 / ZH {len(b2):,} 字符)")

    # 2) 摘要词数
    lines = en.split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("## Abstract"))
    j = next(k for k, l in enumerate(lines) if l.startswith("**Keywords"))
    n_abs = len(re.findall(r"\S+", " ".join(lines[i + 1:j])))
    ok &= n_abs <= 250
    w(f"[2] 摘要正文词数      : {n_abs}  {'OK (<=250)' if n_abs <= 250 else '❌ 超 250'}")

    # 3) 图：题注连续 + 首次引用顺序
    i6 = next(k for k, l in enumerate(lines) if l.startswith("## 6. Figure captions"))
    i7 = next(k for k, l in enumerate(lines) if l.startswith("## 7. Table captions"))
    caps = [int(m.group(1)) for m in
            (re.match(r"^\*\*Figure (\d+)\.\*\*", l) for l in lines[i6:i7]) if m]
    caps_ok = caps == list(range(1, len(caps) + 1))
    order = []
    for l in lines[:i6]:
        for m in re.finditer(r"Figures?\s+(\d+)", l):
            v = int(m.group(1))
            if v not in order:
                order.append(v)
    order_ok = order == sorted(order)
    ok &= caps_ok and order_ok
    w(f"[3] 图题注 1..{len(caps)} 连续  : {'OK' if caps_ok else '❌ ' + str(caps)}")
    w(f"    正文首次引用顺序   : {'OK 严格递增' if order_ok else '❌ ' + ' '.join(map(str, order))}")

    # 4) 表：题注连续 + §7 行数
    tabs = [int(m.group(1)) for m in
            (re.match(r"^\*\*Table (\d+)\.\*\*", l) for l in lines[:i7]) if m]
    i8 = next(k for k, l in enumerate(lines) if l.startswith("## 8. References"))
    s7rows = [l for l in lines[i7:i8] if re.match(r"^\| \*\*Table \d+\*\*", l)]
    tabs_ok = tabs == list(range(1, len(tabs) + 1))
    s7_ok = len(s7rows) == len(tabs)
    ok &= tabs_ok and s7_ok
    w(f"[4] 正文表题注 1..{len(tabs)} 连续: {'OK' if tabs_ok else '❌ ' + str(tabs)}")
    w(f"    §7 清单行数        : {len(s7rows)}  {'OK' if s7_ok else '❌ 与题注数不符'}")

    # 5) 图件文件与题注一一对应
    files = sorted(os.listdir(FIGDIR)) if os.path.isdir(FIGDIR) else []
    nums = sorted(int(re.match(r"fig(\d+)_", f).group(1)) for f in files
                  if re.match(r"fig\d+_", f))
    files_ok = nums == list(range(1, len(nums) + 1)) and len(nums) == len(caps)
    ok &= files_ok
    w(f"[5] 图件文件 fig1..{len(nums)}    : {'OK' if files_ok else '❌ ' + str(nums)}"
      f"  (共 {len(files)} 个 PNG)")

    # 6) 残留标记
    bad = {k: len(re.findall(p, en)) for k, p in
           {"@@": r"@@", "TODO(AUTHOR)": r"TODO\(AUTHOR\)",
            "placeholder": r"(?i)placeholder", "email address": r"email address"}.items()}
    left = {k: v for k, v in bad.items() if v}
    w(f"[6] 残留标记          : {left if left else 'OK 无'}")

    # 6b) 参考文献 ↔ 正文引用双向闭合（回应修改清单 P0-1）
    body_end = en.index("## 8. References")
    body = en[:body_end]
    refs = [int(x) for x in re.findall(r"^\s*(\d+)\.\s", en[body_end:], flags=re.M)]
    cited = set()
    for t in re.findall(r"refs?\.\s*[0-9]+(?:\s*[,\u2013-]\s*[0-9]+)*", body):
        nums = [int(x) for x in re.findall(r"[0-9]+", t)]
        if len(nums) == 1:
            cited.add(nums[0])
        elif nums[-1] - nums[0] < 20:
            cited.update(range(nums[0], nums[-1] + 1))
        else:
            cited.update(nums)
    # 记法一致性：引用必须是 "ref." / "refs."（带句点），不能是裸 "ref"/"refs"
    bad_style = re.findall(r"\brefs?(?!\.)\s+[0-9]", body)
    uncited = sorted(set(refs) - cited)
    dangling = sorted(cited - set(refs))
    refs_ok = not uncited and not dangling and not bad_style
    ok &= refs_ok
    w(f"[6b] 参考文献 {len(refs)} 条 / 被引 {len(cited)} 条: "
      f"{'OK 双向闭合' if refs_ok else '❌'}"
      + (f"  未被引用={uncited}" if uncited else "")
      + (f"  引用不存在={dangling}" if dangling else "")
      + (f"  记法缺句点={sorted(set(bad_style))}" if bad_style else ""))

    # 6c) 交叉引用闭合：正文里的 Table Sn / Figure Sn 必须真实存在（回应 P0-2）
    si = rd(SI) if SI else ""
    si_tables = set(re.findall(r"\*\*Table (S\d+)\.\*\*", si))
    si_figs = set(re.findall(r"\*\*Figure (S\d+)\.\*\*", si))
    used_t = set(re.findall(r"\*\*Table (S\d+)\*\*", en))
    used_f = set(re.findall(r"\*\*Figure (S\d+)[a-z]?\*\*", en))
    miss_t = sorted(used_t - si_tables)
    miss_f = sorted(used_f - si_figs)
    if not si:
        w("[6c] 正文→SI 交叉引用  : 跳过（SI 源文件不在本布局）")
        w("[6d] SI 编号与图件映射  : 跳过（SI 源文件不在本布局）")
    else:
        si_ok = not miss_t and not miss_f
        ok &= si_ok
        # 6c-2) SI 自身的图表编号必须连续，且每张都登记在 build_pdf.py 的映射里
        bp = rd(os.path.join(HERE, "build_pdf.py"))
        tn = sorted(int(x[1:]) for x in si_tables)
        fn = sorted(int(x[1:]) for x in si_figs)
        t_seq = tn == list(range(1, len(tn) + 1))
        f_seq = fn == list(range(1, len(fn) + 1))
        mapped = set(int(m) for m in re.findall(r"^\s+(\d+):\s+\"figS", bp, flags=re.M))
        f_map_ok = set(fn) == mapped
        ok &= t_seq and f_seq and f_map_ok
        w(f"[6c] 正文→SI 交叉引用  : "
          f"{'OK' if si_ok else '❌'}"
          + (f"  指向不存在的 SI 表={miss_t}" if miss_t else "")
          + (f"  指向不存在的 SI 图={miss_f}" if miss_f else "")
          + f"  (SI 表 {len(tn)} 张 / SI 图 {len(fn)} 张)")
        w(f"[6d] SI 编号与图件映射  : "
          f"{'OK' if (t_seq and f_seq and f_map_ok) else '❌'}"
          f"  表连续={t_seq}  图连续={f_seq}  FIG_MAP_SI 齐备={f_map_ok}")

    # 6e) 规范守卫（回应第一/三轮审查）：
    #     (a) 摘要不得引用具体图表编号（摘要须自洽）；
    #     (b) 不得再出现 "§4.3 item N" 这类随清单条数漂移的数字引用 — 一律描述式。
    abstract = en[en.index("## Abstract"):en.index("**Keywords:**")]
    abs_figref = re.findall(r"\b(?:Tables?|Figures?)\s+\d+", abstract)
    itemref = re.findall(r"§\d+(?:\.\d+[a-z]?)?\s+item\s+\d+", body)
    guard_ok = not abs_figref and not itemref
    ok &= guard_ok
    w(f"[6e] 引用规约守卫      : {'OK' if guard_ok else '❌'}"
      + (f"  摘要含图表编号={abs_figref}" if abs_figref else "")
      + (f"  残留 §N item M 引用={sorted(set(itemref))}" if itemref else ""))

    # 7) 词数
    sec, counts = "", collections.OrderedDict()
    for l in lines:
        m = re.match(r"^#{2,3} (.+)$", l)
        if m:
            sec = m.group(1)
            continue
        if sec:
            counts[sec] = counts.get(sec, 0) + len(re.findall(r"\S+", l))
    blocks = collections.OrderedDict()
    for k, v in counts.items():
        pre = k.split(".")[0]
        b = {"1": "§1 Introduction", "2": "§2 Methods", "3": "§3 Results",
             "4": "§4 Discussion", "5": "§5 Conclusion", "6": "§6 Figure captions",
             "7": "§7 Table captions", "8": "§8 References"}.get(pre, "§9 Declarations")
        blocks[b] = blocks.get(b, 0) + v
    body = sum(v for k, v in blocks.items() if k[:2] in ("§1", "§2", "§3", "§4", "§5"))
    total = len(re.findall(r"\S+", en))
    w(f"[7] 总词数            : {total:,}")
    for k, v in blocks.items():
        w(f"      {v:6,d}  {k}")
    w(f"      正文 §1–§5 = {body:,}")

    w("\n" + "=" * 72)
    w("结论：" + ("全部通过 ✅" if ok else "存在不一致 ❌ —— 需先修复再投稿"))
    w("=" * 72)
    txt = "\n".join(buf)
    print(txt)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(txt)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
