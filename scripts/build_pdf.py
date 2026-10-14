#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_pdf.py —— 把 Markdown 稿件转成可投稿的 PDF（图表正文内联）。

v5 变更（相对 v4 版）：
  1. 默认源文件改为 ../01_manuscript/Manuscript_v5_EN.md；--zh 用 v5 中文对照稿。
  2. 图片目录改为 ../01_manuscript/figures（不再是 outputs/）。
  3. ★ 插图改为【正文内联】：每张图插入到正文【首次提及 "Figure N" 的段落之后】，
     不再堆在文末图版 —— 符合 Digital Discovery / JCIM 的投稿惯例。
  4. 保留 FIGURE_MANIFEST.md 的权威编号映射：图注编号与 PNG 文件名在多处错位，
     本脚本只按 manifest 的显式映射嵌图，绝不按文件名推断。

流程：
  md --(自写 md->html)--> _build/manuscript.html --(headless Edge print-to-pdf)--> PDF

用法：
  python build_pdf.py              # 英文主稿（默认，插图内联）
  python build_pdf.py --si         # Supplementary Information
  python build_pdf.py --zh         # 中文对照稿（仅工作布局有）
  python build_pdf.py --no-figures # 纯文字版
  python build_pdf.py --plate      # 旧行为：图版放文末

文件名回退（2026-10-13 增）：工作布局的源文件是 Manuscript_v5_EN.md /
SI_extended_results.md，投稿快照包是 Manuscript.md / SI.md（后者在
02_supplementary/）。本脚本按顺序自动探测，两者都能构建，产出各自对应的
*.pdf。HTML 中间产物写系统临时目录，不再落入包内 _build/（该目录按投稿
必读要求不得交付）。
"""
import argparse
import html
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))   # .../03_reproduction_package/scripts
ROOT = os.path.dirname(HERE)                        # .../03_reproduction_package
MANU = os.path.join(ROOT, "..", "01_manuscript")
SI_DIR = os.path.join(ROOT, "..", "02_supplementary")
IMG_DIR = os.path.join(MANU, "figures")


def _first_existing(*paths):
    """Return the first path that exists, or None.

    Working layout first (v5 file names), submission snapshot second.
    """
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _default_pdf(src_md: str) -> str:
    """Manuscript_v5_EN.md -> Manuscript_v5_EN.pdf; Manuscript.md -> Manuscript.pdf."""
    d, base = os.path.split(src_md)
    return os.path.join(d, base[:-3] + ".pdf")


SRC_EN_CANDIDATES = [
    os.path.join(MANU, "Manuscript_v5_EN.md"),      # working layout
    os.path.join(MANU, "Manuscript.md"),            # submission snapshot
]
SRC_ZH_CANDIDATES = [
    os.path.join(MANU, "Manuscript_v5_zh_head_EN_body.md"),
]
SI_SRC_CANDIDATES = [
    os.path.join(MANU, "SI_extended_results.md"),   # working layout
    os.path.join(MANU, "SI.md"),
    os.path.join(SI_DIR, "SI.md"),                  # submission snapshot
    os.path.join(SI_DIR, "SI_extended_results.md"),
]

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

# ---- 图注编号 -> PNG 文件（权威映射，见 FIGURE_MANIFEST.md）----
FIG_MAP = {
    # 2026-10-11（第三档）：正文只保留 8 张图；SHAP×2、within/cross、来源离散、四口径移入 SI
    1:  "fig1_leakage_two_targets.png",
    2:  "fig2_confound_structure.png",
    3:  "fig3_isoreticular_control.png",
    4:  "fig4_nn_distance.png",
    5:  "fig5_pred_vs_actual.png",
    6:  "fig6_penalty_vs_geometry.png",
    7:  "fig7_multitarget_bars.png",
    8:  "fig8_cross_database.png",
}

# ---- SI 图（Supplementary Information）----
SI_SRC = os.path.join(MANU, "SI_extended_results.md")
SI_OUT = os.path.join(MANU, "SI_extended_results.pdf")
FIG_MAP_SI = {
    1: "figS1_shap_family.png",
    2: "figS2_shap_dependence.png",
    3: "figS3_within_vs_cross.png",
    4: "figS4_aggregation_stability.png",
    5: "figS5_aggregation_rules.png",
    6: "figS6_source_confound.png",
    7: "figS7_sg_curve.png",
}

CSS = """
@page { size: A4; margin: 20mm 18mm 18mm 18mm; }
* { box-sizing: border-box; }
body {
  font-family: "Times New Roman", "Georgia", "SimSun", "Microsoft YaHei", serif;
  font-size: 10.5pt; line-height: 1.55; color: #111;
  max-width: 100%; margin: 0;
}
h1 { font-size: 17pt; margin: 0 0 2mm; line-height: 1.3; }
h2 { font-size: 13pt; margin: 7mm 0 2mm; border-bottom: 1px solid #ccc;
     padding-bottom: 1mm; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 5mm 0 1.5mm; page-break-after: avoid; }
h4 { font-size: 10.5pt; margin: 4mm 0 1mm; font-style: italic; page-break-after: avoid; }
p  { margin: 0 0 2.2mm; text-align: justify; }
ul, ol { margin: 1mm 0 2.5mm; padding-left: 6mm; }
li { margin-bottom: 0.8mm; text-align: justify; }
hr { border: none; border-top: 1px solid #bbb; margin: 5mm 0; }
code {
  font-family: "Consolas", "Courier New", monospace; font-size: 9pt;
  background: #f4f4f4; padding: 0.4mm 1mm; border-radius: 1px;
  word-break: break-word;
}
pre { background: #f7f7f7; border-left: 2px solid #999; padding: 2mm 3mm;
      overflow-x: auto; font-size: 8.5pt; line-height: 1.35; }
pre code { background: none; padding: 0; }
blockquote {
  margin: 2mm 0; padding: 1.5mm 3mm; border-left: 3px solid #bbb;
  background: #fafafa; color: #333; font-size: 9.5pt;
}
blockquote p { margin-bottom: 1.2mm; }
table {
  border-collapse: collapse; width: 100%; margin: 2.5mm 0;
  font-size: 8.6pt;
  /* 长表允许跨页，但不在行内断开——避免整张表被推走留下半页空白 */
  page-break-inside: auto;
}
tr { page-break-inside: avoid; page-break-after: auto; }
thead { display: table-header-group; }
th, td { border: 1px solid #bbb; padding: 1.1mm 1.6mm; text-align: left;
         vertical-align: top; }
th { background: #eee; font-weight: 600; }
/* 图表：正文内联 */
figure { margin: 4mm 0 5mm; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; height: auto; border: 1px solid #ddd; }
figcaption { font-size: 8.6pt; line-height: 1.4; text-align: left;
             margin-top: 1.5mm; color: #222; }
.fig-pending {
  border: 1.5px dashed #c00; background: #fff6f6; color: #a00;
  padding: 6mm 4mm; font-size: 9.5pt; text-align: center; margin: 4mm 0;
}
"""


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def inline(text: str) -> str:
    """行内标记（兼容历史转义层；v5 源文件已无转义残留，此层仅作保险）。"""
    text = text.replace("\\*\\*", "**").replace("\\*", "*")
    text = text.replace("\\|", "|").replace("\\_", "_")
    text = esc(text)
    text = re.sub(r"`([^`]+)`", lambda m: "<code>" + m.group(1) + "</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<em>\1</em>", text)
    return text


def split_row(line: str):
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    parts, cur, i = [], "", 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s) and s[i + 1] == "|":
            cur += "|"
            i += 2
            continue
        if s[i] == "|":
            parts.append(cur)
            cur = ""
            i += 1
            continue
        cur += s[i]
        i += 1
    parts.append(cur)
    return [p.strip() for p in parts]


def is_sep_row(line: str) -> bool:
    s = line.strip()
    if not (s.startswith("|") and s.endswith("|")):
        return False
    body = s.replace("|", "").replace(" ", "").replace(":", "").replace("\\", "")
    return bool(body) and set(body) <= {"-"} and "-" in body


def md_to_html(md: str) -> str:
    lines = md.split("\n")
    out = []
    i = 0
    n = len(lines)
    in_ul = in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    while i < n:
        raw = lines[i]
        s = raw.strip()

        if not s:
            close_lists()
            i += 1
            continue

        if s.startswith("```"):
            close_lists()
            buf = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>" + esc("\n".join(buf)) + "</code></pre>")
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            close_lists()
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
            i += 1
            continue

        if re.match(r"^-{3,}$", s):
            close_lists()
            out.append("<hr>")
            i += 1
            continue

        if s.startswith("|") and i + 1 < n and is_sep_row(lines[i + 1]):
            close_lists()
            header = split_row(s)
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                if not lines[i].strip().endswith("|"):
                    break
                rows.append(split_row(lines[i]))
                i += 1
            t = ["<table><thead><tr>"]
            t += ["<th>" + inline(h) + "</th>" for h in header]
            t.append("</tr></thead><tbody>")
            for r in rows:
                r = (r + [""] * len(header))[: len(header)]
                t.append("<tr>" + "".join("<td>" + inline(c) + "</td>" for c in r) + "</tr>")
            t.append("</tbody></table>")
            out.append("".join(t))
            continue

        if s.startswith(">"):
            close_lists()
            qs = []
            while i < n and lines[i].strip().startswith(">"):
                q = lines[i].strip()[1:].strip()
                qs.append(q)
                i += 1
            content = "<br>".join(inline(x) for x in qs)
            out.append("<blockquote><p>" + content + "</p></blockquote>")
            continue

        m = re.match(r"^[-*]\s+(.*)$", s)
        if m:
            if not in_ul:
                close_lists()
                out.append("<ul>")
                in_ul = True
            out.append("<li>" + inline(m.group(1)) + "</li>")
            i += 1
            continue
        m = re.match(r"^\d+\.\s+(.*)$", s)
        if m:
            if not in_ol:
                close_lists()
                out.append("<ol>")
                in_ol = True
            out.append("<li>" + inline(m.group(1)) + "</li>")
            i += 1
            continue

        close_lists()
        buf = [s]
        i += 1
        while i < n:
            nxt = lines[i]
            ns = nxt.strip()
            if (not ns or ns.startswith(("#", "|", ">", "```", "---"))
                    or re.match(r"^[-*]\s+", ns) or re.match(r"^\d+\.\s+", ns)):
                break
            buf.append(ns)
            i += 1
        joined = " ".join(buf)
        out.append("<p>" + inline(joined) + "</p>")

    close_lists()
    return "\n".join(out)


def extract_captions(md: str):
    caps = {}
    for m in re.finditer(r"\*\*Figure (\d+)\.\*\*(.*?)(?=\*\*Figure \d+\.\*\*|##\s*\d|\Z)",
                         md, re.S):
        caps[int(m.group(1))] = re.sub(r"\s+", " ", m.group(2)).strip()
    return caps


def extract_captions_si(md: str):
    """SI 图注：**Figure S1.** …"""
    caps = {}
    for m in re.finditer(r"\*\*Figure S(\d+)\.\*\*(.*?)(?=\*\*Figure S\d+\.\*\*|\n## |\Z)",
                         md, re.S):
        caps[int(m.group(1))] = re.sub(r"\s+", " ", m.group(2)).strip()
    return caps


def build_figure_block_si(num: int, captions: dict, img_dir: str) -> str:
    fname = FIG_MAP_SI.get(num)
    cap_txt = inline(captions.get(num, ""))
    if fname is None:
        return (f'<div class="fig-pending"><b>[Figure S{num} — pending]</b></div>')
    path = os.path.join(img_dir, fname)
    if not os.path.exists(path):
        return (f'<div class="fig-pending"><b>[Figure S{num} — file missing]</b><br>'
                f'{esc(fname)}</div>')
    src = "file:///" + path.replace("\\", "/").lstrip("/")
    return (f'<figure><img src="{src}" alt="Figure S{num}">'
            f'<figcaption><b>Figure S{num}.</b> {cap_txt}</figcaption></figure>')


def insert_si_figures_inline(html_body: str, captions: dict, img_dir: str) -> str:
    insertions = []
    for num in sorted(FIG_MAP_SI):
        m = re.search(r"\bFigure\s+S%d(?!\d)" % num, html_body)
        if not m:
            insertions.append((len(html_body), build_figure_block_si(num, captions, img_dir)))
            continue
        end_pos = len(html_body)
        for tag in ("</p>", "</li>", "</blockquote>", "</table>"):
            k = html_body.find(tag, m.end())
            if k != -1:
                end_pos = min(end_pos, k + len(tag))
                break
        insertions.append((end_pos, build_figure_block_si(num, captions, img_dir)))
    for pos, block in sorted(insertions, key=lambda x: -x[0]):
        html_body = html_body[:pos] + "\n" + block + html_body[pos:]
    return html_body


def build_figure_block(num: int, captions: dict, img_dir: str) -> str:
    fname = FIG_MAP.get(num)
    cap_txt = inline(captions.get(num, ""))
    if fname is None:
        return (f'<div class="fig-pending"><b>[Figure {num} — pending]</b><br>'
                f'<span style="font-size:8.6pt">{cap_txt[:300]}</span></div>')
    path = os.path.join(img_dir, fname)
    if not os.path.exists(path):
        return (f'<div class="fig-pending"><b>[Figure {num} — file missing]</b><br>'
                f'{esc(fname)}</div>')
    src = "file:///" + path.replace("\\", "/").lstrip("/")
    return (f'<figure><img src="{src}" alt="Figure {num}">'
            f'<figcaption><b>Figure {num}.</b> {cap_txt}</figcaption></figure>')


def insert_figures_inline(html_body: str, captions: dict, img_dir: str) -> str:
    """把每张图插到正文首次提及 "Figure N" 的段落之后。"""
    insertions = []  # (position, html)
    for num in sorted(FIG_MAP):
        # 允许 "Figure 12a" / "Figure 14b" 这类带面板后缀的引用
        m = re.search(r"\bFigure\s+%d(?!\d)" % num, html_body)
        if not m:
            insertions.append((len(html_body), build_figure_block(num, captions, img_dir)))
            continue
        end_pos = len(html_body)
        for tag in ("</p>", "</li>", "</blockquote>", "</table>"):
            k = html_body.find(tag, m.end())
            if k != -1:
                end_pos = min(end_pos, k + len(tag))
                break
        insertions.append((end_pos, build_figure_block(num, captions, img_dir)))
    for pos, block in sorted(insertions, key=lambda x: -x[0]):
        html_body = html_body[:pos] + "\n" + block + html_body[pos:]
    return html_body


def find_edge():
    for p in EDGE_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--zh", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--plate", action="store_true",
                    help="旧行为：所有图堆在文末图版")
    ap.add_argument("--src", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--si", action="store_true",
                    help="构建 Supplementary Information（内联 Figure S1）")
    args = ap.parse_args()

    if args.si:
        src_md = args.src or _first_existing(*SI_SRC_CANDIDATES)
        if not src_md:
            sys.exit("找不到 SI 源文件（SI_extended_results.md / SI.md 均不存在）")
        md = open(src_md, encoding="utf-8").read()
        captions = extract_captions_si(md)
        # 图注块（含 *Source data:* 行）从正文移除，改由图内联题注呈现——与主稿 §6 的处理一致
        md_body = re.sub(r"(?ms)^\*\*Figure S\d+\.\*\*.*?^\*Source data:\*[^\n]*\n", "", md)
        # 图题注区（## S<n>. Figure captions，快照包中为 S15）从正文剥离，与主稿 §6 的处理一致
        md_body = re.sub(r"(?m)^##\s*S\d+\.\s*Figure captions.*?(?=^##\s|\Z)",
                         "", md_body, flags=re.S)
        html_body = md_to_html(md_body)
        if not args.no_figures:
            html_body = insert_si_figures_inline(html_body, captions, IMG_DIR)
        tag = "si"
        out_pdf = os.path.abspath(args.out or _default_pdf(src_md))
    else:
        cands = SRC_ZH_CANDIDATES if args.zh else SRC_EN_CANDIDATES
        src_md = args.src or _first_existing(*cands)
        if not src_md:
            sys.exit(f"找不到源文件：{cands[0]} 或其快照回退名均不存在")
        md = open(src_md, encoding="utf-8").read()
        captions = extract_captions(md)
        # 把 §6 图注区整段移除（图注随图内联显示，避免重复）
        md_body = re.sub(r"(?m)^##\s*6\.\s*Figure captions.*?(?=^##\s*\d)", "", md, flags=re.S)
        html_body = md_to_html(md_body)
        if not args.no_figures:
            if args.plate:
                fig_html = ('<h1 style="page-break-before: always;">Figures</h1>'
                            + "\n".join(build_figure_block(n, captions, IMG_DIR)
                                        for n in sorted(FIG_MAP)))
                html_body = html_body + "\n" + fig_html
            else:
                html_body = insert_figures_inline(html_body, captions, IMG_DIR)
        tag = "zh" if args.zh else "en"
        out_pdf = os.path.abspath(args.out or _default_pdf(src_md))

    m_title = re.search(r"^#\s+(.+)$", md, re.M)
    doc_title = html.escape(m_title.group(1).strip()) if m_title else "Manuscript"
    doc = f"""<!DOCTYPE html>
<html lang="{'zh' if args.zh else 'en'}"><head><meta charset="utf-8">
<title>{doc_title}</title><style>{CSS}</style></head>
<body>
{html_body}
</body></html>"""

    # HTML 中间产物写系统临时目录：_build/ 按投稿必读不得交付（2026-10-13 改）
    BUILD = os.path.join(tempfile.gettempdir(), "mof_topholdout_pdf_build")
    os.makedirs(BUILD, exist_ok=True)
    html_path = os.path.join(BUILD, (f"manuscript_{tag}.html"))
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"[1/2] HTML 已生成: {html_path}  ({len(doc):,} 字符)")

    issues = []
    n_bs_star = doc.count(chr(92) + chr(42))
    n_bs_pipe = doc.count(chr(92) + chr(124))
    if n_bs_star:
        issues.append(f"残留反斜杠星号 \\* ×{n_bs_star}")
    if n_bs_pipe:
        issues.append(f"残留转义竖线 \\| ×{n_bs_pipe}")
    print("  ⚠️ " + "; ".join(issues) if issues else "  ✓ 自检: 无残留转义符")

    edge = find_edge()
    if not edge:
        sys.exit("找不到 Edge，无法打印 PDF。HTML 已生成，可手动用浏览器打印。")

    out_pdf = os.path.abspath(out_pdf)

    # 2026-10-11 修复：原实现只检查"目标 PDF 是否存在"，而上一版 PDF 一直在，
    # 于是 Edge 静默失败时也会被报成成功（实测两次构建时间戳完全相同）。
    # 现在改为：打印到临时文件 → 校验存在且大小合理 → 才替换正式文件；
    # 并给 headless Edge 一个独立 user-data-dir，避免它复用正在运行的
    # Edge 实例而直接退出、不产出任何 PDF。
    import shutil

    tmp_pdf = out_pdf + ".tmp.pdf"
    if os.path.exists(tmp_pdf):
        os.remove(tmp_pdf)
    profile = tempfile.mkdtemp(prefix="edge-pdf-")
    cmd = [
        edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        f"--user-data-dir={profile}",
        f"--print-to-pdf={tmp_pdf}",
        "file:///" + html_path.replace("\\", "/").lstrip("/"),
    ]
    try:
        # 不捕获子进程输出（受限环境下管道可能不可用）
        r = subprocess.run(cmd, timeout=300)
    finally:
        shutil.rmtree(profile, ignore_errors=True)
    if not os.path.exists(tmp_pdf) or os.path.getsize(tmp_pdf) < 100_000:
        sys.exit(f"PDF 生成失败 (exit {r.returncode})：没有产出有效的 {tmp_pdf}")
    os.replace(tmp_pdf, out_pdf)
    size = os.path.getsize(out_pdf)
    print(f"[2/2] PDF 已生成: {out_pdf}  ({size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
