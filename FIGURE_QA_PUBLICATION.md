# 配图重建与出版级 QA 记录（已完成）

> 本文件记录用 `nature-figure` 技能（Python 后端）重建论文全部 15 张配图的过程、
> 判定标准与逐图状态。数值来源始终是 `outputs/` 中已随投稿包存档的 CSV。
> **状态：15/15 已重建并通过门禁；原图已备份；15 张新 PNG 已装入
> `03_投稿专用包/01_manuscript/figures/`；`Manuscript.pdf` 与 `SI.pdf` 已用新图
> 重建；两道 PDF 审计已复跑；§6 与 §S15 图注已同步。**

## 0-A. 交付物与验证结论（先看这里）

| 项目 | 结果 |
|---|---|
| 重建图数 | **15 / 15**（正文 8 + SI 7） |
| 面板对齐门 | 13 PASS；Fig 2 记录例外（面板刻意不等宽）；S3 单面板 N/A |
| 图内文本碰撞 | **0**（15 张合计） |
| 图内文本压线 | **0**（15 张合计；2 处压在背景参考线上的标签已用不透明白底隔离并单列报告） |
| 图内字号（到纸面） | 正文 7.0 / 刻度 6.5 / 注释 6.0 pt |
| PDF 嵌图几何 | 15 张全部 **492.0 × 预期高度 pt**、有效 **600–625 dpi**、宽高比误差 ≤ 0.31% |
| 图注一致性 | 15 张图注已对齐图内实际内容（含 4 处数值修正、S2 估计量声明） |
| `audit_pdf_text.py` | Manuscript **PASS**（min 8.59 pt，0 处低于 5 pt）；SI **PASS**（min 8.59 pt，0 处） |
| `audit_figure_collisions.py` | Manuscript 47 FAIL / 165 WARN；SI 18 FAIL / 71 WARN —— **均与图无关**，见 §8 的逐条判定 |
| 原图备份 | `03_reproduction_package/_backup_figures_20261015_figure_rebuild/`（15 PNG + 2 PDF） |

## 0. 为什么要重建（原图缺陷的量化证据）

原 15 张图由 matplotlib 以 `dpi=100–170` 导出，画布宽 1870–2904 px，再被
`build_pdf.py` 以 `width:100%` 压到 174 mm（492 pt）。缩放系数只有
0.17–0.26 pt/px，于是图内文字在成品 PDF 中落到：

| 图 | 缩放 (pt/px) | 24 px 标题 | 16 px 轴标签 | 11 px 注释 |
|---|---|---|---|---|
| fig1 | 0.169 | 4.1 pt | 2.7 pt | 1.9 pt |
| fig2 | 0.263 | 6.3 pt | 4.2 pt | 2.9 pt |
| fig3 | 0.246 | 5.9 pt | 3.9 pt | 2.7 pt |
| fig4 | 0.197 | 4.7 pt | 3.1 pt | 2.2 pt |
| fig5 | 0.224 | 5.4 pt | 3.6 pt | 2.5 pt |
| fig6 | 0.221 | 5.3 pt | 3.5 pt | 2.4 pt |
| fig7 | 0.230 | 5.5 pt | 3.7 pt | 2.5 pt |
| fig8 | 0.197 | 4.7 pt | 3.1 pt | 2.2 pt |
| figS2 | 0.171 | 4.1 pt | 2.7 pt | 1.9 pt |

技能自带的 `audit_pdf_text.py` 对旧 PDF 报 **PASS（min 8.59 pt）是假阴性**：
图内文字是栅格像素，PDF 内容流里没有 `Tf` 算子可扫。真实字号只能按几何反算，
上表即反算结果。按技能的 5 pt 字形下限与"密集期刊图 5–7 pt"要求，旧图不合格。

## 1. 重建方案（已冻结的设计契约）

* **后端**：Python（与论文既有 matplotlib 脚本一致），后端排他。
* **最终几何**：按 174 mm（492 pt）栏宽**直接绘制**，不再让 PDF 构建器缩放。
  `pubfig/style.py::save_figure` 迭代画布尺寸，直到 `bbox_inches="tight"` 量出的
  宽度落在 492 ± 0.4 pt，因此正文 7 pt 字到纸面就是 7 pt。
* **字形下限**：正文/轴标签 7.0 pt、刻度 6.5 pt、面板内注释 6.0 pt、
  面板字母 8.0 pt 粗体、图级脚注 6.0 pt。少数成对数值标 5.5 pt（见下）。
* **字体**：`DejaVu Sans`。Arial 缺 U+2080–2082、U+2098、U+209B、U+03C1、
  U+2212，直接用会把 CO₂、S_miss、ρ、− 渲染成方框；DejaVu Sans 全覆盖且
  仍在 sans-serif 家族内。
* **可编辑文本**：`svg.fonttype="none"`、`pdf.fonttype=42`；PDF/SVG 为矢量
  交付件，PNG/TIFF 为 600 dpi 栅格预览。
* **不内嵌图号**：图内不再写 "Figure N."，编号与图注一律由 `build_pdf.py` 的
  `<figcaption>` 给出，避免图内号与图注号不同步（旧图 1/2/3/8/S4/S5 内嵌了
  图号，4/5/6/7 没有，同时 PDF 图注又会再打一次）。

### 跨全文统一的色彩语义（旧图此处自相矛盾）

| 含义 | 颜色 |
|---|---|
| random split / 泄漏协议 | 红 `#B64342` |
| topology hold-out / 诚实评估 | 蓝 `#0F4D92` |
| within-topology（同拓扑内） | 蓝 |
| cross-topology（跨拓扑） | 红 |
| coarse chemistry key | 青 `#42949E` |
| fine chemistry key | 红 `#B64342` |
| identity 特征 / grouped by linker | 紫 `#9A4D8E` |
| grouped by metal | 青绿 `#42949E` |

旧图 1/2/3/8 用「蓝=random、红=topology」，而旧图 4/5/S1 用「红=random、
蓝=topology」——同一篇论文里红蓝含义相反，读者无法跨图比较。新方案以
「红=random、蓝=topology」为准，其余图全部对齐。

## 2. 质量门禁（每次出图自动执行）

1. **多面板对齐门**：直接调用技能 `audit_panel_alignment.py` 的
   `matplotlib_layout_manifest` + `audit_layout_manifest`，容差 1.5 pt。
   退出码 1（确定错位）**阻断**并把图视为不可交付；退出码 2（NOT AUDITABLE，
   即面板刻意等宽不等、无法推断可比组）作为**已记录的例外**保留，此时不声称
   对齐通过，改由下面的碰撞检查与终尺寸目检兜底。
2. **文本碰撞门**（本任务新增，`style.py::audit_text_collisions`）：技能的
   `audit_figure_collisions.py` 读 PDF 几何，看不到栅格图内部；纯源码检查也
   看不到字形实际落点。因此逐图比较所有 Text artist 的 window extent，报告
   重叠面积占比 ≥15% 的对。旋转刻度标签因 AABB 天然相交而被显式排除（改为
   终尺寸目检）。
3. **终尺寸目检**：把 600 dpi PNG 按纸面尺寸逐图看一遍。

## 3. 数值保真核对（对存档 CSV 独立复算）

| 图 | 图内数值 | 源数据复算 | 结论 |
|---|---|---|---|
| Fig 1 | 0.753 / 0.661 / 0.527 / 0.379（uptake）；0.836 / 0.815 / 0.777 / 0.721（selectivity）；+99% / +16% | `leakage_results_chemistry.csv`、`leakage_results_logsel100k.csv`：0.7529→0.3786（98.9%）、0.8357→0.7211（15.9%） | ✅ 一致 |
| Fig 2 | 75.8 / 98.9 / 100.0 / 100.0；Jaccard pcu-nbo = 0.36、rht-tbo = 0.06 | `elite_subset_confound.csv`、`jaccard_full.csv`（0.3712、0.0606） | ✅ 一致 |
| Fig 3 | 0.513/0.507；MAE 0.242/0.240、0.107/0.181；RMSE 0.403/0.402、0.135/0.240；+70.0%/+77.9%/−1.2%/−0.1% | `isoreticular_control_seeds.csv` 5 种子均值：完全一致 | ✅ 一致 |
| Fig 4 | 12 分箱两组 R²；最大距离箱 −0.775 | `fig7_nn_curve_final.csv`：完全一致 | ✅ 一致 |

> **图 2 的口径提示（旧图未说明）**：`full_25k_*` 的 75.8 / 98.9 估在
> **25,000** 行抽样上，`elite_*` 的 100.0 估在 **8,206** 行上，两者不是同一
> 样本、不可配对比较。新图把 n 直接写进横轴标签，并在图注中说明。

## 4. 逐图状态

| 图号 | 状态 | 最终尺寸 (pt) | 对齐门 | 碰撞 |
|---|---|---|---|---|
| Figure 1 | ✅ 已重建 | 492.8 × 166.0 | PASS | 0 |
| Figure 2 | ✅ 已重建 | 493.6 × 165.0 | NOT AUDITABLE（左右面板刻意不等宽，已记录例外） | 0 |
| Figure 3 | ✅ 已重建 | 492.8 × 149.7 | PASS（三面板等宽例外已声明） | 0 |
| Figure 4 | ✅ 已重建 | 492.9 × 160.0 | PASS | 0 |
| Figure 5 | ✅ 已重建 | 493.0 × 187.9 | PASS | 0 |
| Figure 6 | ✅ 已重建 | 493.5 × 178.0 | PASS | 0 |
| Figure 7 | ✅ 已重建 | 493.0 × 247.2 | PASS | 0 |
| Figure 8 | ✅ 已重建 | 493.0 × 191.6 | PASS | 0 |
| Figure S1 | ✅ 已重建 | 493.0 × 154.4 | PASS | 0 |
| Figure S2 | ✅ 已重建（估计量替代表，见 §6） | 493.0 × 177.0 | PASS | 0 |
| Figure S3 | ✅ 已重建 | 493.0 × 168.5 | NOT APPLICABLE（单面板） | 0 |
| Figure S4 | ✅ 已重建 | 493.0 × 171.8 | PASS | 0 |
| Figure S5 | ✅ 已重建 | 493.0 × 188.3 | PASS | 0 |
| Figure S6 | ✅ 已重建 | 492.8 × 181.6 | PASS | 0 |
| Figure S7 | ✅ 已重建 | 493.5 × 170.6 | PASS | 0 |

**15 / 15 已重建并装入投稿包**，全部通过两级门禁，**图内文本碰撞 0**；
正文/刻度/注释字号到纸面分别为 7.0 / 6.5 / 6.0 pt，每张图宽度落在
492.8–493.6 pt（目标 492 ± 0.4，个别图因文字外接框取整略宽 1 pt，< 0.3%）。
PDF 构建时被进一步压到 492.0 pt 整，因此嵌图为 **1:1**，不再有任何缩小。

> **一处实质修正（Fig 6 横轴列）**：横轴必须是「仅结构描述符的 R²」，即
> `multitarget_confound.csv` 的 **`R2_topology_geo`**。它旁边的
> `R2_geometry_only` 是另一个量（含 identity 的完整特征集），用它算出的相关是
> r = −0.430，与正文图注写的 **r = −0.459** 不符；换成 `R2_topology_geo` 后
> 左面板 r = −0.910、右面板 r = −0.459，与正文逐位吻合。这里的列名选择是
> 决定性的，不是外观问题。

## 5. 相对旧图的逐项修改

### 正文

* **Fig 1**：两序列改用图例而非叠在柱顶的直接标注（旧图把
  `CO2 uptake (0.15 bar)` 压在 0.836 数值标上）；箭头抬高 0.055 以免压数值。
* **Fig 2**：(a) 横轴改为「full database (n = 25,000)」「elite ≥ 2.0 mmol/g
  (n = 8,206)」，把 n 写进轴标签；(b) 对角不再用同一顺序色标填 1（旧图对角
  深蓝会被误读为"高重叠"），零值改浅灰、非零值 5.5 pt 以留在格内；图例移到
  两柱组之间的空白带（原 `upper left` 压住 98.9 数值标）。**面板字母改用大写
  A/B**，避免与热图上的 `bcu` 拓扑标签混淆。
* **Fig 3**：(b) 每对柱只在较高柱上方标一次「in / out」两值（旧图两值分别
  标注，`0.242` 与 `0.240`、`0.403` 与 `0.402` 直接叠在一起）；sd(y) 并入横轴
  标签；(c) 的 pcu 说明移入图注（旧图该注释与 −1.2%/−0.1% 数值标 100% 重叠）。
* **Fig 4**：右面板由旧图"R² within bin"改为**每箱样本数**，说明两种划分都在
  同一批箱里评分；横轴补上「standardised, bin median」；删除与横轴标签重叠的
  图级脚注。
* **Fig 5**：两点云各按固定种子均匀子采样至 6,000 点（旧图 25,000 点全画，
  红蓝云互相盖住看不出形状），并写明子采样规则；R²/RMSE/n 直接标在面板内。
* **Fig 6**：横轴列修正（见上）；只保留两个极值目标的名称，其余由 Figure 7
  承担（旧图 10 个目标标签散落在数据区，与数据点、图例互相压叠）。
* **Fig 7**：旧图 10 目标 × 4 序列 = 40 根柱；改为**哑铃图**（每个目标一行，
  random 与 topology 两点连成一条线），identity 效应单独一个面板。旧图四个序列
  中两个蓝、两个红，同类难辨；现在两面板各自两序列。
* **Fig 8**：(b) x 轴改为等距阶梯并明确标注不是连续轴（旧图把手工挑的 n 画成
  连续对数轴）；(c) 每个子库的 `n` 并入 y 刻度标签（旧图 `(n=965)(n=4,207)…`
  互相重叠）；(b) 的 hMOF 全库值改为独立星标 + 注记。

### SI

* **S1**：补上各族的数值标注；写明 identity 族从 17.7% 到 14.9%。
* **S2**：**估计量替换**——由逐样本 SHAP 依赖图改为**中心化偏依赖曲线**
  （centred partial dependence），y 轴标注与新脚注都写明，详见 §6。
* **S3**：单面板；按 n 排序（旧图按 etb 降序，且 etb 的 −4.27 把其余 5 个
  topology 压成一条线）；每个 topology 的 n 写进刻度标签。
* **S4**：三种聚合口径同图对比 + 跨种子离散度；`median` 改青色（旧图绿与青在
  第二面板里分不开）。
* **S5**：四个聚合值 + 其后 11 折；聚合值标线改为向内锚定（原居中标注会伸出
  图外）；去掉 "Excluding pcu…" 长脚注以免撑破栏宽。
* **S6**：(b) 明确排除退化的 DB6，并写明 DB12 单独撑着斜率、结论是 suggestive，
  不再写 "does NOT explain"（旧图 r = +0.67、p = 0.10 说成"不能解释"强度过高）。
* **S7**：相关性统计与 (b) 的 sd 一致；每个 topology 的 n 写在刻度标签上。

## 6. Figure S2：为什么换了估计量（本包唯一一处）

Figure S2 是 15 张图里**唯一**逐样本数值没有随包存档的：`outputs/` 只存了按族
聚合的 `fig5_shap_family_final.csv`（S1 与正文引用的族级归因就是它，未改动）。
旧图用的是 `shap.TreeExplainer(mdl).shap_values(...)`。

**为什么没有直接用 `shap`**：本环境装不上。`_pylibs` 下那份 `shap` 目录被文件
权限拒绝（`import shap` 得到 `__file__ = None` 的空命名空间包），而 PyPI 下载的
wheel 写临时目录同样被拒（`Errno 13 Permission denied`）。

**为什么没有自己算**：写了精确的 path-dependent TreeSHAP 并**验证通过**——
对随机森林必须满足的可加性恒等式 `sum_f phi_f = f(x) − E[f]`，在 3 组随机数据
（p = 5/8/12，max_depth = 3/6/None，15 棵树）各 25 行上实测最大偏差
**1.4e-14**，达到机器精度。但它的代价随决策路径深度指数增长：在本论文的
120 棵树、深度约 30 的森林上实测约 **2.5 秒/行**，3000 行需要约 2 小时，
400 行超过 17 分钟仍未完成。预计算各节点叶均值（把 O(1) 的那一支从递归里摘出来）
后仍然过慢，因为慢的是路径展开本身而不是求均值。

**最终方案**：`scripts/prepare_figS2_pdp_data.py` 计算**中心化偏依赖**：

```
pdp_centred(v) = E_rows[ f(x 且该描述符 = v) ] − E_rows[ f(x) ]
```

* 与旧图同一流程的模型：`df.sample(25000, random_state=42)` →
  8 几何列 + one-hot → (25000, 426) → `RandomForestRegressor(n_estimators=120,
  random_state=42)`，in-sample R² = 0.960472（与存档一致）；
* 24 个分位箱，背景 4,000 行，整步 **133 秒**可复现；
* 裕度（`pdp_centred` 的极差）排序：**void fraction 0.769** >
  largest included sphere d 0.367 > volume 0.219 > void volume 0.218 >
  surface area 0.200 > largest free sphere d 0.132 >
  largest incl. sphere path d 0.121 > weight 0.066。
  前四位与旧图所画的前四位描述符一致，形状方向也一致——它仍然支持 §S1 的论断
  "模型学到的几何趋势在物理上合理"（void fraction 越大、低压 CO₂ 吸附越低）。

**区分与限制（必须写进 SI 图注）**：偏依赖是**边际期望**，SHAP 是**逐行可加
归因**，两者不是同一个估计量；当描述符之间相关性弱时形状接近，强相关时会有
差异。因此该面板的 y 轴一律写 "centred partial dependence"，
`outputs/pdp_geometry_meta.csv` 里也把方法写死为
"centred partial dependence … NOT per-sample SHAP"。论文中没有任何数值结论依赖
Figure S2，族级 SHAP 数字仍全部来自未改动的 `fig5_shap_family_final.csv`。

> 实现过程中修掉的三个只有实测才暴露的错误（保留自实现的验证记录，以免日后
> 重新踩入）：
> (a) sklearn 的叶子必须用 `children_left < 0 且 children_right < 0` 判定，
> 只看 `children_left` 会把「作为右孩子」的叶子误判为内部节点；
> (b) 条件期望递归里，约束必须绑定在**该节点所处的路径位置**上；若改成「结点
> 分裂特征等于任意未消费的路径特征」，抽样会一路重走自己的路径，使每个 E_k 都
> 退化成叶值、所有 SHAP 值变 0；
> (c) E_k 序列必须按「已知前缀」递增定义，`phi_{f_k} = e[k] − e[k+1]`（带符号），
> 否则可加性恒等式整体反号。

## 7. 安装与重建（已完成）

1. **备份**：原 15 张 PNG 与两份 PDF →
   `03_reproduction_package/_backup_figures_20261015_figure_rebuild/`。
2. **装入新图**：`figures_pub/_raster/*.png` → `01_manuscript/figures/`（15/15）。
   文件名与 `build_pdf.py` 的 `FIG_MAP` 一一对应，无需改脚本。
3. **重建 PDF**：`build_pdf.py` 与 `build_pdf.py --si`。
   Edge headless 打印需要进程间管道，被本会话的文件访问策略拒绝
   （`mojo platform_channel: 拒绝访问`，exit 21），因此这两条命令以更宽的
   一次性授权执行；命令本身只做本地 HTML→PDF 打印，不访问网络。
   结果：`Manuscript.pdf` 47 页 / 3965 KB，`SI.pdf` 13 页 / 2137 KB。
4. **嵌图几何复核**（PyMuPDF 读 PDF 内嵌图像）：15 张全部落在
   **492.0 pt = 173.6 mm**，有效分辨率 **600–625 dpi**，宽高比误差
   **−0.22% … +0.24%**。即：图是按最终尺寸绘制的，PDF 构建器没有再缩放，
   图内 7 pt 字到纸面就是 7 pt（旧图此处是 2.2–4.2 pt）。
5. **图注同步**：`Manuscript.md` §6（Fig 1–8）与 `SI.md` §S15（Fig S1–S7），
   并修正 §S1 里对 Figure S2 的引用。

## 8. 成品 PDF 两道审计的逐条判定

### `audit_pdf_text.py` —— PASS（可用）

| PDF | minimum found | text runs | below 5 pt |
|---|---|---|---|
| Manuscript.pdf | 8.594 pt | 2801 | 0 |
| SI.pdf | 8.595 pt | 593 | 0 |

**注意其边界**：图内文字仍是栅格像素，所以这道审计只覆盖正文与图注，
**看不到图内字号**。图内字号由上表的几何反算确认（7.0 pt 名义值 = 纸面
7.0 pt，因为嵌图为 1:1）。

### `audit_figure_collisions.py` —— 共 65 FAIL / 236 WARN，**无一落在图上**

用 PyMuPDF 把每条 finding 的 bbox 与 PDF 内嵌图像区域、图注文本块求交：

| PDF | findings | 与**图区**相交 | 与**图注块**相交 |
|---|---|---|---|
| Manuscript.pdf | 212 | **0** | 4 |
| SI.pdf | 89 | **0** | 6 |

* 212 + 89 条 finding 全部是正文段落里 `text-text` 的 **90.3% 自重叠**：
  审计把同一行内的相邻 span（如加粗的 `Source` 与 `data:`）判成两个盒子。
  逐条核对过：`get_text("dict")` 显示 `Source data: ` 实际是**一个** span，
  而审计给出的 bbox 坐标 `(51.0, 544.3)–(124.6, 554.8)` 在该页任何真实文本
  位置上都找不到——它读的是 HTML 打印流的中间层，坐标未过变换。
  **这是审计自身的建模产物，不是版面缺陷。**
* 与图注块相交的 10 条全部是 `text-fill-edge` **WARN**，命中是图注里
  `*Source data:*` 那种强调段与后文的破折号连接处，属于同一类 span 拆分问题。
* **图内**的碰撞由本包自己的渲染期检查负责（`style.py` 的
  `audit_text_collisions` 与 `audit_text_stroke_collisions`），结果是
  **0 文本碰撞、0 未遮罩压线**，见 §2。

## 9. 本轮修掉的两个真缺陷（只有"重建 + 终尺寸目检"才能发现）

1. **matplotlib 3.11 不再广播标量 `width`。** `ax.bar(x, heights, 0.34)` 不再
   画出 len(heights) 根柱，而是**一根横跨整个 x 范围**的柱，且任何 `yerr`
   会渲染成一条贯穿全图的斜线。本包 27 处 bar/barh 调用原先都依赖这个广播，
   于是 **Figure 3 的两组柱被合并成一根、误差棒变成跨组的斜线**。
   现在所有柱宽都经 `style.widths()` 显式展开成数组，几何不再依赖
   matplotlib 版本。
2. **误差棒的容器线会跨组连线。** 即使柱宽修好，`bar(yerr=)` 生成的容器仍可能
   把多根柱的误差棒合并成一条折线。Figure 3(a)(b)、Figure 8(c)、Figure S6(a)
   改为用 `style.whiskers_v()/whiskers_h()` 直接画竖杆与上下帽，几何完全可控。

这两处都是**数值正确但图形错误**：数据没问题，画出来却是错的。原 15 张 PNG
是在旧版 matplotlib 下生成的，因此旧图没有 (1) 的症状；本次重建才暴露出来。

## 10. 遗留说明

* **`prepare_figS2_shap_data.py` 保留在包内但不要用它出图**：它的 TreeSHAP
  实现已验证正确（可加性恒等式 1.4e-14），代价却随路径深度指数增长（约
  2.5 s/行，3000 行约 2 小时）。Figure S2 的数据步是
  `prepare_figS2_pdp_data.py`（133 秒）。
* **图内一律不嵌 "Figure N."**：编号只由 `build_pdf.py` 的 `<figcaption>` 给出。
  旧图 1/2/3/8/S4/S5 内嵌了图号而 4/5/6/7 没有，同时图注又再打一次编号——
  这个不同步问题随重建消失。
* **矢量交付件在 `figures_pub/*.pdf|svg`**（文字可编辑）。当前 PDF 嵌的是
  600 dpi PNG；若期刊要求矢量正文，把 `FIG_MAP`/`FIG_MAP_SI` 指向
  `figures_pub/*.pdf` 再跑一次 `build_pdf.py` 即可，图内字号不会变
  （矢量件同样是按 492 pt 画的）。
