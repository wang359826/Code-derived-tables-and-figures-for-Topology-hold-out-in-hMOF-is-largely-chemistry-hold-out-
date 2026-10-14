# 图件清单（Figure manifest）

> 本文件是**图号 ↔ PNG 文件**的权威对照表，供 `build_pdf.py` 嵌图使用
> （正文用 `FIG_MAP`，SI 用 `FIG_MAP_SI`），也是投稿时"图与图注是否相符"的自检表。
>
> ✅ **2026-10-11 第三档后：正文 8 张图（Figure 1–8）+ SI 7 张图（Figure S1–S7）。**
> 正文首次引用顺序仍为 1 → 8 严格递增；SI 图号按 SI 正文首次引用顺序排列。

## 一、正文图（8 张，全部齐备）

| 图号 | 图注主题 | PNG 文件 | 状态 |
|---|---|---|---|
| Figure 1 | 两目标泄漏幅度对比（uptake vs log₁₀ 选择性） | `fig1_leakage_two_targets.png` | ✅ 直接可用 |
| Figure 2 | 混淆结构：单拓扑占比 + Jaccard 热图 | `fig2_confound_structure.png` | ✅ 直接可用（含 (a)(b)） |
| Figure 3 | 受控等网状对照 pcu↔nbo（5 种子） | `fig3_isoreticular_control.png` | ✅ 直接可用（含 (a)(b)(c)） |
| Figure 4 | R² vs 最近邻距离 | `fig4_nn_distance.png` | ✅ §3.6 引用，详解在 SI §S1 |
| Figure 5 | 预测 vs 实测散点（随机 / 留出拓扑） | `fig5_pred_vs_actual.png` | ✅ §3.6 引用 |
| Figure 6 | ΔR² vs 描述符独有 R²（10 目标）**＋右面板：identity 效应对同一横轴（r = −0.459, p = 0.18）** | `fig6_penalty_vs_geometry.png` | ✅ §3.13 引用。**双面板**：右面板此前未在正文出现（2026-10-14 已在 §6 图注中点明）；图件本身未改，只补了题注 |
| Figure 7 | 2×2 设计 10 目标柱状图 | `fig7_multitarget_bars.png` | ✅ §3.14 引用 |
| Figure 8 | ARC-MOF 跨库复现（3 面板） | `fig8_cross_database.png` | ✅ §3.15 引用 |

## 一之二、SI 图（7 张，Figure S1–S7；题注集中在 SI §S15）

| 图号 | 图注主题 | PNG 文件 | 所在 SI 节 |
|---|---|---|---|
| Figure S1 | SHAP 特征族归因 | `figS1_shap_family.png` | §S1 |
| Figure S2 | SHAP 依赖图 | `figS2_shap_dependence.png` | §S1 |
| Figure S3 | 拓扑内 vs 跨拓扑 R² | `figS3_within_vs_cross.png` | §S5 |
| Figure S4 | 口径稳定性（3 规则 × 5 种子） | `figS4_aggregation_stability.png` | §S7 |
| Figure S5 | 留出拓扑 R² × 四口径聚合规则 | `figS5_aggregation_rules.png` | §S8 |
| Figure S6 | ARC-MOF 来源子库离散度（2 面板） | `figS6_source_confound.png` | §S9 |
| Figure S7 | S_g 曲线：未见化学率是否排序留出代价 | `figS7_sg_curve.png` | §S14 |

**小计：正文 8 张 + SI 7 张，全部齐备，两份 PDF 内均无占位框。15 张图全部可由
`run_all.py` 的 level 1–5 产出**（2026-10-13 起）：Figure 1 由
`scripts/plot_main_figure.py`（level 2）生成、再由 `scripts/fig3_fig4_missing.py`
重绘为提交的最终版（只绘图、不重算），Figure S5 同由 `fig3_fig4_missing.py` 产出；
Figure S4 ← `outputs/multi_seed_uptake25k_fix.csv`（level 2 的 CSV，图由 level 4 的
`fig_cross_database.py` 绘制）；Figure S7 ← `outputs/s_g_curve.csv`，由 level 2 末步
`fig_sg_curve.py` 绘制。
**唯一不在任何 level 内的是 `s_g_curve.py`**（Figure S7 / SI Table S9 的**数据**步）：
它要从 148 MB 原始 hMOF 重跑一次 100,000 抽样，成本约等于 level 2 第一步，而其产出的
数值已随包存档，出图并不需要它。

> 🔄 **2026-10-13 重出 6 张图**：上述 2026-10-11 的四轮修订只改了文件名，PNG 内嵌的
> 标题图号仍是旧号（如 `fig2_confound_structure.png` 内写 "Figure 1."）。已修正
> 6 个绘图脚本的内嵌标题（`fig_confound_and_control.py`、`fig_cross_database.py`、
> `fig3_fig4_missing.py`）并从存档 CSV 原样重出：Figure 1、2、3、8、S4、S5。
> 重出仅改内嵌图号，数据、配色、数值标注与旧版逐像素同源；旧版备份于包外
> `C:\temp\fig_regen\old_backup\`。其余 9 张图的内嵌内容不含图号，无需重出。

## 二、必须剔除的图（属于已撤销的臂或未获图注，仅存在于 `outputs/`，不在本包 figures/ 中）

> 本快照包的 `outputs/` 只发货 `figS6_source_confound.png` 与 `figS7_sg_curve.png` 两个
> PNG；下表描述完整 `outputs/` 存档中**不得嵌入 PDF** 的文件，供对照排查。

| 文件 | 为什么剔除 |
|---|---|
| `fig4_pred_vs_actual_smoke.png`、`fig5_shap_family_smoke.png`、`fig6_shap_dependence_smoke.png`、`fig7_nn_distance_smoke.png` / `_smoke2.png` | smoke 版（小样本），有 `_final` 版时不投 |
| `fig1_main_two_targets_demo.png` | demo 版 |
| `fig1_main_two_targets.png` | 已退役的旧文件名：`plot_main_figure.py` 现直接写 `fig1_leakage_two_targets.png`（2026-10-11 起） |
| `leakage_comparison*.png`（9 个） | 早期 v2/v3 泄漏量化臂，已被 v4 主张取代 |
| `loto_per_*.png`（4 个） | 早期 LOTO 图，未进 §6 图注 |
| `fig_correlation_matrix.png`、`fig_target_distribution.png`、`fig_pred_vs_exp.png`、`fig_shap_importance.png` | v1 分析流水线产物，未进 §6 图注 |
| `fig9_screening_utility.png` | 内容有效，但 §6 未给它图注；如需展示筛选效用须补图注 |

> ⚠️ **绝不嵌入** `cgcnn_*` 相关产物（CGCNN 臂已撤销：两侧测试集方差差 7.7 倍、
> 留出侧 MAE 反而更小，结论方向与原主张相反）。目前 outputs 里无 cgcnn 图，只有
> `cgcnn_*.csv/.pt`，但脚本 `cgcnn_mof.py` 仍在，重跑会产生——**不要跑它**。

## 三、编号变更史（供旧材料对照）

1. **2026-10-11 第一轮**：5–9 号图改名，使文件名与当时的图号一致。
2. **2026-10-11 第二轮**：按"图号随首次引用递增"全局重编号（旧 4→新 1、旧 1→新 2 …）。
   脚本：`scripts/renumber_figures_tables.py`。
3. **2026-10-11 第三轮（第二档）**：§3.16 并入 §3.5 后其图被提前引用，遂移入 SI 成为
   Figure S1（正文图号固定为 1–13）。
4. **2026-10-11 第四轮（第三档）**：SHAP×2、within/cross、来源离散、四口径 5 张图移入 SI，
   正文图号 **13 → 8**，SI 图 **1 → 6**（原 S1 重编号为 S4，按 SI 引用顺序重排）。
   脚本：`_内部存档_非投稿/_tier3a_figures.py`、`_tier3a2_rename.py`。
5. **表格编号（第三档后）**：正文 **26 张**表，按出现顺序编号 Table 1–26，§7 为 26 行清单；
   SI 表为 **Table S1–S9**（S1 近邻距离、S2 方差诊断、S3 缓解策略、S4–S5 within/cross、
   S6 十目标面板、S7 多拓扑计数、S8 seed-42 明细、S9 S_g 曲线）。重编号脚本：`_内部存档_非投稿/_renumber_tables.py`。
