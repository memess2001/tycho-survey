# 88 核确认实验收官复盘、事后探索边界、文献审计与 Claude 独立复核交接单

> 日期：2026 年 11 月 25 日 | 分支：`relay-confirm`（建于 `claude/lucid-archimedes-av5rk7` @ `d646b00` 之上）
> 状态：**交接与讨论笔记**（不改 `README.md` 和 `docs/ROADMAP.md`）。预注册确认实验（2,088 运行）与事后探索诊断（600 运行）严格分区存储与记录，绝不混杂。

---

## 0. 分支提交与文件地图（给 Claude 独立审计用）

| Commit | 内容 | 性质 | 关键文件 |
|---|---|---|---|
| `3d3200b` | 预注册定稿与冻结脚本（开跑前提交） | **预注册冻结** | `docs/results/relay_confirmation_preregistration.md`<br>`scripts/analysis_v2/relay_confirm.py`<br>`scripts/analysis_v2/relay_confirm_analyze.py`<br>`ops/linux/run_relay_confirm.sh` |
| `6942207` | 76 格主运行（1,824）+ §2.5 自动补跑（264）+ 仪器检查 + 第 1 批 MLP 低学习率诊断（168） | **预注册结果** + 事后探索 | `docs/results/v2_analysis/relay_confirm_ic_report.json`<br>`docs/results/v2_analysis/relay_confirm_summary.md`<br>`docs/results/v2_analysis/relay_confirm_results.json`<br>`scripts/analysis_v2/relay_mlp_diag.py`<br>`docs/results/v2_analysis/relay_mlp_diag.{md,json}` |
| `8bf0277` | `Attn0 + MLP0` 残差分支联合缩放消融（72 新运行，共 5 臂 120 运行） | **事后探索** | `scripts/analysis_v2/relay_mlp_res_opt.py`<br>`docs/results/v2_analysis/relay_mlp_res_opt.{md,json}` |
| `beb0e07` | 两关压力测试：Gate 1 RoPE（144 运行）+ Gate 2 深度 4L/6L & RoPE 高学习率（216 运行） | **事后探索** | `scripts/analysis_v2/relay_stress_tests.py`<br>`docs/results/v2_analysis/relay_stress_{rope,depth}.{md,json}` |

原始 `.npz` 权重与曲线缓存全部完整保留在 88 核服务器 `s82`（`~/tycho-survey/work/`）：
* 预注册缓存：`work/relay_confirm/`（2,088 主/补跑 `.npz` + `ic/`）
* 事后探索缓存：`work/relay_confirm_mlp_diag/`（240 `.npz`）、`work/relay_stress_tests/`（360 `.npz`）

---

## 1. 预注册确认实验（2,088 运行）总评与核心事实

### 1.1 仪器检查（全部零误差通过）
见 `docs/results/v2_analysis/relay_confirm_ic_report.json`（`torch 2.15.0+cpu`，与探索阶段同版本）：
* **IC-3**：冻结分类器对照 25 个 §3 探索格，`0 / 25` 不符（`pass=True`）。
* **IC-4**：`train_confirm` 对照 `relay_band_pilot.train`（第 600 步），`max_abs_weight_diff = 0.0`、`max_abs_stat_diff = 0.0`（`pass=True`）。
* **IC-1**：同机两组 1,500 步重复运行逐位差 `0.0`（`pass=True`）。
* **IC-2**：服务器上重跑探索阶段 18 个运行（`base`、`lr5e-4`、`ov1_double`，`s300..302, h2 & h4`），结果依次为 `0/6`、`6/6`、`6/6`，**`0 / 18` 翻转**（`pass=True`）。

### 1.2 最终预注册定档（含 §2.5 11 格补跑至 `n=48`，无剩余待补项）
按冻结脚本默认主口径 `source_strict = T + S`（`T + S + M` 与 `T` 在 `relay_confirm_summary.md` 并列报告）：
* **总评**：**`C1 = unconfirmed`、`C2 = shift`、`C3 = unconfirmed`、`C4 = pass_all`**；**Suite C（`d=256` 门控）= `GO`**。
* **各条主比较最终数（`T+S`）**：
  * `P1a` (`A1-2` vs `A1-6`, `n=48`)：`33 vs 18`（`unconfirmed`，差 15 距 `shift` 差 1 个，`McNemar p = 0.00408`；`T` 单口径为 `33 vs 17` 进 `shift`）。
  * `P1b` (`A1-1` vs `A1-7`, `n=24`)：`18 vs 3`（**`flip`**，`p = 6.1e-5`）。
  * `P1c` (`A2-1` vs `A1-6`, `n=48`)：`44 vs 18`（**`shift`**，`p < 1e-6`）；`P1d` (`A2-4` vs `A2-2`, `n=48`)：`24 vs 6`（**`shift`**，`p = 4e-5`）。
  * `P2a` (`A3-1` vs `A1-6`, `n=48`)：**`48 vs 18`**（**`shift`**，`p < 1e-9`）；`P2b` (`A1-1` vs `A3-2`, `n=24`)：`18 vs 0`（**`flip`**）；`P2c` (`A3-3` vs `A1-9`, `n=24`)：`23 vs 0`（**`flip`**）。
  * `P2d` (`A3-5` vs `A1-6`, `n=48`)：**`48 vs 18`**（**`shift`**，`p < 1e-9`），且 **`R_1` 方向 `48 / 48`（100%，中位比 `0.8184`, CI `[0.784, 0.825]`）**；`P2e` (`A1-1` vs `A3-6`, `n=24`)：`18 vs 0`（**`flip`**），**`R_1` 方向 `24 / 24`（100%，中位比 `0.5782`, CI `[0.551, 0.617]`）**。
  * 跨 `A1 + A3` 全部 375 个成形主运行，对齐时刻（`0.3 / lr`）的 `R_1` 区分中继 **ROC-AUC = `0.9210`**（固定 `N=9600` 仅 `0.7017`）。
  * `P3a` (`A4-1` vs `A1-6`, `n=48`)：`45 vs 18`（**`shift`**）；`P3b` (`A4-2` vs `A1-6`, `n=48`)：**`48 vs 18`**（**`shift`**）；`P3c` (`A1-1` vs `A4-3`, `n=24`)：`18 vs 0`（**`flip`**）；`P3d` (`A5-1` vs `A5-2`, `n=48`)：`48 vs 33`（`unconfirmed`，差 15 距 `shift` 差 1 个，`p = 6.1e-5`）。
  * `S3c`（`A1+A3+A4` 480 个新运行样本外检验“谁先过 `0.85` 频带线”）：`469 / 480` 可判（`97.71%` ≥ 90%），与 `T+S` 一致率 **`442 / 469 = 94.24%`**（`T+S+M` 为 `92.96%`），**PASS**。
  * `C4` (`P4a` `22 vs 1`，`P4b` `22 vs 1`，`P4c` **`241 / 242 = 99.59%`** 严格先锁来源后到双键，时间比中位数 **`3.00`, CI `[2.89, 3.16]`**，**`pass_all`**）。
  * **Suite B（预注册探索）**：`B1-U`（`22 vs 1`, `flip`）、`B1-WU`（`22 vs 1`, `flip`，`mid = 16/24`，`c=2` 砸至 `0/24`，`c=0.5` 拉至 `24/24`）、`B1-W`（`13 vs 0`, `shift`；`T+S+M` `16 vs 3` `flip`）、`B3`（`22 vs 10`, `shift`）；**`B2`（4× GELU MLP）出现重大意外：`B2-low` `24/24 R`，`B2-mid/high` `22–24/24 U`（`unstable`）**。
  * **Suite C（宽度桥接 & `d=256` 门控）**：`d=32` 三格全中继（`23–24 R`）；`d=128` `mid` `14 R / 5 T / 5 M` → `mid-c05` **`0 R / 24 T`**；**`d=256` `mid` `19 R / 0 T / 5 U` → `mid-c05` `0 R / 20 T / 1 M / 3 U`**，触发 **`GO`**（且 `d=128` `54/55` 教科书、`d=256` `22/33` 教科书仍呈两步装成，中位时间比随宽度由 `3.16 -> 2.15 -> 1.78` 缩小）。

### 1.3 为什么基准 `A1-6` 落在 `18/48` 而非探索阶段的 `0/6`？（20 种子核验 + 两大真发现）
在 `s82` 上做的独立核验（重跑 `A1-6_h2_s1000` 至第 300 步 `diff = 0.0`，并将探索种子 `s300..302` 顺延至 `s300..309`）：
* `s300, 301, 302`（`h2`）第 300 步恰好连续抽中 3 个中继轨迹（`b0 = 0.764~0.792, b1 = 0.520~0.621, r1 = 1.26~1.45`）；
* 紧随其后的 `s303, 304, 307, 308, 309`（7 个里 5 个）立刻与 `s1000..1011` 完全同分布（`b0 = 0.666~0.701, b1 = 0.752~0.831`），证实 **探索阶段 `0/6` 是小样本连黑抽样运气，绝非代码或机器差异**。

同时，大样本清晰暴露了探索阶段漏掉的两条核心物理规律：
1. **`σ = lr / √B` 并非精确不变量，`lr`（经早盘 `R_1`）的主导作用远强于 `1 / √B`**：
   * `S1b`（同 `σ = 1.25e-4` 三格：`A1-4 5e-4/16`、`A1-3 1e-3/64`、`A1-5 2e-3/256`）`T+S` 依次为 **`17/24`、`10/24`、`5/24`**（精确 `3×2` Freeman-Halton **`p = 0.00227`**），对应对齐 `R_1` 中位数 `1.019`、`1.294`、`1.663`（同 `lr=1e-3` 下 `B=128/64/32` 的 `R_1` 几乎恒定在 `1.25~1.29`）。
2. **头数 `n_head` 是强吸引盆开关（固定 `d_model=64` 下 4 头系统性比 2 头更易进中继）**：
   * 全部 13 个中间区格无一例外（如 `n=48` 上 `A1-6`：`h2 = 14/24` vs `h4 = 4/24`；`A1-2`：`h2 = 20/24` vs `h4 = 13/24`；`A1-1` `n=24`：`h2 = 12/12` vs `h4 = 6/12`）。
   * 并且 `A5-2`（第 `1,000–3,000` 步掩码，`n=48`）出现强烈分化：`h2` 上 `14 -> 15/24`（仅转化 1/10 中继，高度不可逆），`h4` 上 `4 -> 18/24`（转化 14/20 中继，70% 可拉回，与多头保留备用容量一致）。

### 1.4 事前盲测对赌成绩单（Claude vs Antigravity）
* **`A6-7`（最大分歧点：`5e-4/32, proj0` LR `×1/16, 24k` 步）**：实测 `h2: 4/12 T, 8/12 S`，`h4: 12/12 T`（对照 `A6-6`: `h2 1/12 T, h4 11/12 T`；`A6-8`: `24/24 T`）——**Claude 胜出**（光放慢 `proj0` 学习率本身就足以锁死多数 2 头运行，无需缩小初值或慢 `W_v`）。
* **`A3-4`（`2e-3/32, c=0.5`）**：实测 **`21/24 T, 0 R, 3 U`** —— **Antigravity 方向命中**（`c` 相对 `σ` 明显超线性）。
* **`A3-7`（`ov0 × 0.5`）**：实测 **`24/24 T+S`** —— **Antigravity 命中**。
* **`B1-WU-mid`（200 步 warmup）**：实测 **`16/24 T`**，`c=2` 翻至 `0/24` —— **Antigravity 精准命中**。
* **`B2`（4× GELU MLP）**：**两人均未猜中**（实际 MLP 剧烈放大了中继与不稳定陷阱）。

---

## 2. 事后探索突破与严格边界（600 新运行，完全隔离于预注册缓存）

### 2.1 MLP 陷阱与 `Attn0 + MLP0` 超加性联合缩放解救（`relay_mlp_diag.md` + `relay_mlp_res_opt.md`）
* 降低学习率仍难逃 MLP 陷阱：`MLP-lr5e-4`（无 MLP 时 `18/24 T`）掉成 **`0/24 T, 24/24 R`**；连 `MLP-lr3e-4` 仍为 **`3/24 T, 21/24 R`**（`loss = 1.11` vs `0.19`）。
* 加上 `AdamW wd=0.1 + 200` 步 warmup（`MLP-wu-lr1e-3`）彻底治好了不收敛（`U`: `22 -> 0`），但 **`24 / 24` 全部陷入位置中继**（`loss = 1.061`），原因是 `MLP0` 向残差流倾倒 `rms_m0 = 0.0552`（约 `1.5× rms_emb`）。
* **5 臂联合残差缩放消融（`MLP-wu-lr1e-3`, `n=24/arm`）**：
  * `(1.0, 1.0)` 基准：`0/24 T, 24 R`（中位损失 `1.061`）
  * `(0.5, 1.0)` 仅缩 `Attn0`：`7/24 T, 17 R`（`p = 0.0156`）
  * `(1.0, 0.5)` 仅缩 `MLP0`：`1/24 T, 23 R`（`p = 1.0`）
  * **`(0.5, 0.5)` 联合缩 `Attn0 + MLP0`**：**`21/24 T, 3 R`**（`h2 11/12, h4 10/12`，中位损失 **`0.183`**，**`p = 1e-6`** —— 强超加性协同：`+21 >> 7 + 1`！）
  * **`(0.25, 0.25)` 联合缩 `Attn0 + MLP0`**：**`24/24 T, 0 R`**（`12/12 h2, 12/12 h4`，中位损失 **`0.183`**，`p < 1e-7`）

### 2.2 两关压力测试：RoPE（Gate 1）与深度 4L/6L（Gate 2）（`relay_stress_{rope,depth}.md`）
* **Gate 1（标准 RoPE 替代学习型绝对 `pos`，2 层）**：
  * **纯注意力下中继清零**：`ROPE-attn-low` 和 `ROPE-attn-mid` 均为 **`24/24 T, 0 R`**（对比绝对位置 `A1-6` 的 `13/24 R`）。
  * **但加 `4× MLP + AdamW + warmup` 后中继仍在 2 头上出现，且浅层缩放依然清零**：`ROPE-mlp-wu`（`1e-3`）有 **`5/24` 真错位中继**（全在 `h2`，`δ = 14..17, loss = 0.89~1.43`）→ `(0.5, 0.5)` 降至 `0 R / 23 T`，`(0.25, 0.25)` 达 **`0 R / 24 T`**；高学习率 `ROPE-mlp-wu-lr2e-3` 有 **`7/24 R`**（`h2` 仅 `4/12 T`）→ `(0.25, 0.25)` 降至 **`0 R / 22 T`**（`h2` `4 -> 10/12`）。
  * *(分类细节记录：2 层 RoPE 下有 3 个 `loss = 0.17~0.21, decoy ≥ 0.95` 的真教科书运行，其最尖 L1 头是 `δ = -1` 重复 token 头；`relay_stress_tests.py` 已改用行为优先 `decoy ≥ 0.9 -> T` 避免误标。)*
* **Gate 2（深度 4 层 / 6 层，现代 `4× MLP + AdamW + warmup, lr=1e-3`）**：
  * **学习型绝对位置编码**：随深度递减但 4 层仍严重受困——`2L: 0/24 T (24 R)` → **`4L: 12/24 T (9 R, 3 U, h2 仅 1/12 T, loss 0.696)`** → `6L: 18/24 T (1 R, 4 U, loss 0.167)`；浅层缩放 `L0+L1×0.5` 将 4L 中继从 `9 -> 1`、`h2` 从 `1/12 -> 9/12`，但在 4L/6L 的 4 头上略增不收敛 `U`（`3 -> 6~8`、`4 -> 7`），体现真实的可训性权衡。
  * **`4L + RoPE`**：**`24/24 T, 0 R, 0 U`（中位损失 `0.136`）**——说明**同时具备 RoPE 与深度（≥4 层）后，位置中继陷阱自动被过滤干净**。

---

## 3. 联网 Literature Audit 摘要（核查至 2026 年 arXiv）

* **已发表大原理/零件（须正面引用，不可宣称原创）**：
  * 数据多样性选位置捷径 vs 归纳头：`Kawata et al. (NeurIPS 2025, arXiv:2512.18634)`（单层理论，明写两层耦合为未来工作）。
  * 子电路相对学习率竞争：`Nguyen & Reddy (ICLR 2025, arXiv:2412.00104)`、`Gibson, Cui, Reddy (2026, arXiv:2604.12151)`、`Singh et al. (ICML 2024/2025, arXiv:2503.05631)`、`Vashisht & Ramaswamy (ICML 2026, arXiv:2608.06776)`、`Anson & Aitchison (2025, arXiv:2511.21377)`。
  * 早盘关键窗口：`Chen et al. (ICLR 2024, arXiv:2309.07311)`、`Ali (2026, arXiv:2605.04396)`。
  * 残差缩放工程件本身：`ReZero (2021)`、`LayerScale (2021)`、`DeepNorm (2022)`、`Sun et al. Curse of Depth / LayerNorm Scaling (NeurIPS 2025, arXiv:2502.05795)`（均针对超深网络梯度/表示退化，如 *Curse of Depth* 缩深层 `1/√depth` 而保留 L0=1.0）。
  * 归纳头涌现理论：`Olsson et al. (2022)`、`Musat et al. (2025/2026, arXiv:2511.01033, arXiv:2607.11875)`。
  * `NitzanBrain/toy-induction-lockout`（§16.2 提及）：联网检索无公开索引记录，不作正式引用。
* **经审计具备差异化新意的 4 个核心点**：
  1. **浅层残差淹没（MLP 放大陷阱）与 `Attn0 + MLP0` 超加性联合解救**（`0/24 -> 21/24 -> 24/24`，现代 `AdamW + warmup` 下仍成立）——连接 Pre-LN 残差幅值与离散算法吸引盆选择；
  2. **完整拆清两层耦合位置中继吸引盆（正面承接 Kawata 2025 未来工作）并划定全套外推边界**（`d=256` 成立，且首次从电路动力学解释 **为什么 RoPE + ≥4 层在分布内即自动消除该陷阱**：`4L abs 12/24` vs `4L RoPE 24/24`）；
  3. **固定宽度下头数 `n_head` 的双刃剑分岔效应**（绝对位置下 4 头早盘更易入中继、晚窗口更易逆转，而 2 层 RoPE+MLP 下反而 2 头更易入中继）；
  4. **两步装成的 `proj0` 学习率瓶颈**（`241/242`, `3.00×` 延迟，且 `A6-7` 证明单慢 `proj0` LR 即可卡死 2 头第二步）。
* **诚实定位**：不是“现代深层 RoPE LLM 的通用优化大招”（4L+RoPE 已自动免疫），而是 **Mechanistic Optimization Dynamics（机理优化动力学）** 研究——解释浅层残差幅值、MLP、头数、位置编码与深度如何共同决定算法吸引盆选择。若冲顶会主会（ICLR/ICML/NeurIPS），建议补两块拼图（~1 周）：① 第二个不同任务族（如带位置捷径的 Associative Recall / Trigger-Output）；② 1–2 页简洁的一阶微扰/残差信噪比命题分析。

---

## 4. 请 Claude 重点独立审计的清单

1. **代码与预注册合规审计**：逐行检查 `scripts/analysis_v2/relay_confirm.py`、`relay_confirm_analyze.py` 和 `relay_confirmation_preregistration.md`（commit `3d3200b`），核实 76 格配置、种子分配、双口径计算（`T+S` vs `T+S+M` vs `T`）、§2.5 合并 `n=48` 严格规则及主种子隔离有无漏洞。
2. **数据与分类复算**：检查 `relay_confirm_results.json` 与 `relay_confirm_summary.md` 的一致性（如有条件可从 `s82` 抽样 `.npz` 权重独立重分类核对）；复核 `A6-7`、`S1b`、`h2 vs h4` 和 `d=256 GO` 结论。
3. **事后探索代码审计**：重点审查 `scripts/analysis_v2/relay_mlp_res_opt.py`（`mlp_out_scale` 挂钩与 `mlp[2]` 折回）和 `scripts/analysis_v2/relay_stress_tests.py`（RoPE 旋转与 `pos` 置零、多层行为优先分类器），确认 `0/24 -> 24/24` 与 `4L abs 12/24 vs 4L RoPE 24/24` 无实现偏差。
4. **下一步决策意见**：就“补第二任务 + 1–2 页一阶动力学理论冲主会/TMLR”与“收尾归档回到 Tycho 反常清单挑下一站”给出独立评估与分歧点。
