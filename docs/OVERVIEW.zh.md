# Tycho 平台中文概览

> 给你自己看的说明。给 Sugiyama 的英文设计文档是 `docs/DESIGN.md`，开发规格见 `docs/SPEC.md`，
> Mac mini 运维见 `docs/OPS_MACMINI.md`。最初的聊天记录原文保存在 `notes/origin_conversation.md`。

## 一句话

在一台 Mac mini 上 24×7 地训练成千上万个小网络，每次训练都完整记录"发育过程"。所有训练用同一把校准过的尺子
（refined information，RI）测量："哪些变量、在第几步、以几阶的方式开始一起起作用"。结果写进一张可以公开发布的星表。

## 聊天里的概念在代码里对应什么

| 聊天里的说法 | 代码里的东西 | 位置 |
|---|---|---|
| 第谷的仪器（先校准） | RI 估计器，带显著性检验和偏差校正；校准套件 E1–E7 | `tycho/ri/`、`tycho/calibration/`、`tycho calibrate` |
| 天体 | 一次训练，即 (任务, 配置, 种子)，用哈希编号 | `tycho/train/spec.py` |
| 天空 | 一组区域，每个区域是"任务 × 旋钮网格 × 种子数" | `configs/sky_v0.yaml` |
| 测光（便宜，人人有） | 光变曲线：loss、准确率、权重范数、有效秩、eNTK 统计 | `tycho/train/photometry.py` |
| 光谱（贵，挑着拍） | RI 光变曲线：每个候选子集在每个 checkpoint 的 RI | `tycho/ri/estimators.py`、`tycho/survey/followup.py` |
| 重放（比第谷幸运的地方） | CPU 上逐位可复现，可回到任意天体加密观测 | `tycho/train/replay.py` |
| 星表 | SQLite 星表 + 每个天体的压缩文件 | `tycho/survey/catalog.py` |
| 暂现源通道 | 晚激活、熄灭、再激活、纯协同（没有低阶前兆的高阶交互）、阶序倒置，即时推送到手机 | `tycho/analysis/events.py`、`tycho/survey/alerts.py` |
| 粗网格 → 边界加密 → 多种子重复曝光 | 调度器 | `tycho/survey/scheduler.py` |
| agent 每晚质检、归类、提议、写日志；每周周报 | Claude agent。工具只读，写入有上限，全程审计 | `tycho/agent/` |
| 像 SDSS 一样公开发布 | `tycho release --tag DR1` | `tycho/survey/release.py` |

## RI 定义：已按论文原文核对

- 论文 Definition 1：RI_{I,S}(p) = D(p_{I+S} ‖ p_I)。完整链从 {∅}（均匀分布）开始，各项之和等于 D(p‖u)。
  代码 `chain_decomposition(start="empty")` 与此完全一致。
- 论文附录 C.1 给出两个与链无关的标准值，代码都实现了：
  - **边缘 RI**（RI^marg）：巡天用它判定事件。它只需要子集自己的小边缘表，所以能扫描成千上万个子集，
    而且有已知的 χ² 零分布。
  - **条件 RI**（RI^cond）：需要完整联合表，只用于小变量集。
- 论文图 10 的四个数值（0.1308 / 0.0992 / 0.0316 / 0）和图 1 的 XOR 例子都写成了回归测试，结果精确复现
  （`tests/test_ri_paper.py`）。
- 和论文的差异：我们的贪心选择是"强遗传 + 精确增益"版本。论文的 MAHGenTa 用 |J| 打分、30% 弱遗传、
  梯度下降配合 Gibbs/AIS。巡天只用 RI 当测量仪器，不需要学整个分布，所以没有实现 MAHGenTa。

## 阳性对照（答案已知，用来验证仪器）

- **稀疏奇偶**：标签是 k 个输入位的 XOR，所以学会以后，{这 k 位, ŷ} 这个 (k+1) 阶子集的 RI = log 2，所有低阶子集都是 0。
  已知它的动力学是"长平台后突然下降"，可以拿来对时间。
- **马尔可夫语言（带 token 的阳性对照）**：二值 token 源的条件对数几率里预设了交互项，例如滞后 1 的二阶项加
  滞后 {1,3} 的三阶项。探测时上下文 token 独立均匀采样，所以理论上只有预设子集的 RI 为正，其他含 x̂ 的子集严格为 0。
  预期二阶项先被学会，三阶项后被学会。
- **上下文马尔可夫链**（Edelman 等 2024）：unigram → bigram 的阶段跳变，用来对时间。

## 在 Mac mini 上怎么跑

```bash
ops/install_macmini.sh            # 装环境，注册 launchd（开机自启，崩溃自动拉起）
tycho calibrate --full --jobs 10  # 先校准：约 15 个 CPU 小时，10 核 Mac mini 上 2–3 小时
tycho survey enqueue              # 按 sky_v0 放入第一批天体
tycho survey status               # 随时查看
tycho agent nightly --dry-run     # 不花钱，先看 agent 会看到什么
```

需要你准备：Mac mini 本体；按 `ops/pmset_recommendations.sh` 打印出的命令设好电源；`ant auth login`
（让 agent 能调 Claude API，每晚大约不到 1 美元，有费用上限）；可选：在手机上装 ntfy 接收暂现源推送。

## 时间线

| 阶段 | 周 | 产出 |
|---|---|---|
| 1. 仪器 | 0–2 | 校准报告（误差预算）；阳性对照全部复现；和 Sugiyama 确认 RI 约定 |
| 2. 试点巡天 | 2–6 | sky v0 全覆盖，每点 ≥ 3 个种子；调好分类器和暂现源阈值；第一张边界图 |
| 3. 全面巡天 + DR1 | 6–12 | 边界自适应加密；未知与边界天体的 RI 跟进；第一次数据发布；第一批候选经验律 |
| 4. 搭桥 | 12+ | SAE 特征作为 RI 变量（RQ3）；重放加干预；把同一把尺子用到 Pythia 等公开 checkpoint |
