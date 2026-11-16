# Discussion of 16 November 2026: three outside reviews of the project (exploratory)

> Context: on 16 November the researcher asked three AI systems the same question: how good are the taste and the
> depth of this project, what follows from first principles, and what is the top-level view. The three were Claude
> (this Claude Code session, with the whole repository and its data), Gemini (first the README only, then the file
> tree) and Google Antigravity (the whole repository, including the Part B pre-registration draft). The researcher
> then discussed the answers with Claude.
>
> Status: **exploratory. Nothing here changes the plan.** `docs/ROADMAP.md` and `README.md` are untouched. The
> researcher thinks a repositioning is needed, and wants to cross-check it with other AI systems before deciding.
>
> Form: a curated record, not a transcript. The researcher's two statements in §1 are quoted. The reviews by Gemini
> and Antigravity are summarised; their full texts are with the researcher. One remark of a personal nature was left
> out on purpose.
>
> Numbers: `docs/results/v2_hierarchy_projections.md` (with its verification record) and
> `scripts/analysis_v2/discussion_checks.py` → `docs/results/v2_analysis/discussion_checks_2026-11-16.json`. The
> second set are quick checks made during the discussion; they were not recomputed independently.
>
> Kept in Chinese, as the discussion was.

---

## 1. 研究者的两段说明（原话）

**关于 RI 的位置：**

> ri是我的导师的研究成果所以我当时研究计划写的是用ri，但其实在我申请之前我已经构思了这个巡天项目了，也就是我对工具没有偏好，所以ri不是主要的，发现规律才是重要的，可能我在一次次修改中忘记当时ri只是备选方法中的一种而已，如果效果不好就用效果好的就好，我希望从idea到实验我们都遵循第一性原理。

**关于这次讨论怎么记：**

> 我建议先把我们今天的讨论定位探索性的，有价值的结论或者讨论过程内容可以计入探索性笔记里，我还不着急修改roadmap和readme，这并不是说我不同意重新定位或我认为没必要重新定位，我认为有必要，但是我想再和其他ai交叉讨论，也给你看看别的看法。

另外一点背景：课题由研究者自主推进，思路层面的外部校正需要另找来源（§6 第 6 条）。

## 2. 三份评价各自说了什么

**Claude（本会话；读了全部文档、结果和数据）**

- 总评：方法论的品味好于选题的品味；流程深，科学还浅。
- 六条批评：
  1. 项目从形式出发（"一个能跑几个月的平台"），不是从问题出发；
  2. 天文类比用过头：能做实验的领域里，先观测再找规律是较弱的方法；
  3. "第谷之前"的定位不成立：低阶统计量先被学会已有成熟文献（distributional simplicity bias）；
  4. gate 2 把同一个函数的两套坐标当成对手；
  5. 把连续过程切成阈值事件，规律跟着阈值走；
  6. 横向漂移：13 天开了 7 个题目，没有关上一个。
- 建议：按阶分解损失本身；先写预测再测量；巡天的对象换成"一条候选规律的适用范围"。

**Gemini（只看了 README，随后看了目录树；没有看到任何结果）**

- 评分：品味 8.5，看到目录树后 9.2；深度 6.0。
- 称赞：天文类比、两级观测加重放、暂现源、校准先行、五个任务的组合、夜间 agent。
- 批评：有观测无定律；变量是什么（输入上退化成布尔展开，隐藏单元上是基底伪影）；玩具规模。
- 建议：用 µP 做坐标；证明 eNTK 和有效秩是 RI 相变的前兆；用重放做因果干预；调度器主动寻找相边界；用连续估计器
  校验离散化；"跨任务普适拓扑"的叙事；三篇论文（平台与星表，经验律，机制）。

**Antigravity（读了全部仓库，包括 Part B 预注册草稿）**

- 评分：方法论 A−，选题 B；统计与数值 9/10，训练动力学的科学深度 5.5/10。
- 三个硬伤：工具先行；隐喻过度基建；核心科学判断让渡给了 AI 协作者。
- 第一性原理三条：
  1. θ 赢不了 η 是数学上的必然：损失对 logit 的梯度正比于概率残差；头消融里 AND 是概率尺度上的纯二阶交互，
     OR 在对数几率尺度上产生强耦合；
  2. RI 的主场是无法做均匀干预的自然相关分布，头消融的 D = U 用不上它；
  3. 用五六个有算法含义的二值探针变量把表固定在 64 格，就能直接测 Pythia。
- 顶层设计：θ 和 η 做成"双通道光谱仪"；"瞬态捷径、延迟、熄灭"是金矿，用重放做因果干预；三层金字塔（对偶测量学，
  捷径动力学定律，规模锚点）；二进制数据移出 git；冻结外围基建。

## 3. 一致的判断

| 判断 | 谁 |
|---|---|
| 先校准、预注册、认负结果，是项目最强的部分 | 三方 |
| 有测量、没有规律；论文需要一个主张，不能写成系统说明书 | 三方 |
| 变量是什么，是测量合法性的核心问题 | Gemini、Antigravity；Claude 认为自己第一轮说轻了 |
| 基建超前于科学，外围应当冻结 | Claude、Antigravity |
| 不要再把 θ 对 η 当擂台；Part B 不该押注 θ 胜出 | Claude、Antigravity |
| 异常被记下而没有被追（v1 的 Q4：奇偶指数约 2，与 k 无关） | Claude、Antigravity |
| 核心判断过多由 AI 推进，答辩时是弱点 | Claude、Antigravity |
| 重放最有价值的用途是干预 | Gemini、Antigravity；Claude 接受 |

同一天主分支上的独立审查（`docs/results/v2_analysis/v2b_prereg_review.md`）从内部得出了与第五行相同的结论：Part B
草稿里的 R2 不能区分两种视角，R1 的结果取决于一个标注选择，θ 没有可达的胜出路径。

## 4. 分歧，以及仓库的数据怎么说

| 议题 | 论断 | 数据 |
|---|---|---|
| 两级观测、暂现源、agent、任务组合是亮点 | Gemini | v1 起主要区域在训练中原位测 RI，v2a 关掉了跟进和加密；DR0 的 6,791 条和 DR1 的 2,028 条暂现源没有导出结论；五次发布共 8,590 个天体，agent 提议的为 0；任务从 5 个减到 2 个 |
| 调度器应主动寻找相边界 | Gemini | 已经做过，见 §5.1：加密收敛到"泛化时间等于预算"这条线 |
| eNTK 和有效秩是前兆 | Gemini | 96 格里没有一格在 10 % 水平领先；低于 0.8 的 5 格都是先升后降的病态序列（结果笔记 §3） |
| θ 必然输给 η | Antigravity | 不成立，见 §5.3 和 §5.4：双头表上 θ 给 OR 的值接近 0，η 分不出 AND 和 OR；Part B 审查里排除辅助头之后是平局 |
| 瞬态捷径是普适规律，是金矿 | Antigravity | 定性部分已发表（§5.5）；Part B 的 tiled 数据里网络没有停在低阶解上（§5.4） |
| 层次投影预测训练轨迹 | Claude | MLP 上 11 个里 6 个干净；transformer 上强形式不成立（§5.4），需要加条件（§6 第 3 条） |
| 起点是工具 | Claude、Antigravity | 研究者说明巡天构想在先（§1）。"工具"这一半收回；"从形式出发"这一半保留 |

## 5. 讨论中做的核对

### 5.1 边界加密采到了什么（DR1，grokking 区域）

- 689 个天体里 509 个来自边界加密，其中 453 个在新增的中点上。
- 加密天体里只有 76 个（15 %）在 5 万步内到达 90 % 测试准确率，粗网格是 180 个里的 47 个（26 %）。405 个落在
  "没有任何种子到达"的配置上。
- 只有部分种子到达的配置有 9 个。到达的 32 个种子用了预算的 56 % 到 100 %（中位数 92 %），没到达的最终准确率在
  0.81 到 0.90。种子之间的分歧是阈值和预算造成的。
- 同一批数据里没被截断的连续量给出干净的关系。宽 256、lr 0.003、wd 1.0 的切片上，训练比例 0.3625、0.375、0.4、
  0.5 的泛化时间是 25,953、16,667、8,700、2,917 步；1/T 对训练比例是直线（R² 0.999），外推到零在 0.347。按这条线
  0.35 处需要约 16 万步，是预算的 3 倍。lr 从 0.003 降到 0.001，T 变成 2.95、2.98、3.20 倍。
- 限制：直线只有一个切片的四个点，lr 之比只有三处；这个关系在 grokking 文献里是已知的；0.347 依赖 90 % 这个
  阈值，不是相变点。

### 5.2 层次投影与延迟规则

见 `docs/results/v2_hierarchy_projections.md`。要点：数据的"至多 k 阶最佳拟合"可以在训练前算出；DR2a 的网络在 11 个
中间拟合里的 6 个上贴得很近（距离是该拟合到真值距离的 1.3 % 到 6.8 %）；v1 的 lag-4 捷径就是一阶拟合；低阶项造成
的延迟大部分可以用停留处的输出增益解释（按增益加权时间后，1.27–1.59 降到 1.02–1.24），但这条规则不能跨到链式条件。

### 5.3 两个头、均匀消融：AND 和 OR 在两种视角下的值

单位 nats；θ 是"在含两个单项的 logistic 模型上加入成对项"带来的信息，η 是预注册里的有界值。

| 表（两头都去掉 / 只留其一 / 都留） | θ | η |
|---|---|---|
| AND：0.016 / 0.016 / 0.96 | 0.021 | 0.116 |
| OR：0.016 / 0.99 / 0.9999（Antigravity 用的数） | 0.0001 | 0.121 |
| OR：0.016 / 0.96 / 0.96 | 0.047 | 0.116 |
| 只留其一得 0.2 | 0.003 | 0.042 |
| 只留其一得 0.5 | 0.001 | 0.0001 |

- 在对数几率尺度上，带饱和的 AND 和 OR 都接近可加，所以 θ 都小。Antigravity 说冗余头在 θ 下产生强耦合，用它自己
  的数算出来是 0.0001。
- η 给 AND 和 OR 几乎相同的值，只有符号不同。所以"η 把串联组合排第一"不是两个头的代数能推出的，它来自多个冗余头
  的结构。
- 它正确的部分：损失对 logit 的梯度确实正比于概率残差。这正是延迟规则的来源，也解释了网络为什么会停在 k 阶拟合
  附近：在那里，所有至多 k 阶方向上的驱动力恰好为零。

### 5.4 Part B 的数据：网络会不会先停在低阶解上

tiled 布局（词表 64，64 个 token，片段长 4 到 16）下，不做内容匹配的解在可预测 token 上的损失，只由数据分布算出：

| 预测器 | 损失（nats） |
|---|---|
| 均匀 | 4.159 |
| 上下文计数 | 2.25 |
| 在已出现的不同 token 上均匀 | 2.12 |
| "抄 d 个位置之前的 token"的最佳固定混合 | 1.96 |
| 同上，权重随位置变 | 1.56 |

- 1 层对照在 12,000 步时是 1.65 到 1.89，仍在缓慢下降，落在后两行之间。1 层模型实现的就是这个不依赖内容的解。
- 2 层网络（lr 3e-4）到第 300 步为止和 1 层几乎走同一条路（第 250 步 3.46–3.49 对 3.54–3.59，第 300 步
  3.19–3.23 对 3.31–3.38）。到第 400 步，2 层已到 1.56–1.65，1 层还在 3.01–3.06；第 500 步 2 层是 0.61–0.67。
- 在 1.6 到 2.3 这个区间里，2 层网络最多停一个评估点；1 层停 22 到 31 个（两个学习率合计）。
- 所以这里的低阶解比电路学得慢，2 层网络在损失 3.2 到 3.4 之间就离开了，没有先学完它。"先陷入低阶解、再被高阶解
  替换"在这组数据里不成立。
- single 布局（pilot 1）里，没有形成完整电路的 10 个运行停在 0.83 到 1.60。同一布局下不依赖内容的解算出来是
  2.83 到 3.02（按位置抄）和 3.37 到 3.39（计数），所以那个平台不是低阶捷径，而是另一个用到内容的解。其中 6 个在 8,000 到 12,000 步之间还在下降（3 个超过 0.05 nats），另有
  一个运行在 6,000 到 8,000 步之间才从 0.89 掉到 0.27。"卡死"和"只是慢"在 12,000 步的预算下分不开。

### 5.5 文献核对

- Edelman 等（NeurIPS 2024，本仓库 `icl_markov` 任务的来源）摘要里就有：较简单的 unigram 解的存在可能推迟最终
  bigram 解的形成。
- Singh 等（NeurIPS 2023）：上下文学习在训练中先出现、后消失，被权重内学习取代。
- Park 等（ICLR 2025）：把模型行为分解成几种算法（unigram 或 bigram，检索或推断）的竞争，随训练步数和数据多样性
  切换。Antigravity 的"算法探针变量"与这个做法相近。
- 所以"低阶解先出现、推迟高阶解、随后被替换"作为定性现象已经发表。还没有被做掉的可能是定量部分：停在哪里、
  损失多少、停多久。这一点需要研究者自己查清。

## 6. 讨论后的暂定看法（不是决定）

1. **仪器按损失选。** RI 不是主角。交叉熵训练时，对数线性层次给出损失自身的可加分解；平方误差训练时对应的是矩。
   留下的是按阶的层次分解，放下的是逐子集扫描和事件检测。
2. **θ 对 η 的擂台停下。** 三份评价里两份这样说，主分支上的独立审查也从内部得出同样结论。Part B 的预注册草稿
   怎么处理，留给研究者决定（§7 第 7 条）。
3. **候选主线分成两半。**
   - 候选的停留状态和它们的损失可以由数据分布事先算出（MLP 上是各阶最佳拟合；tiled 归纳任务上是不依赖内容的解，
     算出来 1.56 到 1.96，1 层对照实测 1.65 到 1.89）。
   - 网络是否真的停在那里、停多久，由相邻两个解的学习时间尺度之比决定，这取决于架构和优化器。MLP 上低阶学得快，
     所以会停；attention-only transformer 上不依赖内容的解学得慢，所以不停。
   - 第一性原理的含义：梯子的顺序是学习者自己的时间尺度，不是一个与学习者无关的"阶"。
4. **巡天的对象。** 从"训练运行"换成"一条候选规律的适用范围"：同一套校准过的测量，系统地覆盖条件，画出规律在
   哪里成立、在哪里失效。
5. **主动采样。** 不做标签边界上的二分。值得做的顺序是：自适应预算（跑到事件发生）；按假设选点（放在竞争规则
   预测差别最大的地方）；有了预测模型之后按残差加密。
6. **外部校正。** 课题自主推进，AI 协作者会顺着给定的框架往下做，查得出算术错，查不出问法错。可用的来源按成本
   排序：文献；先写预测再跑实验；相邻论文的作者；workshop 投稿；不同模型之间的交叉审查。建议的最小分工：主张和
   预测由研究者手写一页，实现和复算交给 agent。

## 7. 留给下一轮交叉讨论的问题

1. 延迟规则（延迟约等于停留处输出增益的倒数）是否已经包含在 gradient starvation（Pezeshki 等 2021）的分析里？
2. §6 第 3 条的两半，相对 distributional simplicity bias（Refinetti 2023，Belrose 2024，Rende 2024）、Edelman 2024、
   Park 2025 和 Kunin 2025，还剩什么是新的？
3. 对 attention-only transformer，什么量能事先给出每个候选解的学习时间尺度？
4. single 布局里损失 0.8 到 1.6 的那个长平台是什么解？是陷阱还是只是慢？
5. v1 的 Q4：奇偶任务的指数约为 2、与 k 无关。是参考值 n^(k−1) 不适用于这个设置，还是一个发现？
6. 论文主张的最小版本是什么，什么结果算它失败？
7. Part B 的预注册草稿：改目标、改变量，还是停？
8. Antigravity 的"算法探针变量"和"模型到每个候选解的预测分布的距离"，哪一个是通向 Pythia 的更好的观测量？

## 8. 指针

- 结果笔记：`docs/results/v2_hierarchy_projections.md`；复算记录：
  `docs/results/v2_analysis/hierarchy_projections_verification.md`。
- 本次讨论的核对：`scripts/analysis_v2/discussion_checks.py`，
  `docs/results/v2_analysis/discussion_checks_2026-11-16.json`。
- Part B 的现状：`docs/results/v2b_preregistration.md`（草稿），`docs/results/v2_analysis/v2b_prereg_review.md`。
- 文献：
  - Edelman, Edelman, Goel, Malach, Tsilivis (2024). The evolution of statistical induction heads: in-context learning
    Markov chains. *NeurIPS*. arXiv:2402.11004.
  - Singh, Chan, Moskovitz, Grant, Saxe, Hill (2023). The transient nature of emergent in-context learning in
    transformers. *NeurIPS*. arXiv:2311.08360.
  - Park, Lubana, Pres, Tanaka (2025). Competition dynamics shape algorithmic phases of in-context learning. *ICLR*.
    arXiv:2412.01003.
  - 其余见结果笔记的参考文献。
