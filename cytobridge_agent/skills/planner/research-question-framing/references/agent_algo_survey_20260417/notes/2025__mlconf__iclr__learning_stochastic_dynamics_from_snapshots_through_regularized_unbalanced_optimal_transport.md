# Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport

## Metadata
- 标题: Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport
- 作者: Zhenyi Zhang, Tiejun Li, Peijie Zhou
- 年份: 2025
- 正式 venue: ICLR 2025
- PDF 文件名: Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 是；PDF 首页明确标注 “Published as a conference paper at ICLR 2025”
- 是否与 CytoBridge-agent 目标直接相关: 是；直接面向 time-resolved snapshot dynamics、RUOT、单细胞轨迹推断
- 阅读完成状态: 已完成（读过摘要、引言、理论部分、方法、实验、结论）

## 解决了什么已有 gap

这篇文章解决的是一个比“平衡 OT / SB”更贴近真实单细胞数据的问题：当细胞群体存在增长、死亡、分叉扩增时，质量并不守恒，而标准 OT 或标准 SB 往往会通过“错误迁移”去强行解释细胞数变化，结果就是伪转移和错误 fate path。

已有方法的问题主要有三类：

- dynamical OT 通常不建模随机性；
- SB 建模随机性，但大多默认质量守恒；
- 少数 unbalanced 方法需要先验增长率、谱系或额外信息。

这篇论文补上的 gap 是：在没有 growth/death 先验的情况下，直接从稀疏快照中学习连续的 unbalanced stochastic dynamics。

## 理论 / 算法创新点

它的核心创新是把 Regularized Unbalanced Optimal Transport (RUOT) 写成一个更适合深度学习求解的 Fisher regularization 形式，并由此提出 DeepRUOT。

几个关键点：

- 把 RUOT 与 Schrödinger bridge / unbalanced SB 的关系讲清楚，说明 RUOT 可以视作带质量变化的随机动力学问题。
- 用 Fisher regularization 形式把原先带扩散的 SDE 控制问题，转成以 probability-flow ODE 为核心的等价优化问题。
- 通过显式引入 growth/death 项 `g(x,t)`，避免平衡模型用虚假 transport 去解释质量变化。
- 联合学习速度场、增长率和 log-density / score，而不是只学 transport map。

## 具体算法或理论结构怎么设计

理论上，作者先从带扩散的动态 RUOT 形式出发，在 continuity/Fokker-Planck 约束下，推导出等价的 Fisher regularized 目标。这个目标包含：

- 速度场动能项；
- Fisher information 项；
- 与 growth/death 相关的交叉项；
- growth penalty。

然后把问题转为学习三类神经网络：

- `v_θ(x,t)`：probability flow ODE 的 drift；
- `g_θ(x,t)`：增长/死亡率；
- `s_θ(x,t)`：log density 的近似，用于刻画 score / landscape。

训练损失由三部分组成：

- `L_energy`：最小作用量式的能量项，对应 RUOT 主目标；
- `L_recons`：把演化后的分布对齐到后续时间点，包括局部质量匹配与 unbalanced OT 对齐；
- `L_FP`：强制满足 Fokker-Planck / continuity 约束。

训练流程上还用了一个预训练阶段，先做分布重建与 score matching，再进入完整联合优化，以提升稳定性。

## 设计原则是什么

- 把“随机 + 非守恒”当成一等公民，而不是先做平衡 transport 再事后修正。
- 优先寻找等价但更可训练的理论形式。这里最关键的就是把 SDE 问题改写成 ODE + score + growth 的联合学习问题。
- 显式分解不同生物机制：迁移由 `v` 解释，增殖/死亡由 `g` 解释，避免不同机制在一个 transport term 里互相污染。
- 在高维场景下不追求封闭解，而是设计可用神经网络+Monte Carlo+Neural ODE 近似的目标。

## 工程优化或实现性考虑

- 推导出的 Fisher regularization 形式避免了更难算的导数-向量交叉项，比已有 RUOT 形式更容易实现。
- 通过 probability flow ODE 而不是直接神经 SDE 训练，降低数值复杂度。
- 预训练阶段先做 reconstruction 与 score matching，有助于后续 joint training 稳定。
- 引入局部质量匹配，避免只看全局 transport 导致数量对齐粗糙。
- 实验不仅有 synthetic GRN 和 GMM，还有真实 scRNA-seq hematopoiesis 数据，并展示了 Waddington landscape，可解释性强。

## 局限与未解点

- 作者在结论里明确提到，后续仍需要把 latent embedding 学习与 dynamics 联合起来；当前较依赖预先选定的表示空间。
- 也承认算法效率仍可进一步提升，说明目前训练成本并不轻。
- 实验中多数设置依赖特定扩散系数和网络设计，方法虽然完整，但调参与实现复杂度高于标准 FM。
- 论文聚焦单条件时间演化，对多条件共享、跨扰动泛化、外推到未见条件还没有直接解决。

## 为什么能发顶会 / 为什么是重要理论工作

我觉得它能发 ICLR 2025，原因是它同时满足了“理论 gap 明确”和“应用价值直接”：

- 理论上填补了 unbalanced stochastic snapshot dynamics 缺少深度学习求解器的空白；
- 方法上把 growth/death 从 transport 中分离出来，直接解决了单细胞里很典型的伪转移问题；
- 实验上给出了单细胞发展景观、增长模式和定量指标改善，故事完整且有生物解释。

这类工作很重要，因为它把后续大量单细胞动力学方法都绕不开的一个问题摆到了台面上：质量不守恒不是细节，而是建模核心。

## 对单细胞轨迹建模和 CytoBridge-agent 自动设计算法的启发

- agent 在自动设计算法时，第一步就应判断任务是否需要 balanced / unbalanced 建模，而不是默认守恒。
- 当数据存在明显扩增、死亡、采样密度变化时，应显式建模 growth term；否则很容易用错误迁移去解释数量变化。
- 这篇文章提供了一个很强的设计模板：`transport + stochasticity + mass change + density/landscape` 联合学习，而不是只学某一个 transport field。
- 对 CytoBridge 来说，若未来要做更可信的 fate prediction 和 Waddington landscape，这篇工作几乎是必读基线。
