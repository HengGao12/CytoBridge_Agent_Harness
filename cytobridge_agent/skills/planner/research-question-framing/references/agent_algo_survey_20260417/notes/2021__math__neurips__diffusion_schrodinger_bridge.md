# Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling

## Metadata
- 标题: Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling
- 作者: Valentin De Bortoli, James Thornton, Jeremy Heng, Arnaud Doucet
- 年份: 2021
- 正式 venue: NeurIPS 2021
- PDF 文件名: Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 是
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成摘要、引言与核心算法部分阅读

## 解决了什么已有 gap

- 经典 score-based diffusion 依赖很长的前向加噪时间，目的是把终态推到接近高斯，这会带来很多离散步数和误差。
- 而严格的 SB 虽然更一般，但连续状态空间里的求解又不够可扩展。

## 理论 / 算法创新点

- 提出 Diffusion Schrödinger Bridge (DSB)，把 SB 看成路径空间上的熵正则 OT，并用近似 IPF 来求解。
- 证明 score-based reverse SDE 可以被看作 DSB 的第一步，而继续做 IPF 迭代则能进一步逼近真正的桥。
- 这实际上把 diffusion generative modeling 和 Sinkhorn/IPF 统一到一个框架里。

## 具体算法或理论结构怎么设计

- 先选定参考 SDE，再把 SB 写成正反向半桥交替校正的问题。
- 使用 Iterative Proportional Fitting 近似交替更新前向和后向过程。
- 每一步通过学习 time-inhomogeneous drift/score 来逼近对应半桥，从而逐步减少终端边缘的失配。

## 设计原则是什么

- 与其一次性硬学完整桥，不如交替逼近前后半桥。
- 生成/运输问题最好从路径空间出发定义，而不是只从终点分布出发。
- 有限时间桥往往比把分布推到纯高斯再反演更自然，也更省路径长度。

## 工程优化或实现性考虑

- DSB 的实用关键是允许较短时间区间，减少对超长 diffusion horizon 的依赖。
- 通过 IPF 结构复用 score 网络训练，使方法可以落在现有 diffusion 工程栈上。

## 局限与未解点

- 仍然需要迭代式正反向训练，计算代价不低。
- 误差可能在 IPF 迭代中积累，稳定性依赖近似质量。
- 主要针对平衡桥，未处理质量变化。

## 为什么能发顶会 / 为什么是重要理论工作

- 它非常清楚地说明了 diffusion 模型和 Schrödinger bridge 的关系，这在当时是极具影响力的统一观点。
- 同时兼顾理论和可用算法，因此在 NeurIPS 上很自然。

## 对单细胞轨迹建模和 CytoBridge-agent 自动设计算法的启发

- 对 CytoBridge 来说，这篇文章的重要性在于: 不要把 diffusion 和 bridge 看成两条路线，而应把 diffusion 看成 bridge 的特例或第一步近似。
- 如果单次学习不稳定，可以用交替半桥/迭代投影的思路来改进。
- 但如果面向单细胞场景，后续必须继续补上非平衡质量与更高维更少监督时的可扩展性。
