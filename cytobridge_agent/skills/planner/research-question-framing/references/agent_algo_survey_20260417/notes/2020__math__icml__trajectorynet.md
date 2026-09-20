# TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics

## Metadata
- 标题: TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics
- 作者: Alexander Tong, Jessie Huang, Guy Wolf, David van Dijk, Smita Krishnaswamy
- 年份: 2020
- 正式 venue: ICML 2020
- PDF 文件名: TrajectoryNet A Dynamic Optimal Transport Network for Modeling Cellular Dynamics.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 是
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成摘要、引言与方法部分阅读

## 解决了什么已有 gap

- 早期单细胞 OT 方法多是相邻时间点静态匹配，能给配对，但不能给连续时间路径。
- 这类方法也很难表达非线性轨迹、细胞增殖/死亡，以及在流形上的平滑迁移。

## 理论 / 算法创新点

- 把 continuous normalizing flow 和 dynamic optimal transport 连接起来，提出连续时间的分布迁移模型。
- 在 CNF 上加入动力学 OT 约束，使路径不再任意，而更接近最小动能的连续运输。
- 同时引入不平衡增长模块，从而让总体质量变化也能被建模。

## 具体算法或理论结构怎么设计

- 用 neural ODE 参数化速度场，把起始分布连续推进到后续时间点。
- 训练目标同时包含终点分布拟合、路径能量约束、局部流形正则，以及用于建模增殖/消失的 growth 项。
- 因而学到的不是单个时间点之间的 matching，而是一条全局一致的连续动力学。

## 设计原则是什么

- 轨迹推断不能只看端点配对，必须显式约束中间路径的形状。
- 生物过程不是严格质量守恒，增长/消失应该进入模型主体，而不是作为后处理修补。
- 连续时间模型更适合做插值、驱动基因分析和命运追踪。

## 工程优化或实现性考虑

- 利用 neural ODE/CNF 的现成训练框架，使连续路径建模在实践上可行。
- 通过附加 manifold/density regularization 来避免路径偏离观测数据流形。
- 设计了 growth 网络来补偿非平衡效应，这是单细胞场景下很关键的工程点。

## 局限与未解点

- 样本级轨迹仍是确定性的，难表达真实生物系统里的随机扩散。
- 对 latent 表示和正则系数较敏感，若表示空间几何不好，路径会失真。
- 尽管有 growth，整体仍更偏 dynamic OT，而不是严格概率路径桥。

## 为什么能发顶会 / 为什么是重要理论工作

- 它第一次把“连续时间 dynamic OT”明确做成了单细胞轨迹推断主方法。
- 同时既有数学结构，也有明确生物学应用和可视化结果，所以在 ICML 上很有说服力。
- 后续很多 single-cell OT/bridge/flow-matching 工作都在回应它留下的问题。

## 对单细胞轨迹建模和 CytoBridge-agent 自动设计算法的启发

- CytoBridge 的一个关键方向就是继承它的“全局连续动力学”视角，而不是退回相邻时间点拼接。
- 但也必须补上它没解决的随机性与更 principled 的非平衡路径建模，这正是后来 SB/uSB/FM 路线兴起的原因。
- 对 agent 来说，TrajectoryNet 说明好算法设计通常始于把问题定义改对: 从“匹配时间点”改成“学习连续动力学”。
