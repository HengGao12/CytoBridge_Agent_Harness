# Optimal-Transport Analysis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming

## Metadata
- 标题: Optimal-Transport Analysis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming
- 作者: Geoffrey Schiebinger, Jian Shu, Marcin Tabaka, Brian Cleary, Vidya Subramanian, Aaron Solomon, Shawn Liu, Sten Linnarsson, Josef Kostka, Rudolf Jaenisch, Aviv Regev, Eric S. Lander
- 年份: 2019
- 正式 venue: Cell
- PDF 文件名: Optimal-Transport Analysis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已读引言、核心结果、讨论

## 解决了什么生物问题

它要解决的是：在致密时间分辨的重编程实验中，如何从大规模单细胞快照恢复真正的发育/命运轨迹，而不是只给出一个几何上的伪时间。生物学上作者借此系统剖析了成纤维细胞重编程为 iPSC 的多分支命运、关键 TF 和旁分泌作用。

## 数据类型 / 场景

- 315,000 个单细胞转录组，覆盖成纤维细胞到 iPSC 重编程的致密时间序列。
- 典型场景是多时间点 snapshot distribution，且细胞在时间间隔内可能增殖、死亡和分支。
- 明确适合 out-of-equilibrium developmental process，而不是静态平衡群体。

## 核心算法怎么设计

- 作者把发育过程建模为高维表达空间中的时间变分布 `P_t`。
- 真正缺失的是不同时间点之间的 temporal coupling。WOT 用 optimal transport 来近似这个 coupling，核心假设是短时间内细胞移动距离不会太大。
- 与经典 OT 不同，细胞会增殖和死亡，所以作者引入 unbalanced transport，并利用增殖/凋亡 signature 先验来估计 growth rate。
- 先计算相邻时间点的 coupling，再通过连续时间点复合获得长程 ancestor/descendant distribution。
- 不把轨迹限定成单一路径，而是把“某类细胞未来会分布到哪里、祖先来自哪里”定义成分布对象。

## 设计原则是什么

- 轨迹应建模为分布间耦合，而不是单细胞几何曲线。
- 时间信息必须被显式利用；单纯的 manifold 方法会产生违背时间顺序的结果。
- 质量守恒在细胞系统中通常不成立，因此 growth/death 不能省略。
- 对 snapshot data，合理的最优性假设是“短时间最小位移”，而不是全局树结构先验。

## 工程优化 / 训练技巧 / pipeline 设计

- 对 held-out time point 做验证，这是非常强的工程/科学双重检验。
- 参数鲁棒性分析做得比较充分，说明方法不是只在一个调参点上工作。
- 论文把 trajectory、TF 程序和旁分泌信号分析连成完整 pipeline，使方法不仅能“连线”，还能“解释机制”。

## 局限性

- 依赖 Markov 假设，默认未来主要由当前状态决定。
- 主要捕捉 `P_t` 的时变部分；若系统总体分布近似平衡但个体仍在快速循环，OT 会失灵。
- growth rate 来自外部 signature 近似，不是从联合模型中完全自洽地学出来。
- 论文自己也承认，细胞间相互作用和多时间点联合优化仍可继续扩展。

## 为什么能发到这个级别

这是单细胞动力学领域非常关键的一篇论文，因为它把 OT 从抽象数学工具真正变成了可用于大规模生物时间序列分析的工作流，并且明确解决了“细胞会增殖/死亡”这个生物现实问题。再加上 31.5 万细胞的重编程资源本身极强，方法与生物发现相互放大，完全达到 Cell 级别。

## 对 CytoBridge-agent 自动设计算法的启发

- 对 CytoBridge 这类 agent，WOT 是最重要的设计模板之一：时间信息、质量非守恒、ancestor/descendant distribution 都应该是一等公民。
- 自动设计算法时，不要默认输出单一路径；很多问题天然应该输出 fate distribution 或 coupling。
- 如果数据是多时间点快照，agent 应优先考虑 transport 类方法，并主动判断是否需要 growth/death 修正。
