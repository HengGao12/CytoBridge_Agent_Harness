# Unbalanced Optimal Transport: Dynamic and Kantorovich formulations

## Metadata
- 标题: Unbalanced Optimal Transport: Dynamic and Kantorovich formulations
- 作者: Lénaïc Chizat, Gabriel Peyré, Bernhard Schmitzer, François-Xavier Vialard
- 年份: 2018
- 正式 venue: Journal of Functional Analysis, 274(11), 3090-3123
- PDF 文件名: Unbalanced Optimal Transport Dynamic and Kantorovich Formulation.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 是
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么已有 gap

这篇文章补的是 unbalanced OT 里最关键的理论缺口：  
此前已有不少“带源项的动态 formulation”或“局部静态 formulation”，但缺少像经典 OT 那样，把动态问题、Kantorovich 型静态问题、对偶、度量性质和特例模型统一起来的框架。

对单细胞轨迹推断尤其重要，因为细胞群体在时间上通常不守恒，增长和死亡不能再被当成噪声。

## 理论 / 算法创新点

- 提出一大类带源项的动态 unbalanced OT 变分问题。
- 引入 semi-couplings 作为静态对象，用两套边缘质量来同时描述“从哪里出发的质量”和“到哪里结束的质量”。
- 证明动态 formulation 与静态 Kantorovich-like formulation 等价，这是本文最核心的结果。
- 将 Wasserstein-Fisher-Rao (WFR) 纳入统一框架，并推出其静态公式和 entropy-transport 对偶形式。
- 证明 WFR 在合适极限下回到经典 OT，给出 Γ-收敛意义下的连接。

## 具体算法或理论结构怎么设计

文章的抽象层次很高，但设计得非常干净：

1. 动态侧用连续方程加源项 `∂_t ρ + ∇·(ρv) = ρg`。
2. 代价函数写成对 `(ρ, ρv, ρg)` 的凸、正齐次积分泛函，保证问题保持凸性。
3. 静态侧不用单一 coupling，而用一对 semi-couplings 来允许质量创建与销毁。
4. 再用 minimal path cost 把局部“搬运 + 反应”成本转成静态边成本。
5. 最后证明动态与静态完全等价，并把 WFR、partial transport 等视为该框架的特例。

这个结构的真正价值在于复用性非常强：一旦选好局部代价，就能自动生成一套动态/静态/对偶/度量理论。

## 设计原则是什么

- 保留凸性和正齐次性，这样才能同时得到理论与数值上的稳定性。
- 让“移动”和“质量变化”处于同一变分对象中，而不是后处理地加增长项。
- 把复杂模型写成一个能退化回经典 OT 的扩展，而不是另起炉灶。
- 用最小路径成本连接局部动态机制和全局静态耦合。

## 工程优化或实现性考虑

- 作者明确指出，静态 semi-coupling formulation 为 unbalanced OT 打开了新的数值求解入口。
- 对 WFR 这种结构特殊的代价，还能进一步得到更适合算法实现的静态和对偶公式。
- 文章本身不是数值算法论文，但它实际上定义了“后续该怎么设计 solver”的接口。

## 局限与未解点

- 理论建立在相当抽象的凸分析框架上，对具体应用者门槛较高。
- 结果主要针对紧集/星形域等条件，泛化到更复杂状态空间需要额外工作。
- 即使理论允许 transport 与 growth 同时存在，二者在数据上是否可辨识仍然是开放问题。
- 文章没有直接解决高维、神经参数化、稀疏观测下的学习问题。

## 为什么能发顶会 / 为什么是重要理论工作

因为它基本完成了 unbalanced OT 版的“Benamou-Brenier + Kantorovich + 对偶 + 度量 + 极限连接”全家桶。  
这不是某个局部 trick，而是把一个分散发展的方向整理成了可复用的总框架。后来很多单细胞、成像、时空分布学习工作本质上都在调用这里的思想。

## 对单细胞轨迹建模和 CytoBridge-agent 自动设计算法的启发

- 对有增殖/死亡的单细胞时序数据，必须把质量变化放进主模型，而不是只在 OT 之后再估计 growth。
- semi-coupling 思想非常适合表达“哪些细胞质量真的被匹配过去，哪些质量是本地生成/消失的”。
- 设计 novel algorithm 时，优先保留凸性、正齐次性和可退化到经典 OT 的结构，这能显著降低理论与实现风险。
- 这篇文章也提醒一个关键建模问题：growth 项应当是局部的、状态相关的，不能简单做全局缩放，否则会丢失生物解释。
