# Unnormalized optimal transport

## Metadata
- 标题: Unnormalized optimal transport
- 作者: Wilfrid Gangbo, Wuchen Li, Stanley Osher, Michael Puthawala
- 年份: 2019
- 正式 venue: Journal of Computational Physics, 399, 108940
- PDF 文件名: Unnormalized optimal transport.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 是
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么已有 gap

作者想解决的是一个非常务实的 gap：  
经典 OT 要求总质量相等，但许多真实问题不满足；而已有的 WFR/Hellinger-Kantorovich 路线虽然允许不守恒，却引入了空间依赖的源项，导致 Hamilton-Jacobi 结构和数值求解都更复杂。

这篇文章提供了一个更“保守的扩展”：  
允许质量不守恒，但把源项限制成只随时间变、与空间无关，从而尽量保留经典 OT 的解析结构。

## 理论 / 算法创新点

- 在连续方程中加入空间无关的源项 `f(t)`，定义 unnormalized Wasserstein 距离。
- 由于源项设计得足够简单，最优性条件仍然保留经典 OT 的 Hamilton-Jacobi 方程。
- 在此基础上推导出 unnormalized Monge problem、unnormalized Monge-Ampere equation 和 Kantorovich duality。
- 给出基于 Chambolle-Pock 的 primal-dual 数值算法，并强调计算复杂度与标准 OT 同阶。

## 具体算法或理论结构怎么设计

结构非常清晰：

1. 动态 formulation: `∂_t μ + ∇·(μv) = f(t)`。
2. 代价由运输动能和全局质量变化惩罚两部分组成。
3. 由于 HJ 方程保持不变，轨迹仍然沿经典 OT 的特征线结构展开。
4. 于是可以继续导出 Monge 映射、Monge-Ampere 方程和 Kantorovich 对偶。
5. 数值上把问题写成 primal-dual saddle-point，再用 Chambolle-Pock 求解。

可以把它理解成：作者牺牲了源项的表达自由度，换来了“尽量延续经典 OT 整套工具链”。

## 设计原则是什么

- 不要为了扩展能力破坏全部核心结构。
- 如果一个额外自由度会让 HJ / Monge-Ampere / 对偶全部失效，就先考虑受限版本。
- 先保留可解释和可计算的主干，再逐步放宽模型。

## 工程优化或实现性考虑

- 明确面向计算设计，强调和标准 OT 基本相同的复杂度。
- 使用 Chambolle-Pock 一阶 primal-dual 方法，易于实现且可扩展。
- 文章还给出数值例子，证明该 formulation 不只是理论上的可解。

## 局限与未解点

- 最大限制就是源项 `f(t)` 与空间无关，这对单细胞任务通常过于强，会把“状态特异性增殖/死亡”压扁成全局质量变化。
- 因而它更像是一个可计算基线，而不是最终的生物动力学模型。
- 模型不包含随机桥过程、扩散先验或条件生成机制。
- 与更一般 UOT/WFR 相比，表达能力明显受限。

## 为什么能发顶会 / 为什么是重要理论工作

因为它展示了一条非常典型也非常值得学的理论路线：  
不是一味追求最泛化，而是通过恰当约束把“扩展能力”和“解析/计算可解性”做了平衡。  
在 JCP 这种 venue 上，这种“从变分结构直达可实现算法”的论文是很典型的高质量工作。

## 对单细胞轨迹建模和 CytoBridge-agent 自动设计算法的启发

- 这是一个很好的 ablation baseline: 如果只允许全局质量变化，模型能解释多少数据；解释不了的部分，就是状态相关 growth 的证据。
- 它提醒我们，源项设计是决定模型可解释性和可计算性的关键旋钮。
- 如果 agent 以后要自动设计 UOT 类算法，应把“是否允许空间/状态依赖 source”作为核心分支决策，而不是实现细节。
- 从工程上看，这篇文章说明只要保住 HJ 主干，很多现有 OT 数值工具都还能继续复用。
