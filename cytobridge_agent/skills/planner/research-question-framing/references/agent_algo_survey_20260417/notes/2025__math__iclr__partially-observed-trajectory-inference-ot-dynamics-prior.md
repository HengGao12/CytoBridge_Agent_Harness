# Partially Observed Trajectory Inference using Optimal Transport and a Dynamics Prior

## Metadata
- 标题: Partially Observed Trajectory Inference using Optimal Transport and a Dynamics Prior
- 作者: Anming Gu, Edward Chien, Kristjan Greenewald
- 年份: 2025
- 正式 venue: ICLR 2025
- PDF 文件名: Partially Observed Trajectory Inference using Optimal Transport and a Dynamics Prior.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 是
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么已有 gap
此前 OT / Schrödinger bridge 路线的 trajectory inference 大多假设状态是 fully observed 的，也就是建模空间里每个决定动力学的变量都被看到了。但真实问题常常只观测到位置或表达，速度、加速度、调控势能等隐变量并不可见。忽略这点会让轨迹恢复能力受到结构性限制。

## 理论 / 算法创新点
文章把 trajectory inference 扩展到 latent state-space model，提出 PO-MFL。其核心创新是：在隐空间中设定带已知 dynamics prior 的 SDE，再通过 partial observation model 把隐空间粒子与观测空间快照联系起来，同时保持 mean-field Langevin / Schrödinger bridge 这条理论线的最小熵解释。

## 具体算法或理论结构怎么设计
作者设隐变量满足 `dX_t = -Xi(t, X_t)dt - ∇Ψ(t, X_t)dt + τ dB_t`，其中 `Xi` 是已知动力学先验，`Ψ` 是待估势函数。算法在隐空间上维护粒子和 OT-style coupling，用观测模型把隐空间粒子推到观测空间做数据拟合，并给出 observability 条件和部分理论保证。文章特别强调常速度/常加速度这类 tracking 模型可以作为隐藏动力学先验。

## 设计原则是什么
设计原则是“如果某些控制未来演化的重要变量观测不到，就显式把它们作为 latent state 建模，而不是指望 observed-space drift 自己吸收一切”。这和经典状态空间模型的思想一致，只是这里把它放回了 snapshot trajectory inference 和 entropic OT 框架中。

## 工程优化或实现性考虑
工程上它延续了 grid-free MFL 的粒子化思路，因此没有把问题离散到高维网格上。隐空间粒子表示让它可以直接输出 latent trajectory samples，并在某些场景下比 observed-space baseline 更稳，特别是存在动量信息时。

## 局限与未解点
这条路高度依赖 dynamics prior 和 observability 条件是否合理。若隐藏状态设错、观测模型不对、或系统本身不可辨识，latent inference 反而会引入更强偏差。文章也没有把 unbalanced growth、interaction 或多条件迁移纳入主框架。

## 为什么能发顶会 / 为什么是重要理论工作
这篇工作能上 ICLR，核心是它不是简单加 latent variable，而是把状态空间模型、最小熵 trajectory inference 和 OT 粒子化求解严谨地拼在了一起。它说明 trajectory inference 不必局限在观测空间，这对很多科学数据都很关键。

## 对单细胞轨迹建模和 CytoBridge-agent 自动设计算法的启发
对 CytoBridge 来说，这篇文章提醒我们：只在表达空间学一阶速度场可能太弱。很多单细胞过程的“惯性”可能体现在未观测变量里，例如调控活性、染色质预备态、空间驱动或代谢势。自动设计算法时，应把“是否引入 latent momentum / hidden regulatory state”作为显式可搜索决策，而不是默认一阶 observed-state 动力学足够。
