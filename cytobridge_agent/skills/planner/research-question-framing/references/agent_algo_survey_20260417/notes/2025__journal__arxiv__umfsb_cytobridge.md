# Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schrödinger Bridge

## Metadata
- 标题: Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schrödinger Bridge
- 作者: Zhenyi Zhang et al.
- 年份: 2025
- 正式 venue: arXiv preprint
- PDF 文件名: Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schrödinger Bridge.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否，本地 PDF 仍为 arXiv 版本
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
这篇文章直接面向 CytoBridge 的核心问题：如何从稀疏时间快照里同时建模状态转移、群体增长和细胞间相互作用。它试图弥补此前 OT/SB 方法普遍缺少交互项这一明显空白。

## 数据类型 / 场景
输入是 time-resolved snapshot 单细胞数据，适合存在显著 cell-cell communication 或 mean-field effect 的系统。文章同时使用合成 GRN 数据和真实 scRNA-seq 进行验证。

## 核心算法怎么设计
作者先提出 Unbalanced Mean-Field Schrödinger Bridge 理论框架，把 unbalanced stochastic dynamics 与 mean-field interaction 放在统一桥问题中；随后提出 CytoBridge 算法，用神经网络分别参数化 transition velocity、growth rate、log density 和 interaction potential，并通过带权 interacting particles 近似目标函数。

## 设计原则是什么
设计原则非常清晰：如果交互会改变命运动力学，就必须把 interaction 写进主动力学，而不是事后从表达相关性里再推断。另一原则是 unbalanced、stochastic、interaction 三者要统一求解，因为它们在现实系统中彼此耦合。

## 工程优化 / 训练技巧 / pipeline 设计
用 neural networks 参数化四个核心场，并采用 weighted interacting particles 近似密度演化，是其主要工程实现。相比传统 SB 理论，这样的设计更贴近可训练的深度学习 pipeline。

## 局限性
作为 preprint，文章仍需要更广泛的外部验证和更成熟的复杂度分析。交互项引入后，模型的可辨识性、训练稳定性和过拟合风险都会更高；作者也承认真实应用中需要更充分的偏差评估与外部验证。

## 为什么能发到这个级别
它的重要性在于把 CytoBridge 这条路线的理论问题写清楚了：不是简单给 OT 加个 interaction regularizer，而是从 mean-field Schrödinger bridge 出发重新定义任务。对当前领域而言，这是非常少见的完整问题升级。

## 对 CytoBridge-agent 自动设计算法的启发
这篇文章本身就是对 agent 设计目标的直接支撑。若要做真正 novel 的算法，interaction 不应只是附加分析模块，而应作为 bridge dynamics 的原生组成部分，并和 growth、drift 一起联合学习。
