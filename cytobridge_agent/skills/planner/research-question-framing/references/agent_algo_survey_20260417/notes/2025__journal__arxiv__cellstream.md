# CellStream: Dynamical Optimal Transport Informed Embeddings for Reconstructing Cellular Trajectories from Snapshots Data

## Metadata
- 标题: CellStream: Dynamical Optimal Transport Informed Embeddings for Reconstructing Cellular Trajectories from Snapshots Data
- 作者: Yue Ling et al.
- 年份: 2025
- 正式 venue: arXiv preprint
- PDF 文件名: CellStream Dynamical Optimal Transport Informed Embeddings for Reconstructing Cellular Trajectories from Snapshots Data.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否，本地 PDF 仍为 arXiv 版本
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
CellStream关注的不是单纯“怎么在既有 embedding 上做 trajectory inference”，而是“能否把 temporal dynamics 直接写进 embedding 学习过程”。它要解决的是 snapshot 数据噪声大、普通降维忽略时间结构而误导轨迹解释的问题。

## 数据类型 / 场景
面向 time-resolved scRNA-seq snapshots。适合既需要降噪嵌入，又需要 embedding 本身对动态过程敏感的场景。

## 核心算法怎么设计
CellStream把 autoencoder 与 unbalanced dynamical optimal transport 结合起来，联合学习低维表示和动力学。与先降维再做轨迹不同，它让 embedding 训练直接受 dynamical OT 目标约束，从而得到 dynamics-informed embeddings。

## 设计原则是什么
设计原则是“表示学习和动力学学习不能解耦”。作者认为如果 embedding 不保留时间结构，再强的 trajectory inference 也会被前置表示扭曲，因此应该让 transport objective 反向塑造表示空间。

## 工程优化 / 训练技巧 / pipeline 设计
联合训练 autoencoder 和 dynamical OT 是其主要工程点，它用 latent space 降低高维 OT 的计算难度，同时让噪声鲁棒性优于单纯的几何 embedding。作为原型方法，它也很适合作为更大系统中的表示层。

## 局限性
这类联合学习对训练稳定性要求较高，表示层和动力学层若权重平衡不好，容易出现一边主导另一边的情况。作为 preprint，其在大规模真实数据和更多下游任务上的收益还需要更系统的公开验证。

## 为什么能发到这个级别
CellStream代表了一个很值得关注的方向：把 trajectory inference 从下游分析前移到 representation learning 本身。这个视角对于未来自动设计算法很重要，因为很多性能瓶颈其实出在前置表示。

## 对 CytoBridge-agent 自动设计算法的启发
CytoBridge 不必默认 `X_latent` 是外部给定的；完全可以把“为 dynamics 学一个更好的 latent space”本身作为算法设计对象。表示层和 transport 层的联合优化，可能是做出真正新方法的高价值切口。
