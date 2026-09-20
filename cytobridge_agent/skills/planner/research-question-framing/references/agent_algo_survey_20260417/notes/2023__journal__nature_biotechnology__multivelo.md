# Multi-omic single-cell velocity models epigenome-transcriptome interactions and improves cell fate prediction

## Metadata
- 标题: Multi-omic single-cell velocity models epigenome-transcriptome interactions and improves cell fate prediction
- 作者: Chen Li et al.
- 年份: 2023
- 正式 venue: Nature Biotechnology
- PDF 文件名: Multi-omic single-cell velocity models epigenome–transcriptome interactions and improves cell fate prediction.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
MultiVelo要回答的是：仅凭 RNA velocity 为什么常常不够，以及如何把染色质开放状态与转录动态联合起来，更准确地推断命运方向和调控时滞。它瞄准的是多组学时代对“表观-转录耦合动力学”的直接建模。

## 数据类型 / 场景
输入是同一细胞的 chromatin accessibility + unspliced/spliced RNA 多组学数据。作者在 brain、skin、blood 等数据上展示了时滞、coupled/decoupled state 与 fate prediction 的改进。

## 核心算法怎么设计
作者把经典 RNA velocity ODE 扩展成 c-u-s 三变量动力学模型，让 chromatin accessibility、unspliced RNA 和 spliced RNA 联合演化。模型用 probabilistic latent variable framework 去估计各基因的 switch time、速率参数和不同状态，并区分 chromatin 与 transcription 在时间上是 coupled 还是 decoupled。

## 设计原则是什么
核心原则是：命运转变不仅体现在 RNA 层，也体现在调控层的提前或滞后变化里。因此正确的动力学模型应允许不同模态之间存在系统性 time lag，而不是把 ATAC 只当作静态辅助信息。

## 工程优化 / 训练技巧 / pipeline 设计
MultiVelo把基因分成不同 kinetic model，并用 likelihood 比较选择更合适的模型，这让拟合更灵活。作者还把模型打包到可安装软件中，方便做跨数据集比较和 driver gene 分析。

## 局限性
方法需要高质量同细胞多组学数据，而这类数据目前成本高、噪声大、覆盖有限。它仍然是按基因建模，复杂 GRN 级联与跨细胞交互并未显式进入动力学；因此更适合作为细胞内调控延迟的刻画，而不是完整系统动力学。

## 为什么能发到这个级别
这篇工作补上了单细胞动力学里最自然的一块缺失信息，即 epigenome-transcriptome 时间关系。它不是简单做多组学整合，而是把多组学真正写进了 velocity 方程，因此方法学新意很明确。

## 对 CytoBridge-agent 自动设计算法的启发
CytoBridge 后续若接入 multi-omics，最值得做的不是共享 latent space 本身，而是跨模态的时间滞后和因果先后建模。对 fate prediction，早期调控模态往往比 RNA 本身更早暴露方向信息。
