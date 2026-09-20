# CellRank 2: unified fate mapping in multiview single-cell data

## Metadata
- 标题: CellRank 2: unified fate mapping in multiview single-cell data
- 作者: Philipp Weiler et al.
- 年份: 2024
- 正式 venue: Nature Methods
- PDF 文件名: CellRank 2 unified fate mapping in multiview single-cell data.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
CellRank 2要解决的问题是 fate mapping 经常只能依赖某一种 view，例如表达相似性或 RNA velocity，而无法统一使用时间点信息、代谢标记、多模态和跨时间转移。它希望在一个框架里恢复 terminal states、fate probabilities 和 lineage drivers。

## 数据类型 / 场景
适用于 multiview 单细胞数据，包括 expression similarity、RNA velocity、timepoint transitions、metabolic labeling 等。作者强调可扩展到百万级细胞。

## 核心算法怎么设计
CellRank 2延续了基于马尔可夫链和吸收概率的 fate mapping 思路，但把 transition kernels 模块化，允许不同数据视角定义的转移矩阵统一组合。然后再通过宏状态分解与吸收概率计算 terminal states、fate probabilities 和 genes associated with terminal fates。

## 设计原则是什么
关键原则是“view abstraction”：不同模态和不同时间信息都应先变成可组合的 transition view，再用统一的 fate-mapping machinery 处理。这样既保留了可解释的马尔可夫语义，也避免为每种数据重新发明一个全新模型。

## 工程优化 / 训练技巧 / pipeline 设计
CellRank 2很强的一点是可扩展性设计，包括更通用的 kernel API 和能跑到百万级细胞的实现。支持 metabolic labeling 数据来估计 cell-specific transcription/degradation rates，也体现了工程上对新实验模态的快速兼容。

## 局限性
它仍然是基于转移矩阵的 fate mapping，而不是完整的连续生成动力学模型，因此对长期外推、分布重建和 perturbation 模拟的支持有限。不同 view 如何加权组合也会显著影响结果，且该组合本身并非总有明确的生物依据。

## 为什么能发到这个级别
这篇文章的重要性在于把单细胞 fate mapping 从“某一种数据的专用工具”升级为“统一处理多视角 transition evidence 的框架”。在多模态单细胞快速普及的背景下，这种框架化能力非常关键。

## 对 CytoBridge-agent 自动设计算法的启发
CytoBridge 可以借鉴 CellRank 2 的思路，把不同证据源先转为统一的 transition object，再决定是做 OT、SB 还是 absorbing Markov analysis。对 agent 自动设计来说，view modularity 是很有价值的系统设计原则。
