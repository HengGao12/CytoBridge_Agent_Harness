# Spatial transition tensor of single cells

## Metadata
- 标题: Spatial transition tensor of single cells
- 作者: Peijie Zhou et al.
- 年份: 2024
- 正式 venue: Nature Methods
- PDF 文件名: Spatial transition tensor of single cells.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
STT 要补的是空间转录组里最难的一块：如何同时利用空间位置和 RNA splicing 信息恢复 cell-state-specific spatial dynamics，而不是只做空间聚类或无空间的 velocity。作者用它分析 EMT、血液发育、脑和心脏发育中的空间状态迁移。

## 数据类型 / 场景
输入是 spatial transcriptomics，最好还能估计 unspliced/spliced 信息。适合既关心局部空间流线，又关心长程 attractor 级转变路径的发育或病理系统。

## 核心算法怎么设计
STT 构造一个四维 spatial transition tensor，在细胞状态、目标状态、空间位置和方向上联合编码转移倾向。随后通过 spatially constrained random walk，从短时局部 tensor streamlines 到长时 attractor 间 transition paths 两个尺度恢复空间动力学。

## 设计原则是什么
设计上强调 multiscale 和 multistability。作者不满足于输出一个局部速度向量，而是要求同一框架同时覆盖局部流、全局路径和 attractor 结构；这与单纯把 velocity 投影到空间上的做法差别很大。

## 工程优化 / 训练技巧 / pipeline 设计
STT 的一大工程点是先借助 reference scRNA-seq 改善 spatial 数据里 spliced/unspliced 估计，再做 transition tensor 学习。局部 tensor streamlines 加长程 random walk 的组合，也使结果同时适合可视化与全局总结。

## 局限性
它对空间数据质量和 splicing 估计质量要求高，而很多 ST 平台在这方面仍然受限。方法的张量结构较复杂，解释成本和调参成本都不低；对大规模组织切片的扩展性也需要更多实证。

## 为什么能发到这个级别
这篇文章的重要性在于它不是简单把空间加到已有 velocity 工具上，而是为“空间状态转移”单独定义了新的表示对象和 multiscale 推断流程。对空间动力学分析来说，这是很明确的方法学推进。

## 对 CytoBridge-agent 自动设计算法的启发
CytoBridge 若进入空间轨迹建模，可以考虑把 dynamics 表示成 tensor 或 operator，而不必局限于向量场。这样更容易兼顾多吸引子、多路径以及空间约束下的异质迁移模式。
