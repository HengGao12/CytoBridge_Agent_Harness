# stVCR: Reconstructing spatio-temporal dynamics of cell development using optimal transport

## Metadata
- 标题: stVCR: Reconstructing spatio-temporal dynamics of cell development using optimal transport
- 作者: Qiangwei Peng et al.
- 年份: 2024
- 正式 venue: bioRxiv preprint
- PDF 文件名: stVCR Reconstructing spatio-temporal dynamics of cell development using optimal transport.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否，本地 PDF 仍为 bioRxiv 版本
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
stVCR针对 time-series spatial transcriptomics 的一体化重建问题：怎样同时恢复分化、增长、空间迁移，并把不同时间切片的空间坐标对齐到同一坐标系。它直接面向“发育中的细胞不只改表达，也在物理空间中运动”这一现实问题。

## 数据类型 / 场景
输入是单细胞分辨率的 time-series spatial transcriptomics。作者重点展示了在再生/发育场景中对空间迁移与表达变化联合建模的能力。

## 核心算法怎么设计
stVCR是具有 dynamical form、unbalanced setting 且对 rigid-body transformation 不变的 OT 模型。它把基因表达变化、空间迁移、细胞分裂与凋亡统一进同一个时空 OT 框架，并在拟合过程中同时完成多时间点空间配准。

## 设计原则是什么
设计原则很明确：空间迁移不能被当作轨迹分析之后的附加解释，而应和分化、growth 同时进入主模型。另一个原则是配准与 dynamics 联合求解，因为不同时间点坐标系不一致时，先配准再推轨迹会积累错误。

## 工程优化 / 训练技巧 / pipeline 设计
作者把 rigid-body-invariant OT 作为空间部分的核心，使模型可以在不知道统一坐标系的情况下直接学习跨时间点迁移。方法还允许研究表达与空间迁移如何共同影响 growth，这比只输出一个 transport map 更丰富。

## 局限性
讨论里作者明确指出，目前仍需先把高维 gene expression 降到低维再学习 dynamics，这会限制基因层面的精确解释。作为 preprint，文章在更大规模真实数据上的稳定性与计算代价还需要更多公开验证。

## 为什么能发到这个级别
虽然还是 preprint，但这篇工作的问题定义非常强，把空间、时间、增长和配准四件通常分开的事压进了一个 OT 目标里。对时空单细胞建模来说，这是很有潜力的统一框架。

## 对 CytoBridge-agent 自动设计算法的启发
这篇文章对 CytoBridge 的直接启发是：时空数据应优先考虑“dynamics + alignment”联合建模，而不是串行 pipeline。未来若做空间扩展，rigid-body-invariant 或更一般的 geometric-invariant transport 会很关键。
