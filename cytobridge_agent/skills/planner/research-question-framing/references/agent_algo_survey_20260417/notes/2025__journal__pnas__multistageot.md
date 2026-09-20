# MultistageOT: Multistage optimal transport infers trajectories from a snapshot of single-cell data

## Metadata
- 标题: MultistageOT: Multistage optimal transport infers trajectories from a snapshot of single-cell data
- 作者: Erik Dahlin et al.
- 年份: 2025
- 正式 venue: Proceedings of the National Academy of Sciences of the United States of America
- PDF 文件名: MultistageOT Multistage optimal transport infers trajectories from a snapshot of single-cell data.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
MultistageOT要解决的是只有一个 single-cell snapshot 时，怎样在没有时间标签的情况下恢复分化进程，并识别不属于目标分化过程的 outlier cells。它把传统只适合双边耦合的 OT 扩展到单快照内部的多阶段推断。

## 数据类型 / 场景
输入是单时间点 scRNA-seq snapshot。文章主要以体内骨髓造血为例做 pseudotime、fate prediction 和 outlier detection。

## 核心算法怎么设计
MultistageOT把单个 snapshot 拆成多个隐藏的 differentiation stages，在这些 stages 之间做多边际、多阶段 OT，而不是只做 bimarginal transport。它还引入 auxiliary states 来桥接 disconnected cells，从而识别与目标分化过程无关的 outliers。

## 设计原则是什么
设计原则是为“单快照推断”显式加入隐藏时间维度。作者认为若继续沿用双边 OT，就无法表达中间阶段和方向性偏置，因此必须把 temporal progression 直接写成多阶段结构。

## 工程优化 / 训练技巧 / pipeline 设计
文章给出了 entropy-regularized multistage OT 的明确优化形式与 generalized Sinkhorn 风格求解思路，这是其主要工程价值。outlier detection 通过 auxiliary states 完成，也避免了单独再训练异常检测器。

## 局限性
单快照问题本身信息不足，因此结果对图结构、先验方向性和阶段数设定比较敏感。作者在讨论里也指出，低维表示本身有局限，若表达空间投影失真，推断的 temporal progression 也会被带偏。

## 为什么能发到这个级别
这篇工作的亮点在于把“single snapshot trajectory inference”从启发式排序推进到了更明确的多边际 OT 建模，并且兼顾了 outlier 识别这一现实需求。问题定义很硬，数学建模也足够完整。

## 对 CytoBridge-agent 自动设计算法的启发
对 agent 来说，单快照数据不应被简单判定为“不能做 dynamics”。更合理的方向是显式建模隐藏中间阶段，并把 outlier handling 作为主模型的一部分，而不是后处理。
