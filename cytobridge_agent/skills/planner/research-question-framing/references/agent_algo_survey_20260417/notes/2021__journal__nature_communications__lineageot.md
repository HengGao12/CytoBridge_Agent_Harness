# LineageOT is a unified framework for lineage tracing and trajectory inference

## Metadata
- 标题: LineageOT is a unified framework for lineage tracing and trajectory inference
- 作者: Aden Forrow and Geoffrey Schiebinger
- 年份: 2021
- 正式 venue: Nature Communications
- PDF 文件名: LineageOT is a unified framework for lineage tracing and trajectory inference.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
LineageOT解决的是：当单细胞测序和 lineage tracing 同时可得时，怎样把“状态相似性”和“谱系亲缘性”放进同一个轨迹推断框架，尤其是在复杂分叉结构里更准确地恢复祖先-后代关系和 fate coupling。

## 数据类型 / 场景
面向 time-course scRNA-seq 加 lineage tree/clone 信息的场景，既能处理模拟数据，也能处理 C. elegans 之类谱系已知的系统。核心场景是 destructive sampling 下跨时间点对齐祖先和后代细胞。

## 核心算法怎么设计
算法把细胞在 Waddington landscape 上的演化写成 drift + diffusion 的随机过程。它先利用图模型和 lineage tree 估计各后代细胞在上一时间点的祖先状态分布，再把这个“校正后的祖先分布”与上一时间点细胞通过 entropic OT 连接，得到结合了 lineage 与 expression 的 coupling。

## 设计原则是什么
最关键的原则是：lineage 不是后处理注释，而应该直接改变 coupling 的先验。另一原则是把 lineage tree 当成不确定但有信息量的结构，通过概率校正而不是硬匹配进入 OT；这样既保留可解释性，也提升对复杂分叉的辨识能力。

## 工程优化 / 训练技巧 / pipeline 设计
LineageOT把问题拆成“lineage-aware ancestor estimation + OT matching”两步，避免一次性联合求解全部结构。作者还专门测试了对估计误差较大的 lineage tree 的鲁棒性，说明这个 pipeline 适合现实实验噪声环境。

## 局限性
方法默认 lineage tree 已经可用，树的估计误差、缺失和 subsampling 偏差会传导到 coupling。作者也承认更理想的做法是把树推断与轨迹推断联合起来；此外，它主要解决离散时间点之间的 coupling，不直接学习全局连续动力学。

## 为什么能发到这个级别
这篇文章的重要性在于它首次把 lineage tracing 与 trajectory inference 从概念相关推进到统一数学框架，并且清楚说明了为何 lineage 信息能弥补纯状态方法的不可辨识性。对当时迅速增长的 scLT 数据而言，这是非常及时且有方法学深度的工作。

## 对 CytoBridge-agent 自动设计算法的启发
CytoBridge 以后若接入 lineage tracing 或 clonal data，不应把它只当额外特征，而应直接进入 coupling 或 latent prior。更一般地，agent 设计新算法时要优先寻找“能减少不可辨识性”的额外视角，而不是一味堆更复杂的 dynamics 网络。
