# Mapping cells through time and space with moscot

## Metadata
- 标题: Mapping cells through time and space with moscot
- 作者: Dominik Klein et al.
- 年份: 2025
- 正式 venue: Nature
- PDF 文件名: Mapping cells through time and space with moscot.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
moscot 解决的是 OT 在单细胞 genomics 中两大痛点：多模态信息难统一，以及 atlas 级数据无法扩展。作者不仅用它重建了 170 万细胞的小鼠胚胎时间轨迹，还展示了时空联合分析和胰腺内分泌 lineage 解析。

## 数据类型 / 场景
面向 multimodal、time-resolved、spatial transcriptomics 乃至 spatiotemporal 联合数据。适合 atlas 级别的大规模单细胞任务。

## 核心算法怎么设计
moscot把 OT 抽象成一个可组合的软件框架，支持 `time`、`space`、`spatiotemporal` 等不同任务模式。底层通过可扩展 OT 求解器和统一 problem API，把不同模态和不同应用都表示成 OT 问题，再在大规模数据上高效求解。

## 设计原则是什么
它最强的设计原则是框架化与模块化。作者不是只做一个新 OT 目标，而是把 OT 在单细胞中的主要应用统一到一个 scalable abstraction 中，让时间、空间、多模态成为同类对象。

## 工程优化 / 训练技巧 / pipeline 设计
可扩展性是 moscot 最突出的工程贡献，文章明确展示了百万级细胞的可行性。`moscot.spatiotemporal` 还把时间和空间联合起来做动力学分析，体现了框架对新任务的可扩展性。

## 局限性
作为一个强大的框架，moscot 的最终结果高度依赖用户如何定义 cost、coupling 约束和视角组合；框架本身不自动解决可辨识性问题。对缺少有效先验的系统，规模再大也不等于解释一定正确。

## 为什么能发到这个级别
它之所以能上 Nature，不只是算法本身，还因为它把 OT 从一类论文方法变成了可以支撑大规模单细胞图谱研究的基础设施。再加上对真实生物发现的支撑，影响力明显超出单一工具。

## 对 CytoBridge-agent 自动设计算法的启发
CytoBridge 后续若要做自动算法设计，应该学习 moscot 的“problem abstraction”思路。与其不断堆新 loss，不如先把 time/space/multimodal 统一成可组合对象，让 agent 在统一接口上自动选 OT/SB/FM 变体。
