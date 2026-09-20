# Inference of cell state transitions and cell fate plasticity from single-cell with MARGARET

## Metadata
- 标题: Inference of cell state transitions and cell fate plasticity from single-cell with MARGARET
- 作者: Muhammad Haris Khan et al.
- 年份: 2022
- 正式 venue: Nucleic Acids Research
- PDF 文件名: Inference of cell state transitions and cell fate plasticity from single-cell with MARGARET.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
MARGARET关注的是复杂拓扑下的单细胞轨迹和 fate plasticity 推断。它试图解决传统 TI 方法在多分支、环状或复杂几何中拓扑恢复不稳、terminal state 自动识别差、plasticity 难量化的问题。

## 数据类型 / 场景
输入是 scRNA-seq snapshot 数据，重点用于造血、embryoid body differentiation 等存在复杂分支与过渡群体的系统。作者同时做了合成数据和真实数据基准。

## 核心算法怎么设计
MARGARET先做深度无监督 metric learning，学习更适合轨迹恢复的细胞表示；之后在图上利用新的 connectivity measure 和 graph partitioning 重建复杂拓扑，并自动识别 terminal states。进一步地，它把 fate plasticity 从简单树结构推广到更一般的复杂轨迹结构。

## 设计原则是什么
这篇文章的原则是“先学一个对轨迹友好的几何，再在图上做拓扑推断”，而不是直接在原始 embedding 上跑图算法。另一个原则是把 topology recovery、terminal state detection 和 plasticity quantification 视为一个连贯问题，而不是分散的后处理步骤。

## 工程优化 / 训练技巧 / pipeline 设计
作者把深度表示学习和图分割结合起来，使方法能在复杂拓扑上保持较稳健的全局结构恢复。自动 terminal-state 检测和统一 benchmark 指标设计也让方法更容易做系统比较。

## 局限性
MARGARET本质上仍然是基于图和表示学习的静态推断，没有显式学习物理时间下的连续动力学，也不直接建模 birth/death 或 stochastic transport。复杂拓扑下的结果仍然依赖 learned metric 和图构建质量。

## 为什么能发到这个级别
它抓住了一个很实际的痛点：很多 TI 方法在简单树结构上有效，但到了复杂拓扑就明显失真。MARGARET在 topology、terminal state 和 plasticity 三个维度同时提升，而且 benchmark 做得较全面，因此能够在 NAR 这样的工具方法期刊站住脚。

## 对 CytoBridge-agent 自动设计算法的启发
对 agent 来说，复杂拓扑恢复仍是必须单独优化的模块，不能假设所有系统都近似树结构。即便采用生成式动力学，仍值得把 topology robustness 和 plasticity quantification 作为独立评测项。
