# Mapping lineage-traced cells across time points with moslin

## Metadata
- 标题: Mapping lineage-traced cells across time points with moslin
- 作者: Marius Lange et al.
- 年份: 2024
- 正式 venue: Genome Biology
- PDF 文件名: Mapping lineage-traced cells across time points with moslin.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
moslin要解决的是 destructive time-series lineage-tracing 实验里，大量 lineage 信息无法被现有方法充分利用的问题。作者希望用时间结构化的 lineage 与 gene expression 一起恢复 fate probability、decision driver genes 和祖先-后代映射。

## 数据类型 / 场景
输入是跨时间点的 scRNA-seq + lineage tracing 数据。文章用 C. elegans 胚胎发育和 zebrafish 心脏再生做真实案例。

## 核心算法怎么设计
moslin是基于 Gromov-Wasserstein 的模型，用 lineage similarity 和 gene-expression similarity 联合构造跨时间点 coupling。它在 Wasserstein 和 Gromov-Wasserstein 两种 regime 之间插值，以便在 lineage 噪声较大时仍能借助 expression 信息维持稳定映射。

## 设计原则是什么
设计原则是跨时间点整合全部 lineage 信息，而不是逐个时间点局部分析。作者也特别强调 lineage 噪声是现实常态，所以模型必须允许 lineage 与 expression 互相纠错，而非强制一方主导。

## 工程优化 / 训练技巧 / pipeline 设计
作者系统模拟了 barcode silencing rate，并用它来分析 moslin 在不同 lineage 可靠度下的行为，这种鲁棒性分析非常有说服力。模型还能输出 fate probabilities 和 putative driver genes，使其不是只做 coupling 的工具。

## 局限性
方法的优势建立在有时间结构化 lineage data 的前提上，普通 snapshot 数据无法直接受益。GW 型目标在大规模数据上仍有计算压力，而且 lineage 与 expression 的权重选择会影响 coupling 形态。

## 为什么能发到这个级别
它相比 LineageOT 的推进在于：不只处理单对时间点，而且更系统地处理了 destructive time-series 下多时间点 lineage information 的利用。对逐渐普及的 scLT 数据，这个问题非常实际。

## 对 CytoBridge-agent 自动设计算法的启发
对于带 lineage 的系统，agent 应优先设计“跨全部时间点联合求解”的模型，而不是相邻时间点贪心串联。另一个启发是，lineage-noise-aware interpolation 在现实数据里非常重要，不能默认 lineage tree 是真值。
