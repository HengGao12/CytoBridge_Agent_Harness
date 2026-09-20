# DeepVelo: deep learning extends RNA velocity to multi-lineage systems with cell-specific kinetics

## Metadata
- 标题: DeepVelo: deep learning extends RNA velocity to multi-lineage systems with cell-specific kinetics
- 作者: Haotian Cui et al.
- 年份: 2024
- 正式 venue: Genome Biology
- PDF 文件名: DeepVelo deep learning extends RNA velocity to multi-lineage systems with cell-specific kinetics.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
DeepVelo要解决的是 RNA velocity 在多 lineage、时间变速率系统里经常失败的问题。它希望在复杂异质系统中恢复更可信的 cell-specific kinetics、developmental stage 和调控 driver genes。

## 数据类型 / 场景
输入是带 unspliced/spliced 计数的 scRNA-seq，重点适用于多分支分化和病理过程。作者在多个发育和疾病数据上评估了 velocity 与 driver gene 结果。

## 核心算法怎么设计
DeepVelo利用 graph convolution network 学习细胞依赖的转录、剪接和降解速率，而不是采用全局常数。它借助邻近细胞和时间连续性约束，在图上学习 velocity field，并据此恢复 pseudotime、future state 和与分化相关的 driver genes。

## 设计原则是什么
设计原则与 cellDancer 类似，都是放弃全局统一 kinetics，但 DeepVelo更强调用图卷积统一建模多细胞关系。它把“相近细胞应该共享某种局部动力学结构”作为主要归纳偏置。

## 工程优化 / 训练技巧 / pipeline 设计
GCN 设计使 DeepVelo 能在整个细胞图上共享参数，而不必像 per-gene/per-cell 独立拟合那样代价高昂。它还能联合输出时间、速度和 driver genes，形成较完整的 velocity analysis pipeline。

## 局限性
它仍然依赖 RNA velocity 框架本身的观测前提，且图构建和邻域选择对结果影响较大。方法更擅长局部 velocity 学习，对跨条件分布预测和连续生成建模支持不足。

## 为什么能发到这个级别
DeepVelo抓住了 velocity 领域的关键失败模式，并用深图网络给出了一个可扩展替代方案。它兼顾了准确性、driver gene 识别和在复杂系统中的适用性，因此在 Genome Biology 这样的方法期刊里很有竞争力。

## 对 CytoBridge-agent 自动设计算法的启发
对 CytoBridge 来说，局部图结构是学习 cell-dependent kinetics 的天然载体。未来若要做 velocity-aware dynamics，可以把图神经网络当作估计局部 drift/growth 的模块，而把全局 transport 留给更适合的生成框架。
