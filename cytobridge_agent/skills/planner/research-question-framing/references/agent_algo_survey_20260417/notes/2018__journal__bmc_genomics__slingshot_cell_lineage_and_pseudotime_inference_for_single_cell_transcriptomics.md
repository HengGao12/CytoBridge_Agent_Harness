# Slingshot: cell lineage and pseudotime inference for single-cell transcriptomics

## Metadata
- 标题: Slingshot: cell lineage and pseudotime inference for single-cell transcriptomics
- 作者: Kelly Street, Davide Risso, Russell B. Fletcher, Diya Das, John Ngai, Nir Yosef, Elizabeth Purdom, Sandrine Dudoit
- 年份: 2018
- 正式 venue: BMC Genomics
- PDF 文件名: Slingshot cell lineage and pseudotime inference for single-cell transcriptomics.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已读摘要、引言、结论、方法概要

## 解决了什么生物问题

它解决的是：在噪声很高、分支可能不止一条的单细胞表达数据里，如何稳定地恢复 lineage 结构和 pseudotime。文章重点不是新生物发现，而是为后续动态基因分析提供更鲁棒的轨迹骨架。

## 数据类型 / 场景

- 单细胞转录组降维后的连续细胞云。
- 既支持单分支，也支持双分支和三分支结构。
- 适合那些需要在上游已经完成归一化、降维和聚类后，再做 lineage/pseudotime 推断的场景。

## 核心算法怎么设计

- 第一步先做 lineage inference：在 cluster 层面构建 minimum spanning tree，而不是直接在细胞层面连图。
- 第二步再做 pseudotime inference：提出 simultaneous principal curves，在多个分支之间共享前段公共路径，再分别向各终点延展。
- 它允许指定起始 cluster 和部分终末 cluster，提供有限监督；但若不给，也可无监督运行。
- 通过 cluster-level MST 处理全局拓扑，通过 principal curves 处理局部连续几何，把“拓扑发现”和“细胞排序”分成两步。

## 设计原则是什么

- 先抓稳全局结构，再做精细排序。作者明确反对把所有问题都丢给一个端到端曲线拟合器。
- 在 cluster 层面引入适度监督，是噪声数据里的务实折中。
- 分支 pseudotime 不应在 branching point 处不连续，所以 simultaneous curves 必须共享公共主干。
- 方法要能嵌入任意上游流程，而不是绑定专用预处理。

## 工程优化 / 训练技巧 / pipeline 设计

- 与具体归一化、降维和聚类方法解耦，可接在常见 scRNA-seq pipeline 后面。
- 在模拟数据中系统评估不同聚类方法和 cluster 数，对实际用户很有参考价值。
- 计算量相对可控，作者明确指出它足以放入 bootstrap 过程估计不确定性，尽管论文本身默认只给 point estimate。

## 局限性

- 高度依赖上游降维和聚类质量；cluster 切坏了，后续 topology 也会跟着坏。
- 默认只输出点估计，没有内建结构不确定性和时间不确定性的量化。
- 不利用真实时间、RNA kinetics、增长/死亡或 transport 约束，所以仍是几何型 pseudotime 方法。
- 对 disconnected components、强非树结构或循环过程不是其强项。

## 为什么能发到这个级别

它抓住了当时 trajectory inference 的一个实际痛点：很多方法要么不稳，要么分支一复杂就失效。Slingshot 用非常工程化但又清晰的模块化设计，把 branch-aware lineage + pseudotime 做得稳定、可解释、易嵌入 pipeline，因此在社区里很快变成常用基线。

## 对 CytoBridge-agent 自动设计算法的启发

- 自动设计算法时，模块化优先于一体化黑盒：全局拓扑、局部排序、方向信息可以由不同模块分别解决。
- 弱监督是很有价值的设计空间。给 root/terminal state 的轻量先验，往往比完全无监督更稳。
- 对 agent 来说，Slingshot 适合作为“拓扑基线”或“后处理骨架”，再叠加 OT、velocity 或生成式动力学模块。
