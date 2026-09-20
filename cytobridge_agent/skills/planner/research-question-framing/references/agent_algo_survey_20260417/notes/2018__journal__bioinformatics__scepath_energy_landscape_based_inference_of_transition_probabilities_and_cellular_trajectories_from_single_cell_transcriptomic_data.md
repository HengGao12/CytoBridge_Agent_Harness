# scEpath: energy landscape-based inference of transition probabilities and cellular trajectories from single-cell transcriptomic data

## Metadata
- 标题: scEpath: energy landscape-based inference of transition probabilities and cellular trajectories from single-cell transcriptomic data
- 作者: Suoqin Jin, Adam L. MacLean, Tao Peng, Qing Nie
- 年份: 2018
- 正式 venue: Bioinformatics
- PDF 文件名: scEpath energy landscape-based inference of transition probabilities and cellular trajectories from single-cell transcriptomic data.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已读摘要、引言、方法、结果与讨论

## 解决了什么生物问题

作者想解决的问题是：如何把 Waddington landscape 从隐喻变成可计算对象，从而在单细胞数据中推断细胞状态转移概率、谱系关系和 pseudotime。生物应用包括人类早期胚胎、肺上皮发育和成肌分化。

## 数据类型 / 场景

- 单细胞 RNA-seq 快照数据。
- 既考虑线性发育，也考虑分支分化。
- 适用于希望同时得到 landscape、state transition probability、trajectory 和 marker/TF 动态的场景。

## 核心算法怎么设计

- 首先基于基因表达相关性构建 gene-gene interaction network。
- 然后定义 `scEnergy`：在基因网络邻域上，用统计物理/最大熵式的局部表达组织程度来给每个细胞赋能量。
- 用 PCA 和能量距离把细胞嵌入到定量 energy landscape 中。
- 再用 SIMLR 做结构聚类，把高概率区域压成 metacell。
- metacell 间转移概率由两部分组合而成：一部分来自 Boltzmann-Gibbs 概率，一部分来自距离型转移矩阵；之后再按能量下降方向给边定向。
- 最后通过 maximum probability flow 抽出 lineage tree，并用 principal curve 给出 pseudotime。

## 设计原则是什么

- 用“势景 + 转移概率”替代纯几何 pseudotime。
- 方向信息来自能量下降，而不是外部给定时间或 marker。
- 不希望过度依赖 feature selection，因此刻意测试对输入基因数的稳健性。
- 在 cell-level 噪声较大时，把信息先聚合到 metacell，再做概率推断。

## 工程优化 / 训练技巧 / pipeline 设计

- metacell 采用 Tukey’s trimean 聚合，减少极端值影响。
- 用 SIMLR 学 cell-cell similarity，提高对复杂结构的聚类质量。
- 不只输出轨迹，还能输出 pseudotime-dependent genes、TF 网络和 cell-cell communication 分析，形成完整下游 pipeline。
- 论文专门测试了输入基因集合变化下的鲁棒性，这是非常实用的工程考量。

## 局限性

- `scEnergy` 的物理解释很吸引人，但依赖于相关网络、局部邻域和能量单调下降假设；对非单调过程或 reprogramming 场景未必成立。
- 作者在讨论里明确承认，对污染细胞、多初始态和更复杂系统还不够好。
- 依赖聚类与降维，误差会层层传递。
- 仍然不是显式的连续时间生成模型，也没有质量非守恒或随机扩散的严格建模。

## 为什么能发到这个级别

它的价值不在于单点指标领先多少，而在于把一个长期广泛使用的生物学比喻定量化，并把 landscape、transition probability 和 trajectory 三件事连成了一个完整方法链。对当时大量仍停留在“画一条 pseudotime 曲线”的方法来说，这是一个更有理论包装也更有解释性的替代方案。

## 对 CytoBridge-agent 自动设计算法的启发

- landscape 型标量势函数在自动算法设计中很有价值，尤其适合作为 OT/flow/velocity 模型的辅助先验或 model selection 维度。
- 在高噪单细胞问题里，先压成 metacell 再建模，往往是更稳定的工程策略。
- agent 未来若要自动组合算法，可以考虑“势景模块 + 概率图模块 + 连续时间模块”的分层架构，而不是只押一个范式。
