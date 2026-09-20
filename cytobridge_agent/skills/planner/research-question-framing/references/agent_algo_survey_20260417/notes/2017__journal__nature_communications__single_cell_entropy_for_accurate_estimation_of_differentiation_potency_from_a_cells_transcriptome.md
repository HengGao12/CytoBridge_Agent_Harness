# Single-cell entropy for accurate estimation of differentiation potency from a cell’s transcriptome

## Metadata
- 标题: Single-cell entropy for accurate estimation of differentiation potency from a cell’s transcriptome
- 作者: Andrew E. Teschendorff, Tariq Enver
- 年份: 2017
- 正式 venue: Nature Communications
- PDF 文件名: Single-cell entropy for accurate estimation of differentiation potency from a cell’s transcriptome.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已读摘要、引言、核心结果、讨论

## 解决了什么生物问题

它解决的是如何从单细胞转录组中定量估计“分化潜能/可塑性”，并进一步识别干性亚群、药物耐受癌症干细胞和群体内受调控的异质性。它不直接重建长程轨迹，但为 fate/trajectory 分析提供了一个强有力的状态变量。

## 数据类型 / 场景

- 单细胞 RNA-seq，覆盖 hESC、多谱系祖细胞、黑色素瘤微环境、AML、循环肿瘤细胞等。
- 也讨论了 bulk 样本上的扩展使用。
- 适用于需要估计 potency/plasticity，而不希望先做 feature selection 或监督训练的场景。

## 核心算法怎么设计

- 核心量是 signalling entropy rate。作者把单细胞表达谱投到高质量 PPI 网络上，构造一个细胞特异的随机游走过程。
- 基本假设是：如果两个蛋白可互作，且对应基因都高表达，那么这条信号边更可能被“激活”。
- 在这个随机游走上计算全局 entropy rate，作为单细胞 signalling promiscuity 的量化。高熵被解释为高潜能/高可塑性。
- 在此基础上作者进一步提出 SCENT 框架：给细胞分 potency state，识别群体内亚群，再从这些状态和共表达结构中抽出 lineage landmarks。

## 设计原则是什么

- 不从少数 marker 出发，而是从系统层面的网络可达性和信号不确定性出发。
- potency 被定义为“尚未偏向某条确定谱系时的信号多能性”，这比单纯的表达散度更接近 Waddington 图景。
- 不把 single-cell potency 和 population heterogeneity 混为一谈。作者在讨论中反复强调这两个 entropy 不是一回事。
- 先做状态量化，再做群体级组织，是一个典型的“先局部打分、再全局组织”设计。

## 工程优化 / 训练技巧 / pipeline 设计

- 不需要 feature selection 或训练集，这是实际工程上很强的可迁移性优势。
- 通过统一的 PPI 网络和随机游走定义，让不同数据集上的度量具有相对一致的解释。
- SCENT 把 potency state、landmark identification 和 lineage ordering 串成了可复用 pipeline，而不只是给一个分数。

## 局限性

- 对外部网络先验依赖较强，PPI 网络的不完整性和偏置会传导到结果。
- 输出的主要是 potency/plasticity proxy，不是严格的时间动力学，也不显式建模细胞增长、死亡或质量非守恒。
- 如果真实过程受强条件特异网络重写控制，而通用 PPI 不能表达这种差异，分数会失真。
- 对复杂分支、循环和反事实干预，它更像辅助变量而不是完整求解器。

## 为什么能发到这个级别

它把“分化潜能”这个长期停留在概念层面的量，变成了一个可在大规模 scRNA-seq 上直接计算的系统生物学量，并且跨正常发育和癌症体系做了大量验证。更重要的是，它不靠训练、不靠 marker 列表，而是利用网络结构把单细胞分析从纯表达几何推进到 network-aware state estimation，这一点在 2017 年非常新。

## 对 CytoBridge-agent 自动设计算法的启发

- agent 在设计算法时，不应只搜索“如何连线成轨迹”，还应主动搜索“是否存在可解释的状态势函数/分化势评分”。
- 网络先验是强约束源。对于单细胞动力学问题，合适的互作网络或 GRN 先验能显著减少不可辨识性。
- 在自动化算法设计里，potency score 可作为 root 初始化、trajectory 方向约束、counterfactual 可行性检查的辅助模块。
