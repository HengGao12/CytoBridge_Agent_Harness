# PAGA: graph abstraction reconciles clustering with trajectory inference through a topology preserving map of single cells

## Metadata
- 标题: PAGA: graph abstraction reconciles clustering with trajectory inference through a topology preserving map of single cells
- 作者: F. Alexander Wolf, Fiona K. Hamey, Mireya Plass, Jordi Solana, Joakim S. Dahlin, Berthold Göttgens, Nikolaus Rajewsky, Lukas Simon, Fabian J. Theis
- 年份: 2019
- 正式 venue: Genome Biology
- PDF 文件名: PAGA graph abstraction reconciles clustering with trajectory inference through a topology preserving map of single cells.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已读摘要、结果、结论

## 解决了什么生物问题

它要解决的是：单细胞数据常同时包含离散细胞类型和连续过渡过程，而“先聚类”和“直接做 trajectory”两条路线常常互相冲突。PAGA 试图先把全局拓扑搞清楚，再在这个拓扑约束下讨论局部连续变化。

## 数据类型 / 场景

- kNN 图表示的单细胞表达数据。
- 既支持 connected trajectory，也支持 disconnected components。
- 适合大规模、结构复杂、既有 cluster 又有 transition 的数据集。

## 核心算法怎么设计

- 从单细胞 kNN 图出发，先用 Louvain 等方法把细胞分成 partition。
- 再为 partition 间定义一个统计连通性分数，如果跨 partition 边数显著高于随机期望，就认为两者连通。
- 这样得到的 PAGA 图是一个 coarse-grained graph，节点是细胞群，边权是连通置信度。
- 之后可在高置信路径上，结合随机游走距离在每个 partition 内部给细胞排序。
- PAGA 图还可作为 UMAP/ForceAtlas 的初始化，显式保留全局拓扑。

## 设计原则是什么

- 先做拓扑抽象，再做细粒度解释。
- 不把单个细胞路径当真，而是对一群可能路径求平均，以获得统计稳定性。
- 多分辨率分析比单一分辨率更符合真实单细胞数据。
- trajectory inference 的第一要务不是拟合一条平滑曲线，而是不要把 disconnected structure 错连起来。

## 工程优化 / 训练技巧 / pipeline 设计

- 计算代价低，可作为大规模 embedding 的初始化步骤。
- 与 partition 方法解耦，cluster 可以来自 Louvain、人工标注或其他来源。
- 在百万级神经元数据上展示了可扩展性，这是很强的工程信号。
- 作为 Scanpy 生态核心组件，极大提升了实际可用性。

## 局限性

- 结果质量高度依赖 partition 质量和分辨率设置。
- 边权是“拓扑连通性”而非严格的时间转移率，不能直接等同于动力学 coupling。
- 虽能辅助 pseudotime，但本身不是生成式动力学模型。
- 对增长/死亡、随机扩散和跨条件反事实没有显式建模。

## 为什么能发到这个级别

因为它非常精准地解决了社区一个普遍痛点：很多 embedding 看起来漂亮，但 topology 错得很离谱。PAGA 把“保拓扑”提到首位，又给出了可扩展、可集成、可解释的实现，所以迅速成为单细胞分析的基础组件。

## 对 CytoBridge-agent 自动设计算法的启发

- 对 agent 来说，PAGA 提供了很强的“先粗后细”原则：先确认 topology，再决定是否上 OT、velocity 或 flow matching。
- 抽象图层是必要中间层。没有这个层，连续模型很容易在错误拓扑上做精细但无意义的拟合。
- 未来 CytoBridge 若要自动组合方法，PAGA 类 coarse topology 模块应成为默认候选。
