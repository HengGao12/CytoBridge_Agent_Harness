# SLICE: determining cell differentiation and lineage based on single cell entropy

## Metadata
- 标题: SLICE: determining cell differentiation and lineage based on single cell entropy
- 作者: Minzhe Guo, Erik L. Bao, Michael Wagner, Jeffrey A. Whitsett, Yan Xu
- 年份: 2017
- 正式 venue: Nucleic Acids Research
- PDF 文件名: SLICE determining cell differentiation and lineage based on single cell entropy.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已读摘要、引言、核心结果、讨论

## 解决了什么生物问题

作者要解决的是：在没有时间标签、起点终点标注或经典 marker 先验时，如何仅从 scRNA-seq 推断单细胞的分化程度和分支谱系。生物学上它重点验证了肺泡 II 型细胞、骨骼肌成肌细胞、人早期胚胎和胚胎肺间充质等体系，并在小鼠胚胎肺中提出了新的成纤维细胞分支假说。

## 数据类型 / 场景

- 单细胞 RNA-seq，主要是发育/分化过程中的横截面快照数据。
- 既包含相对均一的单分支过程，也包含异质性更强的多分支体系。
- 典型使用场景是“只有表达快照，没有可靠时间起点”的 lineage inference。

## 核心算法怎么设计

- 先定义单细胞熵 `scEntropy`。作者不是直接对原始表达做熵，而是先把基因按 GO 功能分组，再在高表达基因的 bootstrap 子集上估计功能活性分布的熵；高熵被解释为更未分化、更高可塑性。
- 再在 PCA 空间里构建 cell-cell network，并基于网络或聚类识别稳定状态/簇。
- 谱系方向由熵来定：从高熵状态指向低熵状态，不再依赖外部给定 root。
- 轨迹重建提供了两套实现：最短路径式与 principal-curve 式，用来在分支图上给出细胞过渡路径。

## 设计原则是什么

- 用“内生方向信号”替代外部先验。它认为分化方向应该来自细胞状态本身，而不是实验者手动指定起点。
- 把“状态评分”和“路径重建”拆开。先求每个细胞的 potency/order score，再在图上做拓扑推断。
- 用全局转录组无序度近似分化潜能，而不是只依赖少数 marker。
- 面向 branch-aware 场景设计，目标不是只输出一条线性 pseudotime。

## 工程优化 / 训练技巧 / pipeline 设计

- 对低测序质量细胞先做 outlier 剔除，减少熵估计被文库深度拖偏。
- 在不同数据集上允许用 marker gene 或高变基因做 PCA/建图，说明它是一个可插拔 pipeline，而不是端到端黑盒。
- 对 scEntropy 使用 bootstrap 估计，提高了对基因选择扰动的稳定性。
- 状态识别和路径重建都提供备选实现，作者强调“共识”比单一实现更重要。

## 局限性

- 熵定义依赖 GO 功能分组和若干超参数，生物注释质量会直接影响结果。
- 方向性虽然摆脱了时间标签，但其核心假设是“高熵即未分化”，对 reprogramming、应激或循环过程未必总成立。
- 轨迹仍然依赖上游 PCA、建图和聚类质量，没有显式的动态生成模型，也没有增长/死亡项。
- 论文主要在较干净的发展轨迹数据上验证，未充分展示在强批次效应或复杂条件扰动中的鲁棒性。

## 为什么能发到这个级别

这篇文章抓住了 2016-2017 年单细胞轨迹推断的一个核心痛点：很多方法会给出一条“看起来像轨迹”的结构，但方向要靠人指定。SLICE 用 entropy 给出了一个相对可解释、与外部先验弱耦合的方向信号，并且在多套真实数据上做了成功验证，还提出了新的肺间充质分支假说，因此在当时具有明显的方法新意和生物应用价值。

## 对 CytoBridge-agent 自动设计算法的启发

- 在自动设计算法时，可以把“root/state ordering prior”当成一个独立模块，而不是让主模型同时解决所有问题。
- 单个标量 order parameter 虽不足以替代完整动力学，但很适合作为 trajectory/OT/velocity 模型的方向先验或 sanity check。
- 一个可靠 agent 不该只输出单一路径算法；应允许“状态评分模块 + 拓扑模块 + 连续动力学模块”的组合式设计。
