# Single-Cell Entropy to Quantify the Cellular Order Parameter from Single-Cell RNA-Seq Data

## Metadata
- 标题: Single-Cell Entropy to Quantify the Cellular Order Parameter from Single-Cell RNA-Seq Data
- 作者: Jingxin Liu, You Song, Jinzhi Lei
- 年份: 2020
- 正式 venue: Biophysical Reviews and Letters
- PDF 文件名: Single-Cell Entropy to Quantify the Cellular Order Parameter from Single-Cell RNA-Seq Data.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 是
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已读摘要、方法概要、应用结果

## 解决了什么生物问题

这篇文章试图给单细胞转录组定义一个简单的“细胞有序度/干性”标量，用来量化发育进程、癌症进展和细胞分类。它不是完整轨迹方法，但属于为 fate/differentiation 建模提供低维状态变量的工作。

## 数据类型 / 场景

- 人类着床前胚胎、hESC 分化、HNSCC、黑色素瘤、CML 干细胞等 scRNA-seq 数据。
- 适用于需要一个极简、易解释、低参数的状态量，而不是复杂图模型的场景。

## 核心算法怎么设计

- 定义 `scEntropy`：以某个参考细胞或参考表达向量为基准，计算目标细胞与参考之间表达差分的香农熵。
- 熵越大，被解释为转录组越无序、越偏离参考有序状态。
- 在此基础上，作者进一步用 Gaussian mixture model 对 scEntropy 分布做分解，形成 scEGMM，用于无监督分类。
- 文中展示了它与 OCLR stemness、t-SNE、CNV-based malignant label 等之间的对应关系。

## 设计原则是什么

- 尽可能把复杂单细胞状态压缩成单一可解释标量。
- 把“参考状态选择”放在模型最中心，而不是通过复杂网络先验定义 potency。
- 追求低参数、低复杂度和可部署性，而不是追求对复杂分支结构的一次性完整求解。

## 工程优化 / 训练技巧 / pipeline 设计

- 与参考细胞的选择相配合，可以非常灵活地切换问题定义，例如正常对肿瘤、早期对晚期。
- 文章比较了不同基因集合策略，展示其在若干场景下的分类可用性。
- 实现简单，几乎可以当成上游诊断模块或下游解释变量插入其他流程。

## 局限性

- 结果高度依赖 reference cell/reference state 的选择，这是最大脆弱点。
- 与 2017 年 SCENT 相比，它缺少网络层先验，生物解释更轻，系统层面的约束也更弱。
- 更适合作为状态监测/粗分类工具，不足以单独恢复分支拓扑、时间方向和 fate distribution。
- 论文的创新性和验证广度都明显弱于领域代表作。

## 为什么能发到这个级别

它能发表的原因更像是“一个简单但有一定可用性的工具化方法”。核心思想足够直接，应用场景清楚，也有几个数据集上的例子，但从影响力和方法深度看，它更像增量式方法而不是范式工作。

## 对 CytoBridge-agent 自动设计算法的启发

- 简单标量 order parameter 仍然有价值，尤其适合做 root prior、异常检测和模型输出 sanity check。
- 但 agent 不应把这种一维熵量误当成完整动力学；它更适合作为组合模型的一部分，而不是主模型。
- 若要自动设计类似模块，关键不是熵本身，而是如何自动学习更稳健的 reference state。
