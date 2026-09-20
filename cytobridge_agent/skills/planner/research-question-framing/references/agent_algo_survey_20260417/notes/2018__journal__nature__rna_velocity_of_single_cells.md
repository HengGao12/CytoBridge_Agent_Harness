# RNA velocity of single cells

## Metadata
- 标题: RNA velocity of single cells
- 作者: Gioele La Manno, Ruslan Soldatov, Amit Zeisel, Emelie Braun, Hannah Hochgerner, Viktor Petukhov, Katja Lidschreiber, Maria E. Kastriti, Peter Lönnerberg, Alessandro Furlan, Jean Fan, Lars E. Borm, Zehua Liu, David van Bruggen, Jimin Guo, Xiaoling He, Roger Barker, Erik Sundström, Gonçalo Castelo-Branco, Patrick Cramer, Igor Adameyko, Sten Linnarsson, Peter V. Kharchenko
- 年份: 2018
- 正式 venue: Nature
- PDF 文件名: RNA velocity of single cells.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已读摘要、引言、核心方法与结果

## 解决了什么生物问题

这篇文章解决的是单细胞快照数据缺少方向信息的问题。它试图回答：一个细胞当前的表达状态已知时，短时间后它最可能往哪里走。生物上作者用神经嵴、海马发育和人胚脑等体系证明它能恢复 lineage direction 和 fate tendency。

## 数据类型 / 场景

- 常见 scRNA-seq 协议产生的 spliced/unspliced reads。
- 适合发育、再生等小时到天尺度的动态过程。
- 重点是 snapshot data 的局部未来状态预测，而不是长程全局生成模型。

## 核心算法怎么设计

- 先区分 unspliced 和 spliced mRNA 计数。
- 用简单的一阶转录动力学：unspliced 产生 spliced，spliced 再降解。
- 对每个基因估计 steady-state slope `γ`，比较观测到的 `u` 与期望的 `γs` 偏差。
- 若 unspliced 相对过量，说明该基因正在上调；若不足，说明正在下调。把所有基因的这种局部导数合并后，就得到细胞状态的高维速度向量。
- 再把 velocity 投影到 PCA/t-SNE 等低维空间中，可视化未来状态，并用 Markov random walk 给整个 lineage 定向。

## 设计原则是什么

- 方向性应来自实验测量中的动力学残差，而不是只来自流形几何。
- 先做局部短时预测，再用局部向量场解释全局命运方向。
- 尽量利用现有标准 scRNA-seq 协议，不要求额外代谢标记实验。
- 用机制模型打破 snapshot inference 的不可辨识性，这是全文最重要的原则。

## 工程优化 / 训练技巧 / pipeline 设计

- 论文强调不同平台上 intronic reads 都普遍存在，所以方法部署门槛低。
- 通过邻域平滑和向量场可视化，把高维速度映射成可读的低维方向图。
- 还能与 pseudotime 结合，用速度和切线的一致性验证方向是否合理。

## 局限性

- 核心假设是 steady-state 模型和相对统一的剪接动力学；在 transient state 或不同亚群 kinetics 很不一致时会出错。
- 速度只对应短时未来，不能直接替代长程 fate distribution。
- 对 unspliced 计数质量、测序协议和邻域平滑都有依赖。
- 没有显式处理增长/死亡、跨条件扰动或分布级 transport。

## 为什么能发到这个级别

它第一次从标准 scRNA-seq 数据中提炼出“方向”这一关键信息，而且实现方式足够简单、可复现、跨平台。这个贡献直接改变了领域对快照单细胞数据可推断信息边界的认识，因此是典型的范式推动型工作。

## 对 CytoBridge-agent 自动设计算法的启发

- 如果观测中存在能反映局部导数的额外信号，agent 应优先利用它来约束动力学，而不是只做几何插值。
- 局部 velocity 很适合与全局 OT/flow 模型结合：前者提供短时切向约束，后者提供长程分布演化。
- 自动设计算法时，应该显式检查核心假设是否成立；RNA velocity 的后续 scVelo 就是这种“假设失效后升级模型”的典型例子。
