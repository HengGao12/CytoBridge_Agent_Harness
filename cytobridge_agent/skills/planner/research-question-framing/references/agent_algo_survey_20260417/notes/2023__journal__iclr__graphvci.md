# Predicting Cellular Responses with Variational Causal Inference and Refined Relational Information

## Metadata
- 标题: Predicting Cellular Responses with Variational Causal Inference and Refined Relational Information
- 作者: Yulan Gao et al.
- 年份: 2023
- 正式 venue: ICLR 2023
- PDF 文件名: Predicting Cellular Responses with Variational Causal Inference and Refined Relational Information.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 是，PDF 明确标注为 ICLR 2023 conference paper
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
这篇文章面向单细胞 perturbation response prediction，目标是在没有真实观测的情况下预测某个细胞在 counterfactual perturbation 下的转录组响应。它服务的核心生物问题是个体化细胞响应预测与药物/基因干预评估。

## 数据类型 / 场景
输入是带条件、协变量和 perturbation 标签的 scRNA-seq 干预数据，可结合 ATAC 派生的 GRN 先验。作者在 SciPlex、Marson 等 benchmark 和自建 CROP-seq 数据上验证。

## 核心算法怎么设计
graphVCI把问题写成 variational Bayesian causal inference。模型用 encoder-decoder 学习 factual 与 counterfactual outcome 的共享潜变量，再用图卷积/图注意力把 GRN 关系注入编码和解码过程；同时引入 adjacency-matrix updating 在预训练中修正初始 GRN，并设计稳健估计器去估计群体层面的平均 perturbation effect。

## 设计原则是什么
第一原则是把 perturbation prediction 从纯条件生成提升为带因果含义的 counterfactual inference。第二原则是把 GRN 当作结构先验而不是静态特征，并允许模型边学边修正先验图。第三原则是同时关心单细胞级个体预测和群体平均效应估计。

## 工程优化 / 训练技巧 / pipeline 设计
作者用了 key-dependent attention 作为更稳定的解码聚合器，并在预训练阶段做 GRN refinement，这两点都直接改善了 OOD 预测表现。所有模型共享相近的网络宽度、深度和优化超参数，也让 ablation 更可信。

## 局限性
方法表现仍依赖初始 GRN 质量和训练数据覆盖的 perturbation 组合；对于严重分布外的细胞状态或 perturbation 组合，因果假设未必成立。它也主要建模 perturbation 前后分布变化，不直接提供连续时间轨迹。

## 为什么能发到这个级别
这篇工作补的是单细胞干预预测里“结构先验 + 因果表达 + 稳健群体效应估计”这一块空白。它既有方法创新，也有清晰的 OOD 评测和新的数据资源，所以符合当时 ICLR 对方法完整性的要求。

## 对 CytoBridge-agent 自动设计算法的启发
对 CytoBridge 的 counterfactual 模块，一个重要方向是把 causal structure 直接融入生成模型，而不是只做条件生成。另一个启发是：当 agent 设计 perturbation 算法时，最好同时输出 cell-level response 和 population-level effect，避免只在一个尺度上优化。
