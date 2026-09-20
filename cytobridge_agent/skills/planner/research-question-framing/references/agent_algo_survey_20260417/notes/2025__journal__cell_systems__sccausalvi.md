# scCausalVI disentangles single-cell perturbation responses with causality-aware generative model

## Metadata
- 标题: scCausalVI disentangles single-cell perturbation responses with causality-aware generative model
- 作者: Shaokun An et al.
- 年份: 2025
- 正式 venue: Cell Systems
- PDF 文件名: scCausalVI disentangles single-cell perturbation responses with causality-aware generative model.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
scCausalVI解决的是 perturbation 数据里 basal state 与 treatment effect 混在一起的问题。它想回答：怎样把细胞固有状态、处理诱导效应和技术变异拆开，从而更可信地做跨条件单细胞 in silico perturbation。

## 数据类型 / 场景
输入是多条件、多来源的 perturbation scRNA-seq。作者用多组 benchmark 和 COVID-19 数据展示了跨条件预测与 treatment-responsive population 识别能力。

## 核心算法怎么设计
模型把 structural causal modeling 与深生成模型结合起来，显式分离 intrinsic cellular state 和 treatment effect。通过 deep structural causal network 建模 cell-state-specific response mechanism，再做 cross-condition in silico prediction，以此生成假设情景下的表达分布。

## 设计原则是什么
第一原则是 disentanglement 要服从因果语义，而不是只靠无监督因子分解。第二原则是 perturbation 响应是细胞状态依赖的，因此 response module 必须 condition on cell state，而不是只 condition on treatment label。

## 工程优化 / 训练技巧 / pipeline 设计
文章专门评估了不同硬件配置下的训练效率，并控制了与 baselines 一致的 preprocessing 和 latent dimension。它也把 data integration 和 perturbation prediction 放在同一模型里，减少了多模型串联的误差。

## 局限性
尽管引入因果结构，模型依然需要依靠观测数据来支撑识别，面对严重隐藏混杂或未覆盖条件时，因果解释仍有风险。它比传统条件生成更可解释，但并未提供完整的连续时间动力学。

## 为什么能发到这个级别
这篇文章的重要贡献是把因果意识真正写进了单细胞 perturbation 生成模型，而不是事后借用因果术语包装条件生成。对 virtual cell 和 in silico perturbation 方向，这是很自然的一步升级。

## 对 CytoBridge-agent 自动设计算法的启发
CytoBridge 在设计 perturbation 模块时，应优先把 basal state 与 treatment effect 解缠，否则很多“预测成功”其实只是 cell type effect。因果结构不一定要很重，但至少要显式区分这些来源。
