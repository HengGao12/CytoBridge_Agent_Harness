# TFvelo: gene regulation inspired RNA velocity estimation

## Metadata
- 标题: TFvelo: gene regulation inspired RNA velocity estimation
- 作者: Jiachen Li et al.
- 年份: 2024
- 正式 venue: Nature Communications
- PDF 文件名: TFvelo gene regulation inspired RNA velocity estimation.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
TFvelo想解决的是：很多数据里 unspliced/spliced 信号太弱，经典 RNA velocity 无法稳健拟合，但调控因子与靶基因之间的相位延迟仍然存在。它试图借助 TF-target 关系把 velocity 推广到没有明显 splicing 信息的场景。

## 数据类型 / 场景
输入可以是常规 RNA abundance 数据，外加推断得到的 TF-target regulatory information。适合 splicing 信号不足但调控关系相对可用的单细胞数据。

## 核心算法怎么设计
TFvelo把 target gene 的 velocity 建立在 target gene 本身与其潜在 TF 表达的联合变化上，而不是只看 unspliced-spliced phase portrait。算法采用 generalized EM 迭代更新隐状态和动力学参数，并利用 TF-target 共表达的相位延迟来拟合速度。

## 设计原则是什么
原则是用更接近调控机制的信号替代弱观测。作者认为 velocity 本质上是调控驱动的表达变化率，因此在剪接信息不足时，应退回到基因调控层面而不是强行坚持传统 phase portrait。

## 工程优化 / 训练技巧 / pipeline 设计
广义 EM 让模型可以在 noisy regulation prior 下逐步更新参数。它不再硬依赖 unspliced 数据，因此在普通 RNA abundance 数据上的适用面更广，这本身也是很强的工程优势。

## 局限性
方法表现显著依赖 TF-target 先验的准确性，而 regulatory prior 在很多系统中本身并不稳定。它扩大了 velocity 的适用范围，但也把问题转移到了 GRN 质量与调控延迟可识别性上。

## 为什么能发到这个级别
TFvelo提供了一个很新鲜的视角：velocity 不一定只能来自剪接延迟，也可以来自转录调控延迟。这种重新定义问题的方式比单纯再拟合一次 phase portrait 更有方法学价值。

## 对 CytoBridge-agent 自动设计算法的启发
CytoBridge 在做 dynamics inference 时，不应把 velocity 信号来源限定为 RNA kinetics。本质上，任何能提供系统性相位延迟的模态或先验，都可能成为 dynamics 估计器的输入。
