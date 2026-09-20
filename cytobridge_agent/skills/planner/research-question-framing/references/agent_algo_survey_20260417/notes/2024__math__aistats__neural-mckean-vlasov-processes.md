# Neural McKean-Vlasov Processes: Distributional Dependence in Diffusion Processes

## Metadata
- 标题: Neural McKean-Vlasov Processes: Distributional Dependence in Diffusion Processes
- 作者: Haoming Yang, Ali Hasan, Yuting Ng, Vahid Tarokh
- 年份: 2024
- 正式 venue: AISTATS 2024
- PDF 文件名: Neural McKean-Vlasov Processes Distributional Dependence in Diffusion Processes.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 是
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么已有 gap
这篇文章针对的是 neural SDE 领域一个长期空白：大多数模型默认样本路径彼此独立，只适合 Itô-SDE；但很多真实系统是分布依赖的，也就是 drift/diffusion 会受当前总体分布影响。过去缺少面向 MV-SDE 的通用神经参数化和估计方法。

## 理论 / 算法创新点
文章提出一组用于 MV-SDE 的半参数化表示和对应估计器，把 law dependence 显式塞进 drift/diffusion 的网络参数化里。它不仅比较了不同架构，还从表示能力角度讨论了为什么加入 distributional dependence 后，可表达的 probability flow 会比普通 Itô-SDE 更丰富。

## 具体算法或理论结构怎么设计
具体上，它用经验测度、可学习分布表示或 generative model 来构造“对总体分布的输入”，再把这个输入送入 drift/diffusion 网络。文章还提出 mean-field layer 这类结构，把对总体的依赖写成交换不变的聚合形式，以适配 exchangeable 粒子系统。

## 设计原则是什么
设计原则是：如果系统本质上是交互粒子系统，就不要把相互作用偷换成额外噪声或个体 covariates；而是让模型在参数化层面就承认“状态转移依赖总体分布”。同时，这种依赖必须满足置换不变性，否则对群体数据不稳健。

## 工程优化或实现性考虑
工程上它提供的不是一个单一神经网络，而是一组可替换架构和估计器，对不同机器学习任务做了比较。相比纯粒子级 pairwise interaction，这类 law-dependent 聚合更适合大样本总体建模，因为复杂度可以从显式两两交互降下来。

## 局限与未解点
它主要解决的是 MV-SDE 表示和估计，并不直接给出 OT/SB/FM 形式的 trajectory inference 目标。方法默认 exchangeability 和总体分布估计足够稳定；当样本很少、分布表示误差大或系统并不满足平均场近似时，性能会受影响。

## 为什么能发顶会 / 为什么是重要理论工作
它的重要性在于把“分布依赖动力学”从概念推进成了可训练神经模型家族。对于许多涉及群体效应、注意力、交互粒子和细胞通讯的任务，这是一个基础设施式的贡献，而不是只在某个 benchmark 上多提几个点。

## 对单细胞轨迹建模和 CytoBridge-agent 自动设计算法的启发
CytoBridge 未来如果要把 interaction 做实，不能只停留在经验性的 cell-cell communication score 上。这篇文章给出的启发是，可以把 interaction term 直接设计成 law-dependent drift/growth 模块，让网络输入当前细胞群体的分布表示，再去预测单细胞层面的速度和增长。这样比事后再拼接邻域特征更接近 mean-field 理论，也更适合自动化搜索。
