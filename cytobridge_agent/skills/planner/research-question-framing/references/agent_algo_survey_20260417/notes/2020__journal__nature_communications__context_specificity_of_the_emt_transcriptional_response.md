# Context specificity of the EMT transcriptional response

## Metadata
- 标题: Context specificity of the EMT transcriptional response
- 作者: David P. Cook, Barbara C. Vanderhyden
- 年份: 2020
- 正式 venue: Nature Communications
- PDF 文件名: Context specificity of the EMT transcriptional response.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已读摘要、结果、讨论与方法概要

## 解决了什么生物问题

它要回答的是：EMT 到底是不是一个统一的线性转录程序，还是强烈依赖细胞背景和诱导条件。这个问题对任何想建模细胞状态转换、条件扰动和反事实推断的算法都非常关键，因为它直接决定能不能假设存在通用 transition program。

## 数据类型 / 场景

- 103,999 个单细胞，来自 12 组 EMT time course。
- 4 种癌细胞系，3 种诱导因子，并配套做了 kinase inhibitor screens。
- 既有时间维，也有条件维和扰动维，是典型的多条件动态快照数据。

## 核心算法怎么设计

- 这不是新动力学算法论文，核心是一个高质量实验-计算联合框架。
- 实验上通过 MULTI-seq 做大规模 multiplexed scRNA-seq，在统一流程下比较 12 个 EMT time course。
- 计算上对每个条件单独做 pseudotime 排序，随后沿 pseudotime 用广义加性模型等分析差异表达、regulon 活性和 inhibitor effect。
- 关键发现是：任意两种条件下共享的 EMT response genes 平均只有 22%，说明 EMT 高度 context-specific；inhibitor screen 进一步揭示了这些响应的模块性和条件依赖调控。

## 设计原则是什么

- 不把 EMT 当作单轴线性过程，而是当作条件化、模块化的状态转移响应。
- 比起从单一体系抽象“普适 signature”，更重要的是在匹配条件下做系统比较。
- 评价转移状态时必须把 perturbation dependence 纳入设计，而不是只看自然时间序列。
- pseudotime 在这里主要承担“标准化比较轴”的角色，而不是最终机制模型。

## 工程优化 / 训练技巧 / pipeline 设计

- 960 个样本的统一 multiplexing 设计极大减小了批次偏差，是这篇文章最强的工程部分。
- time course 和 inhibitor screen 共用同一处理逻辑，便于直接比较抑制剂对 EMT progression 的影响。
- 使用监督型 pseudotime 和统一 differential modeling，把多条件比较做得非常规整。

## 局限性

- 主要基于癌细胞系，离体系统与体内微环境仍有差距。
- 文章证明了 context specificity，但并没有建立一个可用于长程预测的生成式动力学模型。
- pseudotime 依然是按条件分别拟合的，跨条件对齐仍带有方法选择依赖。
- 讨论中作者也指出，真正的 EMT manifold 及其 across-context alignment 还需要后续工作。

## 为什么能发到这个级别

这篇文章的价值在于用极大规模、强控制的单细胞实验设计，推翻了“EMT 有单一通用程序”的简单叙事。对发育、肿瘤和细胞状态转换领域来说，这是非常重要的认知更新；同时数据规模和实验设计也足够强。

## 对 CytoBridge-agent 自动设计算法的启发

- agent 不能默认所有条件共享同一套 transition vector field；context-conditioned dynamics 应该是默认候选。
- 设计 novel algorithm 时，跨条件泛化与条件特异模块分解应作为显式目标，而不是事后分析。
- 多条件时间序列是检验模型是否真的“理解动力学”而非只会拟合单一数据集的关键基准。
