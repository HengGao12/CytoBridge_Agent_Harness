# Synchronized Optimal Transport for Joint Modeling of Dynamics Across Multiple Spaces

## Metadata
- 标题: Synchronized Optimal Transport for Joint Modeling of Dynamics Across Multiple Spaces
- 作者: Zixuan Cang, Yanxiang Zhao
- 年份: 2025
- 正式 venue: SIAM Journal on Applied Mathematics, 85(1):341-365, 2025
- PDF 文件名: Synchronized Optimal Transport for Joint Modeling of Dynamics Across Multiple Spaces.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么已有 gap
随着多组学和空间组学数据变多，一个系统往往会在多个空间里同时被表征，例如表达空间、染色质空间、物理空间。若分别在每个空间做 dynamical OT，得到的动力学可能互相不一致，缺少跨空间 coherence。

## 理论 / 算法创新点
文章提出 Synchronized Optimal Transport (SyncOT)，在一个主空间上定义动力学，并通过预定义映射把这套动力学诱导到多个次级空间，同时最小化各空间总的动力学代价。换句话说，它不是做多个独立 OT，而是把它们耦合成一个联合问题。

## 具体算法或理论结构怎么设计
数学上，SyncOT 最小化主空间和诱导次级空间中的加权 kinetic energy，总体仍保持凸优化结构。作者使用 staggered grid 离散化，并为两类常见映射关系分别设计 primal-dual 算法，使问题可以数值求解。

## 设计原则是什么
核心设计原则是：多个空间描述的是同一系统，就必须共享一套尽量一致的动力学，而不是允许每个空间各讲各的故事。对于多模态单细胞数据，这是非常强也非常自然的建模原则。

## 工程优化或实现性考虑
工程上，文章没有追求完全一般的多模态对齐，而是要求给定跨空间映射，再利用凸结构和原始-对偶算法稳定求解。这使得方法在理论上较干净，也更容易分析多空间耦合到底带来了什么。

## 局限与未解点
局限主要在于它假设跨空间映射已知或可给定，而真实多组学问题里这往往就是最难的部分。其次，基于网格的数值离散在高维潜空间不够友好；文章也没有把 unbalanced growth 或 stochastic interaction 纳入核心框架。

## 为什么能发顶会 / 为什么是重要理论工作
它的重要性在于第一次把“多空间动力学一致性”写成了明确的 dynamical OT 优化目标，而不是只做经验性多模态对齐。对多组学动力学建模而言，这是一种更高层次的约束。

## 对单细胞轨迹建模和 CytoBridge-agent 自动设计算法的启发
CytoBridge 若要走向多模态/时空联合建模，SyncOT 给出的启发非常直接：不要只在一个 `X_latent` 里塞下所有模态，而要允许不同空间保留各自几何，再通过同步动力学正则把它们绑在一起。自动设计算法时，agent 应把“单潜空间融合”和“多空间同步 transport”作为两条真正不同的设计路线。
