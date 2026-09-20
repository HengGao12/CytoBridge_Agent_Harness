# CellStream: Dynamical Optimal Transport Informed Embeddings for Reconstructing Cellular Trajectories from Snapshots Data

## Metadata
- 标题: CellStream: Dynamical Optimal Transport Informed Embeddings for Reconstructing Cellular Trajectories from Snapshots Data
- 作者: Yue Ling, Peiqi Zhang, Zhenyi Zhang, Peijie Zhou
- 年份: 2026
- 正式 venue: AAAI 2026
- PDF 文件名: CellStream Dynamical Optimal Transport Informed Embeddings for Reconstructing Cellular Trajectories from Snapshots Data.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 是
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么已有 gap
单细胞轨迹推断里，一个非常实际的问题是 embedding 和 dynamics 常常是分开的：先用 PCA/UMAP/scVI 得到低维表示，再在这个固定空间里跑 OT/ODE。这样做的缺点是 latent space 本身并不保证保留时间结构，可能天生就不适合做动力学学习。

## 理论 / 算法创新点
CellStream 的核心贡献是把 autoencoder 与 unbalanced dynamical OT 直接耦合，联合学习 embedding 和 latent cellular dynamics。它不是把动力学当作下游任务，而是让动力学信号反过来塑造 latent space，从而得到 dynamics-informed embedding。

## 具体算法或理论结构怎么设计
模型用 autoencoder 学低维表示，再在潜空间上用连续动力学模块重建细胞流，训练中用 real-time trajectory feedback 约束 latent stream，文中明确写了由 `L_OT`、`L_WFR` 和 `L_Mass` 组成的反馈项。换句话说，latent geometry 不是只靠重构学出来，而是同时被 transport、birth-death 和 mass consistency 拉着走。

## 设计原则是什么
设计原则非常清楚：如果后续任务是动力学建模，那么表示学习阶段就必须纳入动力学约束，而不是把 topology-preserving 或 batch-correction 当作唯一目标。作者也明确反对“先固定 embedding、再做 dynamics”的管线式思路。

## 工程优化或实现性考虑
工程上，联合训练能显著减少固定 embedding 带来的噪声敏感性，文章在模拟数据、真实 scRNA-seq 以及 spatial transcriptomics 上都做了验证。它的现实意义在于：很多 OT 动力学模型之所以不稳，问题不一定出在 transport 本身，而可能出在一开始的 latent space 就学歪了。

## 局限与未解点
这篇工作更偏方法整合，理论可辨识性和泛化分析还比较有限。联合训练也会增加优化耦合难度，若 reconstruction 与 dynamics 目标冲突，训练可能更脆弱。文章没有把 cell-cell interaction 当成核心模块来建模，这一点相对 CytoBridge 仍然偏弱。

## 为什么能发顶会 / 为什么是重要理论工作
它之所以重要，是因为它准确地命中了当前单细胞 OT 路线里一个常被忽视但又非常致命的问题：latent representation 不是中立的。把 embedding 学习和动力学学习真正耦合起来，是单细胞 snapshot 建模非常自然的一步。

## 对单细胞轨迹建模和 CytoBridge-agent 自动设计算法的启发
对 CytoBridge 而言，CellStream 的启发非常直接：`X_latent` 不能被默认视为外部固定输入，agent 应该把“是否联合学习 latent geometry 与 dynamics”作为一级设计变量。特别是在噪声大、时间间隔大、批次强的数据里，先优化表示再优化动力学，往往不如联合训练更可靠。
