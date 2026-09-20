# Deep generative modeling of transcriptional dynamics for RNA velocity analysis in single cells

## Metadata
- 标题: Deep generative modeling of transcriptional dynamics for RNA velocity analysis in single cells
- 作者: Adam Gayoso et al.
- 年份: 2024
- 正式 venue: Nature Methods
- PDF 文件名: Deep generative modeling of transcriptional dynamics for RNA velocity analysis in single cells.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
veloVI针对的是 RNA velocity 在真实数据中常见的两个痛点：不确定性无法量化，以及研究者通常不知道当前数据到底适不适合做 velocity。它希望让 velocity 不再只是一个箭头图，而是一个带 posterior uncertainty 的生成模型输出。

## 数据类型 / 场景
输入是标准带 unspliced/spliced 计数的 scRNA-seq snapshot 数据。作者重点展示了不同预处理流程、不同系统上的鲁棒性和 uncertainty 评估能力。

## 核心算法怎么设计
veloVI把 RNA velocity 写进一个深生成模型里。它学习基因特异的 RNA metabolism dynamical model，并通过变分推断得到每个细胞、每个基因上的 posterior velocity 分布；同时还能扩展到底层转录率随时间变化的情形，因此不再局限于固定参数的经典动力学。

## 设计原则是什么
设计原则是“先把 uncertainty 建模清楚，再谈 velocity 解释”。与只输出点估计的方法不同，veloVI把是否可用、哪里不确定、哪些基因拟合差都纳入同一后验框架，这比单纯追求漂亮向量场更稳健。

## 工程优化 / 训练技巧 / pipeline 设计
它继承了 scvi-tools 系列成熟的变分推断与深生成 pipeline，在预处理变动时仍能保持较稳的拟合质量。posterior uncertainty 本身也成为很实用的质控信号，减少了人工判读 phase portrait 的负担。

## 局限性
尽管不确定性建模更完善，veloVI仍然依赖 RNA metabolism 的参数化假设和相应计数质量。它改善了“何时该信 velocity”这个问题，但没有从根本上解决 snapshot 数据对长时程、跨条件外推的信息不足。

## 为什么能发到这个级别
这篇文章击中了 RNA velocity 社区最常被质疑的点：过度自信和适用性不明。通过把 posterior uncertainty 和 dynamical model 结合起来，它把 velocity 从启发式工具推进成了更严肃的统计生成框架。

## 对 CytoBridge-agent 自动设计算法的启发
对 agent 来说，任何轨迹或 velocity 方法都应该同时输出“结果”和“可信度”。如果新算法不能量化 uncertainty 或 detect failure mode，就很难在自动化系统里安全使用。
