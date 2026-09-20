# ARTEMIS integrates autoencoders and Schrödinger Bridges to predict continuous dynamics of gene expression, cell population, and perturbation from time-series single-cell data

## Metadata
- 标题: ARTEMIS integrates autoencoders and Schrödinger Bridges to predict continuous dynamics of gene expression, cell population, and perturbation from time-series single-cell data
- 作者: Sayali Anil Alatkar and Daifeng Wang
- 年份: 2025
- 正式 venue: Bioinformatics (ISMB/ECCB 2025 Supplement)
- PDF 文件名: ARTEMIS integrates autoencoders and Schrödinger Bridges to predict continuous dynamics of gene expression, cell population, and perturbation from time-series single-cell data.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
ARTEMIS瞄准的是 time-series scRNA-seq 中连续基因表达动力学、细胞群体数量变化和 perturbation 效应的统一预测。作者想解决的不只是“轨迹长什么样”，还包括“群体会如何增减”以及“扰动后过程会怎样改变”。

## 数据类型 / 场景
输入是多个时间点的 scRNA-seq snapshots。文中用 pancreatic beta differentiation、zebrafish embryogenesis 和 EMT 三个数据集评估模型。

## 核心算法怎么设计
ARTEMIS先用 VAE 把高维表达映射到连续 latent space，再在 latent space 中用 unbalanced diffusion Schrödinger bridge 学习 forward-backward SDE。模型额外用一个神经网络估计 time-varying kill rates，从而同时恢复 gene-expression dynamics 和 population change；之后还可做 in silico perturbation 来识别 drift genes。

## 设计原则是什么
设计上强调三个统一：高维数据先压到连续 latent manifold，再用 bridge 过程学习随机连续动力学，同时把 unbalanced mass change 写入主模型。它比许多只关心表达轨迹的方法更接近完整的群体动力学视角。

## 工程优化 / 训练技巧 / pipeline 设计
VAE 预训练加联合训练的策略缓解了直接在原始表达空间求 bridge 的困难。kill-rate 网络是很实用的工程设计，让 population dynamics 不必退化成外部估计；drift gene 分析则让模型结果更容易转化为生物解释。

## 局限性
模型表达能力很强，但对 latent representation 质量依赖也很高；若 VAE 压缩损失了关键结构，后续 bridge 学到的过程也会偏。作为补充会议论文，它的实验广度和大规模可扩展性还不如更成熟的平台型方法。

## 为什么能发到这个级别
ARTEMIS把 VAE、unbalanced SB 和 perturbation prediction 组合成了一个清晰完整的框架，问题定义也紧贴单细胞动态建模前沿。它说明 SB 路线在单细胞中的吸引力已经从纯理论进入可落地方法阶段。

## 对 CytoBridge-agent 自动设计算法的启发
对 CytoBridge 来说，ARTEMIS强化了“latent compression + bridge dynamics + unbalanced mass”这条路线的合理性。若要做更强的 perturbation module，可进一步让 drift/growth/interaction 在同一 latent bridge 中协同学习。
