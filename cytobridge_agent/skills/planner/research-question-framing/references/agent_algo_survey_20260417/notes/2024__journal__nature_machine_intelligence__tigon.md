# Reconstructing growth and dynamic trajectories from single-cell transcriptomics data

## Metadata
- 标题: Reconstructing growth and dynamic trajectories from single-cell transcriptomics data
- 作者: Yutong Sha et al.
- 年份: 2024
- 正式 venue: Nature Machine Intelligence
- PDF 文件名: Reconstructing growth and dynamic trajectories from single-cell transcriptomics data.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
TIGON针对 time-series scRNA-seq 中一个非常核心的问题：不仅要恢复状态转变轨迹，还要同时恢复 cell population growth，并进一步挖掘 temporal GRN 和 communication。作者强调 growth 若被忽略，很多 temporal inference 会系统偏差。

## 数据类型 / 场景
输入是多个时间点的 scRNA-seq snapshots。适用于发育或疾病过程中细胞数量显著变化的系统。

## 核心算法怎么设计
TIGON是 dynamic, unbalanced OT。它把细胞群体密度的演化写成带 velocity 和 growth 的 PDE，并分别用神经网络建模 velocity 与 growth；在求解上使用基于 Wasserstein-Fisher-Rao 距离的 dimensionless formulation，从而在高维表达空间里更可计算。完成拟合后还能回到基因层面分析 growth-related genes、temporal GRN 和 cell-cell communication。

## 设计原则是什么
核心原则是把“状态流动”和“质量变化”同时放进主模型，而不是把 growth 作为后处理或外部先验。另一个原则是为高维 OT 问题设计可训练的深度近似，而不是依赖网格化数值解。

## 工程优化 / 训练技巧 / pipeline 设计
作者系统比较了 PCA、AE、reversible UMAP 等可逆降维方案，以确保 learned dynamics 能回到基因层面解释。WFR-based dimensionless formulation 则是其主要工程创新，使动态非平衡 OT 在高维单细胞场景下变得可训练。

## 局限性
TIGON仍然需要先做降维，且作者在讨论中明确指出不同可逆降维方式会影响 gene-level downstream 解释。方法虽然能做 growth 与 dynamics 的统一恢复，但交互项仍是下游推断而不是主模型中的原生机制。

## 为什么能发到这个级别
它比早期 OT 轨迹方法更进一步，把 growth 从“应不应该考虑”变成“必须和 trajectory 一起学”的主变量，并在方法、数值求解和生物解释上都给出完整故事。这个问题定义与技术实现都比较扎实。

## 对 CytoBridge-agent 自动设计算法的启发
TIGON直接证明了 growth 不是可选增强项，而是单细胞动力学的主体变量之一。CytoBridge 当前路线里把 velocity、growth 和 interaction 分开建模是合理的，但更进一步应考虑这些量在同一桥过程里的联合可辨识性。
