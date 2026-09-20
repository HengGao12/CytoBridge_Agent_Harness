# Generative modeling of single-cell time series with PRESCIENT enables prediction of cell trajectories with interventions

## Metadata
- 标题: Generative modeling of single-cell time series with PRESCIENT enables prediction of cell trajectories with interventions
- 作者: Grace Hui Ting Yeo et al.
- 年份: 2021
- 正式 venue: Nature Communications
- PDF 文件名: Generative modeling of single-cell time series with PRESCIENT enables prediction of cell trajectories with interventions.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
PRESCIENT直指时间序列 scRNA-seq 的核心难题：如何在物理时间上重建细胞分化过程，并预测基因干预后命运轨迹如何改变。作者重点验证了造血和胰岛 beta 细胞分化中的 fate bias 与 intervention 效果。

## 数据类型 / 场景
输入是 time-series scRNA-seq snapshots，可配合 lineage tracing 做验证。方法特别适用于有多个离散时间点、同时伴随增殖差异的分化系统。

## 核心算法怎么设计
PRESCIENT把细胞分化建模成在潜在能量景观上的扩散过程。作者学习一个 potential function，令 drift 为其负梯度，再配合噪声项模拟单细胞随物理时间演化；同时用 birth-death/growth 估计把增殖信息并入模拟，从而能生成未观测时间点、未观测初始条件和 in silico perturbation 下的轨迹。

## 设计原则是什么
第一原则是把“轨迹推断”提升为“可查询的生成模型”，而不是只做 timepoint 之间的 coupling。第二原则是显式建模 stochasticity、physical time 和 proliferation，因为 fate prediction 在这些因素缺失时会系统性失真。第三原则是让 perturbation 通过修改初始表达或轨迹动力学进入同一个生成框架。

## 工程优化 / 训练技巧 / pipeline 设计
作者先对高变基因做 PCA 降维，再在较低维空间学习 potential landscape，显著降低了直接在原始高维表达空间做 SDE 拟合的难度。增殖/凋亡信号通过基因集打分近似估计 growth，也是一种在缺少真实 birth-death 观测时的工程折中。

## 局限性
PRESCIENT仍然依赖势场驱动扩散这一建模假设，复杂非梯度动力学、强交互效应和远离训练分布的干预可能会导致外推不稳定。growth 是由基因集近似给出的，不是直接观测；因此在 proliferation 估计不准时，fate bias 也会受影响。

## 为什么能发到这个级别
这篇文章把“从时间序列单细胞快照里学习一个可生成、可干预、可物理解释的动态模型”落到了实处，还用 lineage tracing 做了严格验证。相比当时偏总结性或耦合性的 TI 方法，它第一次把 stochastic dynamics、growth 和 intervention 放进了同一个可运行框架里。

## 对 CytoBridge-agent 自动设计算法的启发
如果 agent 要自动设计算法，PRESCIENT说明“能否生成未测时间点和未做过的 perturbation”是非常强的评价维度。CytoBridge 一类方法应继续保留生成式 latent dynamics、growth 显式建模和可执行的 intervention 接口，而不是退回到静态 pseudotime。
