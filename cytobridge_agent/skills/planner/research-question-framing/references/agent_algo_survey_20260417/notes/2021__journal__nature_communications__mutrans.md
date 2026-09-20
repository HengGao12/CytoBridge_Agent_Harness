# Dissecting transition cells from single-cell transcriptome data through multiscale stochastic dynamics

## Metadata
- 标题: Dissecting transition cells from single-cell transcriptome data through multiscale stochastic dynamics
- 作者: Peijie Zhou et al.
- 年份: 2021
- 正式 venue: Nature Communications
- PDF 文件名: Dissecting transition cells from single-cell transcriptome data through multiscale stochastic dynamics.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
作者要解决的是：只有 snapshot scRNA-seq 时，怎样把稳定细胞状态和真正处在命运切换中的 transition cells 区分开，并给出跨细胞状态的高概率转变路径。文中在 EMT、iPSC 分化和血液发育等系统里，把“过渡细胞是谁、往哪里走”变成了可以直接回答的问题。

## 数据类型 / 场景
主要面向 snapshot scRNA-seq，且更适合近稳态、吸引子结构明显的细胞命运系统。作者在多个平台和多个生物系统上验证了方法的可迁移性。

## 核心算法怎么设计
MuTrans把数据解释为带噪随机动力系统。它先通过多尺度随机游走构造 cell-fate dynamical manifold，再把稳定细胞看作吸引子 basin，把过渡细胞放在 basin 之间经 saddle 附近通过的路径上；随后用 coarse-grained transition path theory 估计不同命运路径的相对可能性，并输出 transition cells、稳定 cells 以及 driver/marker genes。

## 设计原则是什么
核心原则不是在低维几何上硬拉 pseudotime，而是先给出一个可解释的随机动力学假设，再让 attractor、saddle、transition path 这些物理对象对应到细胞命运问题。另一个原则是多尺度：先在细胞层面刻画局部迁移，再在 cluster/attractor 层面做粗粒化，兼顾分辨率和稳定性。

## 工程优化 / 训练技巧 / pipeline 设计
作者加入了用于大数据预处理的 DECLARE，并把最贵的动力学分析拆成随机游走、membership、粗粒化转移路径几步，降低了直接在高维连续空间求解的难度。实现上也复用了 PyEMMA 一类成熟工具来做 transition-path 计算。

## 局限性
方法依赖两个强假设：数据近似来自稳态分布，以及 drift 可主要由多井势函数逼近。对强非平衡、强增殖死亡、非梯度分量很强的系统，这些假设可能失效；此外，随机游走和粗粒化质量也会显著影响 transition cell 的定位。

## 为什么能发到这个级别
这篇工作的亮点是把随机动力系统和单细胞轨迹推断真正接起来了，而且解决的是当时很多方法没有明确解决的 transition-cell 识别问题。它同时给出理论对象、可运行算法和跨平台实证，因此达到了 Nature Communications 级别的方法文章标准。

## 对 CytoBridge-agent 自动设计算法的启发
对 agent 来说，不能只输出一条平滑轨迹，还要显式区分稳定态、过渡态和高概率跃迁路径。若要做新算法，值得把“transition-cell detection”作为一级目标，并让模型在 latent space 中直接暴露 attractor、barrier 和 path-level summary，而不是只给点到点 coupling。
