# Simulating Fokker-Planck equations via mean field control of score-based normalizing flows

## Metadata
- 标题: Simulating Fokker-Planck equations via mean field control of score-based normalizing flows
- 作者: Yifan Jiang, Di Fang, et al.
- 年份: 2025
- 正式 venue: arXiv preprint arXiv:2506.05723
- PDF 文件名: Simulating Fokker-Planck equations via mean field control of score-based normalizing flows.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否，当前按预印本处理
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么已有 gap
Fokker-Planck 方程是很多随机动力学的连续分布描述，但高维数值求解很难。过去要么依赖网格 PDE 解法，要么用粒子方法做前向模拟，难以和现代 score-based/flow-based 模型自然结合。

## 理论 / 算法创新点
文章把 FP 方程重写成由 drift 和 score 共同定义的 continuity equation，再把“模拟 FP 演化”变成一个 mean field control 问题。最关键的创新是利用 score-based normalizing flow 在确定性轨迹上高效计算 score，从而避免直接求高维 PDE。

## 具体算法或理论结构怎么设计
作者先把 FP 的速度场写成 `b(x,t) - ∇ log ρ_t(x)` 一类形式，再设计一个 MFC 目标，让 learned flow 的速度场去匹配该连续性方程的真实速度场。这样 density 演化通过可训练的概率流实现，文章还对 OU 过程给出了收敛分析。

## 设计原则是什么
设计原则是“把难求的 density PDE 转写成更易训练的 probability flow matching 问题”，并且显式保留 score 这一决定扩散效应的关键量。相比完全数值离散，作者更倾向于利用生成模型框架来承载连续分布演化。

## 工程优化或实现性考虑
工程上它避免了高维网格，主要在轨迹和 score 上做可微计算，因此对 Langevin、underdamped Langevin 和混沌系统都更可扩展。对于需要反复模拟分布演化的任务，这种做法比传统 PDE 求解器更适合嵌入机器学习管线。

## 局限与未解点
它本质上还是“已知动力学下的前向模拟/控制”，不是从 snapshot 数据反推动力学。文章没有处理 unbalanced birth-death 或 interaction；同时方法效果依赖 score 估计精度和流模型训练质量。

## 为什么能发顶会 / 为什么是重要理论工作
它的重要性在于把 Fokker-Planck、mean field control 和 score-based normalizing flow 打通了，给出了一个兼顾理论与可计算性的高维模拟路线。对于任何想把概率流、自由能耗散和控制统一起来的工作，都有参考价值。

## 对单细胞轨迹建模和 CytoBridge-agent 自动设计算法的启发
对 CytoBridge，这篇文章的启发是：如果想更精细地建模随机性和 free-energy 下降，可以考虑把 score 显式引入控制/动力学目标，而不是只学漂移项。它也提醒 agent，在某些场景下“先学可微概率流，再把它解释为 PDE 解”可能比直接解方程更实际。
