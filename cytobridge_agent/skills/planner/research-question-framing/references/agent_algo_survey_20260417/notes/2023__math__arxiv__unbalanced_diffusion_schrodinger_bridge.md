# Unbalanced Diffusion Schrödinger Bridge

## Metadata
- 标题: Unbalanced Diffusion Schrödinger Bridge
- 作者: Matteo Pariset, Ya-Ping Hsieh, Charlotte Bunne, Andreas Krause, Valentin De Bortoli
- 年份: 2023
- 正式 venue: 当前本地版本为 arXiv 预印本；另有 ICML 2023 workshop poster 记录，但非主会正式论文
- PDF 文件名: Unbalanced Diffusion Schrödinger Bridge.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 当前未确认主会/期刊追认版本
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成摘要、引言与方法结构阅读

## 解决了什么已有 gap

- 现有 neural SB/DSB 默认边缘是概率测度，等价于质量守恒，这和生物系统中的 birth/death 完全不匹配。
- 尤其是药物响应、病毒演化、细胞增殖/死亡场景，质量变化不是噪声，而是任务主体。

## 理论 / 算法创新点

- 推导了带 killing 与 birth 项的 SDE 的时间反演，从而把 non-conservative dynamics 正式带进 DSB。
- 提出 unbalanced DSB，并给出两套可扩展训练方案。
- 文章把质量变化内生化到随机过程，而不是仅在边缘上做后补偿。

## 具体算法或理论结构怎么设计

- 扩展参考随机过程，使其允许粒子死亡或出生。
- 在此基础上推导对应的反向过程与 unbalanced IPF 风格目标。
- 神经网络同时学习运输动力学与质量变化相关项，从而拟合任意有限总质量边缘。

## 设计原则是什么

- 非平衡质量应写进 path law，而不是只加一个 scalar correction。
- 若目标系统本身包含 birth/death，就应让前向与反向动力学都感知这些机制。
- 对桥模型来说，时间反演公式决定了能否正确学习非守恒过程。

## 工程优化或实现性考虑

- 文章强调两种可扩展训练方案，这点是其主要工程贡献。
- 并在药物单细胞响应与病毒变体扩散上展示了方法适用性。

## 局限与未解点

- 非平衡桥训练仍比平衡版本更复杂、更敏感。
- birth 与 death 的参数化方式可能存在可辨识性问题。
- 目前仍以两端分布为主，距离一般多时间点非平衡桥还有一步。

## 为什么能发顶会 / 为什么是重要理论工作

- 这篇工作直接命中了 SB 路线在生物应用里最大的短板: 质量守恒假设不成立。
- 即使还未形成主会正式版本，它在方法路线上也极具代表性和启发性。

## 对单细胞轨迹建模和 CytoBridge-agent 自动设计算法的启发

- CytoBridge 若想真正面向生物过程，unbalanced bridge 几乎是必选方向，而不是锦上添花。
- agent 自动设计算法时，应把“质量变化机制是否内生化到随机过程”作为重要判断标准。
- 这篇论文还说明，增长模块最好和桥/反演公式一起学，而不是外挂一个 growth head 就结束。
