# Likelihood Training of Schrödinger Bridge using Forward-Backward SDEs Theory

## Metadata
- 标题: Likelihood Training of Schrödinger Bridge using Forward-Backward SDEs Theory
- 作者: Tianrong Chen, Guan-Horng Liu, Evangelos A. Theodorou
- 年份: 2021
- 正式 venue: ICLR 2022
- PDF 文件名: Likelihood Training of Schrödinger Bridge using Forward-Backward SDEs Theory.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 是
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成摘要、引言、预备知识与方法部分阅读

## 解决了什么已有 gap

- SB 在理论上很优雅，但生成模型社区更熟悉的是 likelihood/score 训练；两者之间当时缺一条清晰的桥。
- 这导致 SB 常被视为“理论替代品”，但不清楚能否像现代生成模型那样进行 principled 的似然式训练。

## 理论 / 算法创新点

- 用 forward-backward SDE 理论把 SB 的最优性条件重写成一组可训练的随机微分方程。
- 在此基础上构造出 likelihood training objective，并证明它把主流 score-based 训练包含为特例。
- 这让 SB 不再只是 IPF/投影算法，也能走“目标函数驱动”的训练路线。

## 具体算法或理论结构怎么设计

- 从 SB 的控制表示出发，引入前向/后向值函数与对应的 FBSDE。
- 用神经网络参数化这些对象，再通过路径积分/似然相关目标进行优化。
- 最终学到的 drift/score 同时满足桥的约束和可训练的概率一致性条件。

## 设计原则是什么

- 如果问题本质上是随机控制，就应该从控制的必要/充分条件推出训练目标。
- 好的神经桥算法不应只靠经验 matching，而应尽可能继承原问题的最优性结构。
- 训练目标最好和已有成熟范式兼容，这样方法才容易真正落地。

## 工程优化或实现性考虑

- 文章的主要工程贡献是把 FBSDE 结构变成可微可训练的 loss。
- 这比纯粹 IPF 更贴近现代生成模型训练流程，也更便于复用现有神经网络实现。

## 局限与未解点

- FBSDE 本身就难，数值稳定性与近似误差仍是现实障碍。
- 仍主要围绕平衡桥与简单参考过程展开。
- 对多时间点、条件迁移和质量变化没有直接解决。

## 为什么能发顶会 / 为什么是重要理论工作

- 它补上了 SB 与 likelihood training 之间的关键缺口，这是一个明确的理论与方法空白。
- 同时又给出可实现的深度学习框架，因此在 ICLR 这种偏方法创新的场合非常有分量。

## 对单细胞轨迹建模和 CytoBridge-agent 自动设计算法的启发

- 对 CytoBridge 来说，这篇文章最大的启发是: loss 不必拍脑袋设计，可以从 FBSDE/HJB/Fokker-Planck 一致性推出。
- 当 agent 自动设计新算法时，应优先寻找“来自最优性条件的训练目标”，而不是先写网络再补 loss。
- 这篇工作也是后续 generalized SB、unbalanced bridge 等方法的关键中间站。
