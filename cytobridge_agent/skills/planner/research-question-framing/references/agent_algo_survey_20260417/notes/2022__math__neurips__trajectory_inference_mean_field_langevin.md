# Trajectory Inference via Mean-field Langevin in Path Space

## Metadata
- 标题: Trajectory Inference via Mean-field Langevin in Path Space
- 作者: Lénaïc Chizat, Stephen Zhang, Matthieu Heitz, Geoffrey Schiebinger
- 年份: 2022
- 正式 venue: NeurIPS 2022
- PDF 文件名: Trajectory Inference via Mean-field Langevin in Path Space.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 是
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成摘要、引言与优化方法部分阅读

## 解决了什么已有 gap

- 之前路径空间上的 min-entropy trajectory inference 理论已经提出，但对应的是无限维凸优化，几乎不可直接算。
- 因此“有理论但无可用算法”成为这一路线的主要瓶颈。

## 理论 / 算法创新点

- 提出一个 grid-free 的求解器，把每个时间点表示成点云，并通过 Schrödinger bridges 耦合。
- 用 noisy gradient descent/mean-field Langevin dynamics 去逼近路径空间最小熵估计器。
- 给出 mean-field 极限下的全局收敛结果，并讨论如何适配质量变化。

## 具体算法或理论结构怎么设计

- 把轨迹推断写成相对 Wiener measure 的最小熵问题。
- 再把无限维对象离散成一组随时间演化的粒子云。
- 粒子更新由路径空间目标驱动，而时间点之间的联系通过 Schrödinger bridge 保持。

## 设计原则是什么

- 直接在 path space 建模比只学习局部 velocity field 更贴近问题本质。
- 如果目标本身是熵最小化，就应让优化过程也围绕该目标展开，而不是退回启发式配对。
- grid-free/particle-based 设计比固定网格更适合高维生物数据。

## 工程优化或实现性考虑

- 主要工程点是 grid-free 粒子化，避免了路径空间离散的维数爆炸。
- 质量变化扩展对单细胞很重要，因为它显式考虑了分支和细胞消失场景。

## 局限与未解点

- 即便是 grid-free，优化仍然偏重，面对超高维 latent 和很多时间点时成本不低。
- 方法更偏优化器，而非通用神经动力学模型。
- 对条件化、干预和多样本迁移的支持不够直接。

## 为什么能发顶会 / 为什么是重要理论工作

- 它把一条非常数学化的 trajectory inference 路线真正做成了可计算算法，而且带有收敛保证。
- 在单细胞轨迹推断里，这类“理论与算法闭环”工作很少，所以很有分量。

## 对单细胞轨迹建模和 CytoBridge-agent 自动设计算法的启发

- CytoBridge 不必只做 neural field fitting，也可以吸收 path-space optimization 的思想。
- 对 agent 来说，这篇文章提醒一个关键点: 有时候真正的创新不在于换模型，而在于把原本不可算的理论目标变成可算求解器。
- 若后续要做多粒子近似或群体层轨迹优化，这篇文章是非常直接的参考。
