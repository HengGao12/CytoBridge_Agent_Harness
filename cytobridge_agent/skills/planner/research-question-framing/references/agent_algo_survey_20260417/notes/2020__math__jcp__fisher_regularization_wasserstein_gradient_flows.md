# Fisher information regularization schemes for Wasserstein gradient flows

## Metadata
- 标题: Fisher information regularization schemes for Wasserstein gradient flows
- 作者: Wuchen Li, Jianfeng Lu, Li Wang
- 年份: 2020
- 正式 venue: Journal of Computational Physics, 2020
- PDF 文件名: Fisher information regularization schemes for Wasserstein gradient flows.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成摘要、引言与方法部分阅读

## 解决了什么已有 gap

- 经典 JKO/Wasserstein gradient flow 数值离散在实际求解时常常非凸、难优化，还容易带来非负性和稳定性问题。
- 对很多 Fokker-Planck/聚集扩散类方程，单纯沿 Benamou-Brenier 形式做离散，计算代价也偏高。

## 理论 / 算法创新点

- 在 JKO 框架里加入 Fisher information 正则，得到一种与 Schrödinger bridge 密切相关的结构化正则方案。
- 该正则项改善了离散子问题的凸性，并自动保持解的非负性。
- 作者进一步证明底层动态公式不需要额外时间插值，因此能明显降维。

## 具体算法或理论结构怎么设计

- 以 Wasserstein-2 gradient flow 的离散时间步为主线，把每一步写成 Benamou-Brenier 动态最优化。
- 在动作项外再加 Fisher information 正则，这个正则可由 energy splitting 推出，并可解释为 SB 风格的噪声化。
- 数值上用 sequential quadratic programming 求每一步子问题，从而稳定求解一系列质量守恒 PDE。

## 设计原则是什么

- 正则化要保留原动力学结构，而不是纯粹为了好优化随意加平滑项。
- 数值方法要优先保证非负性、质量守恒和能量耗散等“物理正确性”。
- 如果某个正则与 SB/熵结构天然一致，它往往比 ad hoc smoothing 更有解释力。

## 工程优化或实现性考虑

- 最大的工程点是“不需要额外时间插值”，直接减少了动态离散的维度。
- SQP 方案比直接处理原始非凸问题更稳定，也更适合高精度 PDE 数值求解。
- 文章在 porous media、nonlinear Fokker-Planck、aggregation-diffusion 等例子上展示了稳定性。

## 局限与未解点

- 主要是 PDE 数值求解框架，不是直接从快照数据学习动力学的统计模型。
- 仍然基于质量守恒 W2 gradient flow，和单细胞里常见的增殖/死亡问题有距离。
- Fisher 正则会引入偏置，在学习式模型中未必总是最优。

## 为什么能发顶会 / 为什么是重要理论工作

- 它把 JKO、Benamou-Brenier 和 Schrödinger bridge 的联系转成了可运行的稳定数值方案。
- 对 Wasserstein PDE 数值分析来说，既有理论结构又有明确计算收益，这是很典型的高质量计算数学工作。

## 对单细胞轨迹建模和 CytoBridge-agent 自动设计算法的启发

- 如果 CytoBridge 后续要显式求解 Fokker-Planck 或 density evolution 子问题，可以优先考虑 Fisher/score 类结构正则，而不是只靠经验平滑。
- 这篇文章提示 agent 在自动设计算法时要把“优化友好性”和“动力学结构保真”绑在一起考虑。
- 对神经桥模型来说，Fisher regularization 也可以被理解为稳定训练和保持分布正性的一个候选模块。
