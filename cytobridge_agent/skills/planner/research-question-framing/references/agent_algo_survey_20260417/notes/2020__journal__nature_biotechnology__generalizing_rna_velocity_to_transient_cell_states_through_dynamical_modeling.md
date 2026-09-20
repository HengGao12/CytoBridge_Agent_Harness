# Generalizing RNA velocity to transient cell states through dynamical modeling

## Metadata
- 标题: Generalizing RNA velocity to transient cell states through dynamical modeling
- 作者: Volker Bergen, Marius Lange, Stefan Peidli, F. Alexander Wolf, Fabian J. Theis
- 年份: 2020
- 正式 venue: Nature Biotechnology
- PDF 文件名: Generalizing RNA velocity to transient cell states through dynamical modeling.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已读摘要、结果、讨论

## 解决了什么生物问题

它要解决的是原始 RNA velocity 在 transient cell state 和 heterogeneous subpopulation kinetics 下经常失效的问题。生物学上它希望更可靠地恢复发育过程中的 lineage direction、latent time 和 driver genes，尤其是在真正非稳态的系统里。

## 数据类型 / 场景

- 含 spliced/unspliced 信息的 scRNA-seq。
- 适用于发育和扰动响应中普遍存在的 transient states。
- 代表性应用包括 dentate gyrus neurogenesis 和 pancreatic endocrinogenesis。

## 核心算法怎么设计

- 不再只拟合 steady-state slope，而是直接求解每个基因的完整转录-剪接-降解 ODE。
- 对每个基因估计转录率、剪接率、降解率，并给每个细胞赋 latent time 与转录状态。
- 用 EM 交替优化：E 步把细胞映到基因相位轨迹上，M 步更新 kinetics 参数。
- 再把基因级 latent times 耦合成一个共享的 universal latent time，作为细胞内部时钟。
- 这样得到的 velocity 不再依赖“观察到完整 steady state”或“各基因共享同一剪接率”的强假设。

## 设计原则是什么

- 当一个有用方法的核心假设被系统性违反时，不该继续打补丁，而应回到机制模型层面重建。
- 仍保持生物可解释性：参数对应转录、剪接、降解，不是纯黑箱。
- velocity 不只是方向箭头，还应该给出 latent time 和 driver genes 这类更高层可解释量。
- 用跨基因共享时间把单基因噪声压下去，是很强的结构性设计。

## 工程优化 / 训练技巧 / pipeline 设计

- 论文提到用近似时间赋值实现约 30 倍加速，这是使方法真正可用的关键工程点。
- 只对显示出清晰动力学的基因求可靠收敛，避免把所有基因都强行纳入。
- 与现有 scVelo 工作流和可视化生态深度整合，降低了落地门槛。

## 局限性

- 尽管比原始 velocity 强很多，仍然假设基因特异的剪接和降解率在模型内是常数，并主要使用 on/off 两态转录框架。
- 更复杂的调控、空间依赖和蛋白翻译层尚未纳入。
- 速度和 latent time 仍主要解决局部动态，不等价于分布级别的 transport 或跨条件反事实推断。

## 为什么能发到这个级别

因为它不是边缘改进，而是对一个极具影响力的方法做了关键纠偏。RNA velocity 2018 年后被大量使用，但其假设问题也很快暴露。scVelo 给出了机制上更合理、工程上又跑得动的升级版，因此是高影响的“第二代标准方法”。

## 对 CytoBridge-agent 自动设计算法的启发

- 自动设计算法时，必须显式检查模型假设失效的征兆，并能升级到更一般的机制模型。
- latent time 是非常值得保留的中间变量，它既能增强可解释性，也能帮助后续生成模型。
- 对 CytoBridge 来说，velocity 类模块若要长期可靠，最好默认采用能处理 transient dynamics 的版本，而不是停留在最初的 steady-state 近似。
