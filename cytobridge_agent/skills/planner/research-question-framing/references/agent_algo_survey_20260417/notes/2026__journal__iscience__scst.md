# Constructing the spatiotemporal atlas of single-cell lineage trajectories in stereotypic biological structures

## Metadata
- 标题: Constructing the spatiotemporal atlas of single-cell lineage trajectories in stereotypic biological structures
- 作者: Ran Wang et al.
- 年份: 2026
- 正式 venue: iScience
- PDF 文件名: Constructing the spatiotemporal atlas of single-cell lineage trajectories in stereotypic biological structures.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
SCST 的目标是从发育中具有固定形态结构的样本里，重建单细胞分辨率的 3D 时空图谱，并进一步 delineate lineage trajectories。它面向的问题不是一般组织整合，而是发育生物学里“空间位置、时间阶段和谱系关系如何统一恢复”。

## 数据类型 / 场景
输入包括 spatial reference transcriptome、scRNA-seq 和时间序列发育样本。文章主要以小鼠原肠胚形成过程为主线，构建 3D spatiotemporal atlas。

## 核心算法怎么设计
SCST workflow 由五步组成：先建立空间坐标系，再用 MDSC Mapping V2 把单细胞映射到 3D 参考空间，随后做 3D 建模、Gradient Sort 优化局部分布，并用 Optimal Spatial Distribution 算法得到单细胞最优坐标。最终再配合 Digital Lineage Tracing 分析时序样本，恢复 spatiotemporal lineage trajectory。

## 设计原则是什么
设计原则是先可靠地恢复单细胞空间位置，再谈谱系与轨迹，因为空间上下文本身就是命运决定的重要约束。另一个原则是将几何建模、平滑映射和时间整合分层处理，而不是一次性端到端黑盒拟合。

## 工程优化 / 训练技巧 / pipeline 设计
MDSC Mapping V2 结合 spatial smoothing 明显提高了 mapping fidelity，后续再用 Gradient Sort 和 Optimal Spatial Distribution 细化 3D 坐标，是很典型的多阶段工程 pipeline。作者还用 RNAscope 等实验结果校验空间注释。

## 局限性
该方法依赖高质量空间参考和相对稳定的几何结构，因此更适合“stereotypic biological structures”，不一定能直接迁移到形态高度可变的组织。它的 pipeline 也较长，误差可能逐层传递。

## 为什么能发到这个级别
这篇文章的亮点是把空间单细胞映射从二维/局部提升到三维、时序和谱系统一重建，并给出了较完整的发育案例。虽然不是顶刊级方法，但在发育时空图谱领域是很扎实的一步。

## 对 CytoBridge-agent 自动设计算法的启发
CytoBridge 若要做时空单细胞建模，固定几何结构系统值得单独设计专门算法。并不是所有问题都需要端到端黑盒模型，分层的 mapping + trajectory pipeline 在某些发育场景中可能更可靠。
