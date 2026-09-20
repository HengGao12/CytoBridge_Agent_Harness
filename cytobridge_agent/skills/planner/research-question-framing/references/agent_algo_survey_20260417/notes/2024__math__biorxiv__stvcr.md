# stVCR: Reconstructing spatio-temporal dynamics of cell development using optimal transport

## Metadata
- 标题: stVCR: Reconstructing spatio-temporal dynamics of cell development using optimal transport
- 作者: Qiangwei Peng, Peijie Zhou, Tiejun Li
- 年份: 2024
- 正式 venue: bioRxiv preprint doi:10.1101/2024.06.02.596937
- PDF 文件名: stVCR Reconstructing spatio-temporal dynamics of cell development using optimal transport.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否，当前按 bioRxiv 预印本处理
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么已有 gap
这篇工作解决的是时间序列空间转录组里一个很具体但很难的问题：过去方法通常只能重建分化或增长，难以同时处理分化、增长和物理空间迁移；更麻烦的是，不同时间点的空间坐标往往不在同一坐标系中，直接做 OT 会失真。

## 理论 / 算法创新点
stVCR 把 dynamical OT、unbalanced setting 和 rigid-body transformation invariance 组合起来，形成一个同时对 gene-expression dynamics 和 spatial migration 建模的框架。它把“空间对齐”不再视为预处理，而是直接放进动力学重建问题里端到端学习。

## 具体算法或理论结构怎么设计
作者对不同模态采用不同 OT 建模：对基因表达使用通常的 Wasserstein OT，对空间坐标使用刚体变换不变的 OT；同时引入 unbalanced 机制处理 cell division/apoptosis。这样模型既能重建细胞状态转移，也能重建空间迁移，并分析表达、空间位置和增长之间的耦合关系。

## 设计原则是什么
设计原则非常值得借鉴：不同数据子空间的几何和不变量不同，就不该用单一 cost 一把梭。stVCR 明确区分 gene space 和 physical space 的 transport 几何，再用共享动力学把它们耦合起来，这比先拼接特征再统一 OT 更合理。

## 工程优化或实现性考虑
工程上它最大的优点是 end-to-end，把对齐、迁移、增长和分化放在同一条计算图里。对 axolotl brain regeneration 的实验说明，这种联合建模比先对齐后推断更有希望恢复真实空间动态。

## 局限与未解点
它默认跨时间点的空间差异主要能由 rigid transformation 吸收，这在形变复杂的组织里可能不够。文章也没有显式建模 cell-cell interaction，只是分析表达和迁移、增长之间的关系。作为预印本，其稳定性和泛化性还需要更多验证。

## 为什么能发顶会 / 为什么是重要理论工作
它重要的地方在于把“多空间、多机制的单细胞动力学”做成了一个统一 OT 问题，而不是简单在已有 scRNA-seq 方法上外挂空间坐标。对于时空单细胞方向，这是非常直接也很难回避的建模升级。

## 对单细胞轨迹建模和 CytoBridge-agent 自动设计算法的启发
stVCR 对 CytoBridge 的关键启发是：当数据来自多个空间时，算法设计应该允许每个空间拥有自己的 transport 几何和不变量，再通过共享潜动力学耦合它们。自动化设计算法时，agent 应该把“单空间统一建模”与“多空间异构 cost 联合建模”作为明确可比较的设计分支。
