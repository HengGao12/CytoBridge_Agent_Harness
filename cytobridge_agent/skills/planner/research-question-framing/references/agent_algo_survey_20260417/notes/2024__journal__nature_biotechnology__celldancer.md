# A relay velocity model infers cell-dependent RNA velocity

## Metadata
- 标题: A relay velocity model infers cell-dependent RNA velocity
- 作者: Shengyu Li et al.
- 年份: 2024
- 正式 venue: Nature Biotechnology
- PDF 文件名: A relay velocity model infers cell-dependent RNA velocity.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
cellDancer要解决的是传统 RNA velocity 采用全局统一动力学参数时，在多阶段、多分支和异质系统里经常失真。它把问题重新定义为：每个细胞都应有本地、细胞依赖的速度和动力学参数。

## 数据类型 / 场景
输入是带 spliced/unspliced 信息的 scRNA-seq，特别适合多 lineage、动力学速率异质、dropout 较重的复杂系统。作者在模拟和多个真实发育/疾病数据上做了验证。

## 核心算法怎么设计
文章提出 relay velocity model，并实现为 cellDancer。对每个基因单独训练一个 DNN，使每个细胞的速度由邻域细胞的局部动力学“接力”传递而来，从而估计 cell-specific transcription、splicing、degradation 参数，并进一步恢复 velocity、pseudotime 和 driver genes。

## 设计原则是什么
核心原则是局部性而不是全局统一速率。作者认为真正需要拟合的不是一个全局 phase portrait，而是沿细胞邻域连续变化的局部动力学场；这种设计天然更适合多速率和多分支场景。

## 工程优化 / 训练技巧 / pipeline 设计
cellDancer按基因并行训练，并支持 multiprocessing，在大数据上把总运行时间从数小时压到几十分钟。对每个基因单独建模虽然开销不小，但换来了在多速率 regime 下更稳的拟合。

## 局限性
方法仍然依赖 RNA velocity 观测前提，且每个基因都要单独训练网络，计算成本和调参负担不低。它擅长恢复局部动力学，但未直接给出全局连续生成模型，对长期外推和分布预测的支持有限。

## 为什么能发到这个级别
这篇工作的价值在于把 RNA velocity 的“全局常速率”瓶颈正面拆掉，并给出一个在模拟和真实多分支系统上都成立的深度学习替代方案。它既提升了准确性，又明确展示了何时传统 velocity 假设会失败。

## 对 CytoBridge-agent 自动设计算法的启发
CytoBridge 若继续做 velocity/transition 子模块，应该优先考虑 state-dependent 或 cell-dependent kinetics，而不是全局共享参数。更进一步，局部动力学和全局 transport 可以设计成分层组合，而不必强迫一个模型同时承担全部尺度。
