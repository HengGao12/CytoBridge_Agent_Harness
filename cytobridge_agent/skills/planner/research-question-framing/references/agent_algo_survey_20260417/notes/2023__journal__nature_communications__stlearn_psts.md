# Robust mapping of spatiotemporal trajectories and cell-cell interactions in healthy and diseased tissues

## Metadata
- 标题: Robust mapping of spatiotemporal trajectories and cell-cell interactions in healthy and diseased tissues
- 作者: Duy Pham et al.
- 年份: 2023
- 正式 venue: Nature Communications
- PDF 文件名: Robust mapping of spatiotemporal trajectories and cell–cell interactions in healthy and diseased tissues.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
这篇文章聚焦于空间组织中的动态变化过程，想同时回答“组织内部的转录状态如何沿空间展开成 trajectory”以及“哪些区域存在更强的 cell-cell interaction”。作者在神经发育、损伤修复和肿瘤进展等场景中展示了这两个问题可以联动分析。

## 数据类型 / 场景
输入是 spatial transcriptomics，可联合 gene expression、物理距离和 tissue morphology 三类信息。适合需要同时分析空间轨迹、交互作用和缺失值问题的组织数据。

## 核心算法怎么设计
作者提出了三个互补模块，核心与本任务最相关的是 PSTS。PSTS 用 spatial graph 在组织内建模 pseudo-time-space，恢复跨组织区域的状态变化关系；SCTP 用空间约束的双层 permutation 检验筛选更可信的 ligand-receptor interaction；stSME 用神经网络加图信息做空间表达修复。

## 设计原则是什么
设计原则是把空间转录组看成“多模态结构数据”，而不是只有表达矩阵。轨迹、交互和补全三件事都围绕同一个空间图展开，说明作者追求的是一套统一的空间分析语义，而非若干独立工具拼接。

## 工程优化 / 训练技巧 / pipeline 设计
把 trajectory、CCI 和 imputation 一并放进 stLearn 软件包，本身就是很强的工程设计，降低了用户在多个工具间来回切换的成本。SCTP 通过空间约束减少假阳性，stSME 则用表达、距离和形态联合补全，提高后续 PSTS/CCI 的稳定性。

## 局限性
PSTS 恢复的是 pseudo-time-space，而不是严格的时间动力学；在空间结构复杂但真正时间顺序不清晰时，解释仍需谨慎。方法也依赖空间图和形态特征质量，对组织切片质量、配准误差和过度平滑较敏感。

## 为什么能发到这个级别
它抓住了空间组学领域一个很有代表性的痛点：分析流程碎片化。作者把空间轨迹、空间交互和空间补全整合成一个相互增强的框架，并在多种疾病和发育场景里验证，因此方法完整度较高。

## 对 CytoBridge-agent 自动设计算法的启发
CytoBridge 若扩展到空间场景，不能只把空间坐标当 covariate；更合理的做法是围绕同一个空间图同时组织 trajectory、interaction 和 denoising。另一个启发是：interaction 模块与 trajectory 模块最好共享底层表示，而不是彼此孤立。
