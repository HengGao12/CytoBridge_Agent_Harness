# Entropic regularisation of unbalanced optimal transportation problems

## Metadata
- 标题: Entropic regularisation of unbalanced optimal transportation problems
- 作者: M. Buze, M. H. Duong
- 年份: 2023
- 正式 venue: 当前本地版本为 arXiv 预印本
- PDF 文件名: Entropic regularisation of unbalanced optimal transportation problems.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 当前未确认正式追认版本
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成摘要、引言与理论结构阅读

## 解决了什么已有 gap

- UOT 里“熵正则化”经常被当成一个统一概念使用，但实际上不同加法位置会得到不同问题。
- 缺少一套系统理论来比较这些 regularization 方案的等价关系、差异与动态后果。

## 理论 / 算法创新点

- 文章指出 UOT 存在两类本质不同的 entropic regularization: 在原空间上加，或在扩展空间上加。
- 提供多种 reformulation，并提出 regularized induced marginal perspective cost 这一分析工具。
- 给出扩展空间正则向未正则 UOT 收敛的结果，并论证为什么 extended-space regularization 往往更可取。

## 具体算法或理论结构怎么设计

- 先回顾 unregularized UOT 的静态原始/对偶/等价形式。
- 再分别对 original-space 与 extended-space 两种正则方案建立等价重写。
- 最后讨论它们向动态形式的翻译及相互比较。

## 设计原则是什么

- 在 UOT 里，正则不只是数值平滑，放置位置会改变问题本身。
- 因此算法设计时必须先明确“你到底在正则哪个对象”，而不能笼统说用了 entropic UOT。
- 对 lifted/extended formulation 的偏好应有理论依据，而不是纯经验。

## 工程优化或实现性考虑

- 文章本身偏理论，但对工程上很关键的一点是：某些 regularization 形式更适合后续动态/算法实现。
- 对需要稳定数值的 UOT 神经方法，这种 formulation-level 的选择往往比网络细节更重要。

## 局限与未解点

- 本文是较初步版本，实验和算法层内容有限。
- 更多停留在静态 UOT 理论，对完整动态桥求解没有给出成熟神经框架。

## 为什么能发顶会 / 为什么是重要理论工作

- 虽然不是顶会文章，但它对 UOT 目标函数设计的澄清非常关键。
- 在这个方向上，很多后续方法容易混淆 regularization 语义，这篇工作正好补这个洞。

## 对单细胞轨迹建模和 CytoBridge-agent 自动设计算法的启发

- CytoBridge 一旦要做 entropic UOT，就必须显式记录采用的是哪一种正则化语义。
- agent 自动设计算法时，正则放置位置应成为配置空间的一部分，而不是固定写死。
- 这篇文章也支持一个更高层的结论: 好算法经常赢在“问题写对”，而不是“优化器调得更猛”。
