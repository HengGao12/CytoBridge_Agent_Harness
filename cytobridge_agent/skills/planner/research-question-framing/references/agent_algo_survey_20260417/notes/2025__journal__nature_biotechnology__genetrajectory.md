# Gene trajectory inference for single-cell data by optimal transport metrics

## Metadata
- 标题: Gene trajectory inference for single-cell data by optimal transport metrics
- 作者: Rihao Qu et al.
- 年份: 2025
- 正式 venue: Nature Biotechnology
- PDF 文件名: Gene trajectory inference for single-cell data by optimal transport metrics.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
GeneTrajectory解决的是并行生物过程同时存在时，cell trajectory 常把多个过程揉在一起，导致关键 gene program 顺序被遮蔽的问题。它把焦点从“细胞轨迹”转向“基因轨迹”，试图直接恢复过程相关基因程序的先后顺序。

## 数据类型 / 场景
输入是 scRNA-seq 数据，适用于存在多个并行过程的系统，如分化叠加 cell cycle 或其他背景程序。作者展示了在髓系成熟和毛囊 dermal condensate differentiation 中的收益。

## 核心算法怎么设计
作者先构建保留细胞流形结构的 cell graph，再把每个基因的表达视作定义在该图上的概率分布。随后计算 gene-gene graph-based Wasserstein distance，构建 gene affinity graph，提取 gene programs，并为每个 program 定义 gene pseudotemporal order。

## 设计原则是什么
这篇文章最重要的原则是：当多个过程叠加时，“对细胞排序”不一定是正确基本单位，“对基因程序排序”可能更自然。作者用 OT 的几何优势来比较基因分布，而不是继续在 cell pseudotime 上强行解读。

## 工程优化 / 训练技巧 / pipeline 设计
为了降低计算成本，文章用了 coarse-grained cell graph 和 sparsified gene affinity graph。它还明确讨论了 branch identification 需要交互式优化，这说明作者意识到纯自动 pipeline 在 gene trajectory 上仍有难点。

## 局限性
讨论部分很诚实地指出，方法不能自动判断每条 gene trajectory 的方向，还需要借助已知 early/late markers。对多过程叠加、一个基因参与多个过程的情况，gene embedding 也可能把 informative genes 和 uninformative genes 混在一起。

## 为什么能发到这个级别
这篇工作胜在重新定义问题而不是只优化老指标。它说明 OT 不仅能做 cell-state transport，也能用来在 gene program 层面恢复动态顺序，这种“换基本对象”的创新很容易打动顶级方法期刊。

## 对 CytoBridge-agent 自动设计算法的启发
CytoBridge 的 driver gene 模块可以从单纯相关性分析升级到“gene program dynamics”分析。对自动算法设计而言，必要时应主动改变建模对象，而不是默认所有问题都要在细胞层面求解。
