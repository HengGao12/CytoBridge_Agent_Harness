# Trajectory Inference with Cell-Cell Interactions (TICCI): intercellular communication improves the accuracy of trajectory inference methods

## Metadata
- 标题: Trajectory Inference with Cell-Cell Interactions (TICCI): intercellular communication improves the accuracy of trajectory inference methods
- 作者: Yifeng Fu et al.
- 年份: 2025
- 正式 venue: Bioinformatics
- PDF 文件名: Trajectory Inference with Cell–Cell Interactions (TICCI) intercellular communication improves the accuracy of trajectory inference methods.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
TICCI要解决的是现有 trajectory inference 大多只看细胞内部表达相似性，忽略发育过程中 intercellular communication 对命运变化的影响。作者希望用 CCI 信息来提高 branch identification 和 temporal ordering 的准确性。

## 数据类型 / 场景
输入是 scRNA-seq 数据，并结合单细胞分辨率的细胞间通信估计。适用于发展过程里细胞间信号对轨迹方向有明显影响的系统。

## 核心算法怎么设计
TICCI先构建 cell-neighborhood matrix，其中边权同时包含表达相似性与 CCI 信息；再用 Louvain partition 找分支、用 scEntropy 估计分化状态，并通过 Chu-Liu 算法建立有向最小二乘模型识别 trajectory branches，最后用改进的 diffusion fitted time 处理非连通拓扑下的 fitted time。

## 设计原则是什么
设计原则是把 cell-cell interaction 从下游注释前移到 trajectory graph construction 本身。作者认为如果轨迹是由信号交换驱动的，那么图的边权就不应只由表达距离决定。

## 工程优化 / 训练技巧 / pipeline 设计
方法把 branch detection、state scoring 和 fitted time 组织成一条清晰 pipeline，避免用户分别选多种工具。把 CCI 直接整合进邻域图，也是一个很实用的工程设计，因为这一步最直接影响后续所有拓扑结果。

## 局限性
TICCI 的收益高度依赖 CCI 估计本身的质量，而单细胞层面的通信推断仍然噪声较大。它依旧是图算法管线，不是连续生成动力学模型，因此对长时程外推和分布预测的支持有限。

## 为什么能发到这个级别
这篇文章的价值在于提出了一个被很多 TI 方法忽略但非常合理的问题：如果 intercellular communication 真会改变轨迹，那么为什么轨迹图不显式用它。这个问题切得准确，也有可复用的算法实现。

## 对 CytoBridge-agent 自动设计算法的启发
CytoBridge 的 interaction 模块不必停留在“分析结果”层面，可以前移成 trajectory inference 的结构先验。未来设计新算法时，值得探索 interaction-aware transport 或 interaction-aware bridge，而不只是独立的通信分析。
