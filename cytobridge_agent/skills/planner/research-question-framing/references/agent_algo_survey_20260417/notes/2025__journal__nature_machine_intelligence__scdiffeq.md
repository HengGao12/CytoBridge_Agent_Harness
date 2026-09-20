# Learning cell dynamics with neural differential equations

## Metadata
- 标题: Learning cell dynamics with neural differential equations
- 作者: Michael E. Vinyard et al.
- 年份: 2025
- 正式 venue: Nature Machine Intelligence
- PDF 文件名: Learning cell dynamics with neural differential equations.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
scDiffEq主要解决的是现有 drift-diffusion 动力学模型常把 diffusion 设成常数，因而无法描述状态依赖的随机性和命运决策点附近的噪声结构。作者想同时恢复 deterministic drift 和 state-dependent diffusion，并用于 fate prediction 与 perturbation simulation。

## 数据类型 / 场景
输入既可以是带 lineage tracing 的多时间点数据，也可以推广到单时间点 snapshot 数据。文中重点验证了造血系统和 CRISPR perturbation 的重建能力。

## 核心算法怎么设计
scDiffEq学习 neural SDE，把细胞动力学写成 drift-diffusion process，并显式分解每个细胞沿轨迹的 drift 与 diffusion。模型利用 lineage-traced data 监督 fate prediction，同时还能在单时间点数据上生成高分辨率 developmental trajectories。

## 设计原则是什么
最重要的原则是：扩散不是常数噪声，而是具有状态依赖结构的生物信号。作者认为如果不显式建模 diffusion，就无法正确解释 fate decision 附近的随机性，也无法准确模拟 perturbation 后的轨迹分岔。

## 工程优化 / 训练技巧 / pipeline 设计
文章做了 drift/diffusion 比例的系统扫描，并将 fate prediction、unseen population reconstruction 和 TF-level correlation 一起作为评估指标。它还强调 batch size 与 dataset size 可独立调节，便于在不同算力环境下训练。

## 局限性
neural SDE 训练仍然比简单 OT 或 Markov 方法更重，且 diffusion 的解释仍建立在所选 latent representation 上。虽然论文展示了强结果，但对更多组织和更强分布移位下的稳定性还需继续验证。

## 为什么能发到这个级别
这篇文章之所以重要，是因为它把单细胞动力学从“只学漂移”推进到了“显式分解漂移与扩散”，并证明 diffusion 在命运决策里不是噪声，而是信息。这个问题定义本身就足够新，而且实验也较完整。

## 对 CytoBridge-agent 自动设计算法的启发
CytoBridge 的新算法不应再把 stochasticity 当 residual error，而应把它视作要被主动建模的对象。特别是在 fate bifurcation 和 perturbation prediction 上，state-dependent diffusion 很可能是关键增益点。
