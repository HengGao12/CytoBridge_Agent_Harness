# Modeling single cell trajectory using forward-backward stochastic differential equations

## Metadata
- 标题: Modeling single cell trajectory using forward-backward stochastic differential equations
- 作者: Kai Zhang et al.
- 年份: 2024
- 正式 venue: PLOS Computational Biology
- PDF 文件名: Modeling single cell trajectory using forward-backward stochastic differential equations.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
这篇文章要解决的是 OT 只能给端点之间静态耦合、很难表达非线性路径形状的问题。作者希望用连续时间的随机微分方程直接建模单细胞发育轨迹，并在多个数据集上优于 Waddington-OT 和 TrajectoryNet。

## 数据类型 / 场景
输入是 scRNA-seq snapshot/time-course 数据。文中以多个公开分化数据集做 benchmark，重点比较复杂轨迹和有限观测下的恢复质量。

## 核心算法怎么设计
作者提出 FBSDE，用 forward SDE 与 backward SDE 交替逼近同一条满足起点与终点分布约束的随机轨迹。该过程还引入 refined approximation 和 mean-field 概念，以更好地贴合真实轨迹并模仿细胞间相互作用效应。

## 设计原则是什么
设计原则是：如果目标是恢复连续轨迹，就不应满足于静态 OT pairing，而应显式学习 time-continuous stochastic process。forward-backward 联立的好处是把“初始分布约束”和“终点分布约束”同时纳入求解，而不是只顾单向拟合。

## 工程优化 / 训练技巧 / pipeline 设计
作者用交替迭代的方式让 forward/backward SDE 收敛到共同解，并加入 penalty/approximation 改善终端约束满足度。整个方法围绕与 WOT、TrajectoryNet 的对比来设计实验，也让收益点很清楚。

## 局限性
尽管引入了 mean-field 概念，方法的交互建模仍较弱，而且训练 FBSDE 的数值复杂度和稳定性都不轻松。它在少数数据集上效果很好，但泛化到更大规模、更复杂多模态场景还需要进一步验证。

## 为什么能发到这个级别
它把 trajectory inference 从“静态匹配”推进到“连续随机过程求解”，并在 benchmark 上清楚展示了非线性轨迹建模的优势。对单细胞动力学建模而言，这是一条颇有数学味道但又落地的路线。

## 对 CytoBridge-agent 自动设计算法的启发
CytoBridge 的一个直接启发是：forward-backward 结构非常适合做 bridge 型动态推断，尤其是需要同时满足起终点分布约束时。另一个启发是，若想建模交互，不要只在 loss 上加小修补，而应在过程层面显式写进 dynamics。
