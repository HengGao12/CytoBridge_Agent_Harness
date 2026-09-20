# Neural Optimal Transport with Lagrangian Costs

## Metadata
- 标题: Neural Optimal Transport with Lagrangian Costs
- 作者: Alexander Tong, Hanyang Zhao, Brandon Amos, et al.
- 年份: 2024
- 正式 venue: arXiv preprint arXiv:2406.00288
- PDF 文件名: Neural Optimal Transport with Lagrangian Costs.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否，当前按预印本处理
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么已有 gap
标准 OT 默认 ground cost 很简单，通常是欧氏距离或其轻微变体。但在很多真实系统里，粒子运动受势场、障碍、曲面几何或动力学约束影响，直接用欧氏 cost 会错配真实 transport 路径。单细胞轨迹推断里，如果细胞状态空间带有明显几何/能垒结构，这个问题同样存在。

## 理论 / 算法创新点
文章把 least-action / Lagrangian cost 引入神经 OT，研究的是“底层 cost 来自一条最小作用量路径”时如何学习 OT map 和对应 geodesic。它不再把 cost 当成静态距离矩阵，而是把 cost 本身视作一条受动力学约束的路径优化结果。

## 具体算法或理论结构怎么设计
方法把轨迹参数化为样条路径，再用神经网络摊销求解两类难点：一是给定端点时的最小作用量路径，二是 Lagrangian cost 下的 c-transform。这样既能输出 transport map，也能输出受几何约束的实际路径，而不是只给一个耦合矩阵。

## 设计原则是什么
核心设计原则是“先把系统物理/几何约束写进 cost，再做 transport”，而不是先用一个通用 OT 解再事后解释。文章强调如果先验几何很重要，ground cost 就不应固定成欧氏度量，而应允许由势场、位置相关度量甚至流形结构决定。

## 工程优化或实现性考虑
工程上的关键点是 amortized optimization。最小作用量路径和 c-transform 原本都很贵，文章通过参数共享把这两个过程摊销成可复用模块，因此能在多样本上重复调用。示例里还展示了障碍物环境、圆形几何等非标准情形，说明方法的表达力确实比普通 OT 强。

## 局限与未解点
它本质上还是静态 OT，不直接解决多时间点 snapshot 的连续动力学学习，也没有 unbalanced growth 或 interaction 机制。另一方面，Lagrangian 设计本身就引入了新的建模责任：如果先验势场或几何设错，反而会比简单 OT 更糟。

## 为什么能发顶会 / 为什么是重要理论工作
它的重要性在于把“学 transport”从纯几何匹配推进到“带动力学先验的 transport”。对很多应用来说，这比继续在欧氏 OT 上做小修小补更关键，因为它直接决定 coupling 是否具备机制意义。

## 对单细胞轨迹建模和 CytoBridge-agent 自动设计算法的启发
对 CytoBridge，这篇文章的启发不是照搬样条求解器，而是要把“先验生物机制写进 cost/行动量”当成一等设计选项。比如可以让 transport 更偏向沿分化轴移动、避开不可达区域、或惩罚不合理跨谱系跳跃。自动设计算法时，agent 不应只搜索网络结构，还应搜索更合理的 action/cost 形式。
