# WFR-FM: Simulation-Free Dynamic Unbalanced Optimal Transport

## Metadata
- 标题: WFR-FM: Simulation-Free Dynamic Unbalanced Optimal Transport
- 作者: Qiangwei Peng, Zihan Wang, Junda Ying, Yuhao Sun, Qing Nie, Lei Zhang, Tiejun Li, Peijie Zhou
- 年份: 2026
- 正式 venue: arXiv preprint arXiv:2601.06810
- PDF 文件名: WFR-FM Simulation-Free Dynamic Unbalanced Optimal Transport.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否，当前按预印本处理
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么已有 gap
flow matching 已经让 balanced transport 的训练变得 simulation-free，但对 unbalanced dynamics 来说，主流方法仍多依赖 ODE/SDE 仿真或较重的变分优化。对于包含增殖、凋亡的单细胞群体，这正是瓶颈所在。

## 理论 / 算法创新点
WFR-FM 直接把 flow matching 推广到 WFR 动态 unbalanced OT。它同时回归 transport velocity 和 growth rate，并证明当损失最小时，模型恢复的正是 WFR 度量下的 dynamic unbalanced optimal transport。也就是说，它把 unbalanced OT、WFR geometry 和 simulation-free regression 串成了闭环。

## 具体算法或理论结构怎么设计
文章先分析 WFR 度量下 Dirac measures 的 geodesic 闭式形式，再据此构造 conditional path。训练时网络同时预测速度场和质量增长/衰减项，通过回归式损失逼近这条条件路径对应的真实动力学。整个过程不需要内环 ODE integration，因此比常规动态 UOT 更轻。

## 设计原则是什么
设计原则和 OT-CFM 一脉相承，但更适合生物数据：既然 unbalanced dynamics 的关键变量有两个，就同时学 velocity 和 growth，而不是把 growth 当作附加后处理。更深一层的原则是，训练目标要尽量贴住底层几何本身，而不是只在经验上模仿轨迹。

## 工程优化或实现性考虑
工程优势非常明显：simulation-free、可扩展、适合多时间点群体快照。文章还给出理论保证，这让它不仅是一个好用 trick，而是一个 geometry-aligned 的训练框架。对单细胞这种样本量大、时间点稀疏且质量不守恒明显的任务，很有现实意义。

## 局限与未解点
它仍然主要处理 transport + birth/death，没有把 cell-cell interaction 纳入核心。WFR 度量中的超参数会影响 transport 与 growth 的权衡，如何在真实生物数据上稳健设定仍是难点。作为预印本，长期泛化和与更复杂 reference process 的结合还没完全展开。

## 为什么能发顶会 / 为什么是重要理论工作
它之所以重要，是因为它几乎正中当前单细胞 OT 路线的核心矛盾：我们既想要 unbalanced 的表达力，又想要 flow matching 的训练效率。WFR-FM 给出的正是这两者的直接结合，而且不是拍脑袋拼接，而是从 WFR 几何推导出来的。

## 对单细胞轨迹建模和 CytoBridge-agent 自动设计算法的启发
对 CytoBridge 来说，WFR-FM 是非常强的候选母框架。它提示 agent 在自动设计算法时，应优先探索“simulation-free velocity + growth joint regression”这一方向，再根据需要外挂 interaction、multi-space 或 latent-state 模块。若要做下一代方法，这很可能比继续加重 path-space 优化更有前景。
