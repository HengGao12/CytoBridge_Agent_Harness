# Efficient trajectory inference in wasserstein space using consecutive averaging

## Metadata
- 标题: Efficient trajectory inference in wasserstein space using consecutive averaging
- 作者: Amartya Banerjee, Harlin Lee, Nir Sharon, Caroline Moosmüller
- 年份: 2024
- 正式 venue: arXiv preprint arXiv:2405.19679
- PDF 文件名: Efficient trajectory inference in wasserstein space using consecutive averaging.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否，当前按预印本处理
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么已有 gap
这篇文章针对的是“只有离散时间点的点云快照，如何在 Wasserstein 几何里恢复连续轨迹”的问题。此前一类方法只做相邻时间点的 OT 线性插值，轨迹不够平滑；另一类神经 ODE 方法能学连续动力学，但训练不稳定、对 stiff dynamics 敏感，也不天然尊重 Wasserstein 空间的内在几何。

## 理论 / 算法创新点
核心创新是把 Lane-Riesenfeld subdivision 和 OT geodesic 结合起来，提出在 Wasserstein 空间里做 consecutive averaging 的 WLR 算法，并进一步给出 4-point 插值方案。它把样条构造从欧氏空间推广到 Wasserstein 空间，给出了在无限维 Wasserstein 空间中的收敛性分析，而不只是经验性插值。

## 具体算法或理论结构怎么设计
做法是先把每个时间点的细胞群体看成离散概率测度，再通过相邻时刻的 OT geodesic 定义“平均”。在此基础上，用 subdivision 反复细化轨迹，形成 B-spline 近似或精确插值。算法是局部迭代式的，不需要一次性求解全局高阶最优化问题，因此比直接做 Wasserstein spline 全局优化更可落地。文章还强调它能自然处理质量分裂，因此对 bifurcation 和 supercell 场景更友好。

## 设计原则是什么
设计原则很明确：第一，轨迹构造要内禀于 Wasserstein 几何，而不是先把点云硬嵌到欧氏样条里；第二，尽量避免难解的全局变分问题，转而使用局部、可迭代、可控平滑度的构造；第三，让方法天然支持粒子分裂，而不是把每条轨迹都强行假设为一对一延续。

## 工程优化或实现性考虑
工程上它的优势是实现简单，主体由反复求解 OT 和 geodesic averaging 组成，超参数可以直接控制平滑度和近似精度。相比需要长时间训练的神经动力学模型，它更像一个稳定的几何算法。代价是需要求很多次 OT，因此当样本量很大时计算仍然不轻。

## 局限与未解点
它本质上仍是几何插值/逼近算法，不直接学习可泛化的随机动力学生成模型，也没有显式建模 growth、interaction 或 unbalanced birth-death。轨迹质量强依赖局部 OT 匹配的合理性；如果时间间隔大、状态变化剧烈或存在复杂选择压力，仅靠 subdivision 可能不足以恢复真实机制。

## 为什么能发顶会 / 为什么是重要理论工作
它重要的地方不在于“神经网络更大”，而在于把 Wasserstein spline 这条理论线和实际可运行的 trajectory inference 连接起来了。文章同时给出算法、几何解释、收敛分析和单细胞实验，说明这不是纯数学玩具，而是能直接服务 snapshot trajectory inference 的方法学工作。

## 对单细胞轨迹建模和 CytoBridge-agent 自动设计算法的启发
对 CytoBridge 来说，这篇文章最有价值的启发是：并不是所有连续轨迹恢复都需要上来就训练全局神经 ODE/SDE。对于高噪声 snapshot 数据，可以先用这种内禀 Wasserstein subdivision 生成稳定的几何先验轨迹，再把它作为更复杂 RUOT/UMFSB/flow matching 模型的初始化、正则项或 sanity check。它也提醒我们，分叉结构最好在概率测度几何里表达，而不是在单细胞一对一配对上硬做。
