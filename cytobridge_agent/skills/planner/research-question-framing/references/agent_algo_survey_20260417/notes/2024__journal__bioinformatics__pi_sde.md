# A physics-informed neural SDE network for learning cellular dynamics from time-series scRNA-seq data

## Metadata
- 标题: A physics-informed neural SDE network for learning cellular dynamics from time-series scRNA-seq data
- 作者: Qi Jiang and Lin Wan
- 年份: 2024
- 正式 venue: Bioinformatics (Proceedings of ECCB 2024)
- PDF 文件名: A physics-informed neural SDE network for learning cellular dynamics from time-series scRNA-seq data.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
PI-SDE试图解决仅靠数据驱动模型学习 Waddington landscape 时，预测和可解释性都不稳定的问题。作者关注如何在 time-series scRNA-seq 上更准确地恢复潜在能量景观，并提升 held-out timepoint prediction 与生物解释性。

## 数据类型 / 场景
输入是 time-series scRNA-seq，目标是学习连续细胞动力学和潜在能量景观。文章在真实分化数据上比较了加入物理先验前后的变化。

## 核心算法怎么设计
方法把 neural SDE 与 Hamilton-Jacobi 方程结合起来。具体做法是在学习势能函数和随机动力学时加入 HJ regularization，相当于把最小作用量原理写进损失函数，使势场、漂移和随机扩散的估计受到物理约束。

## 设计原则是什么
这篇文章最鲜明的原则是“别只靠神经网络拟合，要把已知物理规律直接并入目标函数”。作者认为真正需要被学习的是受物理限制的 landscape，而不是任意 flexible 的 latent dynamics。

## 工程优化 / 训练技巧 / pipeline 设计
HJ regularization 同时起到了物理约束和训练稳定器的作用，作者专门做了学习率与扩散参数的敏感性分析，说明这不是只为解释性服务，也直接改善了训练鲁棒性。模型还能做 in silico TF perturbation，用于验证学到的速度场方向是否合理。

## 局限性
物理先验带来更强约束，但也意味着如果系统并不符合对应势场/HJ 近似，模型可能出现偏置。文章规模还不算大，主要验证的是有限几个分化数据集；对极复杂、强交互和强非势场系统的适用性仍需进一步确认。

## 为什么能发到这个级别
它代表了一类很有意义的方法路线：把 physics-informed learning 引入单细胞动力学，而不只是做纯黑盒深度学习。对于当时越来越强调可解释和泛化稳定性的 single-cell dynamics 领域，这是一个清晰的方向性贡献。

## 对 CytoBridge-agent 自动设计算法的启发
对 agent 来说，物理/数学先验应被视为可组合模块，而不是可有可无的装饰项。设计新算法时，可以系统比较“纯数据驱动”“弱先验约束”“强物理约束”三种模式的收益与偏差。
