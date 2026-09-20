# Energy landscape decomposition for cell differentiation with proliferation effect

## Metadata
- 标题: Energy landscape decomposition for cell differentiation with proliferation effect
- 作者: Jifan Shi, Kazuyuki Aihara, Tiejun Li, Luonan Chen
- 年份: 2022
- 正式 venue: National Science Review
- PDF 文件名: Energy landscape decomposition for cell differentiation with proliferation effect.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
这篇文章要解决的是一个经典但一直没有被处理干净的问题：如果细胞分化过程中存在明显的增殖和死亡，仅靠传统的 Waddington 势景并不能给出内生的分化方向。作者希望在包含 birth-death 的非平衡分化系统里，同时解释“稳定细胞类型在哪里”和“分化方向朝哪里走”。

## 数据类型 / 场景
- 主要是 model-based 场景，不是直接从单细胞数据端到端拟合。
- 动力学对象是带漂移、扩散和 birth-death 项的随机微分系统 / 广义 Fokker-Planck 方程。
- 通过几个典型 GRN 与 T-cell differentiation 示例说明如何解释分化、干性和增殖效应。

## 核心算法怎么设计
- 作者把细胞分化写成带 birth-death 项的随机动力学：`∂_t p = -∇·(bp) + εΔp + Rp`。
- 在这个框架里分别定义有 birth-death 时的稳态分布 `P_U` 和去掉 birth-death 后的稳态分布 `P_0`。
- 然后构造两个势函数：
  - `U(x) = -ε log P_U(x)`，表示 cell-type landscape，对应稳态细胞类型和 basin 稳定性；
  - `V(x) = -ε log(P_0(x)/P_U(x))`，表示 pluripotency landscape，其负梯度给出分化方向。
- 最后再把系统分解成 `b(x) = -∇U(x) - ∇V(x) + f(x)`，其中 `f(x)` 是非梯度 curl 项。
- 为了在高维下可算，作者提出了数值构造方案和 mean-field approximation，用来近似 `P_U`、`P_0` 及对应势景。

## 设计原则是什么
- 把“细胞类型稳定性”和“分化方向性”拆成两个不同的势函数，而不是强行由一个势面同时承担两类含义。
- 把 proliferation / death 当成决定方向性的核心机制，而不是一个小修正项。
- 保留与已有 landscape 理论的连续性：当 `R(x)=0` 时退化回传统势景；有 birth-death 时才新增 `V(x)`。
- 在解释层面优先追求可分解、可视化、可映射到生物概念，而不是只给一个黑箱 drift。

## 工程优化 / 训练技巧 / pipeline 设计
- 论文不是工程系统论文，但给出了低维数值求解和高维 mean-field approximation 两套可操作路径。
- mean-field 近似的设计重点是同时近似 `P_0` 和 `P_U`，避免只估一个稳态后无法稳定构造 `V(x)`。
- 通过多个 toy/biological systems 验证 `U` 与 `V` 的角色分离，提高了框架的可解释性。

## 局限性
- 这是 model-based framework，默认 `b(x)`、扩散强度和 birth-death rate 形式已知；真实单细胞数据里这些量通常并不直接可得。
- mixture weights 与 rare transition rate 的估计仍不够稳，作者明确承认这是开放问题。
- 高维 mean-field approximation 仍依赖较强近似，面对真实复杂细胞状态空间时未必足够精确。
- 它更像是解释和建模框架，不是直接面向 snapshot 单细胞数据的 end-to-end 轨迹学习器。

## 为什么能发到这个级别
因为它补上的不是一个局部技巧，而是 birth-death 条件下 Waddington 势景应如何重写这一根本问题。文章最重要的价值在于把“稳态细胞类型”和“分化方向”从一个势函数中拆开，并给出与已有 landscape / Fokker-Planck / mean-field 理论一致的统一表述。

## 对 CytoBridge-agent 自动设计算法的启发
- 对有明显增殖和死亡的数据，不能再把 growth 仅当作 OT/UOT 里的质量修正；它也会改变“方向场”本身。
- 设计新算法时，可以考虑把“吸引子稳定性”和“命运方向性”拆开建模，例如分别学习 landscape / score 与 directional potential。
- 如果未来要把 CytoBridge 的 growth 模块与 landscape 可解释性做强连接，这篇文章提供了一个很好的理论模板。
- 这篇工作也提醒一个边界：如果 agent 要从数据里自动发明算法，最好寻找既能从 snapshot 学出 `U`，又能从 proliferation / condition 信息约束 `V` 的联合学习方案。
