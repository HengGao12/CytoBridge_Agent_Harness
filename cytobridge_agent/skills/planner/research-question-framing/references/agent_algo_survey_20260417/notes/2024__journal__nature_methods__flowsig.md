# Inferring pattern-driving intercellular flows from single-cell and spatial transcriptomics

## Metadata
- 标题: Inferring pattern-driving intercellular flows from single-cell and spatial transcriptomics
- 作者: Axel A. Almet, Yuan-Chen Tsai, Momoko Watanabe, Qing Nie
- 年份: 2024
- 正式 venue: Nature Methods
- PDF 文件名: Inferring pattern-driving intercellular flows from single-cell and spatial transcriptomics.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
这篇文章要解决的是：单细胞和空间转录组里，细胞间通信并不只是“谁和谁说话”，而是会形成沿着组织传播的因果流。作者希望从 scRNA-seq 或 ST 数据中恢复这种由 inflow signal、细胞内调控模块到 outflow signal 组成的 intercellular flow，用来解释发育、疾病和刺激响应中的模式形成机制。

## 数据类型 / 场景
- 非空间 scRNA-seq：要求有 control vs perturbed、疾病分组或时间条件，便于识别 differential inflow / outflow。
- 空间转录组：可以直接利用空间约束与 CCC 推断结果来估计 received signals。
- 典型任务是细胞间相互作用和模式形成，而不是单纯 pseudotime。

## 核心算法怎么设计
- FlowSig 不直接在原始基因层面做因果图，而是先构造三类中间变量：
  - inflow signals：非空间数据里用 `receptor expression × downstream TF targets` 近似；空间数据里结合 COMMOT 之类工具估 received signal；
  - intracellular GEMs：非空间数据用 pyLIGER，空间数据用 NSF 得到 gene expression modules；
  - outflow signals：配体表达或与外发通信相关的信号变量。
- 然后用图因果学习来恢复依赖结构：
  - 非空间带 perturbation 的数据用 UT-IGSP；
  - 空间数据用 GSP。
- 条件独立检验先得到 CPDAG，再按生物先验把边重新定向为 `inflow -> GEM -> outflow`，允许 GEM-GEM 间双向或未定向关系。
- 为了降低假阳性，非空间场景只保留 differential flowing variables；空间场景只保留 spatially variable variables。
- 最后还用 bootstrap aggregation 提升边稳定性，并用随机森林把 spatial outflow 进一步追溯到上游 TF。

## 设计原则是什么
- 不在原始基因图上硬做因果发现，而是先把问题压缩成有生物意义的中层变量。
- 把因果方向性建立在明确的生物流程先验上，而不是完全无约束地让图算法自由决定。
- 非空间数据缺少“收到多少信号”的直接观测，因此要用 perturbation 信息和 downstream TF 作为替代观测。
- 空间与非空间不是两套方法，而是在同一个 intercellular flow 抽象下替换 inflow 观测构造方式。

## 工程优化 / 训练技巧 / pipeline 设计
- 强依赖已有 CCC 工具输出，但通过变量构造和后续 causal graph，把 CCC 从 pairwise interaction 提升为 flow network。
- differential filtering / spatial variability filtering 显著减少图学习变量数，提高稳定性。
- bootstrap aggregation 是关键工程细节，否则图结构很容易对抽样噪声敏感。
- pyLIGER 与 NSF 的选择也很务实：分别适配非空间和空间场景的 GEM 提取。

## 局限性
- 结果强依赖上游 CCC 推断、GEM 分解和 TF target 先验质量，误差会层层传递。
- 学到的是 CPDAG 加生物先验定向，并不是完全可辨识的真因果图。
- 它解决的是 communication-driven flow，不是连续时间的分布演化模型，因此不能替代 OT/SB/FM 这类 trajectory generator。
- 非空间场景需要 control/perturbation 等条件信息，否则 inflow 估计会更弱。

## 为什么能发到这个级别
因为它把一个长期缺位的问题 formalize 了：从“谁和谁通信”走向“通信如何驱动模式和下游信号流”。方法上又不是简单拼装，而是把 CCC、module decomposition 和 graphical causal modeling 真正串成了一个新对象，并同时支持 scRNA-seq 和 ST。

## 对 CytoBridge-agent 自动设计算法的启发
- CytoBridge 的 interaction 模块不该只输出配体-受体对；更好的目标是输出 `signal inflow -> intracellular program -> signal outflow` 的 flow graph。
- 在自动设计算法时，agent 可以把“原始基因层建模”与“中层变量建模”作为一个显式设计分叉；FlowSig 证明后者在可解释性和可学性上更稳。
- 如果以后要把 trajectory 与 interaction 融合，最自然的方案不是把通信当额外特征拼接，而是把 intercellular flow 作为外场或条件先验去调制单细胞动力学。
- 这篇文章也提示：空间数据里最宝贵的不是坐标本身，而是它让 inflow 更可观测，从而减少因果不可辨识性。
