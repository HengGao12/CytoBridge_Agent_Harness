# A comparison of single-cell trajectory inference methods

## Metadata
- 标题: A comparison of single-cell trajectory inference methods
- 作者: Wouter Saelens, Robrecht Cannoodt, Helena Todorov, Yvan Saeys
- 年份: 2019
- 正式 venue: Nature Biotechnology
- PDF 文件名: A comparison of single-cell trajectory inference methods.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已读摘要、核心结果、总结与讨论

## 解决了什么生物问题

这篇文章解决的不是单一生物问题，而是 trajectory inference 领域本身的“方法选择问题”。面对几十种 TI 工具、不同输入要求和不同输出结构，研究者很难知道在自己的数据上该选谁。对任何要自动设计算法的 agent，这其实是上游决策问题。

## 数据类型 / 场景

- 110 个真实数据集和 229 个合成数据集。
- 覆盖线性、分支、树状、循环、断连等多种拓扑。
- 评估 45 个 trajectory inference 方法的准确性、可扩展性、稳定性和可用性。

## 核心算法怎么设计

- 论文本身不是新 TI 算法，而是先提出一个统一的 trajectory 表示：用 milestone network 表示总体拓扑，再表示细胞在 milestone 之间的位置。
- 作者把不同方法的输出归入 7 类，并为这些输出写统一 converter，这样才能公平比较。
- 评价指标不只看 pseudotime 对不对，还看 topology、scalability、stability、usability。
- 最终给出的不是“唯一冠军”，而是按数据维度和轨迹拓扑给出 method-selection guideline。

## 设计原则是什么

- benchmark 必须先统一接口，否则比较没有意义。
- 不追求单一总榜，而要按任务结构做条件化推荐。
- 合成数据和真实数据要混合使用，因为两者各有盲点。
- 一个好方法不只要准，还要能跑得动、结果稳、用户能用。

## 工程优化 / 训练技巧 / pipeline 设计

- 做了 method wrappers、guidelines app 和 benchmarking pipeline，基础设施建设很完整。
- 明确区分弱先验和强先验，避免因为给某些方法过多先验而产生伪优势。
- 所有 benchmark 资产公开，对后续方法开发形成持续基线。

## 局限性

- benchmark 结果天然受数据集组成、模拟器和评估指标选择影响。
- 领域发展很快，排行榜会过时。
- 即使统一表示后，某些方法的原生优势或偏好也可能被“压平”。
- 文章主要解决“该选哪类方法”，不直接解决 snapshot 动力学不可辨识性本身。

## 为什么能发到这个级别

这是典型的 field-shaping 基础设施论文。2019 年 trajectory inference 方法已经明显过多，社区非常需要一个标准化、可复现、可操作的 benchmark 与指南。它让“选方法”从经验和口碑变成更结构化的判断，因此影响力很大。

## 对 CytoBridge-agent 自动设计算法的启发

- agent 不应该默认存在一个全局最优算法，而应根据预期拓扑、数据规模、时间信息和先验强度做条件化路由。
- 自动化系统需要统一的中间表示层，否则无法比较和组合不同方法。
- 评价标准必须多维化。只看轨迹准确率会让 agent 偏向在真实工程中并不好用的方法。
