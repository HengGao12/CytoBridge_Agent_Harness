# CFM-GP: Unified Conditional Flow Matching to Learn Gene Perturbation Across Cell Types

## Metadata
- 标题: CFM-GP: Unified Conditional Flow Matching to Learn Gene Perturbation Across Cell Types
- 作者: Abrar Rahman Abir et al.
- 年份: 2025
- 正式 venue: arXiv preprint
- PDF 文件名: CFM-GP Unified Conditional Flow Matching to Learn Gene Perturbation Across Cell Types.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否，本地 PDF 仍为 arXiv 版本
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
CFM-GP要解决的是单细胞 perturbation prediction 往往要为每个 cell type 单独训练模型，难以跨细胞类型泛化。它把问题改写为 cell type-agnostic 的基因扰动分布变换学习。

## 数据类型 / 场景
输入是控制组与扰动组的单细胞表达分布，并带 cell type 条件。作者在 SARS-CoV-2、IFN-beta、Panobinostat、lupus 和 StateFate 等五套数据上验证。

## 核心算法怎么设计
方法采用 conditional flow matching，在未扰动分布和扰动分布之间学习连续时间、条件化的流场变换。cell type 条件直接进入模型，使同一个网络可以在所有 cell types 上共享参数，并预测特定 cell type 下的 perturbation response。

## 设计原则是什么
核心原则是统一建模而非 cell-type-specific 模型堆叠。作者把 perturbation 看成条件化分布 transport，而不是针对每个 cell type 做独立回归，这样天然更适合跨细胞类型泛化。

## 工程优化 / 训练技巧 / pipeline 设计
flow matching 比逐步扩散更轻量，训练和采样效率更高，适合高维基因表达分布。统一模型也减少了多 cell type 训练和维护成本，这一点对自动化系统尤其友好。

## 局限性
作为 preprint，它对更极端 OOD cell types、组合 perturbation 和长期时间效应的验证还不充分。方法也更偏分布预测，对机制可解释性和 causal identifiability 的支撑弱于带显式结构先验的模型。

## 为什么能发到这个级别
这篇工作抓住了 flow matching 在生命科学中的一个很自然切入点：用连续流场做单细胞扰动分布预测，并突出跨 cell type 统一建模的优势。问题设定明确，实验覆盖也比较贴合应用。

## 对 CytoBridge-agent 自动设计算法的启发
若要设计新的 perturbation 算法，CFM-GP提示“统一模型跨 cell type 泛化”本身就是强卖点。CytoBridge 可以把 conditional flow matching 作为 perturbation head，与更结构化的 dynamics backbone 组合。
