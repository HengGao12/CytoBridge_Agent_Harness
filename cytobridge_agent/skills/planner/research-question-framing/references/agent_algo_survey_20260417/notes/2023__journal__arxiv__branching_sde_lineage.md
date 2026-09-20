# Trajectory inference for a branching SDE model of cell differentiation

## Metadata
- 标题: Trajectory inference for a branching SDE model of cell differentiation
- 作者: Elias Ventre et al.
- 年份: 2023
- 正式 venue: arXiv preprint
- PDF 文件名: Trajectory inference for a branching SDE model of cell differentiation.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否，本地 PDF 仍为 arXiv 版本
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么生物问题
这篇工作要补的是 gWOT/Waddington-OT 在存在增殖、死亡和 subsampling 时的识别缺口。作者关心的是：能否借助 lineage tree，把“分化动力学”和“净增殖变化”分开，从而更可靠地恢复 branching differentiation landscape。

## 数据类型 / 场景
适用于 time-course 单细胞数据，尤其是同时带有 CRISPR lineage tracing 的场景。文中主要用 branching SDE 模拟数据系统性分析方法在 proliferation/death/subsampling 下的表现。

## 核心算法怎么设计
作者从 gWOT 出发，把细胞演化写成 branching SDE，并把 lineage trees 当作辅助观测来估计祖先-后代关系。在没有死亡和 subsampling 时，它能在不预先知道 proliferation rate 的前提下恢复 branching dynamics；在存在死亡或 subsampling 时，作者明确给出由 lineage 数据带来的偏差项并分析其不可避免性。

## 设计原则是什么
这篇文章的原则是先澄清“哪里可辨识、哪里不可辨识”，再给算法。它不是单纯把 lineage 作为额外特征，而是用它来打破 proliferation 与 differentiation 的混淆，并且把偏差来源显式写出来。

## 工程优化 / 训练技巧 / pipeline 设计
算法延续 gWOT 的计算复杂度级别，没有引入过重的额外代价。作者也用模拟系统系统比较了真实 branching rate、lineage-assisted estimate 等不同基线，这种偏差分解式 benchmark 很有价值。

## 局限性
方法的强项来自 lineage 信息，因此没有 lineage tree 时并不能直接发挥作用。作者也明确指出，一旦存在死亡和 subsampling，偏差是结构性的，不是简单靠更大模型就能学掉的；这说明它更像“识别边界分析 + 受限改进”，不是全能解法。

## 为什么能发到这个级别
这篇文章的重要性在于它补上了 OT 轨迹推断里一个长期被忽略的理论缺口：增殖、死亡与 branching 会怎样破坏可辨识性。它给出的不是经验性 tweak，而是有理论解释的扩展，因此对后续方法设计约束很强。

## 对 CytoBridge-agent 自动设计算法的启发
CytoBridge 如果想可靠建模 growth，就必须区分“可从 snapshot 恢复的信息”和“必须靠 lineage 或额外观测补足的信息”。agent 自动设计算法时，也应优先识别不可辨识因素，而不是默认所有缺失变量都能靠神经网络吸收。
