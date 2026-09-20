# CytoBridge-Agent Literature Synthesis

## 审计范围与最终统计

- 本地 RAG 文献库总量：`242` 篇 PDF
- 首轮启发式候选：`153`
- 最终正式纳入：`93`
- 最终排除：`60`
- 非候选 / 不在正式审计作用域：`89`
- 正式纳入分类：
  - `journal`: `39`
  - `mlconf`: `6`
  - `math`: `48`

正式纳入口径以 `candidate_registry.csv` 为准；预印本转正式发表的判断已在 [publication_status_20260417.md](web_verification/publication_status_20260417.md) 中逐项核验。  
另外我额外亲自读了两篇“边界但有启发”的补充文献：

- [Inferring pattern-driving intercellular flows from single-cell and spatial transcriptomics](notes/2024__journal__nature_methods__flowsig.md)
- [Energy landscape decomposition for cell differentiation with proliferation effect](notes/2022__journal__national_science_review__energy_landscape_decomposition_for_cell_differentiation_with_proliferation_effect.md)

这两篇没有进入正式统计，是因为它们分别更偏“通信因果流”与“model-based landscape 理论”，未完全命中本次“单细胞轨迹 / OT-UOT-SB-FM-mean-field 主线”的正式纳入口径；但它们对后续算法设计仍然有价值。

## 一、领域演化脉络

### 0. 数学地基阶段：从 OT 到 SB，再到 UOT / mean-field

这条路线真正的理论地基不是从单细胞开始，而是从 OT 与 SB 的连续演化开始：

- [THE GEOMETRY OF OPTIMAL TRANSPORTATION](notes/1996__math__acta_mathematica__the_geometry_of_optimal_transportation.md) 给出了最优映射与势函数表征的基本几何。
- [A survey of the Schrödinger problem and some of its connections with optimal transport](notes/2014__math__dcds__a_survey_of_the_schrodinger_problem_and_its_connections_with_optimal_transport.md) 与 [On the Relation Between Optimal Transport and Schrödinger Bridges A Stochastic Control Viewpoint](notes/2016__math__jota__on_the_relation_between_optimal_transport_and_schrodinger_bridges.md) 把 OT 与带噪声参考过程下的最小熵输运打通。
- [Unbalanced Optimal Transport Dynamic and Kantorovich Formulation](notes/2018__math__journal_of_functional_analysis__unbalanced_optimal_transport_dynamic_and_kantorovich_formulations.md) 与 [Unnormalized optimal transport](notes/2019__math__journal_of_computational_physics__unnormalized_optimal_transport.md) 把“质量不守恒”从补丁变成主问题。
- [The mean field Schrödinger problem ergodic behavior, entropy estimates and functional inequalities](notes/2020__math__ptrf__mean_field_schrodinger_problem.md) 把 interaction 带入 SB；这是后来做 cell-cell interaction dynamics 的真正理论前置。

这段历史的关键信号很清楚：  
单细胞领域后来的“growth / death / stochasticity / interaction / spatial-temporal”这些看似生物学特殊性，本质上都对应着 OT 主线里一个被逐步补齐的数学缺口。

### 1. 2017-2019：从伪时间到“方向”和“拓扑”

这一阶段的问题意识是：  
仅靠静态 manifold 几何，无法回答“细胞往哪里去”。

代表工作有：

- [SLICE](notes/2017__journal__nucleic_acids_research__slice_determining_cell_differentiation_and_lineage_based_on_single_cell_entropy.md)、[single-cell entropy](notes/2017__journal__nature_communications__single_cell_entropy_for_accurate_estimation_of_differentiation_potency_from_a_cells_transcriptome.md)  
  解决的是“分化潜能/状态排序”。
- [Slingshot](notes/2018__journal__bmc_genomics__slingshot_cell_lineage_and_pseudotime_inference_for_single_cell_transcriptomics.md)、[PAGA](notes/2019__journal__genome_biology__paga_graph_abstraction_reconciles_clustering_with_trajectory_inference_through_a_topology_preserving_map_of_single_cells.md)  
  解决的是“全局拓扑与分支骨架”。
- [RNA velocity of single cells](notes/2018__journal__nature__rna_velocity_of_single_cells.md)  
  首次把方向性从观测里直接提出来。
- [Optimal-Transport Analysis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming](notes/2019__journal__cell__optimal_transport_analysis_of_single_cell_gene_expression_identifies_developmental_trajectories_in_reprogramming.md)  
  首次把多时间点 snapshot 分布耦合正式做成单细胞主方法。

这一阶段的结构性结论是：

- `pseudotime` 解决排序，但不解决真实时间和分布耦合。
- `velocity` 解决局部方向，但不自动给全局 fate distribution。
- `graph/topology` 解决结构骨架，但不自动给动态机制。
- `OT` 让“相邻时间点分布如何耦合”第一次成为主问题。

也就是说，后面所有强方法几乎都在做一件事：  
把这四块拼起来，而不是只做其中一块。

### 2. 2020-2022：从静态耦合走向连续动力学、非守恒与生成模型

这一阶段是从“相邻时间点配对”走向“学习连续时间动力学”的关键转折。

代表工作：

- [TrajectoryNet](notes/2020__math__icml__trajectorynet.md)  
  把 continuous dynamic OT 正式带进单细胞。
- [Generalizing RNA velocity to transient cell states through dynamical modeling](notes/2020__journal__nature_biotechnology__generalizing_rna_velocity_to_transient_cell_states_through_dynamical_modeling.md)  
  解决了 steady-state velocity 在 transient systems 下的系统失真。
- [PRESCIENT](notes/2021__journal__nature_communications__prescient.md)  
  把时间序列单细胞建成可生成、可 intervention 的 stochastic dynamics。
- [LineageOT](notes/2021__journal__nature_communications__lineageot.md)  
  把 lineage tracing 变成降低 coupling 不可辨识性的核心信息源。
- [Variational Mixtures of ODEs for Inferring Cellular Gene Expression Dynamics](notes/2022__journal__arxiv__velovae.md)  
  说明 latent-time + bifurcation + ODE mixture 是另一条重要路线。
- [Deep Generative Learning via Schrödinger Bridge](notes/2021__math__icml__deep_generative_learning_via_schrodinger_bridge.md)、[Diffusion Schrödinger Bridge](notes/2021__math__neurips__diffusion_schrodinger_bridge.md)、[Deep Generalized Schrödinger Bridge](notes/2022__math__neurips__deep_generalized_schrodinger_bridge.md)  
  开始把 SB 从纯理论变成神经可解器。

这阶段最重要的变化不是“模型变复杂”，而是问题定义升级了：

- 不再满足于“重建轨迹图”。
- 开始要求：
  - 连续时间生成
  - held-out timepoint 预测
  - perturbation/counterfactual
  - growth/death 显式建模
  - 更可信的 latent time

### 3. 2023-2024：从 OT/SB 扩展到 FM、几何、interaction、multiview

这一阶段的明显特征是分叉增多，但方向并不发散，而是在补齐主路线的几个薄弱环节：

#### 3.1 补“速度/转录动力学”缺口

- [veloVI](notes/2024__journal__nature_methods__velovi.md)
- [DeepVelo](notes/2024__journal__genome_biology__deepvelo.md)
- [MultiVelo](notes/2023__journal__nature_biotechnology__multivelo.md)
- [A relay velocity model infers cell-dependent RNA velocity](notes/2024__journal__nature_biotechnology__celldancer.md)
- [TFvelo](notes/2024__journal__nature_communications__tfvelo.md)

这些工作共同说明：  
单纯把 velocity 看成“局部短时导数”已经不够，真正难的是 cell-specific kinetics、multi-omic coupling、regulatory conditioning。

#### 3.2 补“fate aggregation / multiview”缺口

- [CellRank 2](notes/2024__journal__nature_methods__cellrank2.md)
- [moslin](notes/2024__journal__genome_biology__moslin.md)

这类工作把 focus 从“学一个漂亮向量场”转向“怎样把多视角证据整合成更稳的 fate map”。

#### 3.3 补“spatial-temporal / interaction”缺口

- [SpaceFlow](notes/2022__journal__nature_communications__spaceflow.md)
- [Robust mapping of spatiotemporal trajectories and cell-cell interactions in healthy and diseased tissues](notes/2023__journal__nature_communications__stlearn_psts.md)
- [Spatial transition tensor of single cells](notes/2024__journal__nature_methods__stt.md)
- [stVCR](notes/2024__journal__biorxiv__stvcr.md)
- 补充参考：[FlowSig](notes/2024__journal__nature_methods__flowsig.md)

说明 community 已经认识到：  
空间不是“再加一个坐标”，而是会改变可观测性、interaction 结构与命运约束。

#### 3.4 补“生成建模 / 几何 / simulation-free”缺口

- [Conditional Flow Matching Simulation-Free Dynamic Optimal Transport](notes/2023__math__arxiv__conditional_flow_matching.md)
- [Generalized Schrödinger Bridge Matching](notes/2023__math__iclr__generalized_schrodinger_bridge_matching.md)
- [Flow Matching on General Geometries](notes/2024__mlconf__iclr__flow_matching_on_general_geometries.md)
- [Metric Flow Matching for Smooth Interpolations on the Data Manifold](notes/2024__mlconf__neurips__metric_flow_matching_for_smooth_interpolations_on_the_data_manifold.md)
- [GENOT](notes/2023__math__neurips__genot.md)

这一支说明：  
2023 之后最强的算法信号之一，是“如何在保留 SB/OT 结构意义的同时，把训练改造成更像 FM/CFM 这种 simulation-free 回归问题”。

对 ML 理论 / 算法会议而言，这里还有一个更具体的代际变化：

- 2020 前后以 [TrajectoryNet](notes/2020__math__icml__trajectorynet.md) 为代表的路线，更强调 Neural ODE 式的连续动力学仿真；
- 2023 之后以 CFM / GSBM / geometry-aware FM 为代表的路线，更强调把问题改写成 simulation-free 的回归或匹配目标；
- 这不是单纯“训练更快”而已，而是把可优化性、模块化扩展和理论对象的表达方式一起改了。

这点对 CytoBridge 很重要，因为它当前的 package 已经同时覆盖：

- `dynamical_ot` / `unbalanced_ot`：ODE-first
- `ruot` / `cyto_simulation`：hybrid ODE + FM
- `crufm`：FM-first, simulation-free

所以今天再提 ML 顶会方向，不能只把“把 Neural ODE 换成 Flow Matching”当成 novelty 本身；真正的 gap 必须落在 geometry、identifiability、interaction semantics、multi-view constraints 或 routing 上。

### 4. 2025-2026：从“能拟合”走向“更可信的外推、交互和统一框架”

这是离 CytoBridge 当前目标最近的一段。

代表工作：

- [Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport](notes/2025__mlconf__iclr__learning_stochastic_dynamics_from_snapshots_through_regularized_unbalanced_optimal_transport.md)  
  把 stochastic + unbalanced snapshot dynamics 正式推成 ICLR 级问题。
- [ARTEMIS](notes/2025__journal__bioinformatics__artemis.md)  
  把 autoencoder + SB + perturbation 明确揉进单细胞时间序列。
- [Gene trajectory inference for single-cell data by optimal transport metrics](notes/2025__journal__nature_biotechnology__genetrajectory.md)
- [Learning cell dynamics with neural differential equations](notes/2025__journal__nature_machine_intelligence__scdiffeq.md)
- [Mapping cells through time and space with moscot](notes/2025__journal__nature__moscot.md)
- [MultistageOT](notes/2025__journal__pnas__multistageot.md)
- [scCausalVI](notes/2025__journal__cell_systems__sccausalvi.md)
- [Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schrödinger Bridge](notes/2025__math__arxiv__unbalanced-mean-field-schrodinger-bridge.md)
- [CellStream](notes/2026__math__aaai__cellstream.md)

这批论文共同暴露出一个趋势：

- 纯“时间配对”已经不够。
- 纯“生成模型”也不够。
- 真正开始有价值的是把以下几件事联合起来：
  - latent representation
  - continuous dynamics
  - growth/death
  - stochasticity
  - interaction
  - perturbation
  - spatial or lineage constraints

这正是 CytoBridge 要做的方向，所以它并不是拍脑袋出来的，而是站在 2025-2026 主趋势上的。

## 二、反复出现的高质量设计原则

### 1. 先把问题定义收窄，再设计模型

好论文几乎都不是“端到端学所有东西”。

它们先回答：

- 我到底要恢复什么对象？
  - root ordering?
  - local velocity?
  - timepoint coupling?
  - continuous path measure?
  - fate distribution?
  - perturbation response?
  - interaction-driven flow?

对象一旦定义清楚，模型空间就会自动收缩。坏工作常见的问题恰恰是：  
问题没收窄，模型却已经变复杂。

### 2. 方向性必须有来源

整个领域最反复出现的原则是：  
方向不能只靠 embedding 几何脑补。

高质量方向来源主要有五类：

- splicing kinetics / RNA velocity  
  例：[RNA velocity](notes/2018__journal__nature__rna_velocity_of_single_cells.md), [scVelo](notes/2020__journal__nature_biotechnology__generalizing_rna_velocity_to_transient_cell_states_through_dynamical_modeling.md)
- explicit time labels  
  例：[WOT](notes/2019__journal__cell__optimal_transport_analysis_of_single_cell_gene_expression_identifies_developmental_trajectories_in_reprogramming.md), [PRESCIENT](notes/2021__journal__nature_communications__prescient.md)
- lineage tracing  
  例：[LineageOT](notes/2021__journal__nature_communications__lineageot.md), [moslin](notes/2024__journal__genome_biology__moslin.md)
- perturbation / interventional shifts  
  例：[graphVCI](notes/2023__journal__iclr__graphvci.md), [scCausalVI](notes/2025__journal__cell_systems__sccausalvi.md)
- reference process / prior dynamics  
  例：[SB route](notes/2016__math__jota__on_the_relation_between_optimal_transport_and_schrodinger_bridges.md), [PO-MFL](notes/2025__math__iclr__partially-observed-trajectory-inference-ot-dynamics-prior.md)

### 3. 质量守恒 vs 非守恒是一级设计决策

这不是细节，而是第一层路由。

经验规律非常明确：

- 如果系统存在明显 proliferation / death / sampling bias，balanced OT 几乎一定会制造伪迁移。
- 真正强的方法会显式拆开：
  - transport
  - growth/death
  - noise

对应代表：

- [WOT](notes/2019__journal__cell__optimal_transport_analysis_of_single_cell_gene_expression_identifies_developmental_trajectories_in_reprogramming.md)
- [UOT dynamic formulation](notes/2018__math__journal_of_functional_analysis__unbalanced_optimal_transport_dynamic_and_kantorovich_formulations.md)
- [DeepRUOT / RUOT ICLR 2025](notes/2025__mlconf__iclr__learning_stochastic_dynamics_from_snapshots_through_regularized_unbalanced_optimal_transport.md)
- 补充参考：[ELD](notes/2022__journal__national_science_review__energy_landscape_decomposition_for_cell_differentiation_with_proliferation_effect.md)

### 4. 真正好的模型都在做“机制拆分”

最常见的可复用拆分是：

- state representation
- topology
- local direction
- global coupling
- stochasticity
- growth/death
- interaction
- condition sharing

好工作通常不会让一个神经网络去同时吸收这些机制。  
而是显式分层，例如：

- `velocity + pseudotime`
- `coupling + growth`
- `latent ODE + branch mixture`
- `inflow + GEM + outflow`
- `drift + score + density + growth`

这对 agent 设计尤其重要，因为 agent 最擅长搜索模块组合，而不是盲选一个超大黑箱。

### 5. 表示学习与动力学学习最好分开，但最终要联合

近几年一个明显趋势是：

- 早期方法多在固定 PCA/latent 空间里做 dynamics；
- 新工作逐渐意识到 representation 选择会决定 dynamics 是否可学；
- 但完全 joint learning 又容易不可辨识、训练不稳。

因此最佳实践通常是：

- 先有足够稳的表示；
- 再让 dynamics 反过来约束表示；
- 最后形成 joint but structured learning。

代表：

- [CellStream](notes/2026__math__aaai__cellstream.md)
- [TIGON](notes/2024__journal__nature_machine_intelligence__tigon.md)
- [ARTEMIS](notes/2025__journal__bioinformatics__artemis.md)

### 6. 高水平论文都很重视“反证式评估”

好工作几乎都不满足于可视化：

- held-out timepoint reconstruction
- lineage consistency
- perturbation OOD
- branch recovery under known synthetic dynamics
- ablation on growth/noise/interaction
- failure-case discussion

这也是为什么很多论文能上更高层级：  
不是因为模型更深，而是因为验证更像“我要证明这个机制真的必要”。

## 三、与当前 CytoBridge package 的自洽边界

### 1. 现在的 package 已经覆盖到哪一步

如果只看当前 builtin family，CytoBridge 已经不是“只会 Neural ODE 的时间序列包”。

它已经覆盖：

- `dynamical_ot`
  - `velocity`
  - 纯 `neural_ode`
- `unbalanced_ot`
  - `velocity + growth`
  - 纯 `neural_ode`
- `ruot`
  - `velocity + growth + score`
  - `neural_ode -> flow_matching -> neural_ode`
- `crufm`
  - `velocity + growth + score`
  - `simulation-free flow_matching`
- `cyto_simulation`
  - `velocity + growth + score + interaction`
  - interaction-aware hybrid route

换句话说，当前 package 已经具备：

- deterministic transport
- unbalanced mass modeling
- stochastic score-style dynamics
- hybrid ODE/FM pipeline
- simulation-free FM family
- interaction-aware builtin family

所以在这套 package 上提新 idea 时，不能再把以下内容当成“还没人做过”的大 gap：

- 只是给 dynamics 加一个 growth head
- 只是给 stochasticity 加一个 score head
- 只是把 ODE 训练替换成 FM 训练
- 只是把 interaction 当成一个更大的黑箱网络塞进去

### 2. 现在的 package 还没有 first-class 覆盖到哪一步

当前 package 的第一性输入契约仍然主要是：

- `adata.obs["time_point_processed"]`
- `adata.obsm["X_latent"]`

因此它还没有被诚实地描述成以下这些“已经成熟的一等 builtin 能力”：

- 端到端的 time-series spatial transcriptomics workflow
- geometry-aware / manifold-aware FM 默认族
- lineage-aware coupling 默认族
- partial-observation 专用推断默认族
- 通用多组学动力学默认族
- perturbation-first 的默认研究主线

这些方向可以做，但更像下一层 custom algorithm 或新 builtin family，而不是“当前已经直接具备”。

### 3. 这对后续 idea formulation 的含义

因此，和当前 package 自洽的 gap 提法应该更像：

- 现有 builtin family 已经有，但还不够可辨识
- 现有 builtin family 已经有，但缺少 geometry / multi-view / lineage / space 约束
- 现有 builtin family 已经有，但缺少稳定的 router 去判断何时该用哪一类
- 现有 builtin family 已经有，但 interaction 语义还不够可信
- 现有 builtin family 已经有，但 evaluation 和 refusal boundary 还不够强

而不是把问题说成：

- “我们第一次从 ODE 走向 FM”
- “我们第一次支持 stochastic dynamics”
- “我们第一次把 interaction 放进动力学”

这些表述对当前 package 来说都已经不准确。

## 四、还没解决的核心 gap

### 1. 稀疏 snapshot 下的可辨识性仍然不够

即使到 2026，绝大多数方法仍然无法严格回答：

- 漂移和增长是否可分？
- 噪声和未观测异质性是否可分？
- interaction 和 shared latent confounder 是否可分？
- 多个不同动力学是否能解释同一组快照？

这是全领域最大瓶颈，没有之一。

### 2. 缺少真正“可辨识、可路由、可验证”的 unbalanced + stochastic + interaction + condition 框架

现状是：

- 有的工作会做 unbalanced + stochastic
- 有的会做 interaction
- 有的会做 perturbation / condition sharing
- 有的会做 spatial-temporal

真正的缺口已经不只是“把这些项都加进去”。

更难的是：

- 如何让这些项在 sparse snapshots 下仍然可分
- 如何根据数据 regime 自动路由到合适的 family
- 如何证明 interaction 不是 latent confounder 的替身
- 如何让评价协议真的对应科学对象

CytoBridge 当前的 builtin family 已经把这些模块的大框架搭出来了；下一步更大的问题是把它们做得更可信，而不是再写一版“更全”的总模型。

### 3. 跨条件泛化与真实 counterfactual 仍然偏弱

大部分 perturbation 方法仍然面临：

- 训练分布覆盖不足
- 组合扰动爆炸
- context dependence 极强
- 评价指标往往只看相关性或 DE overlap

也就是说，“能预测一个 perturbation response”不等于“学到了可转移机制”。

但对当前 CytoBridge-agent 的默认 scope 来说，这一条更适合放在第二阶段。  
第一阶段更应该先把**非 perturbation 的多时间点 snapshot 单细胞组学**路线做扎实。

### 4. 时间外推仍然不稳

很多方法能做：

- 插值
- 近邻时间点外推

但真正跨长时间跨度外推时，常见问题是：

- drift 爆炸
- growth term 累积偏差
- branch probability 漂移
- latent space 漏出 data manifold

### 5. interaction 建模仍多停留在“后处理”

很多单细胞方法仍然是：

- 先学 trajectory
- 再单独做 CCC

这导致 interaction 对 dynamics 只是注释，不是生成机制。  
FlowSig、TICCI、UMFSB 这类工作说明 community 已经开始意识到这个问题，但还没有形成成熟统一范式。

### 6. 空间与时间还没真正统一

当前大量方法是：

- 时间上强，空间上弱
- 或空间上强，时间上弱

真正同时处理：

- migration
- proliferation
- differentiation
- local interaction
- measurement resolution mismatch

的模型仍然很少，而且 benchmark 也不足。  
这也和当前 package 的现状一致：interaction builtin 已有，但 time-series spatial transcriptomics 仍不是成熟的一等默认工作流。

### 7. 工程上仍缺少“稳定、可复现、可自动路由”的方法库

这也是 agent 设计最该切入的实际缺口。

目前研究论文常常：

- 依赖强数据预处理技巧
- 对表示空间非常敏感
- 超参数难路由
- 代码可复现性一般

对于一个全自动 agent，这些不是实现细节，而是决定能不能落地的主问题。

## 五、如果要设计 novel algorithm，最值得切入的方向

### 方向 A：Unbalanced Mean-Field SB / FM with Interaction

最值得做的不是再造一个 balanced OT 变体，而是把：

- unbalanced mass change
- stochastic bridge
- mean-field interaction
- optional condition-specific modulation

统一进一个可训练框架。

这是目前高质量缺口最大的方向，也是 CytoBridge 最自然的主线。

可操作设计：

- 基础动力学：UMFSB / RUOT
- interaction：mean-field term or external field
- condition：shared backbone + condition adapters
- evaluation：held-out time + held-out condition + interaction ablation

### 方向 B：Joint Representation-Dynamics Learning, but Structured

不要再做“先 embedding，后 dynamics”的完全割裂版；
但也不要直接端到端黑箱。

更好的设计是：

- 表示层负责 manifold faithfulness
- 动力学层负责 temporal consistency
- growth / interaction / uncertainty 作为显式 heads
- geometry / metric 作为可搜索先验

可借鉴：

- [CellStream](notes/2026__math__aaai__cellstream.md)
- [Flow Matching on General Geometries](notes/2024__mlconf__iclr__flow_matching_on_general_geometries.md)
- [TIGON](notes/2024__journal__nature_machine_intelligence__tigon.md)

### 方向 C：用多视角信息主动打破不可辨识性

值得重点搜索的多视角组合有：

- time + RNA velocity
- time + lineage
- time + space
- time + multi-omics
- time + interaction

不是每个数据集都有这些信号，但 agent 可以先自动识别哪些视角存在，再动态选择模型。

这比“所有数据都喂同一个大模型”更像真正可发表的算法设计。

### 方向 D：把 interaction 从注释升级为动力学控制项

当前许多方法默认 interaction 是 downstream analysis。  
更有新意的路线是：

- 把 intercellular flow 作为 drift / growth / fate 的条件因子
- 让通信强度影响转移概率，而不是只在最后解释 marker
- 在空间场景里，把 interaction induced field 作为外场

补充启发可看：

- [FlowSig](notes/2024__journal__nature_methods__flowsig.md)
- [TICCI](notes/2025__journal__bioinformatics__ticci.md)
- [Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schrödinger Bridge](notes/2025__math__arxiv__unbalanced-mean-field-schrodinger-bridge.md)

### 方向 E：让模型自己知道“什么时候不该外推”

高水平方法越来越需要：

- 不确定性估计
- 机制失配检测
- support mismatch 警报
- mass conservation / non-conservation mismatch 警报
- geometry mismatch 警报

这类“自我否定机制”非常适合 agent 化，因为它们天然就是模型路由器与评估器的一部分。

## 六、不建议重复造轮子的方向

- 只做新的 pseudotime 排序，没有真实方向信息来源。
- 只做 balanced OT 的小改动，却不处理 growth/death。
- 只做条件生成，却不讨论可辨识性和 OOD。
- 只在已有 latent 空间里学一个更深 drift 网络，但没有机制拆分。
- 只做更花哨的可视化，而不解决 held-out prediction。
- 只讲“虚拟细胞”宏大叙事，没有明确的数学对象、评价协议和失败模式。
- 只把“Neural ODE 改成 Flow Matching”当成完整 novelty，却不说明新的理论对象或新的可学性边界。

这些方向今天很难再支撑真正强的新方法。

## 七、对 CytoBridge-agent 的直接建议

### 1. 把算法搜索空间明确写成模块组合

建议最少包含以下开关：

- `balanced` vs `unbalanced`
- `deterministic` vs `stochastic`
- `observed-state` vs `latent-state`
- `no-interaction` vs `interaction-field` vs `mean-field`
- `single-condition` vs `shared-backbone multi-condition`
- `Euclidean path` vs `geometry-aware path`
- `pairwise coupling` vs `continuous-time generator`

### 2. 先做问题路由，再做训练

训练前让 agent 先判断：

- 时间点是否足够多？
- 是否有明显 growth/death？
- 是否有 velocity / lineage / space / multi-omic 辅助视角？
- 是否目标是 prediction、explanation 还是 counterfactual？

只有这样，agent 才不会在不该用 OT 的地方硬上 OT，也不会在该用 UOT 的地方偷懒用 balanced。

### 3. 把“评价协议”内建进 agent

建议默认输出：

- held-out time prediction
- lineage / marker consistency
- growth plausibility
- branch stability
- uncertainty / refusal note
- failure-case note

没有这些，agent 只是在自动调参，不是在自动设计算法。

### 4. 下一代 CytoBridge 最值得押注的主方向

如果只选一个主方向，我建议是：

`unbalanced stochastic mean-field bridge / flow matching`  
`+ structured latent representation`  
`+ interaction-aware conditioning`  
`+ multi-view identifiability constraints`

原因很简单：

- 它最贴近 2025-2026 的前沿缺口；
- 与现有 CytoBridge 目标连续；
- 同时有数学深度、算法新意和明确生物价值；
- 最容易形成“不是换个 backbone，而是重新定义问题”的工作。

## 八、最后的判断

真正能发出好工作的，不是“把 OT、SB、FM、velocity、lineage、space、perturbation 全堆在一起”。  
真正有效的路线是：

1. 明确主问题对象。
2. 明确最关键的不可辨识性来源。
3. 用最少但最强的附加视角去打破它。
4. 显式拆分 transport / growth / noise / interaction / condition。
5. 用可证伪的评估协议证明这些模块确实必要。

这批文献读下来，最稳定的共识不是“哪个模型最好”，而是：

**高质量算法设计，本质上是在有约束地减少错误解释。**  
谁能最系统地避免把 growth 误解释成 transport、把 interaction 误解释成 latent drift、把 context effect 误解释成共享动力学，谁就更接近下一篇真正重要的论文。
