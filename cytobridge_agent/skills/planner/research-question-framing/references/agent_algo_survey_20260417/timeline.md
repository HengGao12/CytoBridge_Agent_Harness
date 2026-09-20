# Algorithm Design Literature Timeline

All included papers are listed in chronological order.

## 1996

- [THE GEOMETRY OF OPTIMAL TRANSPORTATION](notes/1996__math__acta_mathematica__the_geometry_of_optimal_transportation.md) | `math` | Acta Mathematica, 177(2), 113-161
  贡献: 这篇文章解决的是现代 OT 理论最基础的一层 gap: 在平方欧氏距离之外，更一般的严格凸成本和严格凹距离成本下，Monge-Kantorovich 问题什么时候存在唯一的最优映射，最优映射的几何结构到底是什么。
  重要性: 因为它把 OT 从平方代价的特殊情形推进到了更一般的成本，并且不是弱化问题，而是给出了“存在 + 唯一 + 几何表征”这一整套结论。
  启发: 如果算法想学习“细胞分布从 t 到 t+1 的映射”，优先考虑势函数/势场参数化，而不是直接学任意黑箱映射。

## 2014

- [A survey of the Schrödinger problem and some of its connections with optimal transport](notes/2014__math__dcds__a_survey_of_the_schrodinger_problem_and_its_connections_with_optimal_transport.md) | `math` | Discrete and Continuous Dynamical Systems, 34(4), 1533-1574
  贡献: 这篇文章填的是“概念与理论桥梁”的 gap。
  重要性: 因为它把一个跨概率、控制、OT、统计物理的碎片化主题整理成了可操作的统一框架，而且不是纯文献罗列，还补充了 Markov 结构、Benamou-Brenier 形式和 slowing-down 极限等关键结果。
  启发: 如果已有 RNA velocity、发育方向或物理先验，不该直接做无先验 OT，而应优先考虑“相对参考过程的最小熵修正”。

## 2016

- [On the Relation Between Optimal Transport and Schrödinger Bridges A Stochastic Control Viewpoint](notes/2016__math__jota__on_the_relation_between_optimal_transport_and_schrodinger_bridges.md) | `math` | Journal of Optimization Theory and Applications, 169(2), 671-691
  贡献: 这篇文章解决的核心 gap 是：
  重要性: 因为它不是简单复述 SB 是 OT 的噪声版，而是给出了更细的结构关系：
  启发: 设计单细胞时序模型时，应优先问“有没有 prior dynamics”，例如 RNA velocity、已知发育方向、细胞周期漂移，而不是默认自由 OT。

## 2017

- [Single-cell entropy for accurate estimation of differentiation potency from a cell’s transcriptome](notes/2017__journal__nature_communications__single_cell_entropy_for_accurate_estimation_of_differentiation_potency_from_a_cells_transcriptome.md) | `journal` | Nature Communications
  贡献: 它解决的是如何从单细胞转录组中定量估计“分化潜能/可塑性”，并进一步识别干性亚群、药物耐受癌症干细胞和群体内受调控的异质性。它不直接重建长程轨迹，但为 fate/trajectory 分析提供了一个强有力的状态变量。
  重要性: 它把“分化潜能”这个长期停留在概念层面的量，变成了一个可在大规模 scRNA-seq 上直接计算的系统生物学量，并且跨正常发育和癌症体系做了大量验证。更重要的是，它不靠训练、不靠 marker 列表，而是利用网络结构把单细胞分析从纯表达几何推进到 network-aware state estimation，这一点在 2017 年非常新。
  启发: agent 在设计算法时，不应只搜索“如何连线成轨迹”，还应主动搜索“是否存在可解释的状态势函数/分化势评分”。

- [SLICE determining cell differentiation and lineage based on single cell entropy](notes/2017__journal__nucleic_acids_research__slice_determining_cell_differentiation_and_lineage_based_on_single_cell_entropy.md) | `journal` | Nucleic Acids Research
  贡献: 作者要解决的是：在没有时间标签、起点终点标注或经典 marker 先验时，如何仅从 scRNA-seq 推断单细胞的分化程度和分支谱系。生物学上它重点验证了肺泡 II 型细胞、骨骼肌成肌细胞、人早期胚胎和胚胎肺间充质等体系，并在小鼠胚胎肺中提出了新的成纤维细胞分支假说。
  重要性: 这篇文章抓住了 2016-2017 年单细胞轨迹推断的一个核心痛点：很多方法会给出一条“看起来像轨迹”的结构，但方向要靠人指定。SLICE 用 entropy 给出了一个相对可解释、与外部先验弱耦合的方向信号，并且在多套真实数据上做了成功验证，还提出了新的肺间充质分支假说，因此在当时具有明显的方法新意和生物应用价值。
  启发: 在自动设计算法时，可以把“root/state ordering prior”当成一个独立模块，而不是让主模型同时解决所有问题。

## 2018

- [Slingshot cell lineage and pseudotime inference for single-cell transcriptomics](notes/2018__journal__bmc_genomics__slingshot_cell_lineage_and_pseudotime_inference_for_single_cell_transcriptomics.md) | `journal` | BMC Genomics
  贡献: 它解决的是：在噪声很高、分支可能不止一条的单细胞表达数据里，如何稳定地恢复 lineage 结构和 pseudotime。文章重点不是新生物发现，而是为后续动态基因分析提供更鲁棒的轨迹骨架。
  重要性: 它抓住了当时 trajectory inference 的一个实际痛点：很多方法要么不稳，要么分支一复杂就失效。Slingshot 用非常工程化但又清晰的模块化设计，把 branch-aware lineage + pseudotime 做得稳定、可解释、易嵌入 pipeline，因此在社区里很快变成常用基线。
  启发: 自动设计算法时，模块化优先于一体化黑盒：全局拓扑、局部排序、方向信息可以由不同模块分别解决。

- [scEpath energy landscape-based inference of transition probabilities and cellular trajectories from single-cell transcriptomic data](notes/2018__journal__bioinformatics__scepath_energy_landscape_based_inference_of_transition_probabilities_and_cellular_trajectories_from_single_cell_transcriptomic_data.md) | `journal` | Bioinformatics
  贡献: 作者想解决的问题是：如何把 Waddington landscape 从隐喻变成可计算对象，从而在单细胞数据中推断细胞状态转移概率、谱系关系和 pseudotime。生物应用包括人类早期胚胎、肺上皮发育和成肌分化。
  重要性: 它的价值不在于单点指标领先多少，而在于把一个长期广泛使用的生物学比喻定量化，并把 landscape、transition probability 和 trajectory 三件事连成了一个完整方法链。对当时大量仍停留在“画一条 pseudotime 曲线”的方法来说，这是一个更有理论包装也更有解释性的替代方案。
  启发: landscape 型标量势函数在自动算法设计中很有价值，尤其适合作为 OT/flow/velocity 模型的辅助先验或 model selection 维度。

- [Unbalanced Optimal Transport Dynamic and Kantorovich Formulation](notes/2018__math__journal_of_functional_analysis__unbalanced_optimal_transport_dynamic_and_kantorovich_formulations.md) | `math` | Journal of Functional Analysis, 274(11), 3090-3123
  贡献: 这篇文章补的是 unbalanced OT 里最关键的理论缺口：
  重要性: 因为它基本完成了 unbalanced OT 版的“Benamou-Brenier + Kantorovich + 对偶 + 度量 + 极限连接”全家桶。
  启发: 对有增殖/死亡的单细胞时序数据，必须把质量变化放进主模型，而不是只在 OT 之后再估计 growth。

- [RNA velocity of single cells](notes/2018__journal__nature__rna_velocity_of_single_cells.md) | `journal` | Nature
  贡献: 这篇文章解决的是单细胞快照数据缺少方向信息的问题。它试图回答：一个细胞当前的表达状态已知时，短时间后它最可能往哪里走。生物上作者用神经嵴、海马发育和人胚脑等体系证明它能恢复 lineage direction 和 fate tendency。
  重要性: 它第一次从标准 scRNA-seq 数据中提炼出“方向”这一关键信息，而且实现方式足够简单、可复现、跨平台。这个贡献直接改变了领域对快照单细胞数据可推断信息边界的认识，因此是典型的范式推动型工作。
  启发: 如果观测中存在能反映局部导数的额外信号，agent 应优先利用它来约束动力学，而不是只做几何插值。

## 2019

- [Optimal-Transport Analysis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming](notes/2019__journal__cell__optimal_transport_analysis_of_single_cell_gene_expression_identifies_developmental_trajectories_in_reprogramming.md) | `journal` | Cell
  贡献: 它要解决的是：在致密时间分辨的重编程实验中，如何从大规模单细胞快照恢复真正的发育/命运轨迹，而不是只给出一个几何上的伪时间。生物学上作者借此系统剖析了成纤维细胞重编程为 iPSC 的多分支命运、关键 TF 和旁分泌作用。
  重要性: 这是单细胞动力学领域非常关键的一篇论文，因为它把 OT 从抽象数学工具真正变成了可用于大规模生物时间序列分析的工作流，并且明确解决了“细胞会增殖/死亡”这个生物现实问题。再加上 31.5 万细胞的重编程资源本身极强，方法与生物发现相互放大，完全达到 Cell 级别。
  启发: 对 CytoBridge 这类 agent，WOT 是最重要的设计模板之一：时间信息、质量非守恒、ancestor/descendant distribution 都应该是一等公民。

- [PAGA graph abstraction reconciles clustering with trajectory inference through a topology preserving map of single cells](notes/2019__journal__genome_biology__paga_graph_abstraction_reconciles_clustering_with_trajectory_inference_through_a_topology_preserving_map_of_single_cells.md) | `journal` | Genome Biology
  贡献: 它要解决的是：单细胞数据常同时包含离散细胞类型和连续过渡过程，而“先聚类”和“直接做 trajectory”两条路线常常互相冲突。PAGA 试图先把全局拓扑搞清楚，再在这个拓扑约束下讨论局部连续变化。
  重要性: 因为它非常精准地解决了社区一个普遍痛点：很多 embedding 看起来漂亮，但 topology 错得很离谱。PAGA 把“保拓扑”提到首位，又给出了可扩展、可集成、可解释的实现，所以迅速成为单细胞分析的基础组件。
  启发: 对 agent 来说，PAGA 提供了很强的“先粗后细”原则：先确认 topology，再决定是否上 OT、velocity 或 flow matching。

- [Unnormalized optimal transport](notes/2019__math__journal_of_computational_physics__unnormalized_optimal_transport.md) | `math` | Journal of Computational Physics, 399, 108940
  贡献: 作者想解决的是一个非常务实的 gap：
  重要性: 因为它展示了一条非常典型也非常值得学的理论路线：
  启发: 这是一个很好的 ablation baseline: 如果只允许全局质量变化，模型能解释多少数据；解释不了的部分，就是状态相关 growth 的证据。

- [A comparison of single-cell trajectory inference methods](notes/2019__journal__nature_biotechnology__a_comparison_of_single_cell_trajectory_inference_methods.md) | `journal` | Nature Biotechnology
  贡献: 这篇文章解决的不是单一生物问题，而是 trajectory inference 领域本身的“方法选择问题”。面对几十种 TI 工具、不同输入要求和不同输出结构，研究者很难知道在自己的数据上该选谁。对任何要自动设计算法的 agent，这其实是上游决策问题。
  重要性: 这是典型的 field-shaping 基础设施论文。2019 年 trajectory inference 方法已经明显过多，社区非常需要一个标准化、可复现、可操作的 benchmark 与指南。它让“选方法”从经验和口碑变成更结构化的判断，因此影响力很大。
  启发: agent 不应该默认存在一个全局最优算法，而应根据预期拓扑、数据规模、时间信息和先验强度做条件化路由。

## 2020

- [Single-Cell Entropy to Quantify the Cellular Order Parameter from Single-Cell RNA-Seq Data](notes/2020__journal__biophysical_reviews_and_letters__single_cell_entropy_to_quantify_the_cellular_order_parameter_from_single_cell_rna_seq_data.md) | `journal` | Biophysical Reviews and Letters
  贡献: 这篇文章试图给单细胞转录组定义一个简单的“细胞有序度/干性”标量，用来量化发育进程、癌症进展和细胞分类。它不是完整轨迹方法，但属于为 fate/differentiation 建模提供低维状态变量的工作。
  重要性: 它能发表的原因更像是“一个简单但有一定可用性的工具化方法”。核心思想足够直接，应用场景清楚，也有几个数据集上的例子，但从影响力和方法深度看，它更像增量式方法而不是范式工作。
  启发: 简单标量 order parameter 仍然有价值，尤其适合做 root prior、异常检测和模型输出 sanity check。

- [TrajectoryNet A Dynamic Optimal Transport Network for Modeling Cellular Dynamics](notes/2020__math__icml__trajectorynet.md) | `math/mlconf` | ICML 2020
  贡献: 早期单细胞 OT 方法多是相邻时间点静态匹配，能给配对，但不能给连续时间路径。
  重要性: 它第一次把“连续时间 dynamic OT”明确做成了单细胞轨迹推断主方法。
  启发: CytoBridge 的一个关键方向就是继承它的“全局连续动力学”视角，而不是退回相邻时间点拼接。

- [Fisher information regularization schemes for Wasserstein gradient flows](notes/2020__math__jcp__fisher_regularization_wasserstein_gradient_flows.md) | `math` | Journal of Computational Physics (2020)
  贡献: 经典 JKO/Wasserstein gradient flow 数值离散在实际求解时常常非凸、难优化，还容易带来非负性和稳定性问题。
  重要性: 它把 JKO、Benamou-Brenier 和 Schrödinger bridge 的联系转成了可运行的稳定数值方案。
  启发: 如果 CytoBridge 后续要显式求解 Fokker-Planck 或 density evolution 子问题，可以优先考虑 Fisher/score 类结构正则，而不是只靠经验平滑。

- [Generalizing RNA velocity to transient cell states through dynamical modeling](notes/2020__journal__nature_biotechnology__generalizing_rna_velocity_to_transient_cell_states_through_dynamical_modeling.md) | `journal` | Nature Biotechnology
  贡献: 它要解决的是原始 RNA velocity 在 transient cell state 和 heterogeneous subpopulation kinetics 下经常失效的问题。生物学上它希望更可靠地恢复发育过程中的 lineage direction、latent time 和 driver genes，尤其是在真正非稳态的系统里。
  重要性: 因为它不是边缘改进，而是对一个极具影响力的方法做了关键纠偏。RNA velocity 2018 年后被大量使用，但其假设问题也很快暴露。scVelo 给出了机制上更合理、工程上又跑得动的升级版，因此是高影响的“第二代标准方法”。
  启发: 自动设计算法时，必须显式检查模型假设失效的征兆，并能升级到更一般的机制模型。

- [Context specificity of the EMT transcriptional response](notes/2020__journal__nature_communications__context_specificity_of_the_emt_transcriptional_response.md) | `journal` | Nature Communications
  贡献: 它要回答的是：EMT 到底是不是一个统一的线性转录程序，还是强烈依赖细胞背景和诱导条件。这个问题对任何想建模细胞状态转换、条件扰动和反事实推断的算法都非常关键，因为它直接决定能不能假设存在通用 transition program。
  重要性: 这篇文章的价值在于用极大规模、强控制的单细胞实验设计，推翻了“EMT 有单一通用程序”的简单叙事。对发育、肿瘤和细胞状态转换领域来说，这是非常重要的认知更新；同时数据规模和实验设计也足够强。
  启发: agent 不能默认所有条件共享同一套 transition vector field；context-conditioned dynamics 应该是默认候选。

- [Entropic Optimal Transport between Unbalanced Gaussian Measures has a Closed Form](notes/2020__math__neurips__entropic_ot_unbalanced_gaussians.md) | `math/mlconf` | NeurIPS 2020
  贡献: 高斯 OT 有经典闭式解，但熵正则 OT 和非平衡熵正则 OT 一直主要靠 Sinkhorn 数值求解，缺少可解析基准。
  重要性: 它第一次给出了非平凡的熵正则 OT 闭式解，理论含金量很高。
  启发: 如果 CytoBridge 在 latent 空间里局部接近高斯，可以用这篇工作的闭式结果校验 UOT 模块是否学对了质量变化和耦合方向。

- [The mean field Schrödinger problem ergodic behavior, entropy estimates and functional inequalities](notes/2020__math__ptrf__mean_field_schrodinger_problem.md) | `math` | Probability Theory and Related Fields (2020)
  贡献: 经典 Schrödinger bridge 主要针对独立布朗粒子，不能直接处理带相互作用的大群体粒子系统。
  重要性: 它把 mean-field 相互作用正式带入 SB 理论，明显扩展了经典路径熵输运的适用边界。
  启发: 如果 CytoBridge 未来要建模 cell-cell interaction，最合理的理论底座不是普通 SB，而是 mean-field SB。

- [A Machine Learning Framework for Solving High-Dimensional Mean Field Game and Mean Field Control Problems](notes/2020__math__pnas__a_machine_learning_framework_for_high_dimensional_mean_field_game_and_mean_field_control_problems.md) | `math` | Proceedings of the National Academy of Sciences, 117(17), 9183-9193
  贡献: 这篇文章解决的是高维 mean-field / optimal control 数值求解的 gap。
  重要性: 因为它抓住了一个真正重要的 bottleneck: 不是再加一点理论修补，而是把高维 MFG/MFC 从“原则上可写出 PDE”推进到“在标准工作站上可近似求解”。
  启发: 对高维 latent state 的连续时间动力学学习，优先考虑“potential + characteristic + PDE penalty”这条路，而不是只做离散时间回归。

## 2021

- [Likelihood Training of Schrödinger Bridge using Forward-Backward SDEs Theory](notes/2021__math__iclr__likelihood_training_schrodinger_bridge_fbsde.md) | `math/mlconf` | ICLR 2022
  贡献: SB 在理论上很优雅，但生成模型社区更熟悉的是 likelihood/score 训练；两者之间当时缺一条清晰的桥。
  重要性: 它补上了 SB 与 likelihood training 之间的关键缺口，这是一个明确的理论与方法空白。
  启发: 对 CytoBridge 来说，这篇文章最大的启发是: loss 不必拍脑袋设计，可以从 FBSDE/HJB/Fokker-Planck 一致性推出。

- [Deep Generative Learning via Schrödinger Bridge](notes/2021__math__icml__deep_generative_learning_via_schrodinger_bridge.md) | `math/mlconf` | ICML 2021
  贡献: 生成模型里 SB 很有理论吸引力，但早期更多停留在概念层面，缺少成体系的深度学习训练方案。
  重要性: 它较早把 SB 真正引入深度生成学习主舞台，且不是单纯复述理论，而是给出可跑的训练方案。
  启发: CytoBridge 可以把这篇文章看作“把 SB 变成神经网络训练对象”的起点。

- [Dissecting transition cells from single-cell transcriptome data through multiscale stochastic dynamics](notes/2021__journal__nature_communications__mutrans.md) | `journal` | Nature Communications
  贡献: 作者要解决的是：只有 snapshot scRNA-seq 时，怎样把稳定细胞状态和真正处在命运切换中的 transition cells 区分开，并给出跨细胞状态的高概率转变路径。文中在 EMT、iPSC 分化和血液发育等系统里，把“过渡细胞是谁、往哪里走”变成了可以直接回答的问题。
  重要性: 这篇工作的亮点是把随机动力系统和单细胞轨迹推断真正接起来了，而且解决的是当时很多方法没有明确解决的 transition-cell 识别问题。它同时给出理论对象、可运行算法和跨平台实证，因此达到了 Nature Communications 级别的方法文章标准。
  启发: 对 agent 来说，不能只输出一条平滑轨迹，还要显式区分稳定态、过渡态和高概率跃迁路径。若要做新算法，值得把“transition-cell detection”作为一级目标，并让模型在 latent space 中直接暴露 attractor、barrier 和 path-level summary，而不是只给点到点 coupling。

- [Generative modeling of single-cell time series with PRESCIENT enables prediction of cell trajectories with interventions](notes/2021__journal__nature_communications__prescient.md) | `journal` | Nature Communications
  贡献: PRESCIENT直指时间序列 scRNA-seq 的核心难题：如何在物理时间上重建细胞分化过程，并预测基因干预后命运轨迹如何改变。作者重点验证了造血和胰岛 beta 细胞分化中的 fate bias 与 intervention 效果。
  重要性: 这篇文章把“从时间序列单细胞快照里学习一个可生成、可干预、可物理解释的动态模型”落到了实处，还用 lineage tracing 做了严格验证。相比当时偏总结性或耦合性的 TI 方法，它第一次把 stochastic dynamics、growth 和 intervention 放进了同一个可运行框架里。
  启发: 如果 agent 要自动设计算法，PRESCIENT说明“能否生成未测时间点和未做过的 perturbation”是非常强的评价维度。CytoBridge 一类方法应继续保留生成式 latent dynamics、growth 显式建模和可执行的 intervention 接口，而不是退回到静态 pseudotime。

- [LineageOT is a unified framework for lineage tracing and trajectory inference](notes/2021__journal__nature_communications__lineageot.md) | `journal` | Nature Communications
  贡献: LineageOT解决的是：当单细胞测序和 lineage tracing 同时可得时，怎样把“状态相似性”和“谱系亲缘性”放进同一个轨迹推断框架，尤其是在复杂分叉结构里更准确地恢复祖先-后代关系和 fate coupling。
  重要性: 这篇文章的重要性在于它首次把 lineage tracing 与 trajectory inference 从概念相关推进到统一数学框架，并且清楚说明了为何 lineage 信息能弥补纯状态方法的不可辨识性。对当时迅速增长的 scLT 数据而言，这是非常及时且有方法学深度的工作。
  启发: CytoBridge 以后若接入 lineage tracing 或 clonal data，不应把它只当额外特征，而应直接进入 coupling 或 latent prior。更一般地，agent 设计新算法时要优先寻找“能减少不可辨识性”的额外视角，而不是一味堆更复杂的 dynamics 网络。

- [Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling](notes/2021__math__neurips__diffusion_schrodinger_bridge.md) | `math/mlconf` | NeurIPS 2021
  贡献: 经典 score-based diffusion 依赖很长的前向加噪时间，目的是把终态推到接近高斯，这会带来很多离散步数和误差。
  重要性: 它非常清楚地说明了 diffusion 模型和 Schrödinger bridge 的关系，这在当时是极具影响力的统一观点。
  启发: 对 CytoBridge 来说，这篇文章的重要性在于: 不要把 diffusion 和 bridge 看成两条路线，而应把 diffusion 看成 bridge 的特例或第一步近似。

- [The most likely evolution of diffusing and vanishing particles Schrodinger Bridges with unbalanced marginals](notes/2021__math__siam_jco__unbalanced_schrodinger_bridges_diffusing_vanishing_particles.md) | `math` | SIAM Journal on Control and Optimization (2022)
  贡献: 之前面对 unequal-mass marginals，很多方法直接在连续性方程中加 source/sink，做法偏经验化。
  重要性: 它把“非平衡桥为什么应该是 bridge，而不是 ad hoc 源汇 PDE”讲得非常清楚。
  启发: 对细胞死亡、细胞流失、筛选存活等问题，CytoBridge 更适合显式学习 hazard/killing 机制。

## 2022

- [The Schrödinger Bridge between Gaussian Measures has a Closed Form](notes/2022__math__aistats__gaussian_schrodinger_bridge_closed_form.md) | `math` | AISTATS 2023
  贡献: 静态 Gaussian OT 有闭式解，但动态 Gaussian Schrödinger bridge 没有相应解析公式。
  重要性: 它补上了 Gaussian SB 这一块长期缺失的解析解，理论上非常整洁。
  启发: CytoBridge 可以把 Gaussian SB 当作 latent 局部近似的 sanity check。

- [Neural Lagrangian Schrödinger Bridge Diffusion Modeling for Population Dynamics](notes/2022__math__iclr__neural_lagrangian_schrodinger_bridge.md) | `math/mlconf` | ICLR 2023
  贡献: TrajectoryNet 一类 dynamic OT/CNF 方法能给连续路径，但路径是确定性的，难表达真实群体中的随机扩散行为。
  重要性: 它抓住了 TrajectoryNet 路线最明显的短板: 确定性过强。
  启发: CytoBridge 若要比纯 CNF 路线更贴近生物过程，应优先考虑 stochastic latent dynamics。

- [Variational Mixtures of ODEs for Inferring Cellular Gene Expression Dynamics](notes/2022__journal__arxiv__velovae.md) | `mlconf` | ICML 2022 (PMLR 162)
  贡献: 这篇工作瞄准的是基因表达动力学恢复，尤其是 latent time 不可观测、单祖细胞向多子命运分叉时，如何从 snapshot 数据里恢复每个细胞所处的动态阶段并预测未来状态。它试图把 RNA velocity 的思想推广成更完整的生成动力学模型。
  重要性: 它填补了传统 RNA velocity 与深生成时间恢复之间的 gap，把 latent time、future prediction 和 biochemistry-informed dynamics 放进同一个模型里。对当时 RNA velocity 方向来说，这是一条很自然且技术上有说服力的升级路线。
  启发: 若 agent 要设计连续动力学算法，VeloVAE提示一个重要策略：用弱生物机制约束去限制生成模型，而不是完全依赖数据拟合。对分叉系统，也可以考虑“连续变化的动力学族”而非硬切 branch-specific models。

- [Identifying multicellular spatiotemporal organization of cells with SpaceFlow](notes/2022__journal__nature_communications__spaceflow.md) | `journal` | Nature Communications
  贡献: SpaceFlow主要解决空间转录组中的“表达相似性”和“空间位置”如何联合建模，进而恢复随空间变化展开的伪时序/谱系模式。作者用它识别了发育心脏中的 evolving lineage，也分析了肿瘤-免疫区域的空间演化关系。
  重要性: 这篇工作的贡献在于把 ST 数据中的空间结构从“可视化背景”提升为“轨迹推断信号”，并用深图网络给出统一实现。对于刚快速发展的空间组学领域，这是一个很自然也很有用的方法学推进。
  启发: 如果 agent 未来要覆盖空间轨迹问题，空间约束应该在 latent representation 学习阶段就被显式编码，而不是只在结果可视化时使用。另一个启发是可以把空间和时间当作两种可交换但不等价的结构先验，设计统一的 multiview dynamics。

- [Deep Generalized Schrödinger Bridge](notes/2022__math__neurips__deep_generalized_schrodinger_bridge.md) | `math/mlconf` | NeurIPS 2022
  贡献: 经典 SB 擅长处理无交互或简单参考过程下的分布匹配，但对 mean-field game 里的相互作用和状态代价支持不足。
  重要性: 这是把 SB 从“分布桥”推进到“带交互和状态代价的群体控制”的关键扩展。
  启发: 如果 CytoBridge 后续要做 cell-cell interaction 或 condition-specific control，DeepGSB 提供了很强的模板。

- [Trajectory Inference via Mean-field Langevin in Path Space](notes/2022__math__neurips__trajectory_inference_mean_field_langevin.md) | `math/mlconf` | NeurIPS 2022
  贡献: 之前路径空间上的 min-entropy trajectory inference 理论已经提出，但对应的是无限维凸优化，几乎不可直接算。
  重要性: 它把一条非常数学化的 trajectory inference 路线真正做成了可计算算法，而且带有收敛保证。
  启发: CytoBridge 不必只做 neural field fitting，也可以吸收 path-space optimization 的思想。

- [Inference of cell state transitions and cell fate plasticity from single-cell with MARGARET](notes/2022__journal__nucleic_acids_research__margaret.md) | `journal` | Nucleic Acids Research
  贡献: MARGARET关注的是复杂拓扑下的单细胞轨迹和 fate plasticity 推断。它试图解决传统 TI 方法在多分支、环状或复杂几何中拓扑恢复不稳、terminal state 自动识别差、plasticity 难量化的问题。
  重要性: 它抓住了一个很实际的痛点：很多 TI 方法在简单树结构上有效，但到了复杂拓扑就明显失真。MARGARET在 topology、terminal state 和 plasticity 三个维度同时提升，而且 benchmark 做得较全面，因此能够在 NAR 这样的工具方法期刊站住脚。
  启发: 对 agent 来说，复杂拓扑恢复仍是必须单独优化的模块，不能假设所有系统都近似树结构。即便采用生成式动力学，仍值得把 topology robustness 和 plasticity quantification 作为独立评测项。

## 2023

- [Simulation-free Schrödinger bridges via score and flow matching](notes/2023__math__aistats__simulation_free_schrodinger_bridges.md) | `math` | AISTATS 2024
  贡献: 之前很多 SB 神经算法需要显式模拟学到的随机过程，训练很重，且高维生物数据上容易失真。
  重要性: 它把两个当时最热的训练范式 score matching 和 flow matching 真正统一到 SB 框架下。
  启发: 这篇论文对 CytoBridge 的启发非常直接: “静态耦合 + simulation-free 动态拟合”是现实可行的高维路线。

- [Predicting Cellular Responses with Variational Causal Inference and Refined Relational Information](notes/2023__journal__iclr__graphvci.md) | `mlconf` | ICLR 2023
  贡献: 这篇文章面向单细胞 perturbation response prediction，目标是在没有真实观测的情况下预测某个细胞在 counterfactual perturbation 下的转录组响应。它服务的核心生物问题是个体化细胞响应预测与药物/基因干预评估。
  重要性: 这篇工作补的是单细胞干预预测里“结构先验 + 因果表达 + 稳健群体效应估计”这一块空白。它既有方法创新，也有清晰的 OOD 评测和新的数据资源，所以符合当时 ICLR 对方法完整性的要求。
  启发: 对 CytoBridge 的 counterfactual 模块，一个重要方向是把 causal structure 直接融入生成模型，而不是只做条件生成。另一个启发是：当 agent 设计 perturbation 算法时，最好同时输出 cell-level response 和 population-level effect，避免只在一个尺度上优化。

- [Generalized Schrödinger Bridge Matching](notes/2023__math__iclr__generalized_schrodinger_bridge_matching.md) | `math/mlconf` | ICLR 2024
  贡献: 现有 matching/diffusion/flow 方法大多显式规定中间边缘，只适合动能最小化或简单运输目标。
  重要性: 它把 bridge matching 从“求桥”推进到“求带任务结构的桥”，拓宽了方法边界。
  启发: CytoBridge 如果要把生物先验写进动力学，GSBM 提供了很自然的模板: 把先验写成 state cost。

- [A Computational Framework for Solving Wasserstein Lagrangian Flows](notes/2023__math__icml__wasserstein_lagrangian_flows.md) | `math/mlconf` | ICML 2024
  贡献: SB、UOT、带物理约束的 OT 等问题都可以视作某种分布路径上的作用量最小化，但现有求解器通常是为单一问题手工打造。
  重要性: 这篇工作最强的地方不是某个单点技巧，而是给出一个统一抽象，把零散方法压缩进同一范式。
  启发: 这是最接近 CytoBridge-agent 核心目标的论文之一: 先定义 action family，再自动组合 kinetic/potential terms。

- [Multi-omic single-cell velocity models epigenome–transcriptome interactions and improves cell fate prediction](notes/2023__journal__nature_biotechnology__multivelo.md) | `journal` | Nature Biotechnology
  贡献: MultiVelo要回答的是：仅凭 RNA velocity 为什么常常不够，以及如何把染色质开放状态与转录动态联合起来，更准确地推断命运方向和调控时滞。它瞄准的是多组学时代对“表观-转录耦合动力学”的直接建模。
  重要性: 这篇工作补上了单细胞动力学里最自然的一块缺失信息，即 epigenome-transcriptome 时间关系。它不是简单做多组学整合，而是把多组学真正写进了 velocity 方程，因此方法学新意很明确。
  启发: CytoBridge 后续若接入 multi-omics，最值得做的不是共享 latent space 本身，而是跨模态的时间滞后和因果先后建模。对 fate prediction，早期调控模态往往比 RNA 本身更早暴露方向信息。

- [Robust mapping of spatiotemporal trajectories and cell–cell interactions in healthy and diseased tissues](notes/2023__journal__nature_communications__stlearn_psts.md) | `journal` | Nature Communications
  贡献: 这篇文章聚焦于空间组织中的动态变化过程，想同时回答“组织内部的转录状态如何沿空间展开成 trajectory”以及“哪些区域存在更强的 cell-cell interaction”。作者在神经发育、损伤修复和肿瘤进展等场景中展示了这两个问题可以联动分析。
  重要性: 它抓住了空间组学领域一个很有代表性的痛点：分析流程碎片化。作者把空间轨迹、空间交互和空间补全整合成一个相互增强的框架，并在多种疾病和发育场景里验证，因此方法完整度较高。
  启发: CytoBridge 若扩展到空间场景，不能只把空间坐标当 covariate；更合理的做法是围绕同一个空间图同时组织 trajectory、interaction 和 denoising。另一个启发是：interaction 模块与 trajectory 模块最好共享底层表示，而不是彼此孤立。

- [Diffusion Schrödinger Bridge Matching](notes/2023__math__neurips__diffusion_schrodinger_bridge_matching.md) | `math/mlconf` | NeurIPS 2023
  贡献: 之前 SB 数值法不是维度扩展性差，就是迭代误差累积明显。
  重要性: 它解决的是 SB 数值法中最核心的“怎么又准又能扩展”问题。
  启发: 如果 CytoBridge 发现一阶段 simulation-free 方法不够准，DSBM 这种迭代精修思路很值得采用。

- [Generative Entropic Neural Optimal Transport To Map Within and Across Spaces](notes/2023__math__neurips__genot.md) | `math/mlconf` | NeurIPS 2024
  贡献: 传统离散 OT 在单细胞基因组学里很常用，但扩展性、隐私、out-of-sample 推断都很差。
  重要性: 它把 OT/FM 真正按单细胞需求重新设计了，而不是简单把视觉方法移植过来。
  启发: 对 CytoBridge 而言，GENOT 说明“先求静态耦合、再学随机条件映射”是一条很强的工程路线。

- [Conditional Flow Matching Simulation-Free Dynamic Optimal Transport](notes/2023__math__arxiv__conditional_flow_matching.md) | `math` | arXiv preprint arXiv:2302.00482
  贡献: CNF 过去多依赖 simulation-based maximum likelihood，训练成本高且不稳定。
  重要性: 它把 CNF 的训练范式改写了，影响非常大。
  启发: CytoBridge 若追求可扩展和稳定，CFM 是非常自然的候选基础模块。

- [I textsuperscript{2}SB Image-to-Image Schrödinger Bridge](notes/2023__math__arxiv__i2sb.md) | `math` | arXiv preprint arXiv:2302.05872
  贡献: 标准条件 diffusion 仍然从噪声开始生成，没有充分利用“源端点本身已经很有信息”的场景。
  重要性: 它展示了 SB 在高维条件生成里可以被做得既可解释又高效。
  启发: 若 CytoBridge 未来要做 perturbation/control 这类“已知源状态到目标状态”的条件预测，I2SB 的思路很值得借鉴。

- [Entropic regularisation of unbalanced optimal transportation problems](notes/2023__math__arxiv__entropic_regularisation_unbalanced_ot.md) | `math` | arXiv preprint arXiv:2305.02410
  贡献: UOT 里“熵正则化”经常被当成一个统一概念使用，但实际上不同加法位置会得到不同问题。
  重要性: 虽然不是顶会文章，但它对 UOT 目标函数设计的澄清非常关键。
  启发: CytoBridge 一旦要做 entropic UOT，就必须显式记录采用的是哪一种正则化语义。

- [Unbalanced Diffusion Schrödinger Bridge](notes/2023__math__arxiv__unbalanced_diffusion_schrodinger_bridge.md) | `math/mlconf` | arXiv preprint arXiv:2306.09099 (ICML 2023 workshop poster only)
  贡献: 现有 neural SB/DSB 默认边缘是概率测度，等价于质量守恒，这和生物系统中的 birth/death 完全不匹配。
  重要性: 这篇工作直接命中了 SB 路线在生物应用里最大的短板: 质量守恒假设不成立。
  启发: CytoBridge 若想真正面向生物过程，unbalanced bridge 几乎是必选方向，而不是锦上添花。

## 2024

- [Neural McKean-Vlasov Processes Distributional Dependence in Diffusion Processes](notes/2024__math__aistats__neural-mckean-vlasov-processes.md) | `math` | AISTATS 2024
  贡献: 这篇文章针对的是 neural SDE 领域一个长期空白：大多数模型默认样本路径彼此独立，只适合 Itô-SDE；但很多真实系统是分布依赖的，也就是 drift/diffusion 会受当前总体分布影响。过去缺少面向 MV-SDE 的通用神经参数化和估计方法。
  重要性: 它的重要性在于把“分布依赖动力学”从概念推进成了可训练神经模型家族。对于许多涉及群体效应、注意力、交互粒子和细胞通讯的任务，这是一个基础设施式的贡献，而不是只在某个 benchmark 上多提几个点。
  启发: CytoBridge 未来如果要把 interaction 做实，不能只停留在经验性的 cell-cell communication score 上。这篇文章给出的启发是，可以把 interaction term 直接设计成 law-dependent drift/growth 模块，让网络输入当前细胞群体的分布表示，再去预测单细胞层面的速度和增长。这样比事后再拼接邻域特征更接近 mean-field 理论，也更适合自动化搜索。

- [A physics-informed neural SDE network for learning cellular dynamics from time-series scRNA-seq data](notes/2024__journal__bioinformatics__pi_sde.md) | `journal` | Bioinformatics (Proceedings of ECCB 2024)
  贡献: PI-SDE试图解决仅靠数据驱动模型学习 Waddington landscape 时，预测和可解释性都不稳定的问题。作者关注如何在 time-series scRNA-seq 上更准确地恢复潜在能量景观，并提升 held-out timepoint prediction 与生物解释性。
  重要性: 它代表了一类很有意义的方法路线：把 physics-informed learning 引入单细胞动力学，而不只是做纯黑盒深度学习。对于当时越来越强调可解释和泛化稳定性的 single-cell dynamics 领域，这是一个清晰的方向性贡献。
  启发: 对 agent 来说，物理/数学先验应被视为可组合模块，而不是可有可无的装饰项。设计新算法时，可以系统比较“纯数据驱动”“弱先验约束”“强物理约束”三种模式的收益与偏差。

- [DeepVelo deep learning extends RNA velocity to multi-lineage systems with cell-specific kinetics](notes/2024__journal__genome_biology__deepvelo.md) | `journal` | Genome Biology
  贡献: DeepVelo要解决的是 RNA velocity 在多 lineage、时间变速率系统里经常失败的问题。它希望在复杂异质系统中恢复更可信的 cell-specific kinetics、developmental stage 和调控 driver genes。
  重要性: DeepVelo抓住了 velocity 领域的关键失败模式，并用深图网络给出了一个可扩展替代方案。它兼顾了准确性、driver gene 识别和在复杂系统中的适用性，因此在 Genome Biology 这样的方法期刊里很有竞争力。
  启发: 对 CytoBridge 来说，局部图结构是学习 cell-dependent kinetics 的天然载体。未来若要做 velocity-aware dynamics，可以把图神经网络当作估计局部 drift/growth 的模块，而把全局 transport 留给更适合的生成框架。

- [Mapping lineage-traced cells across time points with moslin](notes/2024__journal__genome_biology__moslin.md) | `journal` | Genome Biology
  贡献: moslin要解决的是 destructive time-series lineage-tracing 实验里，大量 lineage 信息无法被现有方法充分利用的问题。作者希望用时间结构化的 lineage 与 gene expression 一起恢复 fate probability、decision driver genes 和祖先-后代映射。
  重要性: 它相比 LineageOT 的推进在于：不只处理单对时间点，而且更系统地处理了 destructive time-series 下多时间点 lineage information 的利用。对逐渐普及的 scLT 数据，这个问题非常实际。
  启发: 对于带 lineage 的系统，agent 应优先设计“跨全部时间点联合求解”的模型，而不是相邻时间点贪心串联。另一个启发是，lineage-noise-aware interpolation 在现实数据里非常重要，不能默认 lineage tree 是真值。

- [Flow Matching on General Geometries](notes/2024__mlconf__iclr__flow_matching_on_general_geometries.md) | `mlconf` | ICLR 2024
  贡献: Flow Matching 在 2023 年之后迅速成为连续生成建模的核心范式，但其标准形式基本建立在欧氏空间上。对流形数据，已有方法要么训练时依赖昂贵仿真，要么需要分数/散度近似、在高维下方差大，要么只适用于简单几何。本文补上的 gap 是：如何把 FM/CFM 的“直接回归目标向量场、尽量 simulation-free”的优点，推广到一般 Riemannian 几何，尤其是 mesh 这类复杂几何。
  重要性: 它能发 ICLR 2024，我觉得原因很明确：
  启发: 如果单细胞状态空间的有效几何明显非欧氏，直接用线性插值很可能学到离开数据流形的伪轨迹；应把“路径几何”作为独立模块来设计。

- [A relay velocity model infers cell-dependent RNA velocity](notes/2024__journal__nature_biotechnology__celldancer.md) | `journal` | Nature Biotechnology
  贡献: cellDancer要解决的是传统 RNA velocity 采用全局统一动力学参数时，在多阶段、多分支和异质系统里经常失真。它把问题重新定义为：每个细胞都应有本地、细胞依赖的速度和动力学参数。
  重要性: 这篇工作的价值在于把 RNA velocity 的“全局常速率”瓶颈正面拆掉，并给出一个在模拟和真实多分支系统上都成立的深度学习替代方案。它既提升了准确性，又明确展示了何时传统 velocity 假设会失败。
  启发: CytoBridge 若继续做 velocity/transition 子模块，应该优先考虑 state-dependent 或 cell-dependent kinetics，而不是全局共享参数。更进一步，局部动力学和全局 transport 可以设计成分层组合，而不必强迫一个模型同时承担全部尺度。

- [TFvelo gene regulation inspired RNA velocity estimation](notes/2024__journal__nature_communications__tfvelo.md) | `journal` | Nature Communications
  贡献: TFvelo想解决的是：很多数据里 unspliced/spliced 信号太弱，经典 RNA velocity 无法稳健拟合，但调控因子与靶基因之间的相位延迟仍然存在。它试图借助 TF-target 关系把 velocity 推广到没有明显 splicing 信息的场景。
  重要性: TFvelo提供了一个很新鲜的视角：velocity 不一定只能来自剪接延迟，也可以来自转录调控延迟。这种重新定义问题的方式比单纯再拟合一次 phase portrait 更有方法学价值。
  启发: CytoBridge 在做 dynamics inference 时，不应把 velocity 信号来源限定为 RNA kinetics。本质上，任何能提供系统性相位延迟的模态或先验，都可能成为 dynamics 估计器的输入。

- [Reconstructing growth and dynamic trajectories from single-cell transcriptomics data](notes/2024__journal__nature_machine_intelligence__tigon.md) | `journal` | Nature Machine Intelligence
  贡献: TIGON针对 time-series scRNA-seq 中一个非常核心的问题：不仅要恢复状态转变轨迹，还要同时恢复 cell population growth，并进一步挖掘 temporal GRN 和 communication。作者强调 growth 若被忽略，很多 temporal inference 会系统偏差。
  重要性: 它比早期 OT 轨迹方法更进一步，把 growth 从“应不应该考虑”变成“必须和 trajectory 一起学”的主变量，并在方法、数值求解和生物解释上都给出完整故事。这个问题定义与技术实现都比较扎实。
  启发: TIGON直接证明了 growth 不是可选增强项，而是单细胞动力学的主体变量之一。CytoBridge 当前路线里把 velocity、growth 和 interaction 分开建模是合理的，但更进一步应考虑这些量在同一桥过程里的联合可辨识性。

- [CellRank 2 unified fate mapping in multiview single-cell data](notes/2024__journal__nature_methods__cellrank2.md) | `journal` | Nature Methods
  贡献: CellRank 2要解决的问题是 fate mapping 经常只能依赖某一种 view，例如表达相似性或 RNA velocity，而无法统一使用时间点信息、代谢标记、多模态和跨时间转移。它希望在一个框架里恢复 terminal states、fate probabilities 和 lineage drivers。
  重要性: 这篇文章的重要性在于把单细胞 fate mapping 从“某一种数据的专用工具”升级为“统一处理多视角 transition evidence 的框架”。在多模态单细胞快速普及的背景下，这种框架化能力非常关键。
  启发: CytoBridge 可以借鉴 CellRank 2 的思路，把不同证据源先转为统一的 transition object，再决定是做 OT、SB 还是 absorbing Markov analysis。对 agent 自动设计来说，view modularity 是很有价值的系统设计原则。

- [Deep generative modeling of transcriptional dynamics for RNA velocity analysis in single cells](notes/2024__journal__nature_methods__velovi.md) | `journal` | Nature Methods
  贡献: veloVI针对的是 RNA velocity 在真实数据中常见的两个痛点：不确定性无法量化，以及研究者通常不知道当前数据到底适不适合做 velocity。它希望让 velocity 不再只是一个箭头图，而是一个带 posterior uncertainty 的生成模型输出。
  重要性: 这篇文章击中了 RNA velocity 社区最常被质疑的点：过度自信和适用性不明。通过把 posterior uncertainty 和 dynamical model 结合起来，它把 velocity 从启发式工具推进成了更严肃的统计生成框架。
  启发: 对 agent 来说，任何轨迹或 velocity 方法都应该同时输出“结果”和“可信度”。如果新算法不能量化 uncertainty 或 detect failure mode，就很难在自动化系统里安全使用。

- [Spatial transition tensor of single cells](notes/2024__journal__nature_methods__stt.md) | `journal` | Nature Methods
  贡献: STT 要补的是空间转录组里最难的一块：如何同时利用空间位置和 RNA splicing 信息恢复 cell-state-specific spatial dynamics，而不是只做空间聚类或无空间的 velocity。作者用它分析 EMT、血液发育、脑和心脏发育中的空间状态迁移。
  重要性: 这篇文章的重要性在于它不是简单把空间加到已有 velocity 工具上，而是为“空间状态转移”单独定义了新的表示对象和 multiscale 推断流程。对空间动力学分析来说，这是很明确的方法学推进。
  启发: CytoBridge 若进入空间轨迹建模，可以考虑把 dynamics 表示成 tensor 或 operator，而不必局限于向量场。这样更容易兼顾多吸引子、多路径以及空间约束下的异质迁移模式。

- [Metric Flow Matching for Smooth Interpolations on the Data Manifold](notes/2024__mlconf__neurips__metric_flow_matching_for_smooth_interpolations_on_the_data_manifold.md) | `mlconf` | NeurIPS 2024
  贡献: 这篇文章抓住了 flow matching 在轨迹推断里的一个非常关键的缺陷：经典 CFM/OT-CFM 的条件路径默认是欧氏直线，而真实系统动力学，尤其单细胞发育过程，通常支持在弯曲数据流形上。直线插值会把训练点放到“离数据很远、模型最不确定”的区域，导致中间时间点重建失真，学出的向量场也偏离真实轨迹。
  重要性: 这篇论文能发 NeurIPS 2024，核心原因是它指出并解决了一个非常真实、而且过去 FM 社区没有认真处理的 failure mode：欧氏直线插值在真实数据流形上常常是错的。
  启发: 对单细胞 snapshot dynamics，路径设计不该默认直线；“中间态是否仍落在可信细胞流形附近”本身就是首要建模问题。

- [Modeling single cell trajectory using forward-backward stochastic differential equations](notes/2024__journal__plos_computational_biology__fbsde.md) | `journal` | PLOS Computational Biology
  贡献: 这篇文章要解决的是 OT 只能给端点之间静态耦合、很难表达非线性路径形状的问题。作者希望用连续时间的随机微分方程直接建模单细胞发育轨迹，并在多个数据集上优于 Waddington-OT 和 TrajectoryNet。
  重要性: 它把 trajectory inference 从“静态匹配”推进到“连续随机过程求解”，并在 benchmark 上清楚展示了非线性轨迹建模的优势。对单细胞动力学建模而言，这是一条颇有数学味道但又落地的路线。
  启发: CytoBridge 的一个直接启发是：forward-backward 结构非常适合做 bridge 型动态推断，尤其是需要同时满足起终点分布约束时。另一个启发是，若想建模交互，不要只在 loss 上加小修补，而应在过程层面显式写进 dynamics。

- [Improving and generalizing flow-based generative models with minibatch optimal transport](notes/2024__math__tmlr__minibatch-ot-cfm.md) | `math` | Transactions on Machine Learning Research (TMLR), 2024
  贡献: 这篇文章补的是 flow-based generative model 训练上的一个关键缺口：传统 CNF 最大似然训练需要数值模拟，成本高且不稳定；早期 flow matching 又通常依赖高斯源分布和特定桥接构造，不够统一，也没有把 dynamic OT 和 Schrödinger bridge 很好地放到 simulation-free 的同一框架里。
  重要性: 这篇文章的价值在于把 flow matching、dynamic OT、Schrödinger bridge 和 CNF 训练统一到了一个更干净的目标函数下，而且这个目标在实践上明显更稳定、计算上明显更便宜。它不是只多了一个 trick，而是重新组织了这条技术路线的训练方式。
  启发: 对 CytoBridge 最直接的启发是：速度场学习可以优先设计成“基于 coupling 的监督回归问题”，而不是一上来做昂贵的路径仿真或内外层优化。更具体地说，可以先用 OT/RUOT/WFR 构造局部桥接对，再做 simulation-free velocity/growth regression。这样既保留 OT 的几何先验，又更利于自动化大规模算法搜索。

- [Latent Schrödinger Bridge Diffusion Model for Generative Learning](notes/2024__math__arxiv__latent-schrodinger-bridge-diffusion.md) | `math` | arXiv preprint arXiv:2404.13309
  贡献: 这篇文章想解决的是高维 diffusion / Schrödinger bridge 建模的统计与计算瓶颈。直接在原始高维空间做 bridge 学习既难训练，也很难给出清晰误差界；同时很多实际任务已经有现成的大模型或预训练表示，如何把这些表示和 bridge 理论拼起来，过去并不清楚。
  重要性: 它的重要性在于给“潜空间做 bridge”这件事补了理论地基。很多工作都在这么做，但很少有人把表示学习误差、bridge 逼近误差和最终生成误差串起来分析。对于后续任何 latent SB 方法，这都是一个值得参考的理论模板。
  启发: CytoBridge 的输入契约本来就依赖潜表示，这篇文章说明 latent-space bridge 是一条合理路线，但前提是潜空间必须为动态任务而学，而不是只为重构或批次校正而学。对自动设计算法来说，一个核心决策应变成：先评估 latent geometry 是否保留时间/命运结构，再决定是否使用 SB/FM/UOT，而不是默认潜空间永远可靠。

- [Efficient trajectory inference in wasserstein space using consecutive averaging](notes/2024__math__arxiv__efficient-trajectory-inference-consecutive-averaging.md) | `math` | arXiv preprint arXiv:2405.19679
  贡献: 这篇文章针对的是“只有离散时间点的点云快照，如何在 Wasserstein 几何里恢复连续轨迹”的问题。此前一类方法只做相邻时间点的 OT 线性插值，轨迹不够平滑；另一类神经 ODE 方法能学连续动力学，但训练不稳定、对 stiff dynamics 敏感，也不天然尊重 Wasserstein 空间的内在几何。
  重要性: 它重要的地方不在于“神经网络更大”，而在于把 Wasserstein spline 这条理论线和实际可运行的 trajectory inference 连接起来了。文章同时给出算法、几何解释、收敛分析和单细胞实验，说明这不是纯数学玩具，而是能直接服务 snapshot trajectory inference 的方法学工作。
  启发: 对 CytoBridge 来说，这篇文章最有价值的启发是：并不是所有连续轨迹恢复都需要上来就训练全局神经 ODE/SDE。对于高噪声 snapshot 数据，可以先用这种内禀 Wasserstein subdivision 生成稳定的几何先验轨迹，再把它作为更复杂 RUOT/UMFSB/flow matching 模型的初始化、正则项或 sanity check。它也提醒我们，分叉结构最好在概率测度几何里表达，而不是在单细胞一对一配对上硬做。

- [Neural Optimal Transport with Lagrangian Costs](notes/2024__math__arxiv__neural-optimal-transport-lagrangian-costs.md) | `math` | arXiv preprint arXiv:2406.00288
  贡献: 标准 OT 默认 ground cost 很简单，通常是欧氏距离或其轻微变体。但在很多真实系统里，粒子运动受势场、障碍、曲面几何或动力学约束影响，直接用欧氏 cost 会错配真实 transport 路径。单细胞轨迹推断里，如果细胞状态空间带有明显几何/能垒结构，这个问题同样存在。
  重要性: 它的重要性在于把“学 transport”从纯几何匹配推进到“带动力学先验的 transport”。对很多应用来说，这比继续在欧氏 OT 上做小修小补更关键，因为它直接决定 coupling 是否具备机制意义。
  启发: 对 CytoBridge，这篇文章的启发不是照搬样条求解器，而是要把“先验生物机制写进 cost/行动量”当成一等设计选项。比如可以让 transport 更偏向沿分化轴移动、避开不可达区域、或惩罚不合理跨谱系跳跃。自动设计算法时，agent 不应只搜索网络结构，还应搜索更合理的 action/cost 形式。

- [stVCR Reconstructing spatio-temporal dynamics of cell development using optimal transport](notes/2024__math__biorxiv__stvcr.md) | `math` | bioRxiv preprint doi:10.1101/2024.06.02.596937
  贡献: 这篇工作解决的是时间序列空间转录组里一个很具体但很难的问题：过去方法通常只能重建分化或增长，难以同时处理分化、增长和物理空间迁移；更麻烦的是，不同时间点的空间坐标往往不在同一坐标系中，直接做 OT 会失真。
  重要性: 它重要的地方在于把“多空间、多机制的单细胞动力学”做成了一个统一 OT 问题，而不是简单在已有 scRNA-seq 方法上外挂空间坐标。对于时空单细胞方向，这是非常直接也很难回避的建模升级。
  启发: stVCR 对 CytoBridge 的关键启发是：当数据来自多个空间时，算法设计应该允许每个空间拥有自己的 transport 几何和不变量，再通过共享潜动力学耦合它们。自动化设计算法时，agent 应该把“单空间统一建模”与“多空间异构 cost 联合建模”作为明确可比较的设计分支。

## 2025

- [Regularized unbalanced optimal transport as entropy minimization with respect to branching brownian motion](notes/2021__math__arxiv__regularized_unbalanced_ot_branching_brownian_motion.md) | `math` | Astérisque 458 (2025)
  贡献: 经典 Schrödinger problem 对应的是平衡 regularized OT，但非平衡 regularized OT 的路径空间含义长期不清楚。
  重要性: 它补上了 UOT 最缺的那块理论地基: regularized UOT 到底对应什么样的随机路径问题。
  启发: CytoBridge 如果要认真做非平衡桥，质量变化模块应建立在有 branching/killing 语义的参考过程上。

- [Trajectory Inference with Cell–Cell Interactions (TICCI) intercellular communication improves the accuracy of trajectory inference methods](notes/2025__journal__bioinformatics__ticci.md) | `journal` | Bioinformatics
  贡献: TICCI要解决的是现有 trajectory inference 大多只看细胞内部表达相似性，忽略发育过程中 intercellular communication 对命运变化的影响。作者希望用 CCI 信息来提高 branch identification 和 temporal ordering 的准确性。
  重要性: 这篇文章的价值在于提出了一个被很多 TI 方法忽略但非常合理的问题：如果 intercellular communication 真会改变轨迹，那么为什么轨迹图不显式用它。这个问题切得准确，也有可复用的算法实现。
  启发: CytoBridge 的 interaction 模块不必停留在“分析结果”层面，可以前移成 trajectory inference 的结构先验。未来设计新算法时，值得探索 interaction-aware transport 或 interaction-aware bridge，而不只是独立的通信分析。

- [ARTEMIS integrates autoencoders and Schrödinger Bridges to predict continuous dynamics of gene expression, cell population, and perturbation from time-series single-cell data](notes/2025__journal__bioinformatics__artemis.md) | `journal` | Bioinformatics (ISMB/ECCB 2025 Supplement)
  贡献: ARTEMIS瞄准的是 time-series scRNA-seq 中连续基因表达动力学、细胞群体数量变化和 perturbation 效应的统一预测。作者想解决的不只是“轨迹长什么样”，还包括“群体会如何增减”以及“扰动后过程会怎样改变”。
  重要性: ARTEMIS把 VAE、unbalanced SB 和 perturbation prediction 组合成了一个清晰完整的框架，问题定义也紧贴单细胞动态建模前沿。它说明 SB 路线在单细胞中的吸引力已经从纯理论进入可落地方法阶段。
  启发: 对 CytoBridge 来说，ARTEMIS强化了“latent compression + bridge dynamics + unbalanced mass”这条路线的合理性。若要做更强的 perturbation module，可进一步让 drift/growth/interaction 在同一 latent bridge 中协同学习。

- [scCausalVI disentangles single-cell perturbation responses with causality-aware generative model](notes/2025__journal__cell_systems__sccausalvi.md) | `journal` | Cell Systems
  贡献: scCausalVI解决的是 perturbation 数据里 basal state 与 treatment effect 混在一起的问题。它想回答：怎样把细胞固有状态、处理诱导效应和技术变异拆开，从而更可信地做跨条件单细胞 in silico perturbation。
  重要性: 这篇文章的重要贡献是把因果意识真正写进了单细胞 perturbation 生成模型，而不是事后借用因果术语包装条件生成。对 virtual cell 和 in silico perturbation 方向，这是很自然的一步升级。
  启发: CytoBridge 在设计 perturbation 模块时，应优先把 basal state 与 treatment effect 解缠，否则很多“预测成功”其实只是 cell type effect。因果结构不一定要很重，但至少要显式区分这些来源。

- [Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport](notes/2025__mlconf__iclr__learning_stochastic_dynamics_from_snapshots_through_regularized_unbalanced_optimal_transport.md) | `mlconf` | ICLR 2025
  贡献: 这篇文章解决的是一个比“平衡 OT / SB”更贴近真实单细胞数据的问题：当细胞群体存在增长、死亡、分叉扩增时，质量并不守恒，而标准 OT 或标准 SB 往往会通过“错误迁移”去强行解释细胞数变化，结果就是伪转移和错误 fate path。
  重要性: 我觉得它能发 ICLR 2025，原因是它同时满足了“理论 gap 明确”和“应用价值直接”：
  启发: agent 在自动设计算法时，第一步就应判断任务是否需要 balanced / unbalanced 建模，而不是默认守恒。

- [Meta Flow Matching Integrating Vector Fields on the Wasserstein Manifold](notes/2025__math__iclr__meta-flow-matching.md) | `math/mlconf` | ICLR 2025
  贡献: 标准 flow matching 学的是“单个样本如何从源分布走到目标分布”，默认粒子彼此独立，且通常只能针对一个固定起始分布工作。对于真实生物群体过程，这不够，因为不同患者、不同条件下的初始细胞群体不同，且细胞间相互作用会影响动力学。
  重要性: 这篇工作之所以能上 ICLR，不只是因为用了 GNN，而是因为它把 flow matching 的对象从“点”推进到了“分布族”，这对生物群体动力学是非常自然也非常关键的一步。它为从 patient-specific / cohort-specific 数据学习人口级动力学提供了清晰范式。
  启发: 对 CytoBridge 来说，这篇文章最重要的启发是：interaction 和 context 不能只靠在单细胞特征里拼几个邻域统计量来解决。更合理的做法是把整个初始细胞群体、微环境或处理条件编码成一个 distribution-level context，再让速度场、增长项或得分函数显式依赖它。这样 agent 在自动设计算法时，才真正具备“同一方法泛化到不同样本背景”的能力。

- [Modeling Complex System Dynamics with Flow Matching Across Time and Conditions](notes/2025__mlconf__iclr__modeling_complex_system_dynamics_with_flow_matching_across_time_and_conditions.md) | `mlconf` | ICLR 2025
  贡献: 标准 FM/CFM 只处理两个边缘分布。真实单细胞实验，尤其 perturbation screen，往往同时有多个时间点和多个条件，而且缺测组合很多。把问题拆成若干 pairwise OT-CFM 虽然简单，但会丢掉两类关键信息：
  重要性: 它能发 ICLR 2025，我觉得原因在于它把 FM 从“两个边缘之间的生成建模工具”，推进成了“多时间、多条件系统动力学建模工具”。
  启发: agent 不应默认把多时间点问题拆成相邻 pairwise 子任务；这会浪费跨时间点的一致性信息。

- [Partially Observed Trajectory Inference using Optimal Transport and a Dynamics Prior](notes/2025__math__iclr__partially-observed-trajectory-inference-ot-dynamics-prior.md) | `math/mlconf` | ICLR 2025
  贡献: 此前 OT / Schrödinger bridge 路线的 trajectory inference 大多假设状态是 fully observed 的，也就是建模空间里每个决定动力学的变量都被看到了。但真实问题常常只观测到位置或表达，速度、加速度、调控势能等隐变量并不可见。忽略这点会让轨迹恢复能力受到结构性限制。
  重要性: 这篇工作能上 ICLR，核心是它不是简单加 latent variable，而是把状态空间模型、最小熵 trajectory inference 和 OT 粒子化求解严谨地拼在了一起。它说明 trajectory inference 不必局限在观测空间，这对很多科学数据都很关键。
  启发: 对 CytoBridge 来说，这篇文章提醒我们：只在表达空间学一阶速度场可能太弱。很多单细胞过程的“惯性”可能体现在未观测变量里，例如调控活性、染色质预备态、空间驱动或代谢势。自动设计算法时，应把“是否引入 latent momentum / hidden regulatory state”作为显式可搜索决策，而不是默认一阶 observed-state 动力学足够。

- [Wasserstein Flow Matching Generative modeling over families of distributions](notes/2025__math__icml__wasserstein-flow-matching.md) | `math/mlconf` | ICML 2025
  贡献: 传统 flow matching 默认“样本是点”，也就是从噪声点生成数据点。但在单细胞和空间组学里，很多对象本身就是分布，例如一个样本的细胞群体、一个微环境邻域、一个时间点的人口状态。把这些对象拆成独立点会丢掉分布内部的几何结构。
  重要性: 它能发 ICML，核心原因是把 flow matching 的基本对象升级了。过去 FM 在点空间里已经很成熟，但对科学数据来说，分布才是真正的一等对象。WFM 把这件事系统化了，而且和单细胞应用有直接连接。
  启发: CytoBridge 很适合吸收 WFM 的视角：把每个时间点/样本的细胞群体当成一个 distribution object，再学分布级流，而不是只学单细胞局部速度。对自动设计算法来说，这意味着 agent 需要同时考虑“点级动力学模型”和“群体级生成模型”两条路线，不能默认前者总是更合适。

- [Steering Large Agent Populations using Mean-Field Schrodinger Bridges with Gaussian Mixture Models](notes/2025__math__arxiv__mean-field-schrodinger-bridges-gaussian-mixtures.md) | `math` | IEEE Control Systems Letters (2025)
  贡献: Mean-field Schrödinger bridge 的通用数值解法通常要么依赖空间离散，维度一高就崩；要么依赖神经网络和随机优化，算得慢且缺少明确性能保障。另一方面，真实边界分布经常不是单高斯，而是多峰结构。
  重要性: 它的重要性在于给 MFSB 提供了一条结构化、可控、可带约束的近似求解路线。即使假设较强，这种“用 mixture 包装解析子问题”的思路对大规模群体控制和 bridge 计算都很有价值。
  启发: 对 CytoBridge，这篇文章提示可以把复杂细胞群体先做 mixture-level 抽象，再在 mixture 间求 tractable bridge，而不是直接对全部细胞端到端黑箱训练。对于自动化算法设计，这给出了一类很值得探索的折中路线：先做群体压缩，再做结构化桥接。

- [Schrödinger Bridge with Quadratic State Cost is Exactly Solvable](notes/2024__math__arxiv__schrodinger-bridge-quadratic-state-cost.md) | `math` | IEEE Transactions on Automatic Control (2025)
  贡献: 经典 Schrödinger bridge 在热核参考过程下结构很漂亮，但一旦希望路径靠近某个名义状态、或者希望显式表达状态依赖的“生灭/惩罚”，求解就会迅速变难。对单细胞动力学来说，这恰好对应“想鼓励系统靠近某些生物学可行区域”的需求。
  重要性: 它的重要性在于示范了一类非平凡的 SB 扩展仍能保持解析可解，并把 state regularization、reaction-diffusion 和桥接理论连接起来。对想做 unbalanced / growth-aware bridge 的人，这是一块很有价值的理论基石。
  启发: CytoBridge 如果想把“靠近稳定命运盆地”或“远离不可信区域”的偏好显式写进桥接模型，可以借鉴这种 state-cost regularized bridge 思路。更重要的是，它提醒我们在设计算法时要优先寻找能保留解析结构的特殊形式，因为这类结构往往能带来远超黑箱模型的稳定性和解释性。

- [Score-based Neural Ordinary Differential Equations for Computing Mean Field Control Problems](notes/2024__math__arxiv__score-neural-odes-mean-field-control.md) | `math` | Journal of Computational Physics (2025)
  贡献: 高维 mean field control 的一个核心难点是：优化控制时不仅需要 density，还经常需要沿轨迹计算 score，甚至二阶信息。传统方法要么依赖网格求解 HJB/FP 方程，要么在高维上数值不稳，难以和现代生成建模的可微框架结合。
  重要性: 它的重要性在于把 score-based generative machinery 与 mean-field control 更系统地接通了，并提供了能在高维里运行的结构化神经 ODE 方案。对于任何想把 FP/HJB/MFC 路线和生成模型结合的人，这篇文章都很有参考价值。
  启发: CytoBridge 如果未来想把 mean-field control 或 score-based regularization 纳入核心算法，可以借鉴这篇文章“显式演化 score”的思路。特别是当我们希望同时控制 transport、uncertainty 和 free-energy 下降时，单纯学一个速度场往往不够，可能需要把 score 或 curvature 也纳入被学习状态。

- [Mapping cells through time and space with moscot](notes/2025__journal__nature__moscot.md) | `journal` | Nature
  贡献: moscot 解决的是 OT 在单细胞 genomics 中两大痛点：多模态信息难统一，以及 atlas 级数据无法扩展。作者不仅用它重建了 170 万细胞的小鼠胚胎时间轨迹，还展示了时空联合分析和胰腺内分泌 lineage 解析。
  重要性: 它之所以能上 Nature，不只是算法本身，还因为它把 OT 从一类论文方法变成了可以支撑大规模单细胞图谱研究的基础设施。再加上对真实生物发现的支撑，影响力明显超出单一工具。
  启发: CytoBridge 后续若要做自动算法设计，应该学习 moscot 的“problem abstraction”思路。与其不断堆新 loss，不如先把 time/space/multimodal 统一成可组合对象，让 agent 在统一接口上自动选 OT/SB/FM 变体。

- [Gene trajectory inference for single-cell data by optimal transport metrics](notes/2025__journal__nature_biotechnology__genetrajectory.md) | `journal` | Nature Biotechnology
  贡献: GeneTrajectory解决的是并行生物过程同时存在时，cell trajectory 常把多个过程揉在一起，导致关键 gene program 顺序被遮蔽的问题。它把焦点从“细胞轨迹”转向“基因轨迹”，试图直接恢复过程相关基因程序的先后顺序。
  重要性: 这篇工作胜在重新定义问题而不是只优化老指标。它说明 OT 不仅能做 cell-state transport，也能用来在 gene program 层面恢复动态顺序，这种“换基本对象”的创新很容易打动顶级方法期刊。
  启发: CytoBridge 的 driver gene 模块可以从单纯相关性分析升级到“gene program dynamics”分析。对自动算法设计而言，必要时应主动改变建模对象，而不是默认所有问题都要在细胞层面求解。

- [Learning cell dynamics with neural differential equations](notes/2025__journal__nature_machine_intelligence__scdiffeq.md) | `journal` | Nature Machine Intelligence
  贡献: scDiffEq主要解决的是现有 drift-diffusion 动力学模型常把 diffusion 设成常数，因而无法描述状态依赖的随机性和命运决策点附近的噪声结构。作者想同时恢复 deterministic drift 和 state-dependent diffusion，并用于 fate prediction 与 perturbation simulation。
  重要性: 这篇文章之所以重要，是因为它把单细胞动力学从“只学漂移”推进到了“显式分解漂移与扩散”，并证明 diffusion 在命运决策里不是噪声，而是信息。这个问题定义本身就足够新，而且实验也较完整。
  启发: CytoBridge 的新算法不应再把 stochasticity 当 residual error，而应把它视作要被主动建模的对象。特别是在 fate bifurcation 和 perturbation prediction 上，state-dependent diffusion 很可能是关键增益点。

- [MultistageOT Multistage optimal transport infers trajectories from a snapshot of single-cell data](notes/2025__journal__pnas__multistageot.md) | `journal` | Proceedings of the National Academy of Sciences of the United States of America
  贡献: MultistageOT要解决的是只有一个 single-cell snapshot 时，怎样在没有时间标签的情况下恢复分化进程，并识别不属于目标分化过程的 outlier cells。它把传统只适合双边耦合的 OT 扩展到单快照内部的多阶段推断。
  重要性: 这篇工作的亮点在于把“single snapshot trajectory inference”从启发式排序推进到了更明确的多边际 OT 建模，并且兼顾了 outlier 识别这一现实需求。问题定义很硬，数学建模也足够完整。
  启发: 对 agent 来说，单快照数据不应被简单判定为“不能做 dynamics”。更合理的方向是显式建模隐藏中间阶段，并把 outlier handling 作为主模型的一部分，而不是后处理。

- [Synchronized Optimal Transport for Joint Modeling of Dynamics Across Multiple Spaces](notes/2025__math__siamjam__synchronized-optimal-transport.md) | `math` | SIAM Journal on Applied Mathematics, 85(1):341-365, 2025
  贡献: 随着多组学和空间组学数据变多，一个系统往往会在多个空间里同时被表征，例如表达空间、染色质空间、物理空间。若分别在每个空间做 dynamical OT，得到的动力学可能互相不一致，缺少跨空间 coherence。
  重要性: 它的重要性在于第一次把“多空间动力学一致性”写成了明确的 dynamical OT 优化目标，而不是只做经验性多模态对齐。对多组学动力学建模而言，这是一种更高层次的约束。
  启发: CytoBridge 若要走向多模态/时空联合建模，SyncOT 给出的启发非常直接：不要只在一个 `X_latent` 里塞下所有模态，而要允许不同空间保留各自几何，再通过同步动力学正则把它们绑在一起。自动设计算法时，agent 应把“单潜空间融合”和“多空间同步 transport”作为两条真正不同的设计路线。

- [Plug-in estimation of Schrödinger bridges](notes/2024__math__arxiv__sinkhorn-bridge-plugin-estimation.md) | `math` | SIAM Journal on Mathematics of Data Science (2025)
  贡献: 现有 Schrödinger bridge 学习方法通常需要交替训练前向/后向 drift，并在训练中反复模拟 SDE，计算代价高，也缺少明确的统计收敛保证。对做科学建模的人来说，这意味着即便模型跑出来了，也不容易知道估计误差到底来自哪里。
  重要性: 这是一篇典型的“把复杂问题还原成简单问题”的重要理论工作。它把 SB 估计从昂贵神经仿真拉回到静态 entropic OT，并给出统计保证，这对整个 bridge 学习方向都很有启发意义。
  启发: 对 CytoBridge 来说，这篇文章最大的启发是：可以先解静态 EOT/RUOT 势函数，再把它们当作动态模型的解析初始化、蒸馏目标或正则项，而不一定从随机 drift 网络直接学起。尤其在相邻时间点 bridge 构造上，这种 plug-in 思路可能显著提升稳定性和可解释性。

- [Kinetic Optimal Transport (OTIKIN) -- Part 1 Second-Order Discrepancies Between Probability Measures](notes/2025__math__arxiv__otikin-second-order-discrepancy.md) | `math` | arXiv preprint arXiv:2502.15665
  贡献: 经典 Wasserstein OT 是一阶的，只看位置分布和速度场，对“位置-速度联合状态”或带惯性的粒子系统表达力有限。很多动态系统并不是纯扩散或一阶梯度流，而更接近带加速度约束的 kinetic system；这时一阶 W2 几何就不够用了。
  重要性: 它的重要性在于给 kinetic OT 单独开了一条正统理论线，而不是停留在“给 W2 加点速度标签”的直觉层面。对于任何想把 momentum、hidden velocity 或 second-order mechanics 引入 population dynamics 的工作，这是上游理论资源。
  启发: CytoBridge 目前主要还是一阶群体动力学视角。OTIKIN 提醒我们，如果未来希望引入 latent velocity、transcriptional momentum 或更接近物理惯性的 hidden state，底层 transport 几何也许该升级到 phase-space。自动设计算法时，可以把“是否需要二阶状态变量”作为一个真正的模型分叉点，而不是只在一阶框架里微调。

- [Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schrödinger Bridge](notes/2025__math__arxiv__unbalanced-mean-field-schrodinger-bridge.md) | `math` | arXiv preprint arXiv:2505.11197
  贡献: 这篇文章正面瞄准了 CytoBridge 自己要解决的问题：从稀疏时间分辨 snapshot 数据中学习连续随机动力学，同时允许 unbalanced growth/death，并把 cell-cell interaction 纳入主模型。此前 OT、SB 和 RUOT 线的多数方法能处理其中一部分，但很少能把三者放到同一框架里。
  重要性: 这篇工作重要的原因不只是“效果更好”，而是它把当前单细胞动力学建模里最关键的三件事首次较完整地捆绑起来：unbalanced、stochastic 和 interaction。对领域发展来说，这比继续在只含 transport 的模型上打补丁更有路线意义。
  启发: 这篇文章本身就是 CytoBridge 路线的直接来源。对 agent 自动设计算法而言，它提供的核心启发是：搜索空间不应只包含 velocity network 的变化，还应系统性搜索 growth penalty、interaction parameterization、Fisher regularization 强度、以及 mean-field 假设的形式。真正 novel 的算法，很可能出现在这几块的更优组合上。

- [Variational Regularized Unbalanced Optimal Transport Single Network, Least Action](notes/2025__math__arxiv__variational-ruot.md) | `math` | arXiv preprint arXiv:2505.11823
  贡献: RUOT 已经能同时表达随机性和质量不守恒，但许多现有实现并没有把最优必要条件真正写进模型结构，导致学到的解不一定满足 least-action 原理，训练也容易不稳。另一个实际问题是 WFR 常用的二次 growth penalty 在生物上未必合理。
  重要性: 这篇工作的重要之处在于，它不是又提出一个更大的网络，而是把 RUOT 的数学结构真正前移到模型设计层。对单细胞动力学这种机制导向任务，这类“结构优先”的改进通常比纯经验调参更有长期价值。
  启发: 对 CytoBridge 而言，Var-RUOT 的启发非常关键：搜索空间应尽量围绕最优性条件压缩，而不是放任模型过度自由。自动设计算法时，可以优先探索“由势函数诱导动力学”的结构化家族，并把 growth penalty 设计本身当成会影响生物合理性的核心变量。

- [Simulating Fokker-Planck equations via mean field control of score-based normalizing flows](notes/2025__math__arxiv__fokker-planck-mean-field-control-score-flows.md) | `math` | arXiv preprint arXiv:2506.05723
  贡献: Fokker-Planck 方程是很多随机动力学的连续分布描述，但高维数值求解很难。过去要么依赖网格 PDE 解法，要么用粒子方法做前向模拟，难以和现代 score-based/flow-based 模型自然结合。
  重要性: 它的重要性在于把 Fokker-Planck、mean field control 和 score-based normalizing flow 打通了，给出了一个兼顾理论与可计算性的高维模拟路线。对于任何想把概率流、自由能耗散和控制统一起来的工作，都有参考价值。
  启发: 对 CytoBridge，这篇文章的启发是：如果想更精细地建模随机性和 free-energy 下降，可以考虑把 score 显式引入控制/动力学目标，而不是只学漂移项。它也提醒 agent，在某些场景下“先学可微概率流，再把它解释为 PDE 解”可能比直接解方程更实际。

- [CFM-GP Unified Conditional Flow Matching to Learn Gene Perturbation Across Cell Types](notes/2025__journal__arxiv__cfm_gp.md) | `journal` | arXiv preprint arXiv:2508.08312
  贡献: CFM-GP要解决的是单细胞 perturbation prediction 往往要为每个 cell type 单独训练模型，难以跨细胞类型泛化。它把问题改写为 cell type-agnostic 的基因扰动分布变换学习。
  重要性: 这篇工作抓住了 flow matching 在生命科学中的一个很自然切入点：用连续流场做单细胞扰动分布预测，并突出跨 cell type 统一建模的优势。问题设定明确，实验覆盖也比较贴合应用。
  启发: 若要设计新的 perturbation 算法，CFM-GP提示“统一模型跨 cell type 泛化”本身就是强卖点。CytoBridge 可以把 conditional flow matching 作为 perturbation head，与更结构化的 dynamics backbone 组合。

- [CellFlow enables generative single-cell phenotype modeling with flow matching](notes/2025__journal__biorxiv__cellflow.md) | `journal` | bioRxiv preprint doi:10.1101/2025.04.11.648220
  贡献: CellFlow关注更广义的单细胞 phenotype generation：给定复杂 perturbation 条件，能否生成未来的细胞表型分布，并据此指导开发、药物处理和 organoid engineering。它试图把虚拟筛选提升到单细胞异质性层面。
  重要性: 这篇 preprint 把 flow matching 明确落到单细胞 phenotypic screen 和 organoid engineering 上，应用场景抓得很准。它说明新一代生成模型正在从“分子设计”进一步渗透到“细胞状态设计”。
  启发: 对 agent 来说，CellFlow提示 perturbation 模块可以直接面向 protocol design，而不只服务于事后解释。未来若做自动实验规划，生成异质群体分布会比预测均值更有决策价值。

## 2026

- [CellStream Dynamical Optimal Transport Informed Embeddings for Reconstructing Cellular Trajectories from Snapshots Data](notes/2026__math__aaai__cellstream.md) | `math` | AAAI 2026
  贡献: 单细胞轨迹推断里，一个非常实际的问题是 embedding 和 dynamics 常常是分开的：先用 PCA/UMAP/scVI 得到低维表示，再在这个固定空间里跑 OT/ODE。这样做的缺点是 latent space 本身并不保证保留时间结构，可能天生就不适合做动力学学习。
  重要性: 它之所以重要，是因为它准确地命中了当前单细胞 OT 路线里一个常被忽视但又非常致命的问题：latent representation 不是中立的。把 embedding 学习和动力学学习真正耦合起来，是单细胞 snapshot 建模非常自然的一步。
  启发: 对 CytoBridge 而言，CellStream 的启发非常直接：`X_latent` 不能被默认视为外部固定输入，agent 应该把“是否联合学习 latent geometry 与 dynamics”作为一级设计变量。特别是在噪声大、时间间隔大、批次强的数据里，先优化表示再优化动力学，往往不如联合训练更可靠。

- [Trajectory inference for a branching SDE model of cell differentiation](notes/2023__journal__arxiv__branching_sde_lineage.md) | `journal` | Journal of Mathematical Biology (2026)
  贡献: 这篇工作要补的是 gWOT/Waddington-OT 在存在增殖、死亡和 subsampling 时的识别缺口。作者关心的是：能否借助 lineage tree，把“分化动力学”和“净增殖变化”分开，从而更可靠地恢复 branching differentiation landscape。
  重要性: 这篇文章的重要性在于它补上了 OT 轨迹推断里一个长期被忽略的理论缺口：增殖、死亡与 branching 会怎样破坏可辨识性。它给出的不是经验性 tweak，而是有理论解释的扩展，因此对后续方法设计约束很强。
  启发: CytoBridge 如果想可靠建模 growth，就必须区分“可从 snapshot 恢复的信息”和“必须靠 lineage 或额外观测补足的信息”。agent 自动设计算法时，也应优先识别不可辨识因素，而不是默认所有缺失变量都能靠神经网络吸收。

- [WFR-FM Simulation-Free Dynamic Unbalanced Optimal Transport](notes/2026__math__arxiv__wfr-flow-matching.md) | `math` | arXiv preprint arXiv:2601.06810
  贡献: flow matching 已经让 balanced transport 的训练变得 simulation-free，但对 unbalanced dynamics 来说，主流方法仍多依赖 ODE/SDE 仿真或较重的变分优化。对于包含增殖、凋亡的单细胞群体，这正是瓶颈所在。
  重要性: 它之所以重要，是因为它几乎正中当前单细胞 OT 路线的核心矛盾：我们既想要 unbalanced 的表达力，又想要 flow matching 的训练效率。WFR-FM 给出的正是这两者的直接结合，而且不是拍脑袋拼接，而是从 WFR 几何推导出来的。
  启发: 对 CytoBridge 来说，WFR-FM 是非常强的候选母框架。它提示 agent 在自动设计算法时，应优先探索“simulation-free velocity + growth joint regression”这一方向，再根据需要外挂 interaction、multi-space 或 latent-state 模块。若要做下一代方法，这很可能比继续加重 path-space 优化更有前景。

- [Constructing the spatiotemporal atlas of single-cell lineage trajectories in stereotypic biological structures](notes/2026__journal__iscience__scst.md) | `journal` | iScience
  贡献: SCST 的目标是从发育中具有固定形态结构的样本里，重建单细胞分辨率的 3D 时空图谱，并进一步 delineate lineage trajectories。它面向的问题不是一般组织整合，而是发育生物学里“空间位置、时间阶段和谱系关系如何统一恢复”。
  重要性: 这篇文章的亮点是把空间单细胞映射从二维/局部提升到三维、时序和谱系统一重建，并给出了较完整的发育案例。虽然不是顶刊级方法，但在发育时空图谱领域是很扎实的一步。
  启发: CytoBridge 若要做时空单细胞建模，固定几何结构系统值得单独设计专门算法。并不是所有问题都需要端到端黑盒模型，分层的 mapping + trajectory pipeline 在某些发育场景中可能更可靠。

