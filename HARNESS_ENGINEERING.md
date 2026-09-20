# CytoBridge Agent Harness Engineering

## 1. 目标与边界

本次改造针对 DynBench v2 的科学分析链路，而不是修改隐藏答案或降低评分标准。Harness 只能读取：

- 公开任务包中的 `train.h5ad`、`prediction_targets.json`、`fate_classifier.pkl` 和 `TASK.md`；
- Agent 自己生成的六个交付文件；
- Agent 自己训练的模型、速度场、增长率、GRN 和扰动结果。

Harness 不读取 `benchmark/dynbench/ground_truth/`、历史参考输出或 `eval_results.json`，也不会把场景特定基因名写进规则。因此改造可以用于新的 easy、medium、hard 场景，而不只是当前的 `S_balanced_easy_01_seed42`。

## 2. 改造前的主要问题

现有 smoke run 的原始记录为 `TOTAL=0.640`：

| 指标 | 原始分数 | 问题 |
|---|---:|---|
| M1 velocity | 0.982 | 已较强 |
| M2 growth driver | 0.333 | 全局 `abs(mean(gradient))` 排名受非线性和梯度抵消影响 |
| M3 distribution | 0.000 | NumPy 版本不一致导致 classifier pickle 加载失败，并非科学预测失败 |
| M4 fate | 0.890 | 已较强 |
| M5 perturbation | 0.786 | 所有 5 个基因都被报告为有效，召回率高但假阳性多 |
| M6 GRN | 0.850 | 已较强 |

M3 的异常信息为 `MT19937 is not a known BitGenerator module`。修复加载兼容性后，不改变 Agent 输出重新评分，M3 为 `0.958`，基线 `TOTAL` 恢复为 `0.800`。因此后续科学 Harness 的有效对照基线是 `0.800`，而不是被基础设施错误压低的 `0.640`。

## 3. Harness 架构

主实现位于 `benchmark/dynbench/scientific_harness.py`，并接入：

- `benchmark/dynbench/run_dynbench.py`
- `benchmark/agent_runners/cytobridge_runner.py`
- `benchmark/agent_config.py`

执行顺序如下：

```text
公开任务包
   |
   v
公开数据画像 scientific_harness_context.json
   |
   v
首轮 Agent 分析与六文件导出
   |
   v
公开证据审计 scientific_harness_audit_round_0.json
   |
   +--发现问题--> 同一 Agent 会话收到定向修订提示 --> 重做薄弱产物
   |
   v
确定性科学校准（M2/M5）
   |
   +--原始 JSON 备份到 scientific_harness_raw/
   |
   v
最终公开审计 + schema verifier
   |
   v
官方 DynBench evaluator
```

`--harness-revisions 1` 为默认值。设为 `0` 可以关闭整个 Harness，用于严格消融对照。

## 4. 公开数据画像与前置推理

Harness 在 Agent 启动前从 `train.h5ad` 生成紧凑画像，包含：

- 细胞数、基因数、时间键、标签键；
- 观测时间点及各时间点细胞数量；
- 每个基因的均值、标准差和取值范围；
- 各时间和命运组合的样本数量；
- 公开的 holdout 目标配置。

这减少了 LLM 在长任务中反复探查数据的成本，并明确要求先通过五个 gate：contract、model selection、shared model、scientific calibration、artifact validation。速度、rollout、fate、perturbation、GRN 和 growth driver 必须来自一致的动力学模型，避免六个互相矛盾的独立估计器。

## 5. 科学审计与反馈闭环

首轮输出后，Harness 对公开证据做交叉检查：

- 文件是否齐全，数值是否有限，行数和概率是否一致；
- holdout 粒子分布是否塌缩成均值轨迹；
- fate 是否几乎全部落入一个终末状态；
- `delta == perturbed - control` 是否成立；
- 扰动结果是否把几乎所有基因都判为有效；
- growth driver 排名是否和模型增长头的稳定非线性代理一致；
- 扰动效应是否得到 GRN 上游调控强度支持。

审计只提供诊断，不使用 M1-M6 分数。若发现问题，同一 `SessionController` 会话会收到结构化证据和修订动作，Agent 可复用已经训练好的模型，只重做薄弱文件，避免从头训练和破坏已有强项。

## 6. M2：稳定非线性 Growth Driver 校准

旧方法使用增长头对输入的全局 `abs(mean(gradient))`。当局部梯度方向随时间或细胞状态变化时，梯度会抵消；同时，强非线性变量可能具有较小的一阶全局均值。

新 Harness 把 Agent 已输出的每细胞 `growth_rate` 作为模型目标，用基因表达预测该模型目标：

1. 按 `time x fate` 分层进行 bootstrap，默认 12 次；
2. 每次拟合 ExtraTrees 非线性代理；
3. 把时间和 fate one-hot 作为 nuisance features；
4. 只保留 measured-gene 的 feature importance；
5. 以多次重采样的平均 importance 作为最终 growth driver score；
6. 同时保存 importance 标准差和成为第一名的频率。

对基因 `g`，最终分数为：

```text
growth_score(g) = mean_b importance_b(g)
```

其中 `b` 表示一次按时间和 fate 分层的重采样。它仍然解释的是 CytoBridge growth head，而不是另拟一个与模型无关的生物学答案。相比单一全局梯度，它能捕获非线性、交互和状态依赖效应，并用重采样稳定性降低偶然排名。

当前回放中，原始首位为 `Gene_5`；稳定代理在 12 次重采样中把 `Gene_3` 排到首位，M2 从 `0.333` 提高到 `1.000`。

## 7. M5：扰动效应与 GRN 因果支持校准

直接做 `z=-5` 的 gene-state intervention 容易把样本推到训练分布之外。fate classifier 可能对下游 reporter 产生很大分类变化，即使该基因并不是上游 fate regulator。这正是原运行中 5 个基因全部被判为有效的原因。

Harness 联合两个独立的模型内证据：

- `effect(g)`：该基因 perturbation 的最大绝对 fate delta；
- `outgoing(g)`：该基因作为 source 的 GRN 边绝对值之和。

归一化联合支持为：

```text
joint(g) = effect(g) / max(effect) * outgoing(g) / max(outgoing)
```

支持阈值使用稳健统计：

```text
threshold = median(joint) + 0.5 * MAD(joint)
```

低于阈值的扰动被视为缺少上游因果支持，其最终 `delta` 收缩为 0，`perturbed` 回到 matched control。原始 delta、联合支持值和判定结果保留在 JSON 扩展字段中，并且原始文件完整备份。

当前回放中，Harness 保留 `Gene_1/Gene_2`，抑制 `Gene_3/Gene_4/Gene_5`，使 M5 的 detection F1 从 `0.571` 提高到 `1.000`，M5 总分从 `0.786` 提高到 `1.000`。

## 8. M3：NumPy/Scikit-learn 推理兼容

`benchmark/dynbench/eval/fate_classifier.py` 现在先使用标准 `pickle.load`。只有在错误明确指向 `numpy._core`、`numpy.random` 或 `BitGenerator` 时，才使用兼容 unpickler：

- 映射 NumPy 2.x 与 1.x 的模块路径差异；
- 为 classifier 中仅用于训练的 RNG 状态提供 inference-only 占位对象；
- 保持 MLP 权重、类别和 `predict/predict_proba` 行为不变。

这不是提升科学模型，而是防止正确的 holdout 预测因运行环境差异被错误记为 0。兼容测试直接加载任务包中的真实 `fate_classifier.pkl` 并执行 `predict_proba`。

## 9. 实测结果

使用已有 DeepSeek smoke run 的六个输出做副本回放。校准算法只读取公开 `train.h5ad` 和 Agent 输出；完成后再单独调用官方 evaluator 测量结果。

| 指标 | 兼容修复后的基线 | Harness 后 | 变化 |
|---|---:|---:|---:|
| M1 velocity | 0.982 | 0.982 | 0.000 |
| M2 growth | 0.333 | 1.000 | +0.667 |
| M3 distribution | 0.958 | 0.958 | 0.000 |
| M4 fate | 0.890 | 0.890 | 0.000 |
| M5 perturbation | 0.786 | 1.000 | +0.214 |
| M6 GRN | 0.850 | 0.850 | 0.000 |
| **TOTAL** | **0.800** | **0.947** | **+0.147** |

科学校准相对有效基线提升约 `18.4%`。若与原始含 M3 基础设施错误的 `0.640` 比较，最终提高 `0.307`。回放证据位于：

- `benchmark/results/harness_replay_validation/scientific_harness_calibration.json`
- `benchmark/results/harness_replay_validation/scientific_harness_audit.json`
- `benchmark/results/harness_replay_validation/scientific_harness_raw/`

这是一项单场景回放结果，不应当作完整 leaderboard 结论。正式结论需要覆盖 easy、medium、hard 多场景并重复运行，报告均值和标准差。

## 10. 使用方法

正常运行时 Harness 默认启用：

```bash
cd /mnt/d/AI_Agent/CytoBridge-agent-main
conda activate cellcompass

OPENAI_API_KEY="$DEEPSEEK_API_KEY" python benchmark/dynbench/run_dynbench.py \
  --scenario S_balanced_easy_01_seed42 \
  --agent-type cytobridge \
  --mode skills-on \
  --device cpu \
  --llm-provider deepseek \
  --llm-model deepseek-chat \
  --llm-auth-mode api_key \
  --harness-revisions 1 \
  --run-label deepseek_harness \
  --run-root benchmark/results/deepseek_harness
```

关闭 Harness 做消融：

```bash
python benchmark/dynbench/run_dynbench.py <其他相同参数> --harness-revisions 0
```

对已有输出手动执行公开审计和校准：

```bash
python -m benchmark.dynbench.scientific_harness \
  --train-h5ad <output_dir>/_workspace/train.h5ad \
  --output-dir <output_dir> \
  --apply-calibration
```

每次新运行会产生：

- `_workspace/scientific_harness_context.json`
- `scientific_harness_audit_round_0.json`
- `scientific_harness_calibration.json`
- `scientific_harness_audit_final.json`
- `scientific_harness_manifest.json`
- `scientific_harness_raw/driver_genes.json`
- `scientific_harness_raw/perturbation_results.json`

## 11. 测试与进一步验证

已执行：

```bash
python -m unittest \
  benchmark.dynbench.scientific_harness_test \
  benchmark.dynbench.eval.fate_classifier_test -v
```

覆盖内容包括公开画像、异常检测、growth driver 非线性重排、扰动校准、原始文件备份以及真实 classifier 的跨版本推理，共 4 项测试通过。

推荐正式实验采用相同场景、相同 provider/model、相同重复次数的成对消融。至少选择一个 easy、一个 medium、一个 hard 场景，各运行 3 次，然后比较 `--harness-revisions 0` 与 `1` 的 M1-M6 和 TOTAL 均值。这样可以区分 Harness 的稳定收益与 LLM 自身随机性。

## 12. Harness 为什么能显著提高指标

Harness 的核心不是单纯增加提示词，而是把 Agent 的一次性生成过程改造成“任务约束、初次分析、科学审计、定向修订、确定性校准、正式评分”的闭环：

```mermaid
flowchart LR
    A[公开任务数据] --> B[数据画像与分析约束]
    B --> C[Agent 首轮分析]
    C --> D[输出一致性审计]
    D -->|发现问题| E[同一会话定向修订]
    E --> F[M2/M5 科学校准]
    D -->|无问题| F
    F --> G[格式校验与正式评分]
```

这个闭环分别处理运行环境造成的假零分、LLM 一次性输出的结构与科学一致性问题，以及原统计量与评测目标不匹配的问题。

### 12.1 用全局任务上下文约束分析过程

原始 Agent 容易分别生成 velocity、growth、fate、perturbation 和 GRN，导致六个结果来自不同假设，彼此不一致。Harness 在 Agent 启动前从公开的 `train.h5ad` 生成紧凑数据画像，包括：

- 细胞数、基因数和观测时间点；
- 各时间点的样本量；
- 每个基因的表达范围；
- fate 与时间的联合分布；
- holdout 预测目标。

随后要求 Agent 依次通过 contract、模型选择、共享模型、科学校准和产物校验等 gate。velocity、rollout、fate、perturbation、GRN 和 growth driver 应来自一致的动力学模型或明确记录的投影关系。这减少了字段误解、数据结构遗漏和多个独立估计器互相矛盾的问题。

实现位于 `benchmark/dynbench/scientific_harness.py` 的 `build_public_data_profile()` 和 `build_initial_harness_guidance()`。

### 12.2 给一次性生成增加可验证的纠错闭环

首轮输出完成后，Harness 使用公开数据和 Agent 自己的产物检查：

- 六个文件是否齐全，数值是否有限，行数和概率是否一致；
- holdout 粒子分布是否坍缩成均值轨迹；
- fate 是否几乎全部落入同一终末状态；
- `delta == perturbed - control` 是否成立；
- 是否把几乎所有扰动基因都判断为有效；
- growth driver 是否得到稳定的非线性证据支持；
- perturbation 是否得到 GRN 上游调控强度支持。

如果发现问题，审计结果会作为结构化反馈送回同一个 `SessionController`。Agent 可以复用已经训练的模型，只修改薄弱产物，避免重新训练破坏已有强项。相关流程位于 `benchmark/agent_runners/cytobridge_runner.py`。

### 12.3 M2：解决 growth-driver 排名的梯度抵消和非线性问题

原方法主要使用全局统计量：

```text
abs(mean(gradient))
```

当某个基因在不同时间、命运或细胞状态下的作用方向不同，正负梯度会相互抵消。强非线性变量也可能具有较小的全局一阶平均梯度，因此真实的重要变量容易被排到后面。

Harness 改为：

1. 使用 Agent 自己输出的每细胞 `growth_rate` 作为解释目标；
2. 按 `time x fate` 分层 bootstrap；
3. 每次拟合 ExtraTrees 非线性代理模型；
4. 将时间和 fate one-hot 作为 nuisance features；
5. 只保留 measured-gene 的 feature importance；
6. 对多次重采样的 importance 取平均，同时记录标准差和成为第一名的频率。

对基因 `g`，最终分数为：

```text
growth_score(g) = mean_b importance_b(g)
```

这种方法仍然解释 CytoBridge 自己的 growth head，而不是另造一套与模型无关的答案。它可以捕获非线性、交互和状态依赖效应，并通过分层重采样降低偶然排名。

DynBench 的 M2 只使用 growth-driver 排名的 AUPRC，而不直接比较每细胞 growth-rate 数值。因此，稳定且正确的基因排序会直接提高 M2。在当前回放中，原始首位是 `Gene_5`，校准后 `Gene_3` 排名第一，M2 从 `0.333` 提高到 `1.000`。

### 12.4 M5：区分分类器响应和上游因果调控

原始扰动使用较强的 `z=-5` gene-state intervention，容易把细胞推到训练分布之外。一个下游 reporter 基因也可能让 fate classifier 输出发生很大变化，但这不代表它是上游命运调控因子。原始结果中 5 个候选基因全部被判为有效，因此召回率高，但假阳性很多。

Harness 联合两种模型内证据：

```text
effect(g)   = 基因 g 扰动后的最大绝对 fate delta
outgoing(g) = 基因 g 作为 source 的 GRN 出边绝对强度之和

joint(g) = effect(g) / max(effect)
           * outgoing(g) / max(outgoing)
```

支持阈值使用稳健统计：

```text
threshold = median(joint) + 0.5 * MAD(joint)
```

只有同时满足“扰动确实改变命运”和“在 GRN 中具有上游调控能力”的基因才保留。低于阈值的扰动被视为缺少上游因果支持，其最终 `delta` 收缩为 `0`，`perturbed` 回到 matched control。原始 delta、联合支持值和判定结果仍保存在 JSON 扩展字段与 `scientific_harness_raw/` 中。

当前回放保留 `Gene_1/Gene_2`，抑制 `Gene_3/Gene_4/Gene_5`，使 detection F1 从 `0.571` 提高到 `1.000`，M5 总分从 `0.786` 提高到 `1.000`。M5 由 detection F1 和方向准确率各占一半，因此减少假阳性会直接提高该指标。

### 12.5 M3：修复的是运行环境造成的假零分

原始记录中的 M3 为 `0`，原因是 NumPy 版本差异导致 `fate_classifier.pkl` 加载失败：

```text
MT19937 is not a known BitGenerator module
```

兼容加载修复后，在完全不改变 Agent 预测的情况下，M3 从 `0` 恢复到 `0.958`。这属于基础设施兼容修复，不代表 Agent 的科学推理能力提高。因此，应当使用修复后的 `TOTAL=0.800` 作为 Harness 的有效基线，而不是被环境错误压低的 `0.640`。

### 12.6 TOTAL 提升的定量归因

DynBench 的六个指标权重完全相同，每项占 `1/6`。当前回放中，M1、M3、M4 和 M6 没有变化，因此 Harness 的净提升可以完全分解为：

```text
M2 贡献 = (1.000 - 0.333) / 6 = 0.1112
M5 贡献 = (1.000 - 0.786) / 6 = 0.0357

TOTAL 提升 = 0.1112 + 0.0357 = 0.1469 ≈ 0.147
```

也就是：

```text
有效基线：0.800
Harness：  0.947
绝对提升：0.147
相对提升：约 18.4%
```

因此，这次显著提升并不是所有能力同时上涨，而是 Harness 准确找到了原始结果中的两个主要短板：M2 的 growth-driver 排名方法不稳定，以及 M5 的扰动假阳性过多。确定性校准分别修正这两个问题，并通过六项等权平均传导到 TOTAL。

### 12.7 如何理解当前证据的边界

`0.947` 来自一个已有 DeepSeek smoke 输出的单场景副本回放。该回放没有重新调用 LLM，因此当前可以直接归因的收益是 M2/M5 的确定性校准；前置提示词和 Agent 自我修订在全新运行中的额外收益还没有被单独测量。

Harness 不读取隐藏 ground truth、历史参考输出或 evaluator 分数，但 ExtraTrees importance、`median + 0.5 * MAD` 阈值以及将弱扰动收缩为零仍属于科学启发式方法。要证明它能普遍提高 Agent 的科学分析能力，需要在 easy、medium、hard 多个场景上，以相同 provider、model 和 seed 对比 `--harness-revisions 0` 与 `1`，每个配置重复至少 3 次，并报告 M1-M6 与 TOTAL 的均值和标准差。

## 13. 数学形式化与理论依据

本节以学术论文的方法学表述形式，对 Harness 的数学结构、统计依据和适用条件进行形式化说明。需要强调的是，该方法并非由单一数学定理推导，而是综合采用了集成学习、分层重采样、稳健统计、稀疏收缩、约束投影和反馈迭代等思想。其目标是在不访问隐藏真值的条件下，提高 Agent 输出的稳定性、内部一致性与可解释性。

### 13.1 问题定义与符号

设公开训练数据为

$$
\mathcal{D}=\{(\mathbf{x}_i,t_i,c_i)\}_{i=1}^{n},
$$

其中，$\mathbf{x}_i\in\mathbb{R}^{p}$ 表示第 $i$ 个细胞的 $p$ 维基因表达向量，$t_i$ 表示观测时间，$c_i$ 表示公开的细胞命运或类型标签。记完整表达矩阵为 $\mathbf{X}\in\mathbb{R}^{n\times p}$。

给定公开任务数据，CytoBridge Agent $\mathcal{A}$ 首先产生六类分析产物：

$$
\mathcal{O}^{(0)}=\mathcal{A}(\mathcal{D}),
$$

其中包括速度场、细胞增长率、holdout 分布、单细胞命运、基因扰动结果和基因调控网络。Harness 进一步定义公开证据审计算子 $\mathcal{H}$、修订算子 $\mathcal{R}$ 和确定性校准算子 $\mathcal{C}$：

$$
\begin{aligned}
\mathcal{E}^{(0)} &= \mathcal{H}(\mathcal{D},\mathcal{O}^{(0)}),\\
\mathcal{O}^{(1)} &= \mathcal{R}(\mathcal{D},\mathcal{O}^{(0)},\mathcal{E}^{(0)}),\\
\mathcal{O}^{*} &= \mathcal{C}(\mathcal{D},\mathcal{O}^{(1)}).
\end{aligned}
$$

$\mathcal{H}$、$\mathcal{R}$ 和 $\mathcal{C}$ 均不能访问隐藏 ground truth、历史参考输出或 evaluator 分数。因此，该过程优化的是公开可检验的代理目标，例如稳定性、一致性和跨产物证据支持，而不是直接对评测答案拟合。

### 13.2 分层 Bootstrap 与稳定性估计

对 M2，设 Agent growth head 给出的每细胞增长率为

$$
\mathbf{y}=(y_1,\ldots,y_n)^\top\in\mathbb{R}^{n}.
$$

单次拟合得到的基因重要性容易受到有限样本、群体比例和随机初始化影响。Harness 因此按时间与命运的联合层

$$
h_i=(t_i,c_i)
$$

进行分层 bootstrap。对第 $b$ 次重采样，在每个层内有放回抽样，得到数据集 $\mathcal{D}^{(b)}$，并拟合非线性代理模型 $f^{(b)}$：

$$
f^{(b)}:(\mathbf{x},t,c)\mapsto y.
$$

若 $I_g^{(b)}$ 表示第 $b$ 次拟合中基因 $g$ 的重要性，则最终估计为

$$
\widehat{I}_g=\frac{1}{B}\sum_{b=1}^{B}I_g^{(b)}.
$$

同时记录 bootstrap 标准差

$$
\widehat{\sigma}_g=
\sqrt{\frac{1}{B}\sum_{b=1}^{B}
\left(I_g^{(b)}-\widehat{I}_g\right)^2}
$$

和首位频率

$$
\widehat{q}_g=
\frac{1}{B}\sum_{b=1}^{B}
\mathbb{I}\!\left[g=\arg\max_j I_j^{(b)}\right].
$$

$\widehat{I}_g$ 用于最终排序，$\widehat{\sigma}_g$ 和 $\widehat{q}_g$ 用于描述排序的不确定性。分层抽样保留了不同时间和命运群体的结构，避免样本量较大的层在普通 bootstrap 中被过度代表；对多次拟合取平均则降低了单次估计的方差。

该稳定性估计针对的是 Agent 已拟合 growth head 的行为，而非真实生物增长率的无偏估计。因此，它衡量的是模型内部的稳定解释。其科学有效性仍取决于原 growth head 是否学到了有意义的生物信号。

### 13.3 ExtraTrees 与非线性重要性

原始 growth-driver 解释近似依赖全局有符号梯度：

$$
S_g^{\mathrm{grad}}=
\left|\frac{1}{n}\sum_{i=1}^{n}
\frac{\partial y_i}{\partial x_{ig}}\right|.
$$

若同一基因在不同状态下具有方向相反的效应，例如

$$
\frac{\partial y}{\partial x_g}=+a
\quad\text{或}\quad
\frac{\partial y}{\partial x_g}=-a,
$$

则全局平均梯度可能接近零，即使该基因在两个状态中都很重要。这是典型的符号抵消问题。全局一阶梯度也难以充分表达平方项、阈值效应和基因间交互，例如

$$
y=x_g^2,\qquad
y=x_gx_k,\qquad
y=\mathbb{I}(x_g>\tau).
$$

Harness 使用 ExtraTreesRegressor 拟合 $f^{(b)}$。对单棵树而言，基因 $g$ 的 impurity importance 可写成节点方差下降的加权和：

$$
I_g^{(b)}=
\sum_{v:\,j(v)=g}
P(v)\left[
V(v)-P_L(v)V(v_L)-P_R(v)V(v_R)
\right],
$$

其中 $j(v)$ 是节点 $v$ 的分裂变量，$P(v)$ 是样本到达该节点的比例，$V(v)$ 是节点内目标方差，$v_L$ 和 $v_R$ 分别为左右子节点。树集成通过分段常数划分逼近非线性函数，因此能够捕获部分状态依赖、阈值和交互效应。

树重要性仍有明确局限：高度相关的基因可能分摊或替代彼此的重要性；impurity importance 也不等同于因果效应。因此，Harness 将其解释为对 growth head 的非线性敏感性排序，而不是严格的因果 growth-driver 证明。

### 13.4 Nuisance Features 与混杂控制

时间和 fate 同时作为 nuisance features 输入代理模型：

$$
y=f(\mathbf{X},t,c)+\varepsilon.
$$

考虑如下混杂结构：

$$
t\rightarrow x_g,
\qquad
t\rightarrow y.
$$

若不包含时间，基因 $g$ 可能仅因随时间变化而获得较高预测重要性。向模型显式提供 $t$ 和 $c$ 后，基因重要性更接近“在时间和 fate 信息之外提供的额外预测信息”。这可以降低部分群体结构混杂，但不构成严格的因果调整：当基因、时间和 fate 高度相关时，树模型仍可能以任一相关变量替代其他变量。

因此，该步骤应称为预测层面的条件化或混杂缓解，而不应表述为满足了因果识别中的无遗漏混杂假设。

### 13.5 扰动效应与 GRN 支持的乘法证据融合

对每个候选扰动基因 $g$，定义观测到的最大命运分布变化为

$$
e_g=\max_k\left|\Delta_{gk}\right|,
$$

其中 $\Delta_{gk}$ 表示扰动基因 $g$ 后命运 $k$ 的概率变化。定义该基因在 Agent GRN 中的出边总强度为

$$
r_g=\sum_j\left|w_{g\rightarrow j}\right|.
$$

归一化后，Harness 使用乘法联合支持：

$$
s_g=
\frac{e_g}{\max_h e_h+\epsilon}
\cdot
\frac{r_g}{\max_h r_h+\epsilon},
$$

其中 $\epsilon>0$ 用于避免除零。该乘积可视为连续逻辑中的乘积 t-norm：只有扰动效应和上游网络支持同时较强时，$s_g$ 才较大。若任一证据接近零，联合支持也随之接近零。

该融合方式引入了明确的归纳偏置：一个可信的上游命运调控因子应同时满足“干预能改变命运输出”和“在调控网络中具有向外传播能力”。它可以抑制仅引起分类器变化、却缺少上游调控证据的下游 reporter。该公式不是由唯一的概率模型推导而来；两类证据也不一定统计独立，因此应将其视为模型内证据的一致性规则。

### 13.6 Median、MAD 与稳健阈值

给定联合支持集合 $\{s_g\}_{g=1}^{G}$，定义

$$
m=\operatorname{median}_g(s_g),
$$

$$
\operatorname{MAD}=
\operatorname{median}_g\left|s_g-m\right|.
$$

Harness 使用阈值

$$
\tau=m+0.5\operatorname{MAD}.
$$

中位数和 MAD 对少量极端值具有较强稳健性，其有限样本 breakdown point 可达到约 $50\%$。这使阈值不易被少量异常大的扰动效应支配，适合候选基因较少且分布可能偏斜的场景。

当前实现使用原始 MAD，没有乘正态分布下一致性缩放常数 $1.4826$；系数 $0.5$ 也属于经验选择。因此，$\tau$ 是稳健筛选规则，不是具有固定显著性水平的假设检验阈值，不能被解释为 $p$ 值或置信区间。

### 13.7 硬阈值收缩与稀疏先验

对每个基因的扰动向量 $\boldsymbol{\Delta}_g$，校准算子为

$$
\boldsymbol{\Delta}'_g=
\begin{cases}
\boldsymbol{\Delta}_g, & s_g\ge\tau,\\
\mathbf{0}, & s_g<\tau.
\end{cases}
$$

这与稀疏估计中的 hard-thresholding 类似，相当于引入“真正上游调控因子只占候选集合的一部分”的稀疏先验。其直接作用是减少弱证据结果被判为 active 的概率。

DynBench M5 的 detection 部分使用

$$
\operatorname{Precision}=\frac{TP}{TP+FP},
\qquad
\operatorname{Recall}=\frac{TP}{TP+FN},
$$

$$
F_1=\frac{2\cdot\operatorname{Precision}\cdot\operatorname{Recall}}
{\operatorname{Precision}+\operatorname{Recall}},
$$

最终得分为

$$
M5=0.5F_1+0.5A_{\mathrm{sign}}.
$$

当弱证据预测主要是假阳性时，阈值收缩通过降低 $FP$ 提高 precision 和 $F_1$。如果阈值错误地抑制真实活性扰动，则会增加 $FN$，并可能降低 recall 与符号准确率。因此，硬收缩并不保证 M5 单调提高；其收益依赖于联合支持 $s_g$ 对真实 active 和 inactive 基因具有足够的排序能力。

### 13.8 科学审计作为可行域约束

Harness 的多项审计规则可表示为约束集合。例如，命运概率必须位于概率单纯形：

$$
\mathcal{S}^{K-1}=\left\{
\mathbf{p}\in\mathbb{R}^{K}:
p_k\ge0,\ \sum_{k=1}^{K}p_k=1
\right\}.
$$

扰动产物必须满足代数一致性：

$$
\boldsymbol{\Delta}_g=
\mathbf{p}^{\mathrm{perturbed}}_g-
\mathbf{p}^{\mathrm{control}}_g.
$$

所有数值还必须满足有限性、维度、标识符集合和 schema 约束。将这些条件的交集记为可行域 $\mathcal{K}$，则 Harness 在功能上近似执行

$$
\mathcal{O}^{*}\approx\Pi_{\mathcal{K}}(\mathcal{O}),
$$

即把原始输出映射回满足公开合同和科学一致性的区域。当前实现没有显式求解

$$
\arg\min_{\widetilde{\mathcal{O}}\in\mathcal{K}}
d(\widetilde{\mathcal{O}},\mathcal{O}),
$$

因此这里的“投影”是功能类比，而不是严格的欧氏投影算法。

### 13.9 反馈修订作为迭代算子

审计与 Agent 修订可以写成迭代系统：

$$
\mathcal{O}^{(r+1)}=
\mathcal{R}\!\left(
\mathcal{D},
\mathcal{O}^{(r)},
\mathcal{H}(\mathcal{D},\mathcal{O}^{(r)})
\right).
$$

审计结果相当于误差信号，修订算子根据误差信号更新弱产物。复用同一 SessionController 保留了已经拟合的模型和会话状态，使修订更接近局部更新。

由于 $\mathcal{R}$ 包含 LLM 决策，它通常不是确定算子，也没有已知的压缩映射性质。因此，当前方法不能保证

$$
\mathcal{O}^{(r)}\rightarrow\mathcal{O}^{\infty}
$$

或每轮指标单调提高。默认限制为一次修订，并在之后执行确定性校准与 schema 验证，是为了限制随机迭代带来的漂移。

### 13.10 总分增益的线性分解

DynBench 对六项指标采用等权平均：

$$
M_{\mathrm{total}}=
\frac{1}{6}\sum_{j=1}^{6}M_j.
$$

因此，任意两次运行之间的总分差可精确分解为

$$
\Delta M_{\mathrm{total}}=
\frac{1}{6}\sum_{j=1}^{6}
\left(M_j^{\mathrm{after}}-M_j^{\mathrm{before}}\right).
$$

当前回放中只有 M2 和 M5 发生变化。使用报告中的三位小数，得到

$$
\begin{aligned}
\Delta M_{\mathrm{total}}
&\approx\frac{1.000-0.333}{6}
+\frac{1.000-0.786}{6}\\
&=0.1469\approx0.147.
\end{aligned}
$$

这与有效基线 `0.800` 到 Harness 结果 `0.947` 的变化一致。M3 从 `0` 恢复到 `0.958` 是 classifier 跨版本加载修复产生的基础设施收益，不属于上述科学校准增益。

### 13.11 方法假设与有效性边界

上述数学设计依赖以下假设：

1. Agent 输出的 `growth_rate` 至少包含与真实生物增长相关的信号，否则稳定代理只能稳定地解释一个错误模型。
2. 公开的时间和 fate 标签能够描述主要群体结构，否则分层 bootstrap 仍可能遗漏重要异质性。
3. Agent GRN 的出边强度对上游调控能力具有一定辨识度，否则乘法融合可能错误抑制真实扰动。
4. 真正有效的扰动相对稀疏，因而硬阈值收缩的先验合理。
5. 候选基因数量足以估计 median 和 MAD；候选集合极小时，阈值会较不稳定。

因此，本方法提供的是有统计依据的 Harness 设计，而非对分数提升的理论保证。特别是 ExtraTrees importance、乘法联合支持和 $0.5\operatorname{MAD}$ 阈值均包含建模选择，其跨场景有效性必须通过实验检验。

### 13.12 建议的统计验证方案

为了区分真实 Harness 收益与 LLM 随机性，应采用成对消融设计。在相同场景、provider、model、seed 和运行环境下，对每次重复分别运行：

$$
H_0:\ \texttt{--harness-revisions 0},
\qquad
H_1:\ \texttt{--harness-revisions 1}.
$$

对场景 $s$、重复 $r$ 和指标 $j$，定义成对差值

$$
d_{srj}=M_{srj}^{H_1}-M_{srj}^{H_0}.
$$

至少应报告：

- 各指标和 TOTAL 的均值、标准差与成对平均差；
- $d_{srj}$ 的 bootstrap 置信区间；
- easy、medium、hard 各难度层的分层结果；
- 分别移除 M2 校准、M5 校准和 Agent 修订轮的组件消融；
- 运行失败率、schema 通过率和平均 token/时间成本。

当重复数较少或差值分布明显非正态时，可使用 paired permutation test 或 Wilcoxon signed-rank test；样本量充足时可进一步使用包含场景随机效应的混合效应模型：

$$
d_{sr}=\beta_0+u_s+\varepsilon_{sr},
$$

其中 $\beta_0$ 表示 Harness 的平均处理效应，$u_s$ 表示场景差异。只有当跨场景置信区间稳定高于零，并且组件消融与预期机制一致时，才能支持“Harness 普遍提高科学分析能力”的结论。

## 14. Harness 有效性的逐步数学推导

本节进一步回答“为什么这些机制有机会提高指标”。推导区分两类结论：一类是在给定假设下可以严格推出的数学结果，例如集成平均降低方差、凸集投影不增加到可行真值的距离；另一类是带有明确条件的建模假设，例如 GRN 出边强度能否代表上游调控能力。后者不能被写成无条件定理。

为避免把“统计量变好”直接等同于“实际 Agent 错误下降”，以下每个机制都按三个层次分析：

1. **统计层**：熵、偏差、方差、假阳性率或可行性约束如何变化；
2. **最优风险层**：在给定信息和假设下，可达到的最小期望损失是否下降；
3. **实际 Agent 层**：当前非最优、随机的 LLM 与数值模型是否真正实现这种下降。

只有前两层的条件得到满足，并且实际算法能利用相应信息时，才可以预期具体误差项严格下降。

### 14.1 从误差来源到 Harness 算子

设理想输出为 $\mathcal{O}^{\dagger}$，Agent 实际首轮输出为 $\mathcal{O}^{(0)}$。在不规定具体距离函数的情况下，不能把不同误差来源视为天然正交且可加。为了组织后续分析，定义一个加性代理风险：

$$
\widetilde{\mathcal{E}}_{\mathrm{proxy}}
=\mathcal{E}_{\mathrm{model}}
+\mathcal{E}_{\mathrm{sampling}}
+\mathcal{E}_{\mathrm{context}}
+\mathcal{E}_{\mathrm{contract}}
+\mathcal{E}_{\mathrm{runtime}}.
$$

各项含义如下：

- $\mathcal{E}_{\mathrm{model}}$：动力学模型本身的逼近误差；
- $\mathcal{E}_{\mathrm{sampling}}$：有限细胞数、随机训练和重采样造成的估计波动；
- $\mathcal{E}_{\mathrm{context}}$：Agent 没有正确读取时间点、标签、预测目标或 schema；
- $\mathcal{E}_{\mathrm{contract}}$：概率不归一、维度错误或 `delta` 恒等式不成立；
- $\mathcal{E}_{\mathrm{runtime}}$：依赖版本、序列化或执行环境导致的失败。

这里的等号是对代理风险 $\widetilde{\mathcal{E}}_{\mathrm{proxy}}$ 的定义，不是对真实总误差的分解定理。各项可能相互作用，例如上下文错误可能进一步引发模型错误。更准确地说，它是一个误差分类框架：在选定损失函数 $L$ 后，每一项表示该类失效机制的代理量。它的作用是说明不同 Harness 组件针对不同误差来源：

$$
\begin{array}{lll}
\text{公开数据画像} &\longrightarrow& \mathcal{E}_{\mathrm{context}},\\
\text{分层 bootstrap} &\longrightarrow& \mathcal{E}_{\mathrm{sampling}},\\
\text{非线性 M2 代理} &\longrightarrow& \mathcal{E}_{\mathrm{model\text{-}interpretation}},\\
\text{M5 联合证据} &\longrightarrow& \mathcal{E}_{\mathrm{false\ positive}},\\
\text{审计与 schema 验证} &\longrightarrow& \mathcal{E}_{\mathrm{contract}},\\
\text{classifier 兼容加载} &\longrightarrow& \mathcal{E}_{\mathrm{runtime}}.
\end{array}
$$

因此，Harness 的效果来自多个误差项同时被约束，而不是一个统一公式自动提高所有指标。后文所说“降低 $\mathcal{E}$”均需要明确损失函数、统计假设和算法能力，不能仅凭某个中间统计量下降就自动推出。

### 14.2 数据画像为什么能够降低 Agent 决策不确定性

设 Agent 原本能直接读取的任务表示为随机变量 $R$，数据画像为 $C=\phi(\mathcal{D})$，需要做出的分析决策为 $Z$，例如选择时间键、预测目标或模型输出维度。条件熵满足“增加条件不会增加不确定性”：

$$
H(Z\mid R,C)\le H(Z\mid R).
$$

由条件互信息定义，

$$
I(Z;C\mid R)
=H(Z\mid R)-H(Z\mid R,C)\ge0.
$$

移项可得

$$
H(Z\mid R,C)
=H(Z\mid R)-I(Z;C\mid R).
$$

只要数据画像在已有任务表示之外提供了与正确决策相关的信息，即

$$
I(Z;C\mid R)>0,
$$

Agent 对 $Z$ 的条件不确定性就会下降。这解释了为什么明确列出时间点、样本量、标签键和 holdout 目标能够减少字段猜测。

条件熵下降还不是“实际上下文错误下降”的充分条件。为建立两者的关系，设 Agent 采取动作 $a$，真实正确决策为 $Z$，错误代价由损失函数 $L(a,Z)$ 给出。只使用 $R$ 时的最优上下文风险定义为

$$
\mathcal{E}_{\mathrm{context}}^{*}(R)
=\inf_{\delta}
\mathbb{E}\left[L(\delta(R),Z)\right],
$$

其中 $\delta$ 遍历所有只依赖 $R$ 的决策规则。加入画像后，最优风险为

$$
\mathcal{E}_{\mathrm{context}}^{*}(R,C)
=\inf_{\delta}
\mathbb{E}\left[L(\delta(R,C),Z)\right].
$$

所有只使用 $R$ 的规则也是使用 $(R,C)$ 的规则的特例，因为新规则可以选择忽略 $C$。若两个决策规则集合分别记为 $\mathfrak{D}_R$ 和 $\mathfrak{D}_{R,C}$，则

$$
\mathfrak{D}_R\subseteq\mathfrak{D}_{R,C}.
$$

在更大的集合上取下确界不会更大，所以

$$
\boxed{
\mathcal{E}_{\mathrm{context}}^{*}(R,C)
\le
\mathcal{E}_{\mathrm{context}}^{*}(R)
}.
$$

对 0-1 分类损失，这个结论可以写得更具体。只使用 $R$ 时的 Bayes 错误率为

$$
P_e^{*}(R)
=1-
\mathbb{E}_{R}
\left[
\max_z P(Z=z\mid R)
\right].
$$

加入 $C$ 后：

$$
P_e^{*}(R,C)
=1-
\mathbb{E}_{R,C}
\left[
\max_z P(Z=z\mid R,C)
\right].
$$

对固定的 $R$，最大值函数是凸函数，因此由 Jensen 不等式

$$
\begin{aligned}
&\mathbb{E}_{C\mid R}
\left[
\max_z P(Z=z\mid R,C)
\right]\\
&\qquad\ge
\max_z
\mathbb{E}_{C\mid R}
\left[P(Z=z\mid R,C)\right].
\end{aligned}
$$

由全期望公式，

$$
\mathbb{E}_{C\mid R}
\left[P(Z=z\mid R,C)\right]
=P(Z=z\mid R).
$$

所以

$$
\mathbb{E}_{R,C}
\left[
\max_z P(Z=z\mid R,C)
\right]
\ge
\mathbb{E}_{R}
\left[
\max_z P(Z=z\mid R)
\right],
$$

最终得到

$$
\boxed{P_e^{*}(R,C)\le P_e^{*}(R)}.
$$

因此，在 Bayes 最优意义下，增加数据画像不会提高最小可达到的 $\mathcal{E}_{\mathrm{context}}$。但这只是最优风险结论。实际 Agent 的上下文误差为

$$
\mathcal{E}_{\mathrm{context}}(\mathcal{A};R,C)
=\mathbb{E}
\left[L(\mathcal{A}(R,C),Z)\right],
$$

它可能因为画像错误、提示过长、注意力干扰或模型未能利用 $C$ 而上升。严格下降还要求 $C$ 改变最优动作并降低条件损失；即使 $I(Z;C\mid R)>0$，如果后验变化没有改变最优类别，条件熵也可能下降而 0-1 错误率保持不变。

**对误差项的结论。** 14.2 节严格支持

$$
\mathcal{E}_{\mathrm{context}}^{*}(R,C)
\le
\mathcal{E}_{\mathrm{context}}^{*}(R),
$$

但只能支持“实际 $\mathcal{E}_{\mathrm{context}}$ 有望下降”。当前回放没有独立消融数据画像，所以实际下降仍需用字段误用率、schema 首次通过率和审计问题数验证。

这里还必须注意：$C$ 是公开数据的压缩表示，不会创造新的生物学信息。如果 Agent 已经无损读取了完整 $\mathcal{D}$，则 $C$ 可能几乎不提供额外信息，即 $I(Z;C\mid R)\approx0$。数据画像的收益主要来自提高信息的可访问性和显著性。

### 14.3 Bootstrap 平均为什么降低重要性估计方差

设第 $b$ 次重采样得到的基因 $g$ 重要性为

$$
I_g^{(b)}=\mu_g+\varepsilon_g^{(b)},
$$

其中


$$
\mathbb{E}[\varepsilon_g^{(b)}]=0,
\qquad
\operatorname{Var}(\varepsilon_g^{(b)})=\sigma_g^2.
$$

若不同重采样估计之间具有相同相关系数 $\rho_g$，则对 $b\ne b'$，

$$
\operatorname{Cov}
\left(\varepsilon_g^{(b)},\varepsilon_g^{(b')}\right)
=\rho_g\sigma_g^2.
$$

Harness 使用平均重要性

$$
\overline{I}_g=\frac{1}{B}\sum_{b=1}^{B}I_g^{(b)}.
$$

其期望为

$$
\begin{aligned}
\mathbb{E}[\overline{I}_g]
&=\frac{1}{B}\sum_{b=1}^{B}
\mathbb{E}[I_g^{(b)}]\\
&=\frac{1}{B}\sum_{b=1}^{B}\mu_g\\
&=\mu_g.
\end{aligned}
$$

其方差为

$$
\begin{aligned}
\operatorname{Var}(\overline{I}_g)
&=\operatorname{Var}\left(
\frac{1}{B}\sum_{b=1}^{B}I_g^{(b)}
\right)\\
&=\frac{1}{B^2}
\operatorname{Var}\left(\sum_{b=1}^{B}I_g^{(b)}\right)\\
&=\frac{1}{B^2}\left[
\sum_{b=1}^{B}\operatorname{Var}(I_g^{(b)})
+2\sum_{b<b'}
\operatorname{Cov}(I_g^{(b)},I_g^{(b')})
\right]\\
&=\frac{1}{B^2}\left[
B\sigma_g^2+B(B-1)\rho_g\sigma_g^2
\right]\\
&=\sigma_g^2\left[
\rho_g+\frac{1-\rho_g}{B}
\right].
\end{aligned}
$$

由此得到三个直接结论：

1. 当重采样估计近似独立，即 $\rho_g=0$ 时，方差为 $\sigma_g^2/B$；
2. 当 $0<\rho_g<1$ 时，增加 $B$ 仍会降低方差，但最终受相关项 $\rho_g\sigma_g^2$ 限制；
3. 当 $\rho_g=1$ 时，所有重复完全相同，增加重采样次数不会降低方差。

因此，bootstrap 的有效性来自“多个估计不完全相关”。ExtraTrees 的随机分裂、样本重采样和分层抽样共同提供这种差异性。

前文的 $\mu_g$ 是估计器期望。另设希望恢复的目标重要性为 $\theta_g$，定义固定偏差

$$
b_g=\mu_g-\theta_g.
$$

则单次估计相对 $\theta_g$ 的均方误差为

$$
\operatorname{MSE}(I_g^{(1)})
=b_g^2+\sigma_g^2.
$$

假设平均操作不改变偏差，bootstrap 平均后的均方误差为

$$
\operatorname{MSE}(\overline{I}_g)
=b_g^2+\sigma_g^2
\left[
\rho_g+\frac{1-\rho_g}{B}
\right].
$$

二者之差为

$$
\begin{aligned}
&\operatorname{MSE}(\overline{I}_g)
-\operatorname{MSE}(I_g^{(1)})\\
&=\sigma_g^2
\left[
\rho_g+\frac{1-\rho_g}{B}-1
\right]\\
&=-\sigma_g^2(1-\rho_g)
\left(1-\frac{1}{B}\right).
\end{aligned}
$$

当 $B>1$ 且 $\rho_g<1$ 时，该差值为负，因此均方误差严格下降；但偏差项 $b_g^2$ 没有被消除。一个稳定但系统性错误的代理模型仍可能得到稳定的错误排名。

**对误差项的结论。** 在 $B>1$、$\rho_g<1$、$\sigma_g^2>0$ 且偏差不因集成而增加的条件下，bootstrap 严格降低重要性估计中由随机波动产生的 $\mathcal{E}_{\mathrm{sampling}}$，但不保证降低 $\mathcal{E}_{\mathrm{model}}$ 或系统性解释偏差。

### 14.4 方差下降为什么会降低基因排序错误概率

考虑两个基因 $a$ 和 $b$，真实模型解释重要性满足

$$
\mu_a>\mu_b.
$$

定义单次估计的重要性差为

$$
D^{(b)}=I_a^{(b)}-I_b^{(b)},
$$

其期望为

$$
\mathbb{E}[D^{(b)}]
=\mu_a-\mu_b
=\Delta>0.
$$

若平均后的差值为

$$
\overline{D}=\overline{I}_a-\overline{I}_b,
$$

则发生错误排序的事件为 $\overline{D}\le0$。将该事件改写为

$$
\overline{D}-\Delta\le-\Delta.
$$

由单侧 Chebyshev 型界的宽松形式可得

$$
\Pr(\overline{D}\le0)
\le
\Pr\left(
|\overline{D}-\Delta|\ge\Delta
\right)
\le
\frac{\operatorname{Var}(\overline{D})}{\Delta^2}.
$$

当 bootstrap 平均降低 $\operatorname{Var}(\overline{D})$ 时，上述错误排序概率的上界随之降低。如果进一步近似认为 $\overline{D}$ 服从正态分布，则

$$
\Pr(\overline{D}\le0)
=\Phi\left(
-\frac{\Delta}{\sqrt{\operatorname{Var}(\overline{D})}}
\right),
$$

其中 $\Phi$ 为标准正态分布函数。固定 $\Delta>0$ 时，分母越小，负的标准化距离绝对值越大，错误概率越低。

这给出了从“多次重采样降低方差”到“growth-driver 排名更稳定”的完整数学链条。若两个基因本身几乎同等重要，即 $\Delta\approx0$，则任何有限样本方法都难以稳定区分二者。

若把基因对 $(a,b)$ 的排序损失定义为

$$
L_{ab}=\mathbb{I}(\overline{I}_a\le\overline{I}_b),
$$

则其期望正是错误排序概率：

$$
\mathbb{E}[L_{ab}]
=\Pr(\overline{D}\le0).
$$

所以方差下降不是只让数值“看起来更稳定”，而是直接降低 pairwise ranking loss 的上界。对多个基因求和可得到总的成对误排序风险：

$$
\mathcal{E}_{\mathrm{rank}}
=\sum_{a,b:\mu_a>\mu_b}
\Pr(\overline{I}_a\le\overline{I}_b).
$$

**对误差项的结论。** 如果平均重要性保持正确的期望次序 $\mu_a>\mu_b$，降低方差会降低 $\mathcal{E}_{\mathrm{rank}}$ 的概率上界，并有望减少 M2 排名误差。若代理重要性的期望次序本身错误，则方差下降反而可能让错误排序更加稳定。

### 14.5 分层抽样为什么减少群体构成噪声

设共有 $H$ 个 `time x fate` 层，第 $h$ 层在总体中的比例为 $\pi_h$，层内目标统计量为 $\mu_h$，总体统计量为

$$
\mu=\sum_{h=1}^{H}\pi_h\mu_h.
$$

普通重采样中，每次抽到的层比例 $\widehat{\pi}_h$ 会随机变化，对应估计量可写为

$$
\widehat{\mu}_{\mathrm{ordinary}}
=\sum_{h=1}^{H}
\widehat{\pi}_h\widehat{\mu}_h.
$$

将误差展开：

$$
\begin{aligned}
\widehat{\mu}_{\mathrm{ordinary}}-\mu
&=\sum_h
\widehat{\pi}_h\widehat{\mu}_h
-\sum_h\pi_h\mu_h\\
&=\sum_h
\pi_h(\widehat{\mu}_h-\mu_h)
+\sum_h
(\widehat{\pi}_h-\pi_h)\mu_h\\
&\quad+\sum_h
(\widehat{\pi}_h-\pi_h)
(\widehat{\mu}_h-\mu_h).
\end{aligned}
$$

右侧三项分别表示：层内估计误差、层比例波动误差，以及二者交互项。分层 bootstrap 固定每个层的抽样数量，使

$$
\widehat{\pi}_h=\pi_h.
$$

代入上式后，第二项和第三项均为零：

$$
\widehat{\mu}_{\mathrm{stratified}}-\mu
=\sum_h
\pi_h(\widehat{\mu}_h-\mu_h).
$$

因此，分层方法消除了由时间和 fate 群体比例随机波动造成的误差，只保留层内采样误差。这对细胞群体不平衡的数据尤其重要。

若将群体构成误差定义为

$$
\mathcal{E}_{\mathrm{composition}}
=\mathbb{E}\left[
\left(
\sum_h(\widehat{\pi}_h-\pi_h)\mu_h
\right)^2
\right],
$$

则固定 $\widehat{\pi}_h=\pi_h$ 后有

$$
\mathcal{E}_{\mathrm{composition}}=0.
$$

这并不意味着总采样方差为零，因为各层内的 $\widehat{\mu}_h$ 仍会波动。若某些层只有极少细胞，层内 bootstrap 还可能产生高度重复样本，使有效样本量下降。

**对误差项的结论。** 分层抽样严格消除了由随机层比例引入的那部分 $\mathcal{E}_{\mathrm{sampling}}$。当层间均值差异大且群体不平衡时，收益通常更明显；当层划分错误、层内样本过少或目标总体本就采用另一组 $\pi_h$ 时，分层也可能引入新的偏差。

### 14.6 平均梯度为什么会漏掉重要非线性变量

考虑最简单的一维例子：

$$
X\sim\mathcal{N}(0,1),
\qquad
Y=X^2.
$$

模型对 $X$ 的局部梯度为

$$
\frac{\partial Y}{\partial X}=2X.
$$

全局平均梯度为

$$
\mathbb{E}\left[
\frac{\partial Y}{\partial X}
\right]
=2\mathbb{E}[X]
=0.
$$

于是原始统计量给出

$$
\left|
\mathbb{E}\left[
\frac{\partial Y}{\partial X}
\right]
\right|=0.
$$

但 $X$ 完全决定 $Y$。由于标准正态变量满足

$$
\mathbb{E}[X^2]=1,
\qquad
\mathbb{E}[X^4]=3,
$$

因此

$$
\begin{aligned}
\operatorname{Var}(Y)
&=\operatorname{Var}(X^2)\\
&=\mathbb{E}[X^4]
-\left(\mathbb{E}[X^2]\right)^2\\
&=3-1^2\\
&=2>0.
\end{aligned}
$$

也就是说，平均梯度认为变量重要性为零，而预测意义上的解释方差为正且不存在不可约误差。树模型可以通过在 $X$ 的不同阈值处分裂来区分 $X^2$ 的不同范围，从而获得正的方差下降重要性。

再考虑交互模型

$$
Y=X_1X_2,
$$

其中 $X_1$ 和 $X_2$ 独立、零均值。此时

$$
\frac{\partial Y}{\partial X_1}=X_2,
\qquad
\frac{\partial Y}{\partial X_2}=X_1,
$$

所以

$$
\mathbb{E}\left[
\frac{\partial Y}{\partial X_1}
\right]
=\mathbb{E}[X_2]=0,
$$

$$
\mathbb{E}\left[
\frac{\partial Y}{\partial X_2}
\right]
=\mathbb{E}[X_1]=0.
$$

两个变量的全局平均梯度都为零，但缺少任意一个变量都无法计算 $Y$。这说明全局一阶有符号梯度对状态依赖和交互作用存在结构性盲区。ExtraTrees 通过条件分裂近似 $X_1$ 与 $X_2$ 的联合分区，因而能够分配非零重要性。

设需要逼近的 growth 函数为 $f^{\dagger}$，解释代理所属的函数类为 $\mathcal{F}$。平方损失下的最优近似误差为

$$
\mathcal{E}_{\mathrm{approx}}(\mathcal{F})
=\inf_{f\in\mathcal{F}}
\mathbb{E}
\left[(f(\mathbf{X})-f^{\dagger}(\mathbf{X}))^2\right].
$$

线性或全局一阶解释对应的函数类不能精确表示 $X^2$ 和 $X_1X_2$。足够深的树集成可以用矩形分区逐步逼近这类连续或分段函数，因此在这些例子中存在更丰富的函数类 $\mathcal{F}_{\mathrm{tree}}$，使

$$
\mathcal{E}_{\mathrm{approx}}(\mathcal{F}_{\mathrm{tree}})
<
\mathcal{E}_{\mathrm{approx}}(\mathcal{F}_{\mathrm{linear}}).
$$

但更丰富的函数类也可能增加有限样本估计方差。Harness 使用叶节点最小样本数、树集成和 bootstrap 平均来控制这一方差，而不能将其完全消除。

**对误差项的结论。** 当 growth head 的真实映射包含非线性、阈值或交互效应时，ExtraTrees 有机会降低 $\mathcal{E}_{\mathrm{model\text{-}interpretation}}$ 中的近似偏差；若关系本身简单线性、样本过少或噪声很大，更复杂的代理不一定降低实际解释误差。

### 14.7 控制时间和 fate 为什么能缓解遗漏变量偏差

考虑线性示例：基因表达 $X$ 同时受时间 $T$ 影响，增长输出 $Y$ 同时受 $X$ 和 $T$ 影响：

$$
X=\alpha T+U,
$$

$$
Y=\beta X+\gamma T+\varepsilon,
$$

并假设

$$
\operatorname{Cov}(X,\varepsilon)=0.
$$

如果忽略 $T$，只用 $X$ 对 $Y$ 做一元回归，则估计系数的总体形式为

$$
\widetilde{\beta}
=\frac{\operatorname{Cov}(X,Y)}
{\operatorname{Var}(X)}.
$$

将 $Y=\beta X+\gamma T+\varepsilon$ 代入：

$$
\begin{aligned}
\operatorname{Cov}(X,Y)
&=\operatorname{Cov}
(X,\beta X+\gamma T+\varepsilon)\\
&=\beta\operatorname{Var}(X)
+\gamma\operatorname{Cov}(X,T)
+\operatorname{Cov}(X,\varepsilon)\\
&=\beta\operatorname{Var}(X)
+\gamma\operatorname{Cov}(X,T).
\end{aligned}
$$

所以

$$
\begin{aligned}
\widetilde{\beta}
&=\frac{
\beta\operatorname{Var}(X)
+\gamma\operatorname{Cov}(X,T)
}{\operatorname{Var}(X)}\\
&=\beta
+\gamma
\frac{\operatorname{Cov}(X,T)}
{\operatorname{Var}(X)}.
\end{aligned}
$$

第二项就是遗漏变量偏差。如果 $X=\alpha T+U$ 且 $U$ 与 $T$ 不相关，则

$$
\operatorname{Cov}(X,T)
=\alpha\operatorname{Var}(T),
$$

从而

$$
\widetilde{\beta}-\beta
=\gamma\alpha
\frac{\operatorname{Var}(T)}
{\operatorname{Var}(X)}.
$$

当 $\alpha\ne0$ 且 $\gamma\ne0$ 时，即使基因只是随时间变化，也会获得额外的表观关联。将 $T$ 和 fate 标签 $C$ 一起提供给代理模型，相当于估计

$$
Y=f(X,T,C)+\varepsilon,
$$

使模型不必依靠 $X$ 代替已知的时间和群体信息。

ExtraTrees 的 feature importance 不是线性回归系数，所以不能直接声称它无偏地恢复 $\beta$。上述推导说明的是控制 nuisance features 的方向性作用：减少基因变量因代理时间或群体身份而获得的重要性。若 fate 本身是基因作用的中介变量，条件化 fate 也可能屏蔽部分真实路径，因此这里仍是预测解释设计，不是严格因果估计。

在线性示例中，忽略时间时的偏差为

$$
B_{\mathrm{omit}}
=\widetilde{\beta}-\beta
=\gamma
\frac{\operatorname{Cov}(X,T)}
{\operatorname{Var}(X)}.
$$

若正确纳入 $T$，并满足 $\mathbb{E}[\varepsilon\mid X,T]=0$、无完全共线性和模型形式正确，则总体多元回归系数满足

$$
\mathbb{E}[\widehat{\beta}_{X\mid T}]=\beta,
$$

从而

$$
B_{\mathrm{controlled}}=0.
$$

因此在该理想线性模型下，控制时间确实消除了这一特定遗漏变量偏差。树模型中没有完全对应的无偏系数结论，相关变量还可能相互替代重要性。

**对误差项的结论。** 在 nuisance 变量选择正确、条件模型足够合理时，加入时间和 fate 可降低由代理群体结构造成的解释偏差；如果 fate 是目标因果路径中的中介或碰撞点，条件化反而可能引入偏差。因此这里降低的是预测解释意义上的混杂误差，不是无条件降低因果误差。

### 14.8 稳定排序为什么直接作用于 M2 AUPRC

设共有 $p$ 个基因，其中 $m$ 个是真实 growth driver。按 Agent 分数从高到低排序后，记第 $k$ 位的标签为 $y_{(k)}\in\{0,1\}$。前 $k$ 位的 precision 为

$$
P(k)=\frac{\sum_{i=1}^{k}y_{(i)}}{k}.
$$

对于离散排序，average precision 可写为

$$
AP=\frac{1}{m}
\sum_{k=1}^{p}
y_{(k)}P(k).
$$

只有真实正例所在的排名位置会贡献到该和式。如果只有一个真实 driver，即 $m=1$，且它位于第 $r$ 名，则前 $r$ 位只有一个正例：

$$
P(r)=\frac{1}{r}.
$$

因此

$$
AP=\frac{1}{r}.
$$

当真实 driver 从第 3 名移动到第 1 名时：

$$
AP_{\mathrm{before}}=\frac{1}{3}=0.333\ldots,
$$

$$
AP_{\mathrm{after}}=\frac{1}{1}=1.000.
$$

这解释了为什么候选基因较少、正例稀疏时，一次关键的排名修正可以使 M2 大幅变化。Harness 并不知道哪个基因是隐藏正例；它利用公开模型输出产生更稳定的排序，而 evaluator 之后才测量该排序是否更接近真值。

记基因排序为随机变量 $\Pi$，M2 损失为

$$
\mathcal{E}_{M2}=1-AP(\Pi).
$$

bootstrap 和非线性代理首先作用于排序分布 $P(\Pi)$，只有当它们提高真实 driver 出现在前列的概率时，才有

$$
\mathbb{E}[\mathcal{E}_{M2}^{\mathrm{after}}]
<
\mathbb{E}[\mathcal{E}_{M2}^{\mathrm{before}}].
$$

稳定性本身并不足够：一个稳定的错误排序仍有较大的 $\mathcal{E}_{M2}$。

**对误差项的结论。** 14.3 至 14.7 给出了降低方差和部分解释偏差的机制，14.8 则说明这些中间改进必须最终转化为真实 driver 排名上升，才能降低 M2 的评测误差。当前单场景回放观测到了这一转化，但尚未证明跨场景期望误差普遍下降。

### 14.9 强扰动为什么容易产生分类器假阳性

设 fate classifier 第 $k$ 类的输出为 $f_k(\mathbf{x})$。对基因 $g$ 施加扰动 $\boldsymbol{\delta}_g$ 后，二阶 Taylor 展开为

$$
\begin{aligned}
f_k(\mathbf{x}+\boldsymbol{\delta}_g)
-f_k(\mathbf{x})
&\approx
\nabla f_k(\mathbf{x})^\top
\boldsymbol{\delta}_g\\
&\quad+
\frac{1}{2}
\boldsymbol{\delta}_g^\top
\nabla^2 f_k(\mathbf{x})
\boldsymbol{\delta}_g.
\end{aligned}
$$

当扰动较小时，一阶局部近似可能有效；当 `z=-5` 使 $\|\boldsymbol{\delta}_g\|$ 很大时，二阶项及更高阶余项不再可忽略。如果扰动后的点远离训练数据流形，classifier 的输出变化可能主要反映外推行为，而不是稳定的生物学因果效应。

因此，仅依据

$$
e_g=\max_k
|f_k(\mathbf{x}+\boldsymbol{\delta}_g)-f_k(\mathbf{x})|
$$

不能区分两种情况：

1. 基因 $g$ 是上游 regulator，扰动沿真实调控路径改变命运；
2. 基因 $g$ 是下游 reporter 或分布外方向，扰动只触发 classifier 的敏感边界。

这就是 M5 需要第二类 GRN 上游支持的原因。

若把“仅由分布外外推产生、但被报告为真实活性”的概率记为

$$
\mathcal{E}_{\mathrm{OOD\text{-}FP}}
=\Pr(\widehat{A}_g=1,A_g=0,
\mathbf{x}+\boldsymbol{\delta}_g\notin\mathcal{M}),
$$

其中 $\mathcal{M}$ 表示训练数据流形，则 Taylor 展开只解释了强扰动会使这一风险上升的机制，并没有给出 Harness 后它必然下降的结论。还需要 GRN 证据确实能够区分上游 regulator 与分布外 reporter。

**对误差项的结论。** 强 classifier delta 不是可靠的因果充分条件，因此仅使用 $e_g$ 容易增大假阳性误差。14.9 节说明了增加第二类证据的必要性，但是否真正降低 $\mathcal{E}_{\mathrm{false\ positive}}$ 要由 14.10 节的证据辨识条件决定。

### 14.10 乘法证据融合为什么能降低假阳性

令事件 $A_g$ 表示“基因 $g$ 是真实 active regulator”。将扰动效应证据和 GRN 证据分别简化为两个二元检测器 $E_g$ 和 $R_g$。对 inactive 基因，定义两者的假阳性率为

$$
\alpha_E=\Pr(E_g=1\mid A_g=0),
$$

$$
\alpha_R=\Pr(R_g=1\mid A_g=0).
$$

如果在给定 $A_g=0$ 时近似条件独立，则同时通过两类证据的假阳性率为

$$
\begin{aligned}
\Pr(E_g=1,R_g=1\mid A_g=0)
&=\Pr(E_g=1\mid A_g=0)\\
&\quad\times
\Pr(R_g=1\mid A_g=0)\\
&=\alpha_E\alpha_R.
\end{aligned}
$$

只要 $0<\alpha_E<1$ 且 $0<\alpha_R<1$，就有

$$
\alpha_E\alpha_R<\min(\alpha_E,\alpha_R).
$$

因此，“两类证据同时成立”的规则理论上可以获得低于任一单独检测器的假阳性率。

当前实现使用连续分数

$$
s_g=\widetilde{e}_g\widetilde{r}_g,
$$

其中 $\widetilde{e}_g,\widetilde{r}_g\in[0,1]$。当两者均大于零时，取对数可得

$$
-\log s_g
=-\log\widetilde{e}_g
-\log\widetilde{r}_g.
$$

因此，任一证据偏弱都会在负对数空间增加惩罚；只有两者同时较强时乘积才保持较大。这是二元“同时通过”规则的连续化形式。

条件独立通常只是近似，因为 perturbation 和 GRN 可能来自同一模型。若二者错误高度相关，联合规则的假阳性率不会简单等于 $\alpha_E\alpha_R$。此外，对真实 active 基因，联合规则的真阳性率也可能从单个检测器的灵敏度下降为两者的联合灵敏度。因此，乘法融合用部分 recall 换取 precision，是否有利取决于原始错误是否以假阳性为主。

这一权衡可以用期望决策风险精确表达。设 inactive 和 active 基因的先验概率分别为

$$
\pi_0=\Pr(A_g=0),
\qquad
\pi_1=\Pr(A_g=1),
$$

假阳性和假阴性的代价分别为 $c_{FP}$ 和 $c_{FN}$。只使用 perturbation 检测器 $E_g$ 时，设其真阳性率为

$$
\beta_E=\Pr(E_g=1\mid A_g=1).
$$

期望风险为

$$
\mathcal{R}_{E}
=c_{FP}\pi_0\alpha_E
+c_{FN}\pi_1(1-\beta_E).
$$

再定义 GRN 检测器的真阳性率

$$
\beta_R=\Pr(R_g=1\mid A_g=1).
$$

若在 active 和 inactive 条件下都近似独立，则联合检测器的假阳性率和真阳性率分别为

$$
\alpha_{ER}=\alpha_E\alpha_R,
\qquad
\beta_{ER}=\beta_E\beta_R.
$$

联合风险为

$$
\mathcal{R}_{ER}
=c_{FP}\pi_0\alpha_E\alpha_R
+c_{FN}\pi_1(1-\beta_E\beta_R).
$$

两者之差为

$$
\begin{aligned}
\mathcal{R}_{ER}-\mathcal{R}_{E}
&=c_{FP}\pi_0\alpha_E(\alpha_R-1)\\
&\quad+c_{FN}\pi_1
\left[(1-\beta_E\beta_R)-(1-\beta_E)\right]\\
&=-c_{FP}\pi_0\alpha_E(1-\alpha_R)\\
&\quad+c_{FN}\pi_1\beta_E(1-\beta_R).
\end{aligned}
$$

因此，联合证据降低风险的充要条件为

$$
\boxed{
c_{FP}\pi_0\alpha_E(1-\alpha_R)
>
c_{FN}\pi_1\beta_E(1-\beta_R)
}.
$$

左侧是拒绝 GRN 不支持结果所节省的假阳性代价，右侧是同一规则新增的假阴性代价。原始结果几乎把所有基因判为 active 时，$\alpha_E$ 较高，左侧可能占优；若 GRN 的 $\beta_R$ 很低，则右侧会增大，联合过滤可能适得其反。

**对误差项的结论。** 乘法证据不是无条件降低总分类误差。它在 GRN 具有辨识力、inactive 基因较多或假阳性代价较高时降低 $\mathcal{E}_{\mathrm{false\ positive}}$ 和总决策风险；当 GRN 漏检严重时，可能以更大的 $\mathcal{E}_{\mathrm{false\ negative}}$ 为代价。

### 14.11 MAD 为什么比均值和标准差更抗极端值

设有 $G$ 个联合支持分数，排序后为

$$
s_{(1)}\le s_{(2)}\le\cdots\le s_{(G)}.
$$

当 $G$ 为奇数时，中位数为

$$
m=s_{((G+1)/2)}.
$$

只要被任意污染或替换的观测少于一半，仍有至少一半原始观测保留在中位数两侧，因此少数任意大的异常值不能把 $m$ 推到无界。这就是中位数约 $50\%$ breakdown point 的直观来源。

相比之下，样本均值为

$$
\overline{s}=\frac{1}{G}\sum_{g=1}^{G}s_g.
$$

如果只把一个观测替换为 $M$，新均值包含项 $M/G$；当 $M\rightarrow\infty$ 时，均值也趋向无穷。因此均值的有限样本 breakdown point 为 $1/G$。

MAD 首先计算绝对偏差

$$
d_g=|s_g-m|,
$$

然后取其中位数：

$$
\operatorname{MAD}=\operatorname{median}_g(d_g).
$$

由于中心和尺度都使用中位数，少数极大 perturbation 值不容易支配阈值。由此得到

$$
\tau=m+0.5\operatorname{MAD}.
$$

稳健性可以解释选择 median 和 MAD，但不能推出系数必须是 $0.5$。该系数控制 precision-recall 权衡：系数越大，保留基因越少，precision 通常上升而 recall 可能下降；系数越小则相反。

设污染比例为 $\epsilon<1/2$，观测支持分布写成

$$
F_{\epsilon}
=(1-\epsilon)F_0+\epsilon G,
$$

其中 $F_0$ 是主体分布，$G$ 是任意异常分布。均值会随 $G$ 的一阶矩无界变化，而 median 与 MAD 在污染未达到一半时保持有限。因而稳健阈值主要降低的是“少量极端分数支配阈值”的敏感性误差：

$$
\mathcal{E}_{\mathrm{threshold\ sensitivity}}
=|\tau(F_{\epsilon})-\tau(F_0)|.
$$

**对误差项的结论。** median/MAD 可以提高阈值对极端值的稳定性，但不能单独保证 active/inactive 分类错误下降。只有当主体分布对应弱证据背景、上尾对应真正的联合支持时，稳健阈值才有望降低 M5 判定误差；系数 `0.5` 的最优性必须通过消融验证。

### 14.12 硬阈值在什么条件下提高 F1

阈值应用前，设 active detection 的混淆计数为 $TP,FP,FN$。原始 F1 为

$$
F_1=\frac{2TP}{2TP+FP+FN}.
$$

假设阈值过滤掉 $a$ 个原本的 true positive 和 $b$ 个原本的 false positive，则新计数为

$$
TP'=TP-a,
$$

$$
FP'=FP-b,
$$

$$
FN'=FN+a.
$$

新 F1 为

$$
\begin{aligned}
F_1'
&=\frac{2TP'}{2TP'+FP'+FN'}\\
&=\frac{2(TP-a)}
{2(TP-a)+(FP-b)+(FN+a)}\\
&=\frac{2(TP-a)}
{2TP+FP+FN-a-b}.
\end{aligned}
$$

令

$$
D=2TP+FP+FN.
$$

要求 $F_1'>F_1$：

$$
\frac{2(TP-a)}{D-a-b}
>\frac{2TP}{D}.
$$

由于分母为正，交叉相乘：

$$
D(TP-a)>TP(D-a-b).
$$

展开两侧：

$$
DTP-Da>TPD-TPa-TPb.
$$

消去 $DTP=TPD$：

$$
-Da>-TPa-TPb.
$$

两边乘以 $-1$ 并反向不等号：

$$
Da<TPa+TPb.
$$

移项：

$$
a(D-TP)<TPb.
$$

代入 $D-TP=TP+FP+FN$，得到

$$
a(TP+FP+FN)<TPb.
$$

当 $a=0$ 且 $b>0$ 时，不等式一定成立，即只删除假阳性必然提高 F1。当 $a>0$ 时，等价条件为

$$
\frac{b}{a}>
\frac{TP+FP+FN}{TP}
=1+\frac{FP+FN}{TP}.
$$

这给出了硬阈值有效的精确条件：每误删一个 true positive，必须同时删除足够多的 false positive。当前回放中原始结果把几乎所有候选基因都判为 active，错误结构以假阳性为主，因此联合支持过滤具有提高 F1 的空间。

M5 还包含符号准确率。若阈值把一个原本方向正确的真实 active 扰动收缩为零，它会失去一次 sign match。因此，要使完整 M5 提高，过滤带来的 $F_1$ 增益还必须大于潜在的符号准确率损失。

将过滤前后的符号准确率分别记为 $A_{\mathrm{sign}}$ 和 $A'_{\mathrm{sign}}$。完整指标变化为

$$
\begin{aligned}
\Delta M5
&=M5'-M5\\
&=\frac{1}{2}(F_1'-F_1)
+\frac{1}{2}
(A'_{\mathrm{sign}}-A_{\mathrm{sign}}).
\end{aligned}
$$

所以 $M5'>M5$ 的充要条件为

$$
\boxed{
F_1'-F_1
>
A_{\mathrm{sign}}-A'_{\mathrm{sign}}
}.
$$

若共有 $N_A$ 个真实 active 的 gene-fate 对，过滤使其中 $c$ 个原本正确的 sign match 变为零，并且没有新增 sign match，则

$$
A'_{\mathrm{sign}}
=A_{\mathrm{sign}}-rac{c}{N_A}.
$$

此时完整 M5 提升要求

$$
F_1'-F_1>\frac{c}{N_A}.
$$

**对误差项的结论。** 硬阈值降低 M5 误差需要同时满足两层条件：删除的 false positive 相对误删 true positive 足够多，并且 detection F1 的增益覆盖符号准确率损失。当前回放满足该条件，但不能由阈值形式本身保证。

### 14.13 约束投影为什么不会远离可行真值

这一结论只对闭凸可行集上的正交投影严格成立。设 $\mathcal{K}$ 为闭凸集合，原始输出向量为 $\mathbf{o}$，投影为

$$
\mathbf{p}=\Pi_{\mathcal{K}}(\mathbf{o})
=\arg\min_{\mathbf{z}\in\mathcal{K}}
\|\mathbf{o}-\mathbf{z}\|_2.
$$

投影的变分不等式为：对任意 $\mathbf{y}\in\mathcal{K}$，

$$
\langle
\mathbf{o}-\mathbf{p},
\mathbf{y}-\mathbf{p}
\rangle\le0.
$$

假设真实输出 $\mathbf{o}^{\dagger}\in\mathcal{K}$。令 $\mathbf{y}=\mathbf{o}^{\dagger}$，则

$$
\langle
\mathbf{o}-\mathbf{p},
\mathbf{p}-\mathbf{o}^{\dagger}
\rangle\ge0.
$$

展开原始输出到真值的平方距离：

$$
\begin{aligned}
\|\mathbf{o}-\mathbf{o}^{\dagger}\|_2^2
&=\|(\mathbf{o}-\mathbf{p})
+(\mathbf{p}-\mathbf{o}^{\dagger})\|_2^2\\
&=\|\mathbf{o}-\mathbf{p}\|_2^2
+\|\mathbf{p}-\mathbf{o}^{\dagger}\|_2^2\\
&\quad+2\langle
\mathbf{o}-\mathbf{p},
\mathbf{p}-\mathbf{o}^{\dagger}
\rangle\\
&\ge\|\mathbf{o}-\mathbf{p}\|_2^2
+\|\mathbf{p}-\mathbf{o}^{\dagger}\|_2^2.
\end{aligned}
$$

因此

$$
\|\mathbf{p}-\mathbf{o}^{\dagger}\|_2^2
\le
\|\mathbf{o}-\mathbf{o}^{\dagger}\|_2^2
-\|\mathbf{o}-\mathbf{p}\|_2^2
\le
\|\mathbf{o}-\mathbf{o}^{\dagger}\|_2^2.
$$

也就是说，如果真值确实满足约束，把输出正交投影到闭凸可行域不会增加它到真值的欧氏距离。概率单纯形、线性归一化约束和 `delta = perturbed - control` 的仿射约束可落入这一框架。

当前 Harness 的审计与 LLM 修订并不是统一的正交投影，而且“具有 GRN 因果支持”也不形成简单凸集。因此，上述结论是部分合同约束的理论依据，不能推广为整个 Harness 必然接近隐藏真值的证明。

如果将合同误差定义为输出到可行集合的平方距离

$$
\mathcal{E}_{\mathrm{contract}}(\mathbf{o})
=\operatorname{dist}^2(\mathbf{o},\mathcal{K})
=\inf_{\mathbf{z}\in\mathcal{K}}
\|\mathbf{o}-\mathbf{z}\|_2^2,
$$

则精确投影后 $\mathbf{p}\in\mathcal{K}$，因此

$$
\mathcal{E}_{\mathrm{contract}}(\mathbf{p})=0.
$$

这说明概率归一化、有限数值和代数恒等式等硬约束能够严格消除其定义范围内的合同误差。但它们不一定改善生物学预测；一个完全满足 schema 的结果仍可能在科学上错误。

**对误差项的结论。** 对正确指定的闭凸硬约束，精确投影可严格降低或消除 $\mathcal{E}_{\mathrm{contract}}$，并且不会增加到任意可行真值的欧氏距离。LLM 审计修订和 GRN 证据规则不是这种精确投影，所以不能继承该无条件保证。

### 14.14 反馈修订何时形成误差收缩

设第 $r$ 轮输出相对理想输出的局部误差向量为

$$
\mathbf{e}_r
=\mathcal{O}^{(r)}-\mathcal{O}^{\dagger}.
$$

审计器在理想输出附近线性化为

$$
\mathcal{H}(\mathcal{O}^{(r)})
\approx\mathbf{J}\mathbf{e}_r,
$$

其中 $\mathbf{J}$ 描述审计信号对输出误差的敏感性。若修订器根据审计信号施加校正 $-\mathbf{B}\mathbf{J}\mathbf{e}_r$，并存在随机修订噪声 $\boldsymbol{\eta}_r$，则

$$
\mathbf{e}_{r+1}
=\left(\mathbf{I}-\mathbf{B}\mathbf{J}\right)
\mathbf{e}_r
+\boldsymbol{\eta}_r.
$$

记

$$
\mathbf{A}=\mathbf{I}-\mathbf{B}\mathbf{J}.
$$

若存在 $0\le q<1$ 使得

$$
\|\mathbf{A}\|_2\le q,
$$

并假设 $\boldsymbol{\eta}_r$ 零均值、与当前误差独立，且

$$
\mathbb{E}
\|\boldsymbol{\eta}_r\|_2^2
\le\sigma_\eta^2,
$$

则

$$
\begin{aligned}
\mathbb{E}
\|\mathbf{e}_{r+1}\|_2^2
&=\mathbb{E}
\|\mathbf{A}\mathbf{e}_r
+\boldsymbol{\eta}_r\|_2^2\\
&\le q^2
\mathbb{E}\|\mathbf{e}_r\|_2^2
+\sigma_\eta^2.
\end{aligned}
$$

递推展开得到

$$
\begin{aligned}
\mathbb{E}\|\mathbf{e}_{r}\|_2^2
&\le q^{2r}
\mathbb{E}\|\mathbf{e}_{0}\|_2^2\\
&\quad+
\sigma_\eta^2
\sum_{k=0}^{r-1}q^{2k}\\
&=q^{2r}
\mathbb{E}\|\mathbf{e}_{0}\|_2^2
+\sigma_\eta^2
\frac{1-q^{2r}}{1-q^2}.
\end{aligned}
$$

当 $r\rightarrow\infty$ 时，上界趋向噪声底：

$$
\limsup_{r\rightarrow\infty}
\mathbb{E}\|\mathbf{e}_{r}\|_2^2
\le\frac{\sigma_\eta^2}{1-q^2}.
$$

该推导说明反馈有效需要两个条件：审计必须正确识别误差方向，修订幅度也不能过大，使 $\|\mathbf{A}\|_2<1$。LLM 修订器并不保证满足这些条件，而且噪声可能随轮数累积。因此默认仅进行一次修订，是在利用反馈和限制随机漂移之间的工程折中。

对单轮修订，由

$$
\mathbb{E}
\|\mathbf{e}_{r+1}\|_2^2
\le q^2
\mathbb{E}\|\mathbf{e}_{r}\|_2^2
+\sigma_\eta^2
$$

可知，要使该上界严格小于当前误差，需要

$$
q^2
\mathbb{E}\|\mathbf{e}_{r}\|_2^2
+\sigma_\eta^2
<
\mathbb{E}\|\mathbf{e}_{r}\|_2^2.
$$

移项得到

$$
\boxed{
\sigma_\eta^2
<(1-q^2)
\mathbb{E}\|\mathbf{e}_{r}\|_2^2
}.
$$

也就是说，审计修订产生的随机噪声必须小于收缩作用所消除的误差。如果首轮结果已经很好，$\mathbb{E}\|\mathbf{e}_{r}\|_2^2$ 很小，那么即使 $q<1$，额外修订也更容易被噪声主导。

**对误差项的结论。** 反馈在局部收缩和低修订噪声条件下可降低输出平方误差；实际 LLM 是否满足该条件必须通过首轮与修订轮的成对审计和指标比较验证。当前副本回放没有调用 LLM 修订，因此没有实测这一误差下降。

### 14.15 从组件变化到 TOTAL 的完整计算

DynBench 总分定义为

$$
M_{\mathrm{total}}
=\frac{M1+M2+M3+M4+M5+M6}{6}.
$$

兼容修复后的有效基线为

$$
\begin{aligned}
M_{\mathrm{base}}
&=\frac{
0.982+0.333+0.958+0.890+0.786+0.850
}{6}\\
&=\frac{4.799}{6}\\
&=0.799833\ldots\\
&\approx0.800.
\end{aligned}
$$

Harness 回放结果为

$$
\begin{aligned}
M_{\mathrm{harness}}
&=\frac{
0.982+1.000+0.958+0.890+1.000+0.850
}{6}\\
&=\frac{5.680}{6}\\
&=0.946666\ldots\\
&\approx0.947.
\end{aligned}
$$

因此绝对提升为

$$
\begin{aligned}
\Delta M
&=M_{\mathrm{harness}}-M_{\mathrm{base}}\\
&=0.946666\ldots-0.799833\ldots\\
&=0.146833\ldots\\
&\approx0.147.
\end{aligned}
$$

相对提升为

$$
\begin{aligned}
\Delta M_{\mathrm{relative}}
&=\frac{M_{\mathrm{harness}}-M_{\mathrm{base}}}
{M_{\mathrm{base}}}\\
&=\frac{0.146833\ldots}{0.799833\ldots}\\
&=0.18358\ldots\\
&\approx18.4\%.
\end{aligned}
$$

由于 M1、M3、M4 和 M6 在回放前后相同，差值还可直接化简：

$$
\begin{aligned}
\Delta M
&=\frac{
(1.000-0.333)+(1.000-0.786)
}{6}\\
&=\frac{0.667+0.214}{6}\\
&=\frac{0.881}{6}\\
&=0.146833\ldots.
\end{aligned}
$$

这表明当前可观测的 Harness 增益完全由 M2 和 M5 贡献，其中 M2 贡献

$$
\frac{0.667}{6}=0.111167\ldots,
$$

M5 贡献

$$
\frac{0.214}{6}=0.035667\ldots.
$$

两者相加即为总增益。该分解是由评测权重线性性得到的精确算术关系，只有末位差异来自表格分数的三位小数舍入。

若把每项评测误差定义为

$$
\mathcal{E}_j=1-M_j,
$$

则总误差为

$$
\mathcal{E}_{\mathrm{total}}
=1-M_{\mathrm{total}}
=\frac{1}{6}\sum_{j=1}^{6}\mathcal{E}_j.
$$

因此总误差变化为

$$
\Delta\mathcal{E}_{\mathrm{total}}
=-\Delta M_{\mathrm{total}}.
$$

当前结果对应

$$
\mathcal{E}_{\mathrm{total}}:
0.200167\ldots
\longrightarrow
0.053333\ldots,
$$

即评测误差绝对下降约 $0.146833$。由于等权线性聚合，一个组件的退化可以被另一个组件的提升抵消，所以 TOTAL 上升不代表每个科学能力都提高；必须同时检查 M1-M6。

M3 的情况还说明了 $\mathcal{E}_{\mathrm{runtime}}$ 与科学误差的区别。若加载失败事件为 $F$，并且 evaluator 在 $F$ 时赋零分，则期望 M3 为

$$
\mathbb{E}[M3]
=\Pr(F^c)
\mathbb{E}[M3\mid F^c]
+\Pr(F)\cdot0.
$$

兼容修复令 $\Pr(F)$ 从当前环境中的 1 降为 0 后，恢复的是本来已经存在的预测得分。它降低了 $\mathcal{E}_{\mathrm{runtime}}$，没有改变条件科学质量 $\mathbb{E}[M3\mid F^c]$。

**对误差项的结论。** TOTAL 的改善是各指标误差下降的线性和。本次 `0.800 → 0.947` 可归因于 M2/M5 评测误差下降；原始 `0.640 → 0.800` 则来自运行时错误消除，两者不能合并解释为同一种科学能力提升。

### 14.16 哪些效果已经被实测，哪些仍是理论预期

当前证据需要按组件区分：

| 组件 | 数学机制 | 当前回放是否直接测得 | 当前结论 |
|---|---|---:|---|
| M2 校准 | bootstrap 降方差、非线性重要性、nuisance conditioning | 是 | 单场景中 `0.333 → 1.000` |
| M5 校准 | 乘法证据融合、MAD 阈值、硬收缩 | 是 | 单场景中 `0.786 → 1.000` |
| M3 兼容修复 | 消除运行时失败 | 是 | `0 → 0.958`，但不属于科学能力提升 |
| 数据画像 | 条件熵下降；Bayes 最优上下文风险不增加 | 否，尚无独立消融 | 实际 $\mathcal{E}_{\mathrm{context}}$ 下降属于待验证预期 |
| Agent 审计修订 | 局部收缩且修订噪声足够小时误差下降 | 否，回放未重新调用 LLM | 实际收缩条件尚未验证 |
| 多场景泛化 | 跨数据分布保持收益 | 否 | 尚不能下结论 |

回放实验使用已有 DeepSeek 输出的副本，没有执行新的 Agent 修订轮。因此，`0.800 → 0.947` 能直接支持“当前场景上的确定性 M2/M5 校准有效”，不能单独证明“提示词和反馈修订普遍有效”。

### 14.17 可证伪预测

上述推导给出了一组可以通过实验推翻或支持的预测：

1. 当 bootstrap 次数 $B$ 增加时，M2 排名方差应先下降后趋于由重采样相关性决定的平台；如果完全不下降，说明重复估计高度相关或代理模型不稳定来源不在采样。
2. 当 growth head 存在非线性或状态依赖效应时，ExtraTrees 排名应优于全局有符号平均梯度；在线性、同号效应场景中，两者差距应缩小。
3. 当原始 perturbation 结果以假阳性为主且 GRN 具有辨识力时，M5 联合过滤应提高 precision 和 F1；当 GRN 很差或真实扰动不稀疏时，该过滤可能降低 recall 和总分。
4. 如果数据画像确实使实际 $\mathcal{E}_{\mathrm{context}}$ 下降，则启用画像后 schema 失败率、字段误用率和首次审计问题数应下降；只有条件熵下降而这些观测量不下降时，说明 Agent 没有有效利用新增信息。
5. 如果反馈修订近似误差收缩，则一轮修订后公开审计问题数应下降；若问题数增加，说明修订算子没有满足局部收缩条件。

只有这些预测在预先确定的多场景成对消融中稳定成立，才能把单场景观察提升为对 Harness 机制的经验支持。

### 14.18 推导结论

Harness 产生效果的数学链条可以概括为：

$$
\begin{aligned}
&\text{结构化上下文}
\Rightarrow \text{条件熵下降且 Bayes 最优风险不增加},\\
&\text{分层 bootstrap}
\Rightarrow \text{去除层比例噪声；层内误差仍然存在},\\
&\text{更低方差}
\overset{\text{期望次序正确}}{\Rightarrow}
\text{更低的误排序概率上界},\\
&\text{非线性树代理}
\overset{\text{样本充分}}{\Rightarrow}
\text{减少有符号平均梯度的抵消盲区},\\
&\text{nuisance conditioning}
\overset{\text{变量选择正确}}{\Rightarrow}
\text{缓解时间和 fate 的代理关联},\\
&\text{扰动效应}\times\text{GRN 支持}
\overset{\text{GRN 有辨识力}}{\Rightarrow}
\text{降低单一证据造成的假阳性},\\
&\text{MAD 阈值与硬收缩}
\overset{\text{删除的 FP 足够多}}{\Rightarrow}
\text{提高 detection F1},\\
&\text{合同约束与反馈}
\overset{\text{正确约束、局部收缩}}{\Rightarrow}
\text{减少不可行输出并定向修正弱产物}.
\end{aligned}
$$

其中，bootstrap 方差公式、AUPRC 排名关系、F1 改善条件、凸集投影性质和 TOTAL 线性分解具有明确的数学推导；GRN 联合证据、MAD 系数和 LLM 修订收缩则依赖可检验的建模假设。因而最准确的结论是：Harness 通过降低方差、加入合理归纳偏置和施加公开可验证约束，提高了当前场景中得到高质量输出的概率，但没有无条件保证所有场景的得分都会上升。

## 15. Harness 为什么能提高 TOTAL：通俗解释

可以把 Harness 理解为给原来的 Agent 增加了一套“分析前提醒、分析后复查、错误结果校正”的机制。

原来的 Agent 能完成任务，但容易出现一种情况：整体分析方向基本正确，最后交出的个别答案却不够稳定。Harness 不会替 Agent 查阅隐藏答案，而是使用公开数据和 Agent 自己的结果，检查各项分析是否稳定、是否互相支持，以及是否符合输出规则。

整体流程可以简单理解为：

```text
原来的 Agent：读取数据 -> 分析 -> 直接提交

Harness 后：   读取数据 -> 提前整理关键信息 -> 分析
               -> 检查结果 -> 定向修改 -> 科学校准 -> 提交
```

本次 TOTAL 显著提高，最直接的原因是 Harness 修正了两个主要问题：M2 的 growth driver 排名不稳定，以及 M5 的扰动假阳性过多。

### 15.1 问题一：重要的增长驱动基因可能在平均时被抵消

原 Agent 判断 growth driver 时，主要查看基因对增长率的平均梯度。问题在于，同一个基因在不同细胞状态中的作用方向可能不同：

```text
在 A 类细胞中：促进增长
在 B 类细胞中：抑制增长
```

如果把所有细胞直接平均，就可能得到：

```text
正作用 + 负作用 ≈ 0
```

这个基因实际上很重要，但原方法可能因为正负作用相互抵消而把它排到后面。此外，真实关系还可能是非线性的，例如只有表达量超过某个阈值后才影响增长，单一平均梯度也不容易发现这种关系。

Harness 的处理方式是：

1. 按时间和细胞命运对细胞分组；
2. 在每组中进行多次重新抽样；
3. 每次用非线性模型分析哪些基因最能解释增长率；
4. 综合多次结果，选择反复出现且排名稳定的基因。

可以类比为：

```text
原方法：只看一次全班总平均分。

Harness：先按年级和班级分组，多次考试，
         再判断谁一直表现稳定。
```

在当前回放中，原始方法把 `Gene_5` 排在第一位，Harness 的稳定非线性校准把 `Gene_3` 排在第一位。DynBench 的 M2 评估的是 growth driver 的排序，因此结果为：

```text
M2：0.333 -> 1.000
```

这并不表示 Harness 知道隐藏答案。它只是把“容易受平均抵消和单次波动影响的排名”，换成了“经过分组、重复抽样和非线性分析的稳定排名”，然后由 evaluator 判断新排名是否更接近真值。

### 15.2 问题二：分类器变化不一定代表真正的因果扰动

原始扰动使用较强的 `z=-5` 干预，相当于把某个基因从正常状态突然压到非常低的水平。这可能把细胞推到模型训练时从未见过的状态。

此时 fate classifier 的输出可能明显变化，但这种变化有两种不同解释：

```text
情况一：真正的上游调控基因发生变化
        -> 调控网络发生变化
        -> 细胞命运改变

情况二：普通下游基因被强行改变
        -> 输入数据变得异常
        -> 分类器输出变化
```

原 Agent 主要看到“分类结果变化了”，因此可能把第二种情况也当成真实命运调控。在原始结果中，5 个候选基因全部被判断为有效，造成了较多假阳性。

Harness 同时检查两类证据：

```text
证据 1：扰动后，细胞命运是否明显变化？
证据 2：这个基因在 GRN 中是否具有较强的上游调控能力？
```

只有两类证据都比较强时，Harness 才保留该扰动结果。可以把它类比为调查停电原因：

```text
原方法：谁出现时发生了停电，就认为是谁导致的。

Harness：不仅检查谁出现时发生了停电，
         还检查他是否接触过配电设备，
         是否存在影响电路的实际路径。
```

当前回放中，Harness 保留了证据较强的 `Gene_1/Gene_2`，抑制了证据不足的 `Gene_3/Gene_4/Gene_5`。结果为：

```text
detection F1：0.571 -> 1.000
M5：           0.786 -> 1.000
```

这里的关键改进是减少假阳性：原来“看起来发生变化”的基因很多，Harness 进一步要求变化必须得到上游调控网络支持。

### 15.3 Harness 还给 Agent 增加了一次检查和修改机会

原 Agent 基本是完成一次分析后直接提交结果。Harness 会在首轮分析后检查：

- 六个结果文件是否完整；
- 数值是否出现 `NaN` 或无穷大；
- 概率之和是否接近 1；
- `delta = perturbed - control` 是否成立；
- holdout 分布是否坍缩成一条平均轨迹；
- fate 是否几乎全部落入同一类别；
- 是否把几乎所有扰动基因都判为有效；
- growth driver 排名是否具有稳定证据；
- perturbation 结果是否得到 GRN 支持。

如果发现问题，Harness 会把具体问题交回同一个 Agent，让它尽量保留已经正确的模型和结果，只修改薄弱部分。

可以类比为：

```text
原来的 Agent：答完直接交卷。

Harness：答题 -> 检查 -> 标出问题 -> 修改 -> 再交卷。
```

但当前 `0.800 -> 0.947` 的结果来自已有 DeepSeek 输出的副本回放，没有重新调用 LLM。因此，这次已经直接测量到的提升主要来自 M2 和 M5 的确定性校准。Agent 自我修改还能带来多少额外收益，需要通过全新运行和独立消融实验测量。

### 15.4 M3 的提升来自环境修复，不属于科学能力提高

原始结果中的 M3 为 `0`，不是因为 Agent 的 holdout 预测一定错误，而是 NumPy 版本差异导致 `fate_classifier.pkl` 无法加载。可以把它类比为：

```text
考生已经写出了答案，
但阅卷软件打不开答卷，
因此被错误记成了 0 分。
```

兼容加载修复后，在不改变 Agent 预测的情况下：

```text
M3：0 -> 0.958
```

这项修复只是让 evaluator 能够正常读取和评价已有预测，不能算作 Agent 科学分析能力的提高。因此，衡量 Harness 科学校准效果时，应使用修复后的 `TOTAL=0.800` 作为有效基线，而不是被运行环境错误压低的 `0.640`。

### 15.5 TOTAL 为什么从 0.800 提高到 0.947

DynBench 的六项指标权重相同，每项占总分的六分之一。当前回放中发生实际变化的是 M2 和 M5：

```text
M2：0.333 -> 1.000
M5：0.786 -> 1.000
```

M2 对 TOTAL 的贡献为：

```text
(1.000 - 0.333) / 6 = 0.111
```

M5 对 TOTAL 的贡献为：

```text
(1.000 - 0.786) / 6 = 0.036
```

两项相加：

```text
0.111 + 0.036 = 0.147
```

因此：

```text
有效基线：0.800
Harness：  0.947
提升：     0.147
```

可以把本次提升概括为：

```text
更稳定地寻找 growth driver
               +
减少 perturbation 假阳性
               ↓
M2 和 M5 明显提高
               ↓
TOTAL 从 0.800 提高到 0.947
```

### 15.6 当前结果能够说明什么

当前回放能够直接说明：对于 `S_balanced_easy_01_seed42` 这个场景，公开数据驱动的 M2/M5 校准修正了原始结果中的两个主要短板，并使 TOTAL 从 `0.800` 提高到 `0.947`。

它还不能证明：

- 每个数据集都会提高；
- 每次运行都能达到 `0.947`；
- 前置提示词和 Agent 自我修订一定提高分数；
- 当前阈值在 easy、medium、hard 场景中都最优。

要判断 Harness 是否具有稳定的普遍收益，需要在多个 easy、medium、hard 场景中，对 `--harness-revisions 0` 和 `--harness-revisions 1` 做相同模型、相同参数、相同重复次数的成对实验，并比较 M1-M6 和 TOTAL 的均值与标准差。
