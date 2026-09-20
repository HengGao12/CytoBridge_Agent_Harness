# 面向 CytoBridge Agent 的公开数据科学 Harness

## 摘要

CytoBridge Agent 是一个智能体工作流，用于分析单细胞时序动力学，并产出 DynBench 评估的六类产物：速度场（velocity fields）、生长速率估计与生长驱动因子排序、留出细胞状态分布、单细胞命运预测、扰动响应，以及基因调控网络（GRN）。虽然原始 agent 能够端到端完成这一工作流，但单次执行可能产出"结构上有效、但科学上不稳定"的结果。具体而言，全局有符号梯度聚合可能掩盖状态依赖的生长驱动因子；强扰动可能使命运分类器对分布外（out-of-distribution）状态产生响应；而各产物独立生成时可能缺乏跨任务一致性。

本仓库在 CytoBridge 执行循环之上增加了一个仅依赖公开数据的科学 harness。在推理之前，harness 会汇总公开任务包，并提供指标感知的分析约束。首轮执行后，它会审计六类产出，检查模式有效性、数值一致性、分布坍缩、生长驱动因子证据不稳、以及扰动效应缺乏支撑等问题。检测到的问题会被返回给同一个 agent 会话，进行可配置的修订轮次。最终产出随后经过确定性校准：生长驱动因子使用 agent 自身单细胞生长预测的分层 bootstrap ExtraTrees 代理模型重新排序（将时间与命运作为干扰变量）；扰动效应则通过归一化命运变化幅度与出边 GRN 强度之间的稳健共识来过滤（使用"中位数加 MAD"阈值）。原始产物被保留，以维持来源可追溯性与可审计性。

该 harness 不读取 DynBench 真值、历史参考输出或评估器得分。在对一次现有 DeepSeek 运行（`S_balanced_easy_01_seed42`）的回放中，修复一个独立的 NumPy 分类器加载不兼容问题后，建立了 `0.800` 的有效基线。公开数据校准将 M2 生长驱动因子恢复率从 `0.333` 提升到 `1.000`，将 M5 扰动准确率从 `0.786` 提升到 `1.000`，总分为 `0.947`。M1、M3、M4 和 M6 在科学校准下保持不变。这一 `+0.147` 的结果是来自单场景产物回放的证据，而非一般性基准声明；仍需多场景配对消融实验来估计完整 harness 的平均处理效应。

**关键词：** 智能体科学分析、单细胞动力学、harness engineering、bootstrap 稳定性、扰动分析、基因调控网络、稳健统计、DynBench

## 动机

科学类 agent 可能产出各自看似合理、却无法构成连贯且可复现科学结论的分析。在 DynBench 中，CytoBridge 必须在一次运行中完成若干相互耦合的任务，包括动力学建模、生长驱动因子排序、命运预测、扰动分析以及 GRN 推断。因此，单次执行（single-pass）的工作流可能一直把不稳定的排序、虚假的扰动效应、不一致的产物或输出契约错误保留到最终评估阶段。

harness 被引入，作为围绕这一工作流的、仅依赖公开数据的控制层。它在执行前明确任务要求，审计首轮产出，返回有针对性的修订反馈，并对存在可识别统计弱点的结果应用可复现的校准。这一方法在不读取隐藏真值、不更改评估器、不重新训练底层 CytoBridge 模型的前提下提升了可靠性。

核心问题是：公开数据诊断、稳定性估计以及跨产物一致性检查能否减少自主分析中可避免的科学错误。当前实现为此提供了一个可测试的机制，而其整体有效性仍需通过配对的多场景消融实验来确立。

## 研究目标

harness 针对科学 agent 执行中的三类失败：

1. **上下文与契约失败。** agent 可能忽略时间键、标签结构、留出目标、输出模式或概率约束。
2. **科学解释失败。** 全局梯度汇总可能抵消状态依赖的效应，而强干预可能把分类器的敏感性误当成真实的因果论断。
3. **单次执行失败。** 一个看似合理的首个答案可能在未检查六类产物是否相互一致、是否得到拟合模型支撑的情况下就被提交。

设计目标是仅使用合法基准参与者可获取的信息来减少这些失败。

## 方法

已实现的工作流如下：

```text
公开任务包
        |
        v
公开数据画像与指标感知的引导
        |
        v
CytoBridge Agent 初次执行
        |
        v
科学与契约审计
        |
        +---- 检测到问题 ----> 会话内定向修订
        |                                  |
        +----------------------------------+
        |
        v
确定性的 M2/M5 校准
        |
        v
最终审计、模式校验与 DynBench 评估
```

### 端到端执行协议

上图概括了一个八阶段执行协议。

1. **运行隔离与任务暂存。** `run_dynbench.py` 解析所请求的场景并创建新的输出目录。它只把公开任务输入复制到 `_workspace/` 中：`train.h5ad`、`TASK.md`、`prediction_targets.json`、公开命运分类器以及公开输出验证器。输出目录在运行开始时被重置，以避免把陈旧产物误认为新生成的结果。

2. **公开画像构建。** 若 `harness_revisions > 0`，harness 会读取 `train.h5ad` 并写入 `_workspace/scientific_harness_context.json`。该操作在 LLM 会话开始之前进行。它汇总可观测的数据结构，但不会推断或导入基准答案。

3. **提示增强。** 原始任务提示被扩充，加入画像路径、六文件交付契约、工作区限制、公开验证器命令，以及五道科学关卡：契约识别、模型选择、共享模型一致性、科学校准和产物验证。该引导要求 agent 在公开伪留出集上验证插值、保留分布多样性、量化驱动因子稳定性、使用匹配的扰动对照，并区分上游调控因子与下游报告基因。

4. **初次 agent 执行。** `cytobridge_runner.py` 创建一个 `SessionController`，打开任务会话，并运行初次轮次。agent 负责拟合其模型，并在输出目录中直接物化全部六个必需文件：

   - `velocity_field.csv`；
   - `growth_rates.csv`；
   - `holdout_prediction.csv`；
   - `per_cell_fate.json`；
   - `perturbation_results.json`；
   - `driver_genes.json`。

   harness 不会把合成缺失的主产物作为兜底。如果 agent 无法训练或导出其模型，本次运行即被标记为失败。

5. **公开科学审计。** harness 读取公开训练数据与六个生成的产物。它计算诊断指标，并为每个检测到的问题产出一条结构化 JSON 记录，包含问题代码、严重程度、证据与建议动作。只要存在至少一个问题，审计就会设置 `requires_revision=true`。

6. **定向会话内修订。** 当需要修订时，问题记录被渲染成新的提示并发送给同一个 `SessionController`。agent 被要求保留强产物、复用已拟合的模型、只重新生成较弱或不一致的输出。审计/修订循环在不再有问题、或达到配置的修订上限时停止。默认上限为一轮。

7. **确定性校准与最终审计。** 在修订循环之后，harness 备份原始的驱动因子与扰动 JSON 文件。随后应用下文描述的 M2 生长驱动因子校准与 M5 扰动-GRN 共识规则。对校准后的输出运行最终公开审计，所有动作写入 `scientific_harness_manifest.json`。

8. **契约校验与评分。** 基准校验六个交付物是否都存在、非空、为新生成且可解析。只有在 harness 完成后，官方 DynBench 评估器才会访问其评估资产并计算 M1–M6 与 TOTAL。harness 诊断永远不会收到这些得分。

使用 `--harness-revisions 0` 时，阶段 2、阶段 3 中 harness 特有的引导、阶段 5、6、7 均被禁用。这为配对消融提供了"无 harness"的对照条件。

### 公开数据画像

在 agent 运行之前，`build_public_data_profile()` 汇总 `train.h5ad` 与公开预测目标。它记录：

- 细胞数与已测量基因数；
- 基因名称以及检测到的时间与标签列；
- 各时间点观测到的时间值与细胞计数；
- 每个观测到的 `时间 x 标签` 分层下的细胞计数；
- 全局及逐基因的表达范围、均值与标准差；
- 各观测时间点的平均基因表达；
- 有限值覆盖率与公开留出目标说明。

画像是对公开输入的确定性压缩。它减少了重复的探索性工具调用，并使关键任务变量对 agent 更为突出。它不会在任务包之外添加生物学知识，也不保证 LLM 会正确使用这些信息。

附加的引导把分析组织成五道关卡：

1. **契约关卡：** 识别已测量空间与隐空间、观测时间、留出目标、源细胞、分类器特征顺序与输出模式。
2. **模型选择关卡：** 使用公开证据（可行时包括"留一观测时间点"检验）比较候选的动力学与生长选择。
3. **共享模型关卡：** 从一个连贯的拟合模型（或已文档化的投影）中推导速度、rollout、命运、扰动、GRN 与驱动因子输出。
4. **科学校准关卡：** 检查分布覆盖、驱动因子稳定性、匹配对照扰动与边符号稳定性。
5. **产物关卡：** 运行公开验证器，并在提交前检查数值与跨产物一致性。

### 审计与修订循环

质量控制被划分给公开模式验证器、`audit_public_outputs()` 与最终基准校验器。验证器检查公开文件契约，科学审计将当前输出与公开数据及跨产物证据进行比较，最终校验器则拒绝缺失、为空、陈旧或无法解析的交付物。

科学审计本身检查以下内容：

- 缺失必需输出，以及驱动因子或扰动 JSON 无法读取；
- 生长速率与公开训练细胞的对齐情况；
- 留出预测中的非有限值或分布坍缩；
- 命运预测坍缩到单一终末状态；
- 违反 `delta = perturbed - control` 的情况；
- 近乎普遍的扰动活性；
- 报告的生长驱动因子与稳健非线性代理模型之间的分歧；
- 缺乏上游 GRN 支撑的扰动效应。

生长驱动因子诊断综合了原始表达-生长 Spearman 关联、去除时间与标签效应后的偏关联、agent 报告的排序，以及分层非线性代理模型。只有当报告的领先基因与"在至少 75% 的审计重采样中排名第一"的代理领先基因不一致时，才会触发分歧问题。这避免了因每一次微小排序波动就触发修订。

扰动诊断首先验证恒等式

$$
\Delta_{gk}=p_{gk}^{\mathrm{perturbed}}-p_{gk}^{\mathrm{control}}.
$$

随后计算自适应的活性下限，以及被判定为活跃的候选基因比例。如果至少 80% 的基因被判定为活跃，输出就会被标记为可能缺乏选择性。审计还计算扰动与 GRN 的联合支撑，但不会查阅隐藏标签来判断哪些基因是正确的。

对于留出预测，将逐基因的中位数散布与训练分布进行比较。散布比低于 `0.1` 表明导出的粒子可能已坍缩到一条平均轨迹附近。对于命运预测，当多命运任务中超过 98% 的细胞共享同一 top 命运时，会发出警告。

每个问题被序列化为：

```json
{
  "code": "growth_driver_evidence_disagreement",
  "severity": "warning",
  "summary": "...",
  "evidence": {"reported_top": "...", "stable_surrogate_top": "..."},
  "recommended_action": "..."
}
```

修订提示包含这些问题记录，并明确要求进行模型原生重算、公开伪留出验证，以及保留已经很强的产物。由于复用了同一个 controller 与 session，修订可以访问首轮创建的模型与脚本，而不是重新开始分析。这一机制提供了纠正机会，但 LLM 修订是随机的，并不保证能改善输出。

### 稳定的生长驱动因子校准

设 `X` 为已测量基因表达矩阵，`y` 为 agent 为每个细胞预测的生长速率。对于 bootstrap 重复 `b`，harness 在 `时间 x 命运` 分层内采样细胞，并拟合一个 ExtraTrees 代理模型：

$$
f_b(X, t, c) \rightarrow y.
$$

时间与命运被作为干扰特征提供，只保留已测量基因的重要性。基因 `g` 的校准得分为

$$
S_g=\frac{1}{B}\sum_{b=1}^{B}I_g^{(b)},
$$

其中 $I_g^{(b)}$ 是重复 `b` 中的特征重要性。harness 还记录 bootstrap 标准差以及每个基因排名第一的频率。

这一过程解决了全局有符号梯度聚合的局限性。如果一个基因在某种状态下促进生长、在另一种状态下抑制生长，那么对有符号梯度取平均可能趋近于零，即便该基因具有很强的影响力。非线性代理模型能够表达阈值、交互与状态依赖关系，而 bootstrap 平均则降低了采样方差。所得重要性仍属于模型解释，不应被当作直接的因果识别。

### 扰动-GRN 共识校准

对于扰动目标 `g`，harness 计算最大绝对命运比例变化

$$
e_g=\max_k |\Delta_{gk}|
$$

及其总出边 GRN 强度

$$
r_g=\sum_j |w_{g\rightarrow j}|.
$$

归一化联合支撑为

$$
s_g=
\frac{e_g}{\max_h e_h}
\frac{r_g}{\max_h r_h}.
$$

支撑阈值以稳健方式确定：

$$
\tau=\operatorname{median}(s)+0.5\operatorname{MAD}(s).
$$

效应满足 $s_g<\tau$ 的会被收缩到其匹配对照，从而产生零最终 delta。这一乘法表现为一种连续的逻辑与（AND）：一个被报告的因果效应，必须同时得到干预响应与推断 GRN 中上游传播路径两方面的支撑。该规则是一条科学启发式，而非因果识别定理。当未校准输出被假阳性主导、且 GRN 具有有用的判别信息时，它应当会有所帮助。

### 可复现性与来源追溯

在确定性校准之前，原始的 `driver_genes.json` 与 `perturbation_results.json` 会被复制到 `scientific_harness_raw/`。最终目录记录了初始审计、校准动作、最终审计、阈值、被选中的基因、被抑制的基因以及警告。这使每一次确定性修改都可审查、可回退。

保存的 `prompt.md`、`agent_log.txt` 与 `runner_stdout.log` 记录了执行上下文与 agent 行为。`scientific_harness_manifest.json` 记录了配置的修订上限、每一轮审计、问题代码、校准动作以及最终残留的问题。重新运行校准时会使用保留的原始 JSON，而不是反复校准一个已被修改过的产物。

### 数据访问边界

harness 被有意地与评估真值分离。在画像、审计、修订与校准过程中，它只可读取：

- 从公开任务包复制的文件；
- 本次运行生成的六个产物；
- 由本次 agent 运行创建的模型、脚本与中间文件。

它不读取 `benchmark/dynbench/ground_truth/`、先前的结果目录、历史参考输出或 `eval_results.json`。真值仅在 harness 完成其产物之后，由下游评估器使用。这一分离防止了直接的得分优化，同时仍允许基于公开证据的质量控制。

## 实现

| 组件 | 作用 |
|---|---|
| `benchmark/dynbench/scientific_harness.py` | 公开画像、审计、修订提示、M2/M5 校准与来源追溯 |
| `benchmark/agent_runners/cytobridge_runner.py` | 会话内审计/修订循环与最终校准 |
| `benchmark/dynbench/run_dynbench.py` | CLI 集成、画像生成与基准编排 |
| `benchmark/agent_config.py` | harness 配置传递 |
| `benchmark/dynbench/eval/fate_classifier.py` | 兼容 NumPy 的公开分类器加载 |
| `benchmark/dynbench/scientific_harness_test.py` | 画像、审计与校准测试 |
| `benchmark/dynbench/eval/fate_classifier_test.py` | 跨版本分类器推理测试 |
| `harness.sh` | 面向 WSL/Linux 的可复现 DeepSeek harness 入口 |
| `HARNESS_ENGINEERING.md` | 完整工程原理与数学推导 |

## 评估

所报告的实验回放了某次现有 DeepSeek 冒烟运行（smoke run）的六个产物。校准阶段只读取公开的 `train.h5ad` 与 agent 生成的输出。官方评估器仅在校准之后被调用来测量结果。

| 指标 | 修正基线 | harness 回放 | 变化 |
|---|---:|---:|---:|
| M1 velocity（速度场） | 0.982 | 0.982 | 0.000 |
| M2 growth（生长） | 0.333 | 1.000 | +0.667 |
| M3 distribution（分布） | 0.958 | 0.958 | 0.000 |
| M4 fate（命运） | 0.890 | 0.890 | 0.000 |
| M5 perturbation（扰动） | 0.786 | 1.000 | +0.214 |
| M6 GRN | 0.850 | 0.850 | 0.000 |
| **TOTAL** | **0.800** | **0.947** | **+0.147** |

DynBench 对六个指标赋予相等权重。因此，

$$
\Delta\mathrm{TOTAL}
=\frac{1.000-0.333}{6}
+\frac{1.000-0.786}{6}
\approx0.147.
$$

最初保存的冒烟结果报告 `TOTAL=0.640`，因为 M3 因 `MT19937 is not a known BitGenerator module` 而失败。在 NumPy 兼容性修复之后加载未更改的预测，将 M3 恢复到 `0.958`、总分恢复到 `0.800`。这一基础设施恢复不计入科学推理方面的改进。

回放将 top 生长驱动因子从 `Gene_5` 改为 `Gene_3`。对于扰动，它保留了 `Gene_1` 与 `Gene_2`，并抑制了 `Gene_3`、`Gene_4` 与 `Gene_5` 上缺乏支撑的效应。

## 运行 harness

本项目应在 WSL/Linux 中运行，因为 CytoBridge 的部分代码依赖 `fcntl` 等 POSIX 功能。

```bash
cd /mnt/d/AI_Agent/CytoBridge-agent-main
conda activate cellcompass
bash harness.sh
```

如果 `DEEPSEEK_API_KEY` 尚未导出，`harness.sh` 会通过隐藏终端输入请求它。该 key 只会被导出到进程环境中，不会写入仓库。

等价命令为：

```bash
export DEEPSEEK_API_KEY="<your-key>"
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"

python benchmark/dynbench/run_dynbench.py \
  --scenario S_balanced_easy_01_seed42 \
  --agent-type cytobridge \
  --mode skills-on \
  --device cpu \
  --seed 42 \
  --llm-provider deepseek \
  --llm-model deepseek-chat \
  --llm-auth-mode api_key \
  --harness-revisions 1 \
  --run-label deepseek_harness \
  --run-root benchmark/results/deepseek_harness
```

CLI 当前默认进行一轮 harness 修订。显式使用 `--harness-revisions 0` 可得到无 harness 基线：

```bash
python benchmark/dynbench/run_dynbench.py \
  --scenario S_balanced_easy_01_seed42 \
  --agent-type cytobridge \
  --mode skills-on \
  --device cpu \
  --seed 42 \
  --llm-provider deepseek \
  --llm-model deepseek-chat \
  --llm-auth-mode api_key \
  --harness-revisions 0 \
  --run-label deepseek_no_harness \
  --run-root benchmark/results/deepseek_no_harness
```

## 输出产物

一次完整的 harness 运行会在常规 DynBench 输出之外，额外添加以下来源追溯文件：

```text
_workspace/scientific_harness_context.json
scientific_harness_audit_round_0.json
scientific_harness_calibration.json
scientific_harness_audit_final.json
scientific_harness_manifest.json
scientific_harness_raw/driver_genes.json
scientific_harness_raw/perturbation_results.json
```

DynBench 得分存储在所选运行目录下的 `eval_results.json` 中。

## 验证

聚焦的测试套件为：

```bash
python -m unittest \
  benchmark.dynbench.scientific_harness_test \
  benchmark.dynbench.eval.fate_classifier_test -v
```

这些测试覆盖公开数据画像、审计诊断、非线性生长驱动因子重排序、扰动校准、原始产物备份，以及使用任务包序列化命运分类器进行推理。

## 局限与后续必做工作

当前证据仅限于单场景产物回放。它直接支持该次运行上的确定性 M2/M5 校准，但并未分离出运行前引导或 LLM 修订循环的额外贡献。ExtraTrees 的杂质重要性可能以不可预测的方式在相关基因之间分摊贡献。扰动共识假设出边 GRN 强度具有信息性，且真正活跃的扰动相对稀疏。`median + 0.5 * MAD` 中的系数是经验性的，并不定义统计显著性水平。

更强的评估应当在相同模型、种子与环境下，对 easy、medium、hard 场景配对运行 `--harness-revisions 0` 与 `1`，每个配置至少重复三次。分析应报告逐指标与总分的均值、标准差、配对置信区间、模式失败率、运行时间与 token 成本。独立的组件消融应依次移除 M2 校准、M5 校准与修订轮次。在声称 CytoBridge Agent 的科学分析能力得到普遍提升之前，这些实验是必需的。

## 更多文档

参见 [`HARNESS_ENGINEERING.md`](HARNESS_ENGINEERING.md)，了解完整的实现原理、详细的数学推导、误差分析，以及对所观察到得分提升的通俗解释。
