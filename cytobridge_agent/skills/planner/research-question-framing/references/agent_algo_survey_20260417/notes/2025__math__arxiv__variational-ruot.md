# Variational Regularized Unbalanced Optimal Transport: Single Network, Least Action

## Metadata
- 标题: Variational Regularized Unbalanced Optimal Transport: Single Network, Least Action
- 作者: Qiangwei Peng, Yuchen Li, Peijie Zhou, et al.
- 年份: 2025
- 正式 venue: arXiv preprint
- PDF 文件名: Variational Regularized Unbalanced Optimal Transport Single Network, Least Action.pdf
- 是否由 arXiv/bioRxiv 追认正式发表: 否，当前按预印本处理
- 是否与 CytoBridge-agent 目标直接相关: 是
- 阅读完成状态: 已完成

## 解决了什么已有 gap
RUOT 已经能同时表达随机性和质量不守恒，但许多现有实现并没有把最优必要条件真正写进模型结构，导致学到的解不一定满足 least-action 原理，训练也容易不稳。另一个实际问题是 WFR 常用的二次 growth penalty 在生物上未必合理。

## 理论 / 算法创新点
这篇文章提出 Var-RUOT，用变分方法推导 RUOT 的一阶最优条件，并把这些条件直接塞进参数化和损失函数。最重要的结果是：模型只需要学习一个标量场，而不必分别学习多个动力学量，这使搜索空间明显收缩，也更贴近最优解结构。

## 具体算法或理论结构怎么设计
作者把 action 最小化的必要条件用于重新表达 transport/growth 相关变量，使单一神经网络输出标量势函数即可诱导出所需动力学。训练目标也不再只是经验拟合，而是显式惩罚偏离 RUOT 最优条件的解。文章还单独讨论了 growth penalty 的选择标准，指出经典二次罚项会产生不够生物合理的增长模式。

## 设计原则是什么
设计原则是“先理解最优解应满足什么，再决定网络怎么长”。这和很多深度方法的出发点相反：不是先扔一个通用网络，再靠损失去逼；而是用变分结构先压缩解空间。这对科学问题尤其重要，因为它能减少不合理自由度。

## 工程优化或实现性考虑
工程上，单网络标量势函数让训练更轻，文章也报告了更少训练轮次和更好的稳定性。对 snapshot dynamics 这类本来就容易训练发散的任务，这种结构化简化非常实用。

## 局限与未解点
虽然比一般 RUOT 更稳，但它仍然依赖 growth penalty 和函数空间选择；若这些先验设错，结构化反而会放大偏差。方法目前主要处理 transport + growth，本身没有把 interaction 纳入主框架。

## 为什么能发顶会 / 为什么是重要理论工作
这篇工作的重要之处在于，它不是又提出一个更大的网络，而是把 RUOT 的数学结构真正前移到模型设计层。对单细胞动力学这种机制导向任务，这类“结构优先”的改进通常比纯经验调参更有长期价值。

## 对单细胞轨迹建模和 CytoBridge-agent 自动设计算法的启发
对 CytoBridge 而言，Var-RUOT 的启发非常关键：搜索空间应尽量围绕最优性条件压缩，而不是放任模型过度自由。自动设计算法时，可以优先探索“由势函数诱导动力学”的结构化家族，并把 growth penalty 设计本身当成会影响生物合理性的核心变量。
