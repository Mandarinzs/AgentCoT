# AgentCoT 顶会化研究路线（离线版）

> 说明：当前开发环境无法直接联网抓取最新论文/仓库（外网请求返回 403），因此本文件先基于该方向常见代表工作给出可执行研究路线。恢复联网后可按文末清单补齐系统性 SOTA 扫描。

## 1. 相关工作版图（论文 + 代码）

为保证论文定位清晰，建议把对比对象拆成三层：

1. **推理范式层（CoT/Agent）**  
   代表：Chain-of-Thought、ReAct、Reflexion、Tree-of-Thought、Self-Refine 等。  
   这些方法强在“推理过程可解释”，弱在“对推荐公平性约束与跨群体鲁棒性”关注不足。

2. **推荐公平层（Fairness in RecSys）**  
   代表：重加权、正则化、后处理校准等经典方法。  
   这些方法强在指标优化，弱在“自然语言推理链 + 行动决策 + 公平约束协同优化”。

3. **LLM 推荐层（LLM4Rec / Agent for Rec）**  
   代表：将 LLM 用于用户理解、解释生成、工具调用。  
   这类方法常见问题是：训练目标更偏准确率，公平性往往是附加评估而不是一等公民目标。

## 2. AgentCoT 至少三点“可投稿级”创新（建议作为主贡献）

### 创新 1：公平约束驱动的“双轨迹 CoT”

**核心思路**：每个样本同时生成两条推理链：
- `utility-trace`：最大化推荐相关性/效用；
- `fairness-trace`：显式检查群体暴露、误差差异、机会均等违反项。

在解码阶段做联合打分：
\[
S = \lambda_u S_{utility} + \lambda_f S_{fairness} - \lambda_c C_{constraint-violation}
\]

**比现有方法更好之处**：
- 从“事后评估公平”升级为“推理过程内生公平”；
- 保留可解释推理证据，便于 rebuttal 说明为何该推荐公平；
- 与当前仓库已有 fairness 模块天然兼容（可直接接入训练/评估流水线）。

### 创新 2：因果反事实增强（Counterfactual AgentCoT）

**核心思路**：构造最小改动反事实样本（仅改变敏感属性或其代理变量），要求模型输出保持“语义一致 + 公平一致”。

训练中加入双重一致性损失：
1. **答案一致性**（不该因群体变化而大幅波动）；
2. **推理一致性**（CoT 关键步骤中不出现群体偏见捷径）。

**比现有方法更好之处**：
- 直接提高跨群体稳健性，减少“看起来公平、换群体就崩”的问题；
- 对顶会审稿最敏感的“spurious correlation”给出可量化解决路径；
- 能显著增强 OOD 群体迁移表现（新地区/新年龄层）。

### 创新 3：在线闭环的“约束 bandit + Agent 反思”

**核心思路**：部署阶段将推荐选择看成 constrained contextual bandit：
- 主奖励：点击/转化/满意度；
- 约束：群体曝光下限、误差差异上限、长期满意度均衡；
- Agent 每轮进行自反思（Reflexion-style），调整策略温度与工具调用顺序。

**比现有方法更好之处**：
- 从离线 SOTA 指标扩展到真实系统最关心的长期公平-效用权衡；
- 能解释“为什么本轮策略变化”，提高工业可用性；
- 更符合顶会对“method + system + analysis”完整性的偏好。

## 3. 建议的实验设计（对标顶会标准）

### 3.1 多维基线分组

- **无 CoT 无公平**：标准指令微调推荐模型。
- **仅 CoT**：有推理但无公平目标。
- **仅公平训练**：无推理链，仅 fairness regularization/reweighting。
- **AgentCoT 完整版**：双轨迹 CoT + 反事实一致性 + 在线约束策略。

### 3.2 指标矩阵（必须联合报告）

- 效用：NDCG@K / Recall@K / MRR
- 公平：DP Gap、EO Gap、Worst-group Accuracy、Exposure Parity
- 过程质量：CoT Faithfulness、Self-Consistency
- 代价：推理时延、token 成本、在线 regret

### 3.3 必做消融

- 去掉 fairness-trace；
- 去掉 counterfactual consistency；
- 去掉 online reflexion；
- 单一群体训练 vs 多群体训练；
- 不同 \(\lambda_u, \lambda_f, \lambda_c\) 的 Pareto 前沿。

## 4. 可直接落到当前 AgentCoT 代码库的实现建议

结合仓库现状，建议分三阶段推进：

1. **阶段 A（最小可行创新）**：
   在训练循环中加入 dual-trace loss 与 constraint violation penalty；
2. **阶段 B（反事实增强）**：
   在 data pipeline 增加反事实样本生成与配对 batch；
3. **阶段 C（在线闭环）**：
   在 eval/serving 脚本中模拟 constrained bandit 回放评估。

## 5. 论文写作结构（建议）

1. 任务定义：Fair Agentic Recommendation with CoT Constraints
2. 方法：Dual-Trace CoT + Counterfactual Consistency + Constrained Online Reflexion
3. 理论或命题：公平约束对策略空间收缩与稳定性影响
4. 实验：离线 + 在线模拟 + 错误分析 + 偏见案例
5. 讨论：公平与个性化 trade-off、失败模式、社会影响

## 6. 恢复联网后应补齐的“广泛调研”执行清单

- 抓取近 3 年关键词：`LLM recommender fairness`、`agentic recommendation`、`counterfactual fairness LLM`；
- 统计每篇工作是否开源、核心损失函数、是否有在线实验；
- 构建“方法组件雷达图”：推理、工具使用、公平约束、反事实、在线学习；
- 在附录给出完整对比表（论文、代码链接、许可证、复现难度）。

---

如果你愿意，我下一步可以直接把上面的三点创新拆成：
- 可运行的配置项（`configs/*.yaml`）
- 训练损失接口（`src/agentcot/fairness` + `trainer/loop.py`）
- 一组可复现实验脚本（`scripts/*.sh`）

这样可以从“概念创新”快速进入“可提交实验结果”的状态。
