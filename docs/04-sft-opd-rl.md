# 第三阶段（中）：SFT、在线策略蒸馏与 Agent 强化学习

> 本章解释训练机制和代表性实验配置。所有路径、服务地址、内部工具名和原始 Prompt 均已移除。不同实验使用了不同模型、数据配方和 Runtime，不能把它们误写成同一个 checkpoint 连续训练。

我完成了本章涉及的 SFT、OPD 和 MoE LoRA RL 训练与相关适配。下面从监督信号、数据协议、训推对齐和分布式配置解释这些工作；框架本身提供的训练算法与算子不作为个人原创方法表述。

## 1. 为什么不是只做一次 SFT

第二阶段证明了强模型配合 Skill 可以完成复杂取数，但直接把所有线上请求交给外部模型，会受到成本、延迟和并发限制。第三阶段的目标是把这些行为内化到专用模型中。

我把三种训练放在同一条能力演进线上理解：

```mermaid
flowchart LR
    A[高质量 Teacher 离线轨迹] --> B[SFT 冷启动]
    B --> C[学生具备基本 Skill 和 tool-call 能力]
    C --> D[学生在线 Agent rollout]
    D --> E[教师逐 token 评价: OPD]
    D --> F[业务任务 reward: RL]
    E --> G[更新学生策略]
    F --> G
    G --> D
```

| 维度 | SFT | 在线策略蒸馏（OPD） | LoRA Agent RL |
| --- | --- | --- | --- |
| 训练输入 | 已固定的 Teacher 完整轨迹 | 问题，由当前学生在线执行 | 问题与 reward 标签，由当前策略在线执行 |
| 监督信号 | 标准轨迹 token | 教师对学生实际 token 的 log probability | 工具执行与最终任务 reward |
| 主要目标 | 学会基本格式、Skill 和执行模式 | 在学生自己的状态分布上靠近教师 | 提高真实任务完成质量 |
| 工具是否在训练时执行 | 否，只重放已保存结果 | 是 | 是 |
| 轨迹是否随模型变化 | 否 | 是 | 是 |
| 代表性更新 | 全参数 | 全参数 | LoRA A/B |

一句话记忆：

> SFT 是照着老师做；OPD 是学生先自己做，老师评价学生实际走过的每一步；RL 是学生多做几次，用任务结果比较哪次更好。

## 2. 三条实验线不能混成一条 checkpoint 链

项目中存在三类代表性模型实验：

| 实验线 | 模型形态 | 主要训练方式 | 目的 |
| --- | --- | --- | --- |
| 长上下文 SFT | 命名为 30B 级、训练统计约 35B、每 token 激活约 3B 的 MoE | 8 GPU 全参数微调 | 验证较大模型的多 Skill 行为上限 |
| OPD | 4B 级 Dense 学生 | SFT 后继续做全参数在线蒸馏 | 用教师修正学生自己的在线轨迹 |
| Agent RL | 大型 MoE 基座 | 冻结 Base，仅训练 LoRA | 用任务 reward 优化工具执行能力 |

30B SFT checkpoint 不能直接“继续训练成”4B OPD 学生；它们参数形状不同。DeepSeek 系列 LoRA RL 又是另一套模型结构。可以共享的是数据方法、Harness、Skill、Scorer 和工程经验，而不是直接共享权重。

还要区分模型名称与参数统计：这条 SFT 线的“30B”来自命名，训练资料记录的总参数为约 35B 级别；总参数、可训练参数和每 token 激活参数也不是同一计数。第五章 4B/35B 的历史成绩表没有提供与这里各轮 checkpoint 的完整映射，不能因规模相似就将其分配给 OPD 或 LoRA RL，见 [实验索引](06-experiment-index.md)。

## 3. SFT：模型到底看到了什么

### 3.1 Template 是模型与 Runtime 的文本协议

训练 JSONL 还不是模型最终看到的 token。Template 会把以下内容拼成一个序列：

```text
System
可用工具 Schema
User Query
Assistant thinking
Assistant tool call
Tool result
Assistant 下一轮 thinking / tool call
...
Assistant final answer
```

对 Qwen Agent 模板，一个简化片段可能是：

```text
<|im_start|>system
You are an assistant.
# Tools
...function schemas...
<|im_end|>

<|im_start|>user
查询三个地区的同一经济指标。
<|im_end|>

<|im_start|>assistant
<think>三个指标彼此独立，可以在同一轮查询。</think>
<tool_call>...</tool_call>
<tool_call>...</tool_call>
<tool_call>...</tool_call>
<|im_end|>

<|im_start|>tool
...结果 A...
<|im_end|>
```

同一个 assistant turn 中出现几个 `<tool_call>`，取决于原始轨迹该 turn 中有几个结构化调用。MS-Swift 的模板负责渲染现有分组，不会读懂 think 后自动把三轮调用改成一轮并行。

### 3.2 哪些 token 计算 SFT loss

系统说明、用户问题和工具返回是条件；模型自己生成的 thinking、tool call 和最终回答才是主要监督目标：

```text
system / user / tool result     labels = -100，loss_mask=0
assistant think / call / answer labels = token_id，loss_mask=1
```

SFT 交叉熵为：

$$
L_{SFT}=-\frac{1}{N}\sum_{t=1}^{N}\log P_\theta(y_t\mid x,y_{<t})
$$

代表性脚本使用 `loss_scale=ignore_empty_think`。它只忽略空的 thinking 区域，不代表删除所有 think，也不会改写 Teacher 的 think 内容。非空 thinking 是否训练，仍由样本与模板的 loss mask 决定。

### 3.3 Thinking 模式必须成套记录

`enable_thinking`、是否补空 think 前缀、chat template 和部署时的 reasoning parser 共同决定 token 形式。训练时看过：

```text
<think>...</think><tool_call>...</tool_call>
```

部署却按另一套 JSON 或无 think 协议解析，即使模型生成了训练时正确的 token，Runtime 也可能把它当普通文本，工具不会执行。

所以实验记录不能只写“某模型 checkpoint”，还要写：

- tokenizer 版本；
- chat/agent template；
- thinking 开关；
- tool-call parser；
- stop token；
- Runtime 及其消息协议。

## 4. 长上下文全参数 SFT 的代表性配置

| 参数 | 代表性值 | 作用 |
| --- | ---: | --- |
| GPU 进程 | 8 | 每卡一个训练进程 |
| tuner | full | 更新全部可训练参数 |
| dtype | BF16 | 降低显存，同时保持较大指数范围 |
| epoch | 2 | 完整遍历数据两次 |
| learning rate | `5e-6` | 全参数微调使用较小步长 |
| scheduler | cosine + 5% warmup | 初期稳定、后期逐步衰减 |
| micro batch | 1 / GPU | 受超长序列显存限制 |
| gradient accumulation | 16 | 累积多个 micro batch 后更新 |
| max length | 102,400 token | 覆盖长工具轨迹 |
| overlength | delete | 整条删除，避免截断调用与返回 |
| sequence parallel | 4 | 一条长序列的相关计算拆到 4 卡 |
| DeepSpeed | ZeRO-3 | 参数、梯度和优化器状态分片 |
| attention | FlashAttention | 降低长序列 attention 内存与计算开销 |
| padding | padding-free | 避免无效 padding token 计算 |
| gradient checkpointing | 开启 | 反向时重算部分前向，节省 activation |

8 卡、SP=4 时，数据并行度近似为：

$$
DP=8/4=2
$$

有效全局序列 batch 近似为：

$$
1\;(micro)\times16\;(accumulation)\times2\;(DP)=32
$$

它是序列数，不是固定 token 数。Agent 轨迹长短差异很大，每个 optimizer step 的实际 token 量仍会变化。

## 5. 为什么高显存 GPU 训练 MoE 仍然需要 ZeRO-3

模型名中的 “A3” 只表示每个 token 参与计算的激活参数约为 3B，不表示训练时只需保存 3B 参数。

训练时必须容纳完整模型，并为可训练参数保存梯度和优化器状态。主要开销包括：

```text
BF16 模型权重
+ 梯度
+ Adam 一阶矩
+ Adam 二阶矩
+ 可能存在的 FP32 master state
+ 102.4k 序列的 activation
+ attention、通信和临时 buffer
```

只算 BF16 权重，35B 级参数已经约为 70GB（十进制估算）。加入梯度和 Adam 状态后，静态训练状态会达到数百 GB；102.4k 长上下文的 activation 又是另一块巨大开销。

两种并行解决的是不同问题：

| 技术 | 主要分什么 | 解决的显存 |
| --- | --- | --- |
| ZeRO-3 | 参数、梯度、优化器状态 | 静态模型训练状态 |
| Sequence Parallel | 长序列相关 activation 与计算 | 超长上下文中间激活 |

因此，即使单卡显存很大，MoE 每 token 只激活少量专家，也不能替代 ZeRO-3。MoE 节省的是每 token 计算量，不会自动删掉未激活专家的权重和优化器状态。

## 6. 一次 SFT step 怎样发生

```mermaid
sequenceDiagram
    participant D as JSONL Dataset
    participant T as Template/Tokenizer
    participant G as 8-GPU Trainer
    participant O as Optimizer

    D->>T: messages + tools
    T->>T: 渲染角色、thinking 和 tool-call 协议
    T->>T: tokenize + loss mask + 长度门禁
    T->>G: input_ids / labels
    G->>G: ZeRO-3参数聚合 + SP长序列前向
    G->>G: response token交叉熵
    G->>G: backward，累计16个micro batch
    G->>O: grad clip + AdamW step
    O->>G: cosine scheduler step
```

训练 loss 或 token accuracy 只能说明模型对标准 token 的拟合程度。它不能替代 Agent 评测：模型可能正确生成 XML，却选错 Skill；也可能调用格式合法，却漏掉业务所需指标。

## 7. 为什么先 SFT，再做 OPD

如果基座模型还不能稳定选择 Skill、生成合法 tool call 和结束多轮任务，直接在线 rollout 会得到大量无效轨迹，教师打分和工具执行成本都会被浪费。

SFT 提供一个 Agent cold start：先让学生学会基本动作空间和协议，再在学生自己的状态分布上暴露真实错误。OPD 处理的正是第二步——Teacher 不再提供固定示范，而是评价学生实际走过的 token。

## 8. OPD：学生 response、rollout 与 segment 的区别

| 名称 | 含义 |
| --- | --- |
| 一次 response | 学生收到一次 prompt 后新生成的一段 token，可能是 tool call 或 final answer |
| 一条 rollout | 一个问题从开始到结束的完整多轮 Agent 会话 |
| 一个 segment / Sample | TrajectoryManager 线性化后的训练单元；一条 rollout 可能因 token 漂移产生多个 segment |

```text
prompt_1 = system + user
response_1 = 学生生成 tool call

执行工具后：
prompt_2 = system + user + response_1 + tool_result_1
response_2 = 学生继续调用或最终回答
```

如果把每一轮重新当成独立样本，`response_1` 会在后续 prompt 历史中重复出现，容易被重复训练。

TrajectoryManager 将它整理为：

```text
首轮 system / user               mask=0
学生 response_1                  mask=1
工具结果与下一轮上下文            mask=0
学生 response_2                  mask=1
...
```

环境 token 参与前向条件，但不参与 policy loss；每段学生输出只训练一次。

## 9. 训推来回流转时怎样保持 token 对齐

在线 Agent 的路径跨越多个系统：

```text
Agent Runtime
→ Anthropic/OpenAI Adapter
→ Student SGLang
→ tool call parser
→ 工具环境
→ Runtime 重放历史
→ 下一轮 Student SGLang
```

危险点在于：学生第一轮生成的 token 被解析成消息后，Runtime 下一轮又把历史文本传回来。如果重新渲染和 tokenize，空格、特殊 token 或 tool-call 包装可能发生漂移。

代表性实现做了三层处理：

1. **统一入口模板**：Adapter 使用学生 checkpoint 对应的 tokenizer 与 chat template。
2. **保留真实 token**：每轮直接记录 SGLang 的 `prompt_ids`、`output_ids` 和 rollout logprob，不把解码文本当唯一真相。
3. **前缀比对与分叉**：
   - 完全一致：只追加新尾部；
   - 最近一轮附近的小漂移：realign，重放部分 mask 为 0；
   - 差异过大或过早：fork 新 segment，避免强行错配。

```mermaid
flowchart LR
    A[学生真实 output_ids] --> B[解析为 Runtime 消息]
    B --> C[工具执行并重放历史]
    C --> D[下一轮重新渲染 prompt_ids]
    A --> E{比较已保存 token 前缀}
    D --> E
    E -->|一致| F[追加新增 token]
    E -->|小漂移| G[realign + replay mask 0]
    E -->|大漂移| H[fork segment]
```

这解决的是消息协议往返造成的对齐问题。Actor 更新后还必须把新权重同步回 rollout engine，否则下一批仍由旧学生生成，也会破坏 on-policy 假设。

## 10. OPD 教师怎样评价学生

教师不会重新生成一条“正确回答”。学生完成 rollout 后，系统把最终 `Sample.tokens` 原样送给教师，并设置：

```text
max_new_tokens = 0
return_logprob  = true
```

教师只做 prefill，返回：

$$
\log P_{teacher}(y_t\mid y_{<t})
$$

其中 `y_t` 是学生实际生成的 token。训练位置按相同 token ID 对齐：

```text
同一历史前缀 + 同一个学生 token
学生认为它的概率是多少
教师认为它的概率是多少
```

工具结果位置即使有教师 logprob，也会被 `loss_mask=0` 排除。

这要求教师和学生 tokenizer/词表兼容。系统直接传 token ID；如果同一个 ID 在两边代表不同 token，教师概率就没有意义。

## 11. OPD loss 不是普通的完整词表 KL

对学生实际采样 token：

$$
r_t^{reverseKL}=\log P_{student}(y_t)-\log P_{teacher}(y_t)
$$

纯 OPD 代表性实验把业务 reward 固定为 0，再构造：

$$
A_t=-\beta\left(\log P_{student}(y_t)-\log P_{teacher}(y_t)\right)
$$

当学生比教师更偏爱某个 token，advantage 为负；当教师比学生更认可这个 token，advantage 为正。随后进入 PPO-style clipped policy loss：

$$
ratio_t=\exp(\log P_{new}(y_t)-\log P_{old}(y_t))
$$

$$
L_t=-\min\left(ratio_tA_t,\;clip(ratio_t,1-\epsilon_l,1+\epsilon_h)A_t\right)
$$

代表性配置：

```text
task reward       = 0
opd_kl_coef       = 1.0
reference KL      = 0
entropy bonus     = 0
eps clip low/high = 0.20 / 0.28
```

所以更准确的描述是：

> 使用学生 on-policy token、外部教师 token logprob 构造 reverse-KL advantage，再通过 PPO-style clipped policy update 更新学生。

它复用了 GRPO/PPO 的基础设施，但本轮没有业务 reward，不能与后面的任务奖励 RL 混为一谈。

## 12. OPD 的代表性训练拓扑

| 参数 | 代表性值 |
| --- | ---: |
| prompt 数据 | 约 7.9k 道任务题 |
| 评测集 | 70 题，7 类任务各 10 题 |
| rollout batch | 8 个 prompt |
| samples / prompt | 8 条学生轨迹 |
| global batch | 64 |
| max context | 80k |
| max response | 9,600 |
| learning rate | `1e-6` constant |
| Actor | 4 GPU，TP=2、DP≈2 |
| Student rollout | 2 GPU SGLang |
| Teacher | 独立 SGLang 服务，不占上述 6 GPU |
| max Agent turns | 30 |
| max concurrent rollouts | 12 |

一次训练 step：

```mermaid
sequenceDiagram
    participant C as Ray/Controller
    participant H as Agent Harness
    participant S as Student SGLang
    participant Tool as Skill/Tool
    participant T as Teacher SGLang
    participant A as Megatron Actor

    C->>H: 调度 prompt
    H->>S: 请求当前学生生成
    S-->>H: token + rollout logprob
    H->>Tool: 执行 tool call
    Tool-->>H: tool result
    H->>S: 继续下一轮
    H-->>C: trajectory + segment + loss mask
    C->>T: 学生完整 token，prefill only
    T-->>C: teacher logprob
    C->>A: token / mask / student / teacher logprob
    A->>A: reverse-KL advantage + clipped loss
    A->>A: backward + optimizer step
    A-->>S: 同步新学生权重
```

OPD 日志中任务 reward 为 0 是设计行为，不代表训练没有信号。应检查 teacher/student alignment、loss、有效 token 比例和最终统一 Agent 评测，而不能把 reward=0 当成训练失败。

## 13. Agent RL：题库里为什么没有标准工具轨迹

RL 数据只需要提供：

```json
{
  "prompt": [{"role": "user", "content": "重新编写的金融任务"}],
  "data_source": "indicator_task",
  "reward_model": {"style": "rule"},
  "extra_info": {
    "metadata_json": "{...}",
    "label_json": "{...reward 所需标签...}"
  }
}
```

标签不作为 assistant 标准答案喂给模型。它只在 rollout 完成后供 reward 代码判断任务是否完成。

异构标签先序列化成 JSON 字符串，是为了保持 Parquet 顶层 Schema 稳定：指标任务可能是目标指标集合，表格任务可能是期望单元格，报告任务可能是文档代码。Reward 阶段再按 `data_source` 反序列化。

工具选择、调用数量、并行结构和失败恢复全部由当前 policy 在线生成。

## 14. RL 的完整闭环

```mermaid
flowchart TD
    A[Prompt + reward label] --> B[每题启动多条Agent Session]
    B --> C[Uni-Agent Gateway]
    C --> D[vLLM 当前Policy rollout]
    D --> E[Agent Runtime + Skill + Tool]
    E -->|继续多轮| C
    E --> F[最终答案与完整轨迹]
    C --> G[token ids / rollout logprob / response mask]
    F --> H[按任务类型计算reward]
    H --> I[同题GRPO相对advantage]
    G --> J[PPO old/current logprob]
    I --> K[PPO clipped loss]
    J --> K
    K --> L[Megatron backward]
    L --> M[只更新LoRA A/B]
    M --> N[同步新Adapter到vLLM]
    N --> B
```

系统组件的职责：

| 组件 | 职责 |
| --- | --- |
| verl Trainer | 采样调度、batch、advantage、loss 与训练流程 |
| vLLM Rollout | 使用当前 Base + LoRA 高并发生成 |
| Uni-Agent Gateway | 连接模型与 Agent，保存 token、logprob、mask 等训练真值 |
| Agent Runtime | 驱动多轮 Session、Skill 与工具执行 |
| Reward | 根据轨迹和标签计算任务完成质量 |
| Megatron | 分布式前向、反向与 LoRA 参数更新 |

## 15. Reward 为什么必须按任务类型设计

不同 Skill 的“正确”并不是一种形态：

| 任务类型 | 代表性 reward |
| --- | --- |
| 指标检索 | 实际查询指标集合与目标集合的 Precision / Recall / F1 |
| 表格取数 | 列名、日期、单位和单元格对齐后的 Table F1 |
| 报表定位 | 目标文档代码的 Hit@1 / Hit@5 |
| 新闻检索 | 文档 ID、标题、URL 与要点覆盖 |
| 公告、研报、企业查询 | 多次 LLM-as-a-Judge 后聚合 |

以指标检索为例：

```text
TP = 查到且属于标签的指标
FP = 查了但标签不需要的指标
FN = 标签需要但没有查到的指标

Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
Reward    = F1
```

这样既惩罚漏查，也惩罚为了碰运气而大量查询无关指标。只对最终回答文字做 Judge，可能看不到 Agent 实际调用了错误指标。

### 15.1 指标集合 reward 与业务正确性并不总是等价

第一章中，完整月度值可能足以计算季度总量；如果 reward 严格匹配标签中的季度指标 ID，这条替代路径可能被记为漏查季度指标、又多查月度指标。对于“定位指定指标集合”的任务，这种约束可以合理；对于“给出可计算的业务答案”的任务，它可能只是有偏的代理目标。

因此需要明确评分对象：计入集合的是搜索候选、发出的取数请求，还是成功返回有效数据的指标？是否接受口径一致、可计算的替代指标？探索动作与最终答案如何分别计分？当前公开示例没有给出完整等价规则，不能宣称 Set F1 已覆盖所有业务正确路径。

回归 reward 时，可以使用 [项目地图中的合成 Case](00-project-map.md)，分别检查直接季度取数和月度求和路径。若业务定义接受两者，就应通过可验证的等价映射或计算校验处理；这属于评分设计要求，不代表历史版本已经实现。

### 15.2 Reward 还要跟工具输出协议一起回归

即使 F1 公式正确，parser 读不到当前工具返回中的指标字段，也会使有效调用无法计分。检查时应先验证工具返回能否解析、提取了多少目标、标签是否非空，再计算 reward。字段缺失和真实空集合应有不同状态，不能都静默当作 0。

训练 reward 与离线 scorer 也可能使用不同目标或聚合方式。版本变化后，用固定合成 Case 对比期望集合、解析结果与最终分数，才知道分歧来自模型行为还是评分协议。

## 16. GRPO：同一道题的多次尝试怎样互相比较

代表性实验对每道题启动 8 个独立 Agent Session。设它们的任务 reward 为 `r_1...r_8`：

$$
A_i=\frac{r_i-mean(r_1...r_8)}{std(r_1...r_8)+\epsilon}
$$

关键点是：reward=0.5 不一定得到正 advantage。如果同题其他尝试平均为 0.8，这条轨迹仍然相对较差；如果其他尝试平均为 0.2，它就是正向样本。

如果 8 条轨迹全部同分，组内标准差接近 0，几乎没有相对学习信号。动态难度筛选的意义正是把预算集中到“模型有时能做对、但还不稳定”的题，而不是全对或全错的题。

一条 Session 还可能物化成多个 trajectory segment。实现中应只用每个 Session 的最终段参与 8 次独立 reward 的组内统计，再把该 Session 的 advantage 广播给较早段，避免“物化段更多的 Session”被错误当成更多次独立 rollout。

## 17. Response mask：工具返回很长，但不应该被训练

在线轨迹中的 response 区域可能包含：

```text
assistant tool call           mask=1
工具返回的大段数据              mask=0
环境注入的角色标记与模板胶水      mask=0
assistant 分析和最终回答         mask=1
```

一个 20k token 的物化轨迹，真正参与 policy loss 的模型 token 可能只有几百个。工具结果必须进入上下文，因为后续决策依赖它；但不能让模型学习复述环境自动插入的数据。

标量 reward 通常放在最后一个有效 response token 上，再由 GRPO 得到的序列级 advantage 广播到所有 `response_mask=1` 的 token。

## 18. PPO-clip 怎样更新 LoRA

GRPO 产生 advantage 后，Actor 计算：

$$
ratio_t=\exp(\log P_{current}(y_t)-\log P_{old}(y_t))
$$

并在有效模型 token 上使用非对称 clip，例如 `[0.8, 1.28]`。正 advantage 提高动作概率，负 advantage 降低动作概率；clip 限制一次更新过大。

LoRA 不直接改冻结权重 `W`：

$$
W_{effective}=W+\frac{\alpha}{r}BA
$$

代表性配置：

```text
rank    = 16
alpha   = 32
dropout = 0
scale   = alpha / rank = 2
```

LoRA 覆盖 attention 和大量 MoE 专家投影。即使 rank 只有 16，只要层数、专家数和目标矩阵很多，Adapter 仍可能达到数十亿参数和数 GB；“用了 LoRA”不等于权重文件一定很小。

## 19. MoE 训练中的 R3 Router Replay

同一个 token 在 vLLM rollout 与 Megatron Actor 中，如果被 Router 分到不同专家，那么 current/old logprob 差异中会混入“计算路径不同”的误差。PPO 可能把它误认为策略参数变化。

R3 的处理：

```text
vLLM rollout
→ 保存每层、每个 token 的 Top-K routed experts
→ routed experts 随 trajectory 进入训练 batch
→ Megatron 前向重放 rollout 时的专家选择
→ 在尽量相同的计算路径上比较 logprob
```

重放不能只覆盖最后的回答 token。回答 logits 依赖前面的 Prompt、历史回答和工具上下文形成的 hidden state，因此影响因果前缀的路由也需要对齐。

R3 解决的是“两个推理/训练引擎选了不同专家”。另一类 IS/RS 实验处理的是“rollout policy 与 current policy 已经相差太远”：前者重放路由，后者估计概率比并裁掉过度 off-policy 的 token 或序列。两者不能混为同一个算法开关。

## 20. 两节点 MoE LoRA RL 的代表性并行

| 并行维度 | 代表性值 | 切分对象 |
| --- | ---: | --- |
| TP | 1 | 本轮不再切单个 Dense 矩阵 |
| PP | 2 | Transformer 层分成两个流水段 |
| EP | 8 | MoE 专家分散到 8 个 Expert Rank |
| CP | 8 | 96k 上下文沿序列维度切到 8 卡 |
| rollout TP | 4 | vLLM 推理权重切到 4 卡 |

对于 96k 上下文：

$$
96,000 / CP8 = 12,000\;token/context\;rank
$$

16 张 GPU 主要用于容纳 PP、EP、CP 组合后的大 MoE 和长上下文，而不是简单复制出大量数据并行副本。训练与 rollout 分阶段复用资源，并使用参数、梯度、优化器 offload 以及 FP8 KV cache 控制显存。

## 21. RL 代表性配置与证据边界

| 参数 | 代表性值 |
| --- | ---: |
| Prompt batch | 8 |
| Session / prompt | 8 |
| Epoch | 5 |
| LoRA learning rate | `1e-5` |
| Temperature / top-p / top-k | `1 / 1 / -1` |
| Max Agent turns | 30 |
| Max context | 96k |
| vLLM KV cache | FP8 |
| PPO clip low/high | `0.20 / 0.28` |
| Save / eval interval | 10 step |

不同实验变体还同时改变了数据配方、学习率、长度、micro batch、R3 与 IS/RS 设置。因此，不能只看两条训练曲线就把差异全部归因于某一个机制。要证明 R3、IS 或 rejection sampling 的独立收益，需要单变量消融。

## 22. 三种训练最值得讲清楚的差异

```text
SFT
固定 Teacher 轨迹
→ Template / Tokenize / Response-only CE
→ ZeRO-3 + SP4 全参数更新

OPD
当前学生在线 Agent rollout
→ Teacher 对学生 token 做 prefill
→ reverse-KL advantage
→ PPO-style clipped update

LoRA RL
当前 Policy 每题多次 Agent rollout
→ Skill-specific task reward
→ GRPO group-relative advantage
→ PPO-clip 更新 LoRA
→ Adapter 同步回 rollout engine
```

共同的底线是：训练、rollout 和部署必须使用一致的 tokenizer、template、tool parser 与 Runtime 协议；工具返回必须参与上下文但不参与模型 loss；训练 loss 必须回到统一 Agent 评测中验收。

---

[返回首页](../README.md) · [项目地图](00-project-map.md) · [实验与数据口径](06-experiment-index.md)
