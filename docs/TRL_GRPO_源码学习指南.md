# TRL GRPO 源码学习指南

> 面向目标：读懂 TRL 的 `GRPOTrainer`，能独立解释"GRPO 在源码里每一步到底发生了什么"，
> 并能自己写奖励函数、调组大小、诊断 Reward Hacking。
> 本文档所有路径与行号均基于你服务器上的真实源码：
> `/home/jpzhao_team/zymo/my_env/lib/python3.13/site-packages/trl/`（trl 1.10.0）。
> 行号是阅读锚点，源码更新后会漂移，以函数名为准。

---

## 0. 学习路线总览（建议 1~2 周，每天 30–60 分钟）

```
阶段 0  概念地基      GRPO 数学原理（1 天）
阶段 1  跑通最小实验   exp1 长度奖励实验 + wandb 指标对照（半天，脚本由 ZCode 提供）
阶段 2  源码精读      grpo_config.py → grpo_trainer.py 五个核心方法（3~5 天）
阶段 3  动手验证      exp2 组大小实验：用源码知识解释 k=2/8/16 的差异（1 天）
阶段 4  进阶诊断      exp3 Reward Hacking：从源码角度解释"为什么会 hack"（1 天）
```

核心思想：**每读一段源码，就到 exp1/exp2 的 wandb 曲线上找到对应的现象**。
源码和曲线互相印证，比干读代码快得多。

---

## 1. 阶段 0：概念地基（先建立数学图像，再去读代码）

### 1.1 GRPO 与 PPO 的关系

GRPO（Group Relative Policy Optimization，出自 DeepSeekMath 论文）是 PPO 的简化版：

| 组件 | PPO | GRPO |
|------|-----|------|
| 价值网络（critic） | 需要，估计状态价值 | **不要了** |
| advantage 基线 | 学出来的 V(s) | **同一 prompt 采样 k 个回答的组内平均奖励** |
| KL 惩罚 | 奖励里加 KL 项 | 可选（`beta` 参数，直接加在 loss 里） |

### 1.2 必须内化的三个公式（伪代码级）

```python
# 1) 组相对 advantage：一组 k 个回答，奖励减均值除标准差
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)   # 每个 completion 一个标量

# 2) 策略梯度 + 裁剪（PPO 风格的 surrogate loss）
ratio = exp(logp_new - logp_old)            # 重要性采样比
loss  = -mean( min(ratio * adv, clip(ratio, 1-ε, 1+ε) * adv) )

# 3) 可选 KL 正则（对 ref 模型）
loss += beta * D_KL(policy || ref)
```

**关键直觉**：同一道题采样 k 个回答，好于组平均的加分、差于组平均的减分。
"组"就是免费的 baseline，这就是为什么组大小 k 是文章实验二的核心变量。

### 1.3 阅读材料（按顺序）

1. 公众号文章本身（已读过，建立整体叙事）
2. DeepSeekMath 论文 §4.1（GRPO 原始定义，只需读这一节）
3. TRL 文档 `GRPOTrainer` 页面的 "The GRPO algorithm" 图示（对照 1.2 的公式）

**自检问题**（答得出就进入下一阶段）：
- 为什么 GRPO 不需要 critic？
- 如果一个 group 里所有奖励都相同，advantage 是多少？这个 prompt 还产生梯度吗？（答：全 0，不产生。这正是文章说"只有 mixed-reward group 才有效梯度"的原因）

---

## 2. 阶段 1：跑通最小实验，先有现象再有源码

等 ZCode 把 `trl_experiments/scripts/exp1_minimal_grpo.py` 写好后：

```bash
cd ~/zymo/grpo_qwen3.5_0.8b/trl_experiments
HF_HOME=$HOME/zymo/.hf_cache CUDA_VISIBLE_DEVICES=1 \
  nohup $HOME/zymo/my_env/bin/python scripts/exp1_minimal_grpo.py \
  > logs/exp1.log 2>&1 &
```

打开 wandb 网页版，重点盯三个指标，并记住它们对应源码里的位置：

| wandb 指标 | 含义 | 源码出处（grpo_trainer.py） |
|-----------|------|---------------------------|
| `reward` | 每步奖励均值 | `log()` L3343 |
| `completions/mean_length` | 生成文本平均长度 | `log()` L3343（对 completion_ids 求和） |
| `rewards/<函数名>/mean` | 各奖励函数分别的均值 | `_calculate_rewards()` L1625 |

**这一步你要亲眼看到**：`mean_length` 从随机值在几十步内被拉向 20——
"奖励函数返回什么，策略就优化什么"，这是后面理解 reward hacking 的种子。

---

## 3. 阶段 2：源码精读路线（本指南的核心）

只有两个文件需要读：

- `trl/trainer/grpo_config.py`（1156 行，**先读**，全是带注释的参数定义）
- `trl/trainer/grpo_trainer.py`（3445 行，**只读 5 个方法**，其余跳过）

### 3.1 第一遍：grpo_config.py —— 把"旋钮"和语义对上号

先只看下面这些字段（行号是锚点），逐个读 docstring：

| 参数 | 行号 | 一句话语义 |
|------|------|-----------|
| `num_generations` | L477 | **组大小 k**：每个 prompt 采几个回答；文章实验二的 2/8/16 就是它 |
| `max_completion_length` | L491 | 生成上限；实验三里 length gaming 会顶到它 |
| `generation_batch_size` | L514 | 一次生成多少个 completion（= 每步 prompt 数 × k） |
| `steps_per_generation` | L521 | 采一批数据够几步训练用（采⽣成与更新解耦） |
| `use_vllm` / `vllm_mode` | L580/587 | 生成引擎；我们不开，用 transformers 原生生成 |
| `beta` | L674 | KL 系数；=0 则完全不拉住 ref 模型（exp3 建议 0，让 hack 现象纯粹） |
| `num_iterations` | L682 | 同一批数据重复用几遍（μ，DAPO 论文里的概念） |
| `epsilon` / `epsilon_high` | L686/698 | clip 范围；high>low 是 DAPO 的非对称裁剪 |
| `importance_sampling_level` | L753 | token 级还是 sequence 级算 ratio（GSPO 的争议点） |
| `scale_rewards` | L782 | advantage 除不除组内标准差（`group`/`batch`/`none`） |
| `loss_type` | L794 | loss 变体：grpo/bnpo/dr_grpo 等，长度归一化方式不同 |

**自检问题**：`generation_batch_size=16, num_generations=8` 时一步生成几个 prompt 的回答？
（答：2 个。文章实验二说"k 大则每步覆盖的 prompt 少"，根源就在这里。）

### 3.2 第二遍：grpo_trainer.py 五个核心方法（按数据流顺序）

带着这条主线读：**prompt 进来 → 生成 k 个回答 → 打分 → 变 advantage → 算 loss**

```
① __init__  (L303)          —— 跳读 10 分钟
② _generate_and_score_completions (L2330)  —— 精读，是全文件的心脏
③ _calculate_rewards (L1625)               —— 精读，你写的奖励函数在这被调用
④ _compute_loss (L3061)                    —— 精读，GRPO 数学公式在这落地
⑤ log (L3343)                              —— 略读，wandb 指标从哪来
```

#### ① `__init__`（L303）
只关注三件事：模型和 ref 模型的加载、processing class（tokenizer）、
vLLM 相关的分支直接跳过（我们不用 vLLM）。
**读它的目的**：知道 trainer 持有哪些对象，后面方法里出现的 `self.xxx` 才不陌生。

#### ② `_generate_and_score_completions`（L2330）
- 找到把 prompt 复制 k 份的地方（`num_generations` 的作用点）
- 找到调用 `self._generate(...)`（L2213）得到 completion_ids 的地方
- 找到调用 `self._calculate_rewards(...)` 的地方
- **自检问题**：重复第 i 个 prompt 时怎么保证 k 个回答不一样？（temperature 采样）

#### ③ `_calculate_rewards`（L1625）
- 看清你传给 `GRPOTrainer(reward_funcs=...)` 的函数是怎么被调用的：
  签名约定 `def my_reward(prompts, completions, **kwargs)`，
  dataset 的额外列会通过 kwargs 传进来（`_set_signature_columns_if_needed` L1211）——
  这就是 exp2 里把 `test_list` 传进奖励函数的通道
- 返回值是一个 batch 的奖励张量
- **自检问题**：如果奖励函数内部抛异常，训练会发生什么？（去代码里找答案）

#### ④ `_compute_loss`（L3061）
对照 §1.2 的三个公式逐行找：
- advantage 归一化（对应 `scale_rewards`）
- `ratio = exp(per_token_logps - old_logps)`（对应 `num_iterations>1` 时的重算）
- clip 与 `epsilon_high`
- `beta * KL`（对应 `beta`；`_get_per_token_logps_and_entropies` L1448 提供 logp）
- **自检问题**：completion 长度不同时，loss 里每个 token 的权重一样吗？
  （由 `loss_type` 决定——这是 exp3 里"模型倾向把回复写长"的 loss 侧原因之一）

#### ⑤ `log`（L3343）
略读即可：确认 wandb 上每个指标是在哪算出来的，以后看到曲线异常知道去哪查。

### 3.3 精读时的操作建议

```bash
# 在 VS Code 远程窗口里直接打开（不要复制出来，行号锚点才有效）
code /home/jpzhao_team/zymo/my_env/lib/python3.13/site-packages/trl/trainer/grpo_trainer.py
```

- 用"跳转到定义"追 `self.xxx`，但**每追一层最多 5 分钟**，防止在 transformers 库里迷路
- 在 `_calculate_rewards` 和 `_compute_loss` 里加 `print()` 或断点，用 exp1 的 5 步冒烟跑一次，
  亲眼看看一个 batch 的 shape：`[batch × k, seq_len]`
- 每读完一个方法，用自己的话写 3 句话笔记，写进 `trl_experiments/docs/notes.md`

---

## 4. 阶段 3：用实验二检验源码理解

读完后做 exp2（k=2/8/16）。**要求自己先预测、再看曲线**：

| 预测项 | 用源码知识推理 | 文章实测 |
|--------|---------------|---------|
| k=8 vs k=2 谁的 reward 更高 | k 大 → 组内 baseline 更稳，但 `generation_batch_size/k` → 每步 prompt 少、多样性低 | k=8 最优（0.67） |
| 三组的每步 prompt 数 | = `generation_batch_size / k` | — |
| 哪组每步梯度的 prompt 覆盖广 | 组内全对/全错 → advantage≈0 的比例 | — |

如果实测和预测矛盾，回到 §3.2 的 ② 和 ④ 找原因——这一步才是真的"读懂了"。

## 5. 阶段 4：Reward Hacking 的源码解释（exp3 的思考题）

写 exp3 前先想清楚三个问题，写完实验用曲线验证：

1. 奖励"数 code block 个数"后，模型优化的是 `_calculate_rewards` 返回的那个数——
   源码里有没有任何机制阻止它钻空子？（没有：奖励函数即目标）
2. 真实能力（通过 assert）为什么反而下降？哪个超参数能缓解？（`beta`、KL、clip）
3. `completions/mean_length` 顶格到 `max_completion_length`，说明模型在"刷"什么？

## 6. 时间分配建议

- **今天这半小时**：读 §1 + 打开 grpo_config.py 把 §3.1 的表过一遍（参数 docstring 都很短）
- 本周：跑 exp1 → 精读 §3.2 的 ② ③（最关键的两个方法）
- 下周：④ ⑤ + exp2/exp3 对照验证

## 7. 常见坑速查

| 现象 | 多半是这个原因 |
|------|---------------|
| reward 曲线一直不动 | group 内全对或全错（advantage≈0），调 prompt 难度或 k |
| loss 为 NaN | logp 下溢 / fp16，用 bf16；检查 `max_completion_length` 是否太小 |
| 显存爆 | `generation_batch_size × seq_len` 太大，减 batch 或长度 |
| 曲线和文章对不上 | 框架/版本差异（文章是 TRL 旧版 + vLLM），先对齐 `loss_type`/`scale_rewards` |
| 奖励函数没被调用 | 函数签名不对，`**kwargs` 缺失导致 dataset 列传不进来（L1211） |
