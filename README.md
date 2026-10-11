# GRPO-RLCD-train-FPB-sst5

基于 **GRPO + RLCD（reward / loss with a proper scoring rule for calibration）** 的
小型语言模型校准决策训练实验。项目在 Qwen3-0.6B 上研究：如何让模型在分类决策中
**既答得更准，又给出可信的概率**，并把置信度当作可用的调度信号
（risk-coverage / 级联直答）。

## 1. 方法：RLCD 的两种实现

| 方案 | 校准信号 | 结论 |
| --- | --- | --- |
| **方案 B**（`exp4_rlcd_grpo.py`） | `r = 0.5·r_correct + 0.5·r_cal`，校准项进 reward | 单 token 决策拿不到完整分布，`r_cal` 与 `r_correct` 同信号；组内减均值后校准信号消失 → **无法校准** |
| **方案 A**（`exp5_rlcd_calibrated.py`、`fpb_rlcd.py`） | 多类 **Brier** 作为可微辅助损失：`loss = grpo_loss + λ·Brier` | 校准梯度不依赖组内方差，有梯度步 100%；accuracy 与 ECE 同时改善 |

关键技术点（见 `rlcd/exp5_rlcd_calibrated.py` 头部注释）：

- 覆写 `compute_loss`：`super()` 计算标准 GRPO 项，再二次前向取 completion 首个
  数字 token 位置的候选 logits（`"0"~"4"` 与 `" 0"~" 4"` 共 10 个变体），
  softmax 后合并成 5 维分布，计算多类 Brier 与 one-hot gold 的差；
- 覆写 `_generate_and_score_completions`：把数据集列 `gold` 注回训练 `inputs`
  （TRL 默认只把额外列传给 reward kwargs，直接在 `compute_loss` 里读会 `KeyError`）；
- reward 只保留 `r_correct`，校准信号完全走 loss，避开"组内常量被减掉"的问题。

## 2. 数据集

| 数据集 | 类别 | 规模 | 用途 |
| --- | --- | --- | --- |
| `SetFit/sst5` 影评情感 | 5 类（very negative ~ very positive） | train 8544 / val 1101 / test 2210 | exp4（方案 B）、exp5（方案 A）、λ/G 扫描、数据规模实验 |
| `FinanceInc/auditor_sentiment`（FPB allagree 子集） | 3 类（negative / neutral / positive） | train 3877 / test 969（官方测试集一次性使用） | FPB-GRPO-RLCD E1/E2 与 SFT/CPT 对照 |

## 3. 主要结果

**SST5（exp5，方案 A，λ=0.5 / G=16 / 300 步 / 2 分 51 秒）**

| 指标 | base Qwen3-0.6B | GRPO+RLCD | 变化 |
| --- | --- | --- | --- |
| accuracy | 0.326 | **0.402** | **+7.6 pt** |
| ECE (10 bin) | 0.466 | **0.125** | **↓ 73%** |
| mean confidence | 0.792（过自信） | 0.524（贴近真实 0.40） | 校准成功 |

λ 扫描：0.25 略差，0.5 与 1.0 打平 → **λ=0.5 为甜点**。
G 扫描：**G=8 的 accuracy 最高（0.415）**，G=16 次之；G=32/64 无额外收益
（分类任务组内多样性充足）。数据规模：600→4000 条时 accuracy 0.415→0.462、
ECE 0.134→0.090。

**FPB 金融新闻 3 分类（1 epoch / 1688 步 / 15 分钟）**

| 指标 | base zero-shot | GRPO+RLCD | 变化 |
| --- | --- | --- | --- |
| accuracy | 0.3065 | **0.7513** | **+44.5 pt** |
| ECE (10 bin) | 0.4197 | **0.1702** | **↓ 59%** |

风险-覆盖（级联调度）：目标准确率 ≥90% 时，base 可直答 0%，RLCD 模型可直答 42%；
目标 ≥85% 时可直答 58% —— 置信度可以直接当调度信号使用。

**GRPO 三联实验复现（exp1–exp3）**：奖励函数决定行为（长度被压向目标值）、
组大小 k=8 最优、reward hacking 两版对照（capped / uncapped）与
(lr × temperature × max_length) 消融，详见 `docs/实验小结.md`。

## 4. 目录结构

```
rlcd/                    # RLCD / GRPO 训练、评估与绘图脚本
  common.py              #   公共模块（路径、wandb、MBPP harness）
  exp1_minimal_grpo.py   #   exp1：最小化 GRPO（奖励=长度接近目标）
  exp2_mbpp_grpo.py      #   exp2：MBPP 可验证奖励，k=2/8/16
  exp3_reward_hacking.py #   exp3：reward hacking 对照
  exp4_rlcd_grpo.py      #   RLCD 方案 B（r_cal 进 reward）
  exp5_rlcd_calibrated.py#   RLCD 方案 A（Brier 进 loss，5 类）
  fpb_rlcd.py            #   FPB 3 类 RLCD 训练（1 epoch = 1688 步）
  sft_fpb.py / cpt_fpb.py#   FPB 上的 SFT / CPT 对照
  eval_fpb.py            #   FPB test 评估（accuracy + ECE）
  fpb_risk_coverage.py   #   风险-覆盖曲线（级联调度评估）
  eval_calibration.py / eval_exp5.py / eval_sweep.py
  run_all.sh / run_all_vllm.sh / smoke_run.sh
docs/                    # 实验小结、RLCD 方案文档、TRL GRPO 源码学习指南
scripts/                 # 早期 VERL / LoRA 训练与绘图脚本
data/                    # GSM8K 小样本
experiment_configs/      # 每个实验的 config.json / training_args.bin 等元数据
logs.zip                 # 各次训练的完整日志（trl_experiments/logs/*.log）
legacy_logs.zip          # 早期 VERL / LoRA 训练日志
environment.txt          # 复现所需的关键包版本
```

## 5. 如何复现

1. 准备环境（见 `environment.txt`）：Python 3.12 + PyTorch 2.10 + TRL 1.10.0 +
   Transformers 5.8.1 + vLLM 0.19.1 + PEFT 0.20.0 + datasets 5.0.1 +
   accelerate 1.14.0 + wandb。
2. 下载基座模型 `Qwen/Qwen3-0.6B` 到 `models/Qwen3-0.6B`。
3. 修改脚本中的绝对路径：`rlcd/common.py` 的 `ROOT`、`MODEL_PATH`，
   以及 `rlcd/run_all.sh` 中的 `PY`（原环境路径）。
4. 单实验运行示例：

```bash
cd rlcd
python exp5_rlcd_calibrated.py        # 方案 A：Brier 进 loss（5 类）
python fpb_rlcd.py                    # FPB 3 类，1 epoch
python sft_fpb.py                     # FPB SFT 对照
python eval_fpb.py                    # test 评估：accuracy + ECE
python fpb_risk_coverage.py           # 风险-覆盖曲线
```

## 6. 未上传的内容

模型权重与训练 checkpoint（共约 7.9 GB）**未上传**：GitHub 限制单文件 100 MB，
而单个 `model.safetensors` 为 1.2 GB、优化器状态 `*.distcp` 更大。
这些文件对应 `experiment_configs/` 中各实验目录的配置，可据此重新训练。
基础模型（Qwen3-0.6B / Qwen3.5-0.8B）同样未上传，可从 Hugging Face 重新下载。

## 7. 参考

- 公众号文章《Qwen3-0.6B：从可验证奖励到 Reward Hacking 的 GRPO 三联实验》
  的复现（exp1–exp3 与消融实验）。
- RLCD 配方：在 GRPO 中引入 proper scoring rule（Brier）作为可微校准损失。
