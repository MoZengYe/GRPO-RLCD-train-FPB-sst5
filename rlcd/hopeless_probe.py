# -*- coding: utf-8 -*-
"""探针（HF 批量生成版）：基线模型在 MBPP 前 150 题上各采样 8 次，
统计"零梯度死区"的实际大小——8 次采样全失败的题占比。
（vLLM 离线模式在此环境会挂死，改用 transformers 批量生成。）"""
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/home/jpzhao_team/zymo/grpo_qwen3.5_0.8b/trl_experiments/scripts")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import MODEL_PATH, _score_one, extract_code

N_PROBLEMS = 150
K = 8
BATCH = 128

tok = AutoTokenizer.from_pretrained(str(MODEL_PATH))
ds = load_dataset("google-research-datasets/mbpp", "full", split="train").select(range(N_PROBLEMS))

prompts, tests, ids = [], [], []
for e in ds:
    content = (
        "{text}\n\n"
        "Your code should pass these tests:\n```python\n{tests}\n```\n"
        "Write only the Python function, inside a ```python code block."
    ).format(text=e["text"], tests="\n".join(e["test_list"]))
    p = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    prompts.append(p)
    tests.append(e["test_list"])
    ids.append(e["task_id"])

model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), dtype=torch.bfloat16, device_map="cuda")
model.eval()

# 每个 prompt 复制 K 份，批量生成
all_prompts = [p for p in prompts for _ in range(K)]
all_codes = []
for i in range(0, len(all_prompts), BATCH):
    batch = all_prompts[i : i + BATCH]
    enc = tok(batch, return_tensors="pt", padding=True, padding_side="left").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=512, temperature=0.7, top_p=1.0, do_sample=True,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    gen = out[:, enc["input_ids"].shape[1]:]
    all_codes.extend(tok.batch_decode(gen, skip_special_tokens=True))
    print(f"generated {min(i + BATCH, len(all_prompts))}/{len(all_prompts)}", flush=True)

rows = []
for i in range(N_PROBLEMS):
    codes = [extract_code(c, "last_block") for c in all_codes[i * K : (i + 1) * K]]
    with ThreadPoolExecutor(max_workers=8) as pool:
        rs = list(pool.map(lambda t: _score_one(t[0], t[1]), zip(codes, [tests[i]] * K)))
    rows.append({"task_id": ids[i], "mean": sum(rs) / len(rs),
                 "max": max(rs), "n_zero": sum(1 for r in rs if r == 0)})

hopeless = [r for r in rows if r["max"] == 0]
reachable = [r for r in rows if 0 < r["max"] < 1]
solved_some = [r for r in rows if r["max"] == 1]
means = [r["mean"] for r in rows]

print(f"题数: {len(rows)}")
print(f"死区（8次全失败，无梯度）: {len(hopeless)} 题 = {len(hopeless)/len(rows)*100:.1f}%")
print(f"学习区（有对有错）: {len(reachable)} 题 = {len(reachable)/len(rows)*100:.1f}%")
print(f"至少一次全对: {len(solved_some)} 题 = {len(solved_some)/len(rows)*100:.1f}%")
print(f"基线期望 reward（K样本均值再平均）: {statistics.mean(means):.3f}")
print("死区题号前 20:", [r["task_id"] for r in hopeless[:20]])
