# -*- coding: utf-8 -*-
"""诊断：chat 模板 vs 裸文本 prompt 的 MBPP 基线得分对比（各 8 题）。"""
import sys
sys.path.insert(0, "/home/jpzhao_team/zymo/grpo_qwen3.5_0.8b/trl_experiments/scripts")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import MBPP_PROMPT_TMPL, MODEL_PATH, tests_reward_fn

tok = AutoTokenizer.from_pretrained(str(MODEL_PATH))
ds = load_dataset("google-research-datasets/mbpp", "full", split="train")
model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), dtype=torch.bfloat16, device_map="cuda")
model.eval()

N = 8
texts = [MBPP_PROMPT_TMPL.format(text=ds[i]["text"], tests="\n".join(ds[i]["test_list"])) for i in range(N)]
tests = [ds[i]["test_list"] for i in range(N)]


def gen(prompts):
    outs = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = model.generate(
                **ids, max_new_tokens=400, temperature=0.7, top_p=0.95,
                do_sample=True, pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        outs.append(tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
    return outs


chat_prompts = [
    tok.apply_chat_template([{"role": "user", "content": t}],
                            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    for t in texts
]
r_chat = tests_reward_fn(None, gen(chat_prompts), test_list=tests)
r_raw = tests_reward_fn(None, gen(texts), test_list=tests)
print("chat_rewards:", [round(x, 2) for x in r_chat], "mean", round(sum(r_chat) / N, 3))
print("raw_rewards :", [round(x, 2) for x in r_raw], "mean", round(sum(r_raw) / N, 3))
