# -*- coding: utf-8 -*-
"""诊断 SFT checkpoint：首 token 数字分布 + 贪心生成内容。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch  # noqa: E402
from common import MODEL_PATH  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from fpb_rlcd import PROMPT_TMPL, digit_token_ids3  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/jpzhao_team/zymo/grpo_qwen3.5_0.8b/trl_experiments/outputs/SFT-FPB-E2"

tok = AutoTokenizer.from_pretrained(str(MODEL_PATH))
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.bfloat16, device_map="cuda")
model.eval()

from datasets import load_dataset
ds = load_dataset("FinanceInc/auditor_sentiment", split="test")
digit_ids = torch.tensor(digit_token_ids3(tok)).cuda()
names = ["neg", "neu", "pos"]

for i in [0, 1, 2, 5, 8]:
    ex = ds[i]
    content = PROMPT_TMPL.format(sentence=ex["sentence"])
    prompt = tok.apply_chat_template([{"role": "user", "content": content}],
                                     tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    enc = tok(prompt, return_tensors="pt", padding=True, padding_side="left").to("cuda")
    with torch.no_grad():
        logits = model(**enc).logits[:, -1, :].float()
        d_logits = logits[:, digit_ids]
        p6 = torch.softmax(d_logits, dim=-1)[0]
        p3 = p6.view(2, 3).sum(dim=0)
        gen = model.generate(**enc, max_new_tokens=12, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    gen_text = tok.decode(gen[0][enc.input_ids.shape[1]:], skip_special_tokens=False)
    print(f"[{i}] gold={ex['label']} | 6变体分布={[round(x,3) for x in p6.tolist()]}")
    print(f"    3类聚合={[round(x,3) for x in p3.tolist()]} argmax={p3.argmax().item()}")
    print(f"    贪心生成: {repr(gen_text[:60])}")
