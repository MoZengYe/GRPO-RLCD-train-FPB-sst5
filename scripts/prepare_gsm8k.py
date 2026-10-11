"""Prepare GSM8K into verl GRPO parquet format (subset for smoke run)."""
import argparse
import re
from pathlib import Path

import datasets

def extract_solution(solution_str: str) -> str:
    m = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    assert m is not None, f"no #### answer in: {solution_str[:100]}"
    return m.group(1).replace(",", "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset", default="openai/gsm8k")
    ap.add_argument("--train_samples", type=int, default=512)
    ap.add_argument("--test_samples", type=int, default=128)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    instruction = 'Let\'s think step by step and output the final answer after "####".'
    ds = datasets.load_dataset(args.dataset, "main")
    def process(split, n):
        rows = []
        for idx, ex in enumerate(ds[split]):
            if idx >= n:
                break
            rows.append({
                "data_source": args.dataset,
                "prompt": [{"role": "user", "content": ex["question"] + " " + instruction}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": extract_solution(ex["answer"])},
                "extra_info": {"split": split, "index": idx},
            })
        return datasets.Dataset.from_list(rows)
    process("train", args.train_samples).to_parquet(str(out / "train.parquet"))
    process("test", args.test_samples).to_parquet(str(out / "test.parquet"))
    print(f"OK: {out/'train.parquet'} (train) {out/'test.parquet'} (test)")

if __name__ == "__main__":
    main()
