"""打印 LoRA adapter 里的 A/B 矩阵（名称、形状、参数量）。

用法：
    python scripts/inspect_lora_adapter.py [adapter目录]
默认读取最近一次 LoRA 训练的 checkpoint。
"""

import json
import os
import sys

from safetensors import safe_open

DEFAULT = (
    "/home/jpzhao_team/zymo/grpo_qwen3.5_0.8b/checkpoints"
    "/lora_gsm8k_qwen3_0_6b/global_step_8/actor/model/huggingface/adapter"
)


def shape_numel(shape):
    n = 1
    for x in shape:
        n *= x
    return n


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    cfg_path = os.path.join(path, "adapter_config.json")
    cfg = json.load(open(cfg_path))
    print("adapter dir :", path)
    print("rank(r)     :", cfg["r"])
    print("alpha       :", cfg["lora_alpha"])
    print("dropout     :", cfg["lora_dropout"])
    print("target_modules(HF 命名):", cfg["target_modules"])

    st_path = os.path.join(path, "adapter_model.safetensors")
    f = safe_open(st_path, framework="pt")
    keys = list(f.keys())
    print("tensor 数量 :", len(keys))

    pairs = []
    for k in keys:
        shape = tuple(f.get_slice(k).get_shape())
        if ".lora_A." in k:
            pairs.append(("A", k, shape))
        elif ".lora_B." in k:
            pairs.append(("B", k, shape))

    print("\n前 4 组 A/B（注意形状：A=[rank, in]，B=[out, rank]）：")
    for tag, k, shape in pairs[:8]:
        print(f"  [{tag}] {k}  {shape}")

    layers = sorted({int(k.split(".layers.")[1].split(".")[0]) for _, k, _ in pairs})
    modules = sorted({k.rsplit(".lora_", 1)[0].split(".")[-1] for _, k, _ in pairs})
    print(f"\n覆盖层数   : {len(layers)} (layers {layers[0]}..{layers[-1]})")
    print("覆盖模块   :", modules)

    total = sum(shape_numel(shape) for _, _, shape in pairs)
    print(f"A/B 张量数 : {len(pairs)}  (每个目标层 2 个：A 和 B)")
    print(f"可训练参数 : {total:,}")
    file_mb = os.path.getsize(st_path) / 1024**2
    print(f"adapter 文件: {file_mb:.1f} MB (safetensors)")


if __name__ == "__main__":
    main()
