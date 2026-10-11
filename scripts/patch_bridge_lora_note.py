"""Patch megatron-bridge LoRA for environments without Transformer Engine.

Run ON THE SERVER (the target file lives in ~/zymo/verl_env/...):
  python patch_bridge_lora_note.py

Changes (with backup lora.py.orig_zymo_noTE):
1. filter None out of adapter_types (TELinearAdapter is None without TE)
2. guard two `te.Linear` comparisons with `te is not None`
"""

import shutil
from pathlib import Path

path = Path(
    "/home/jpzhao_team/zymo/verl_env/lib/python3.12/site-packages"
    "/megatron/bridge/peft/lora.py"
)
backup = path.with_suffix(".py.orig_zymo_noTE")

src = path.read_text()
if not backup.exists():
    shutil.copy2(path, backup)

repls = [
    (
        "        adapter_types = (LinearAdapter, LoRALinear, LoRATopKRouter, TELinearAdapter)\n",
        "        adapter_types = tuple(\n"
        "            t for t in (LinearAdapter, LoRALinear, LoRATopKRouter, TELinearAdapter) if isinstance(t, type)\n"
        "        )\n",
    ),
    (
        "            if (isinstance(module, nn.Linear) or (module.__class__ == te.Linear)) and not is_modelopt_linear(module):\n",
        "            if (\n"
        "                isinstance(module, nn.Linear) or (te is not None and module.__class__ == te.Linear)\n"
        "            ) and not is_modelopt_linear(module):\n",
    ),
    (
        "                elif module.__class__ == te.Linear:\n",
        "                elif te is not None and module.__class__ == te.Linear:\n",
    ),
]

for old, new in repls:
    assert src.count(old) == 1, f"pattern not unique: {old[:60]!r} count={src.count(old)}"
    src = src.replace(old, new)

path.write_text(src)
print("patched:", path)
print("backup :", backup)
