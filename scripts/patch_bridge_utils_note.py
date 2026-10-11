"""Patch megatron-bridge peft utils for environments without a real Transformer Engine.

The mcore fallback provides MagicMock placeholders for TE classes, so the
`HAVE_TE` flag becomes True even though the objects are not real types; later
`isinstance(x, <MagicMock>)` then crashes. This patch makes HAVE_TE only true
when the imported objects really are classes, and filters the class tuples.
"""

import shutil
from pathlib import Path

path = Path(
    "/home/jpzhao_team/zymo/verl_env/lib/python3.12/site-packages"
    "/megatron/bridge/peft/utils.py"
)
backup = path.with_suffix(".py.orig_zymo_noTE")

src = path.read_text()
if not backup.exists():
    shutil.copy2(path, backup)

old_have_te = """HAVE_TE = all(
    (
        HAVE_TE_COL_LINEAR,
        HAVE_TE_LN_COL_LINEAR,
        HAVE_TE_ROW_LINEAR,
        HAVE_TE_LINEAR,
        HAVE_TE_COL_GRP_LINEAR,
        HAVE_TE_ROW_GRP_LINEAR,
    )
)
"""
new_have_te = """HAVE_TE = all(
    (
        HAVE_TE_COL_LINEAR,
        HAVE_TE_LN_COL_LINEAR,
        HAVE_TE_ROW_LINEAR,
        HAVE_TE_LINEAR,
        HAVE_TE_COL_GRP_LINEAR,
        HAVE_TE_ROW_GRP_LINEAR,
    )
) and all(
    isinstance(_te_cls, type)
    for _te_cls in (
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
        TELinear,
        TEColumnParallelGroupedLinear,
        TERowParallelGroupedLinear,
    )
)
"""

old_tecl = (
    "TECL = (TEColumnParallelLinear, TELayerNormColumnParallelLinear, TEColumnParallelGroupedLinear)\n"
    "TERL = (TERowParallelLinear, TERowParallelGroupedLinear)\n"
)
new_tecl = (
    "TECL = tuple(\n"
    "    _c\n"
    "    for _c in (TEColumnParallelLinear, TELayerNormColumnParallelLinear, TEColumnParallelGroupedLinear)\n"
    "    if isinstance(_c, type)\n"
    ")\n"
    "TERL = tuple(_c for _c in (TERowParallelLinear, TERowParallelGroupedLinear) if isinstance(_c, type))\n"
)

old_te_linear = "    elif HAVE_TE and isinstance(m, TELinear):  # parallel_mode=\"duplicated\"\n"
new_te_linear = (
    "    elif (\n"
    "        HAVE_TE and isinstance(TELinear, type) and isinstance(m, TELinear)\n"
    "    ):  # parallel_mode=\"duplicated\"\n"
)

for old, new in ((old_have_te, new_have_te), (old_tecl, new_tecl), (old_te_linear, new_te_linear)):
    assert src.count(old) == 1, f"pattern count={src.count(old)}: {old[:50]!r}"
    src = src.replace(old, new)

path.write_text(src)
print("patched:", path)
print("backup :", backup)
