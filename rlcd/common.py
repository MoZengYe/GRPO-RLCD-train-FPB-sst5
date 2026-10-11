# -*- coding: utf-8 -*-
"""公共模块：路径、wandb 环境、MBPP 提示词构建、代码执行 harness。

复现公众号文章《Qwen3-0.6B：从可验证奖励到 Reward Hacking 的 GRPO 三联实验》。
所有读写均限于 /home/jpzhao_team/zymo 内。
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/home/jpzhao_team/zymo/grpo_qwen3.5_0.8b")
MODEL_PATH = ROOT / "models" / "Qwen3-0.6B"
EXP_DIR = ROOT / "trl_experiments"
SANDBOX_DIR = EXP_DIR / "outputs" / "sandbox"
MY_PYTHON = Path("/home/jpzhao_team/zymo/my_env/bin/python")

MBPP_PROMPT_TMPL = (
    "You are an expert Python programmer, and here is your task: {text} "
    "Your code should pass these tests:\n\n{tests}\n"
)

# 作者（Sergio Paniego）的模板：无专家前缀、测试包进代码围栏、显式要求"只写函数"。
# 关键差异是最后一句指令——它把模型逼进"只输出一个代码块"的行为模式，
# 消除散文/无代码回答，抬高起步通过率与每步信号密度。
MBPP_PROMPT_TMPL_AUTHOR = (
    "{text}\n\n"
    "Your code should pass these tests:\n```python\n{tests}\n```\n"
    "Write only the Python function, inside a ```python code block."
)

# 在隔离子进程里执行：先跑模型代码，再逐条 assert 记分，输出 JSON。
# 每条测试的状态三分类：pass（通过）/ assert（断言失败=答案错）/ crash（运行崩溃）
_RUNNER_SRC = r'''
import json, sys
code_path, tests_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(code_path) as f:
    code = f.read()
with open(tests_path) as f:
    tests = [l.strip() for l in f if l.strip()]
ns = {}
exec_err = None
try:
    exec(compile(code, "solution.py", "exec"), ns)
except SyntaxError:
    exec_err = "syntax"
except BaseException:
    exec_err = "runtime"
results = []
if exec_err is None:
    for t in tests:
        try:
            exec(compile(t, "test", "exec"), ns)
            results.append("pass")
        except AssertionError:
            results.append("assert")
        except BaseException:
            results.append("crash")
else:
    results = ["crash"] * len(tests)
with open(out_path, "w") as f:
    json.dump({"exec_err": exec_err, "results": results}, f)
'''


def setup_wandb_env(run_name: str, project: str = "grpo-trl-replica") -> None:
    """wandb 走网页版账号；key 与缓存全部留在 zymo 内。"""
    key_file = Path.home() / "zymo" / ".wandb_key"
    if key_file.exists() and not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = key_file.read_text().strip()
    os.environ["WANDB_PROJECT"] = project
    os.environ["WANDB_NAME"] = run_name
    wandb_dir = EXP_DIR / "logs" / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_dir)
    os.environ["WANDB_CACHE_DIR"] = str(wandb_dir / "cache")
    os.environ["WANDB_CONFIG_DIR"] = str(wandb_dir / "config")
    os.environ["WANDB_DATA_DIR"] = str(wandb_dir / "data")
    os.environ["HF_HOME"] = str(Path.home() / "zymo" / ".hf_cache")


def extract_code(text: str, strategy: str = "join_all") -> str:
    """从 completion 提取 Python 代码。

    join_all：拼接所有代码块（我们最初的策略）；
    last_block：只取最后一个代码块（作者策略，多块时避免拼接出语法错误）。
    """
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if blocks:
        return blocks[-1] if strategy == "last_block" else "\n".join(blocks)
    return text


_RUNNER_LOCK = threading.Lock()
_RUNNER_READY = False


def _ensure_sandbox() -> Path:
    global _RUNNER_READY
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    runner = SANDBOX_DIR / "runner.py"
    with _RUNNER_LOCK:
        if not _RUNNER_READY:
            runner.write_text(_RUNNER_SRC)  # 每进程只写一次（加锁防并发读写竞态）
            _RUNNER_READY = True
    return SANDBOX_DIR


def _run_and_read(code: str, test_list: list, timeout: float = 8.0):
    """在沙箱子进程里执行 code+tests，返回 runner 的 JSON 结果；失败返回 None。"""
    sandbox = _ensure_sandbox()
    if not test_list:
        return None
    workdir = Path(tempfile.mkdtemp(prefix="case_", dir=sandbox))
    try:
        sol, tests, out = workdir / "solution.py", workdir / "tests.txt", workdir / "out.json"
        sol.write_text(code)
        tests.write_text("\n".join(test_list))
        # 最小环境：不泄漏 HF_TOKEN / WANDB_API_KEY 等变量
        env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(workdir),
               "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"}
        try:
            subprocess.run(
                [str(MY_PYTHON), str(sandbox / "runner.py"), str(sol), str(tests), str(out)],
                cwd=str(workdir), env=env, timeout=timeout,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
        except subprocess.TimeoutExpired:
            return None
        if not out.exists():
            return None
        return json.loads(out.read_text())
    finally:
        for p in workdir.iterdir():
            p.unlink(missing_ok=True)
        workdir.rmdir()


def _score_one(code: str, test_list: list, timeout: float = 8.0) -> float:
    """现行口径：通过 assert 的比例（0, 1/3, 2/3, 1）。"""
    data = _run_and_read(code, test_list, timeout)
    if data is None or not test_list:
        return 0.0
    return sum(1 for s in data.get("results", []) if s == "pass") / len(test_list)


def _score_one_staged(code: str, test_list: list, timeout: float = 8.0):
    """阶梯口径（方案1）：区分 解析失败/运行崩溃/跑通但错/部分通过/全对。

    返回 (阶梯分, 真实通过率)。阶梯分:
        解析失败          → 0.0
        全部测试运行崩溃   → 0.1（能解析但运行就崩）
        跑通但全错        → 0.2
        部分通过          → P/n（现行口径）
        全部通过          → 1.0
    """
    data = _run_and_read(code, test_list, timeout)
    if data is None or not test_list:
        return 0.0, 0.0
    n = len(test_list)
    results = data.get("results", [])
    P = sum(1 for s in results if s == "pass")
    C = sum(1 for s in results if s == "crash")
    pass_frac = P / n
    if data.get("exec_err") == "syntax":
        return 0.0, pass_frac
    if P == n:
        return 1.0, pass_frac
    if P == 0:
        return (0.1 if C == n else 0.2), pass_frac
    return P / n, pass_frac


def tests_reward_fn(prompts, completions, test_list=None, extract="join_all", **kwargs):
    """可验证奖励：通过 assert 的比例。逐条独立判分，互不影响。"""
    if test_list is None:
        return [0.0] * len(completions)
    codes = [extract_code(c, extract) for c in completions]
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(lambda t: _score_one(t[0], t[1]), zip(codes, test_list)))


def make_tests_reward_fn(extract: str = "join_all", log_file: str = None, staged: bool = False):
    """生成带提取策略的奖励函数；__name__ 决定 wandb 里的指标名。

    staged=False：现行口径（通过比例）。
    staged=True：阶梯口径（方案1），返回阶梯分；CSV 同时记录 (阶梯分, 真实通过率)
    两列——阶梯分与旧口径分数不可直接比较，跨 run 对比一律用真实通过率列。
    """
    import os

    def _fn(prompts, completions, test_list=None, task_id=None, **kwargs):
        if test_list is None:
            rewards, fracs = [0.0] * len(completions), [0.0] * len(completions)
        else:
            codes = [extract_code(c, extract) for c in completions]
            with ThreadPoolExecutor(max_workers=8) as pool:
                if staged:
                    pairs = list(pool.map(lambda t: _score_one_staged(t[0], t[1]),
                                          zip(codes, test_list)))
                    rewards = [p[0] for p in pairs]
                    fracs = [p[1] for p in pairs]
                else:
                    rewards = list(pool.map(lambda t: _score_one(t[0], t[1]),
                                            zip(codes, test_list)))
                    fracs = rewards
        if log_file and task_id:
            with open(log_file, "a") as f:
                for tid, r, pf in zip(task_id, rewards, fracs):
                    f.write(f"{tid},{r:.4f},{pf:.4f}\n")
        return rewards
    if staged:
        _fn.__name__ = "tests_staged_fn"
    else:
        _fn.__name__ = "tests_reward_fn" if extract == "join_all" else "tests_reward_fn_lastblock"
    return _fn


def code_block_reward_fn(prompts, completions, **kwargs):
    """作者原版 gameable_reward（train_grpo_pathological.py 原样移植）。

    数 ```python 围栏代码块数，min(块数, 4) / 4 归一化——四级跑道
    （0.25/0.5/0.75/1.0）是文章三现象（hacking/length gaming/能力坍塌）的来源。
    此前的 min(块数,1)（把"上限1"误读为最多一块，一步饱和）与无上限版本
    （两级平衡点后冻结）均为对原文公式的误译。
    """
    rewards = []
    for c in completions:
        blocks = re.findall(r"```(?:python)?\n.*?```", c, re.DOTALL)
        rewards.append(min(len(blocks), 4) / 4)
    return rewards


def build_mbpp_dataset(tokenizer, split: str = "train", template: str = None):
    """MBPP full 配置 train=374 题，与文章一致。prompt 预先套 chat template（关闭思考模式）。

    template=None 用我们的模板；传 MBPP_PROMPT_TMPL_AUTHOR 用作者的模板。
    """
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "full", split=split)

    def _fmt(example):
        tests = "\n".join(example["test_list"])
        content = (template or MBPP_PROMPT_TMPL).format(text=example["text"], tests=tests)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        return {"prompt": prompt, "test_list": example["test_list"], "task_id": example["task_id"]}

    ds = ds.map(_fmt, remove_columns=[c for c in ds.column_names if c not in ("test_list", "task_id")])
    return ds


def build_prompts_dataset_with_template(raw_prompts: list, tokenizer):
    from datasets import Dataset

    def _fmt(example):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        return {"prompt": prompt}

    ds = Dataset.from_dict({"prompt": raw_prompts})
    return ds.map(_fmt, remove_columns=["prompt"])
