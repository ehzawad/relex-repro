"""Reward/eval grading worker, kept free of torch/TRL imports.

train_grpo.scored() runs _score in a forkserver pool that preloads only this module. Forking the trainer (fp32 model,
CUDA, colocated vLLM) instead made every worker inherit ~7.2 GiB of commit-accountable memory (measured 2026-11-04:
24 workers ~173 GiB), which on strict-overcommit nodes (vm.overcommit_memory=2) caused MemoryError/ENOMEM crashes.
"""
import multiprocessing
import sys
from pathlib import Path

# The authors' grader runs its symbolic check in a child multiprocessing.Process with a 1 s deadline. In a
# forkserver/spawn worker the default start method would start that child fresh (re-importing sympy, > 1 s), so the check
# times out and correct answers score 0 (measured: 12 of 2000 MATH answers). Forking the small worker keeps it fast.
# It must be set at call time: a pool worker's startup ("prepare") re-applies the parent's start method after import.

HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(HERE / "third_party")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _score(pair):
    if multiprocessing.get_start_method(allow_none=True) != "fork":
        multiprocessing.set_start_method("fork", force=True)
    resp, gold = pair
    from utils import extract_answer_math
    from grader import math_equal
    try:
        return float(bool(math_equal(extract_answer_math(resp), gold, timeout=True)))
    except Exception:
        return 0.0
