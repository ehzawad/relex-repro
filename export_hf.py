"""Export eval generations and ERP agent traces as public Hugging Face datasets (user request 2026-11-03).

  python relex/export_hf.py --out $SCRATCH/.../relex/hf_export [--push]

  ehzawad/relex-repro-math-generations   one row per sampled answer: checkpoint (trajectory, method, window, target,
                                         precision), benchmark, problem index, sample, response, gold, correct, finish.
                                         Problem texts are NOT included (MathArena sets are CC BY-NC-SA); join on
                                         (dataset, problem_index) with the source split.
  ehzawad/relex-repro-erp-agent-traces   one row per opencode episode: checkpoint, task, question, trial, grade, answer,
                                         tool calls, token counts, and the full final conversation; plus the generated
                                         task files and the six synthetic SQLite databases.
Paths and the cluster username are scrubbed from every text field.
"""
import os
import argparse, json, re, shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

D = Path(os.path.expanduser("~/ehz_agent/store/relex"))
SCRUB = [(re.compile(r"~/ehz_agent/store/relex/agent/dbs/"), "dbs/"),
         (re.compile(r"/(?:scratch|sfs|home|tmp)/[^\s\"'\\]*"), "<path>"), (re.compile(r"user"), "<user>")]
TRAJ = {"q15_released": ("authors' released RLVR trajectory", "Qwen/Qwen2.5-Math-1.5B"),
        "q15_own": ("own GRPO run (TRL), fp32 deltas", "Qwen/Qwen2.5-Math-1.5B"),
        "q4b_released": ("authors' released RLVR trajectory", "Qwen/Qwen3-4B-Base")}


def scrub(s):
    if not isinstance(s, str):
        return s
    for p, r in SCRUB:
        s = p.sub(r, s)
    return s


def parse_ckpt(name):
    """method / window / target / precision from checkpoint names like own_relex_c75_t500_fp32, twopoint25_c75_t1000."""
    n = re.sub(r"^(own_|q4_)", "", name); out = {"method": None, "window": None, "target_index": None, "precision": None}
    if n == "base" or "/" in name or name.startswith(("Qwen__", "relex-rlvr__")):
        out["method"] = "released_relex" if "RELEX" in name else "base"; return out
    m = re.match(r"(relexpow|relex|rawls|rank5|rank10|poly3|twopoint0|twopoint25|twopoint|expo1|expo3|recon_u25|rlvr)_?(.*)", n)
    if m:
        out["method"] = {"relexpow": "relex_powerlaw", "rawls": "raw_least_squares", "recon_u25": "rank1_reconstruction"}.get(m.group(1), m.group(1))
        rest = m.group(2)
        if out["method"] == "rlvr":
            out["target_index"] = int(re.match(r"(\d+)", rest).group(1)); return out
        c = re.search(r"c(\d+)", rest); t = re.search(r"t(\d+)", rest); p = re.search(r"(fp32|bf16)", rest)
        out.update(window=int(c.group(1)) if c else None, target_index=int(t.group(1)) if t else None,
                   precision=p.group(1) if p else ("bf16" if name.startswith(("relex", "rawls", "rank", "poly", "two", "expo", "recon")) else None))
    return out


def math_generations(out):
    rows = []
    for traj, (desc, base) in TRAJ.items():
        for d in sorted((D / "evals" / traj).glob("*")):
            for f in sorted(d.glob("*.jsonl")):
                bench = f.stem; meta = parse_ckpt(d.name)
                for l in open(f):
                    if not l.strip():
                        continue
                    r = json.loads(l)
                    rows.append({"trajectory": traj, "trajectory_desc": desc, "base_model": base, "checkpoint": d.name, **meta,
                                 "bench": bench, "dataset": r["ds"], "problem_index": r["i"], "sample_index": r["j"],
                                 "response": scrub(r["resp"]), "gold_answer": scrub(str(r["gold"])), "correct": bool(r["ok"]),
                                 "finish_reason": r["finish"]})
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out / "generations.parquet", compression="zstd")
    (out / "training_logs").mkdir(exist_ok=True)
    shutil.copy(D / "own/q15_run1/train_log.jsonl", out / "training_logs/own_q15_grpo_per_step.jsonl")
    return len(rows)


def agent_traces(out):
    tasks = {}
    for f in (D / "agent").glob("tasks_*.jsonl"):
        for l in open(f):
            t = json.loads(l); tasks[t["id"]] = t
    rows = []
    for summ in sorted((D / "agent/evals").glob("opencode__*.jsonl")):
        tag = summ.stem; run_dir = D / "agent/evals" / tag
        ckpt = tag.replace("opencode__", ""); task_set = "orig37" if ckpt.endswith("_orig37") else ("test_quick" if ckpt.endswith("_quick") else "test")
        ckpt = re.sub(r"_(orig37|quick)$", "", ckpt)
        for l in open(summ):
            e = json.loads(l); t = tasks.get(e["id"], {})
            lj = run_dir / f"{e['id']}.t{e['trial']}.last.json"
            conv = json.loads(lj.read_text()) if lj.exists() else None
            rows.append({"checkpoint": ckpt, "base_model": "Qwen/Qwen3.5-2B", "harness": "opencode 1.19.5", "task_set": task_set,
                         "task_id": e["id"], "category": e.get("cat"), "question": scrub(t.get("question")),
                         "database": Path(t["db"]).name if t.get("db") else "logistics_seed7.db", "trial": e["trial"],
                         "passed": bool(e["pass"]), "end": e.get("end"), "answer": scrub(e.get("answer")),
                         "grade_detail": scrub(json.dumps(e.get("detail"))), "tool_calls": scrub(json.dumps(e.get("mcp_calls"))),
                         "llm_calls": e.get("llm_calls"), "prompt_tokens": e.get("prompt_tokens"),
                         "completion_tokens": e.get("completion_tokens"), "seconds": e.get("seconds"),
                         "final_conversation": scrub(json.dumps(conv)) if conv else None})
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out / "episodes.parquet", compression="zstd")
    (out / "tasks").mkdir(exist_ok=True); (out / "dbs").mkdir(exist_ok=True); (out / "training_logs").mkdir(exist_ok=True)
    for f in (D / "agent").glob("tasks_*.jsonl"):
        with open(out / "tasks" / f.name, "w") as g:
            for l in open(f):
                t = json.loads(l); t["db"] = Path(t["db"]).name if t.get("db") else t.get("db")
                if t.get("split") == "orig37":
                    t["db"] = "logistics_seed7.db"
                g.write(json.dumps(t) + "\n")
    for f in (D / "agent/dbs").glob("*.db"):
        shutil.copy(f, out / "dbs" / f.name)
    shutil.copy(os.path.expanduser("~/ehz_agent/store/erpagent/logistics.db"), out / "dbs/logistics_seed7.db")
    for r in ["run1", "run2"]:
        if (D / f"agent/{r}/train_log.jsonl").exists():
            shutil.copy(D / f"agent/{r}/train_log.jsonl", out / f"training_logs/agent_{r}_grpo_per_step.jsonl")
    return len(rows)


MATH_CARD = """---
license: cc-by-4.0
task_categories: [text-generation]
language: [en]
tags: [math, reasoning, rlvr, grpo, model-extrapolation, relex, reproduction]
size_categories: [100K<n<1M]
configs:
- config_name: default
  data_files: generations.parquet
---
# RELEX reproduction: math eval generations

Every sampled answer from the evaluations in an independent reproduction of **RELEX** (arXiv 2605.21468, *You Only
Need Minimal RLVR Training: Extrapolating LLMs via Rank-1 Trajectories*). Code, pre-registration and findings:
https://github.com/ehzawad/relex-repro (private for now; ask the author).

One row per (checkpoint, benchmark, problem, sample). Checkpoints come from three weight trajectories:
`q15_released` (the authors' released Qwen2.5-Math-1.5B GRPO trajectory), `q15_own` (my own GRPO run with fp32
deltas), `q4b_released` (the authors' Qwen3-4B-Base trajectory). `method` is the predictor (`rlvr` = a real training
checkpoint, `relex` = rank-1 + linear extrapolation, `raw_least_squares`, `rank5`, `twopoint25`, `relex_powerlaw`, ...),
`window` the observed steps 1..window, `target_index` the checkpoint index the predictor targets, `precision` whether the
deltas were fp32 or bf16-rounded.

Benchmarks: `math` = MATH test (5000, greedy, 4096 tokens), `ood` = AIME 2025, AIME 2026, HMMT Feb 2025, AMC 2023,
OlympiadBench (avg@8, T=0.7, top_p 0.8, 4096 tokens), `math16k`/`ood16k` = the authors' Qwen3 settings (sampled, 16K).
Grading uses the authors' grader (MIT). **Problem texts are not included**; join on (`dataset`, `problem_index`) with
the source split (`TianHongZXY/MATH` test, `TianHongZXY/AIME2025` test, `MathArena/aime_2026` train,
`MathArena/hmmt_feb_2025` train, `TianHongZXY/amc23` test, `zwhe99/simplerl-OlympiadBench` test), which keep their own
licences (MathArena sets are CC BY-NC-SA 4.0). Responses were generated by Apache-2.0 Qwen models.
`training_logs/` holds the per-step GRPO statistics of the own run.
"""

AGENT_CARD = """---
license: cc-by-4.0
task_categories: [text-generation]
language: [en]
tags: [agents, tool-use, mcp, sql, grpo, rlvr, opencode, model-extrapolation]
size_categories: [1K<n<10K]
configs:
- config_name: default
  data_files: episodes.parquet
---
# RELEX reproduction: ERP agent traces

Agent episodes from the agentic part of an independent reproduction of **RELEX** (arXiv 2605.21468): Qwen3.5-2B
trained with multi-turn tool-use GRPO (TRL) on a synthetic ERP/logistics database, then evaluated through the
**opencode** agent harness with six MCP tools (list_tables, describe_table, read-only run_sql, update_order_status,
adjust_inventory, set_reorder_point). Code and findings: https://github.com/ehzawad/relex-repro.

One row per episode: `checkpoint` (base, real GRPO steps, RELEX / two-point extrapolations of the run), task id,
category (lookup, aggregate, insight, write, guard), question, trial, `passed`, final answer, grade detail, the tool-call
log, token counts and the full final conversation the harness sent to the model (system prompt, messages, tool calls,
tool results). `task_set`: `test` (300 generated tasks on an unseen database, 2 trials), `orig37` (37 hand-written tasks,
3 trials), `test_quick` (1 trial). `tasks/` has the generated task files with gold SQL or state checks; `dbs/` the six
synthetic SQLite databases (seeds 101-104 train, 201 dev, 301 test) and the original one; `training_logs/` per-step
GRPO statistics. All data is synthetic (no real companies or people).
"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(D / "hf_export")); ap.add_argument("--push", action="store_true")
    a = ap.parse_args(); out = Path(a.out)
    n1 = math_generations(out / "math"); (out / "math/README.md").write_text(MATH_CARD)
    n2 = agent_traces(out / "agent"); (out / "agent/README.md").write_text(AGENT_CARD)
    print("math rows", n1, "agent episodes", n2, flush=True)
    for p in out.rglob("*"):                                   # last check: nothing with the cluster username or scratch paths
        if p.suffix in (".md", ".jsonl"):
            assert "user" not in p.read_text(errors="ignore") or p.parent.name == "training_logs", p
    if a.push:
        from huggingface_hub import HfApi
        api = HfApi()
        for sub, repo in [("math", "ehzawad/relex-repro-math-generations"), ("agent", "ehzawad/relex-repro-erp-agent-traces")]:
            api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
            api.upload_folder(folder_path=str(out / sub), repo_id=repo, repo_type="dataset",
                              commit_message="relex-repro traces (update)")
            print("pushed", repo, flush=True)


if __name__ == "__main__":
    main()
