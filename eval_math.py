"""Evaluate checkpoints with vLLM using the authors' prompt, decoding settings and grader (third_party/, MIT).

  python relex/eval_math.py --models <dir1>,<dir2>,... --bench math [--out <results dir>] [--limit N]

Benchmarks (as in the authors' run_*.sh): math = TianHongZXY/MATH test (5000), greedy, 4096 tokens, 1 sample;
ood = AIME25, AIME26, HMMT-Feb-25, AMC23, OlympiadBench, avg@8 at T=0.7, top_p 0.8, 4096 tokens.
Writes <out>/<model name>/<bench>.jsonl (one line per sample) and <out>/<model name>/<bench>.json (accuracy per dataset).
Skips a (model, bench) that already has its .json.
"""
import argparse, gc, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "third_party"))

PROMPT = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input}\nPlease reason step by step, "
          "and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n")
BENCH = {
    "math": [("TianHongZXY/MATH", "test")],
    "ood": [("TianHongZXY/AIME2025", "test"), ("MathArena/aime_2026", "train"), ("MathArena/hmmt_feb_2025", "train"),
            ("TianHongZXY/amc23", "test"), ("zwhe99/simplerl-OlympiadBench", "test")],
}
SAMPLING = {"math": dict(n=1, temperature=0.0, top_p=1.0, top_k=-1, max_tokens=4096),
            "ood": dict(n=8, temperature=0.7, top_p=0.8, top_k=-1, max_tokens=4096)}


def keys(name, cols):
    n = name.lower()
    if "olympiad" in n:
        return "question", "final_answer"
    if "aime" in n or "amc23" in n or "hmmt" in n:
        return ("problem" if "problem" in cols else "prompt"), "answer"
    return ("problem" if "problem" in cols else "prompt"), ("solution" if "solution" in cols else "answer")


def gold_of(name, a):
    from utils import extract_answer_math
    n = name.lower()
    if "olympiad" in n:
        return a[0] if isinstance(a, list) else a
    if "aime" in n or "amc23" in n or "hmmt" in n:
        return a
    return extract_answer_math(a)


def _score(pair):
    from utils import extract_answer_math
    from grader import math_equal
    resp, gold = pair
    try:
        return bool(math_equal(extract_answer_math(resp), str(gold), timeout=True))
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True); ap.add_argument("--bench", default="math", choices=list(BENCH))
    ap.add_argument("--out", default="$DATA/relex/evals"); ap.add_argument("--limit", type=int)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    a = ap.parse_args()
    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    data = {}
    for name, split in BENCH[a.bench]:
        ds = load_dataset(name, split=split)
        pk, ak = keys(name, ds.column_names)
        rows = [{"q": r[pk], "gold": gold_of(name, r[ak])} for r in ds]
        data[name] = rows[: a.limit] if a.limit else rows
    pool = ProcessPoolExecutor(int(os.environ.get("REWARD_PROCS", "12")))
    for m in [x for x in a.models.split(",") if x]:
        tag = Path(m).name if Path(m).exists() else m.replace("/", "__")
        od = Path(a.out) / tag; od.mkdir(parents=True, exist_ok=True)
        if (od / f"{a.bench}.json").exists():
            print("skip", tag, a.bench); continue
        t0 = time.time()
        llm = LLM(m, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem, max_model_len=4096, seed=0)
        sp = SamplingParams(seed=0, **SAMPLING[a.bench])
        res, lines = {}, []
        for name, rows in data.items():
            outs = llm.generate([PROMPT.replace("{input}", r["q"]) for r in rows], sp, use_tqdm=False)
            pairs = [(o.text, r["gold"]) for r, out in zip(rows, outs) for o in out.outputs]
            ok = list(pool.map(_score, pairs, chunksize=16))
            k = SAMPLING[a.bench]["n"]
            res[name] = round(100 * sum(ok) / len(ok), 2)
            fin = [o.finish_reason for out in outs for o in out.outputs]
            res[name + ":truncated%"] = round(100 * sum(f == "length" for f in fin) / len(fin), 2)
            res[name + ":mean_tokens"] = round(sum(len(o.token_ids) for out in outs for o in out.outputs) / len(fin), 1)
            for i, (r, out) in enumerate(zip(rows, outs)):
                for j, o in enumerate(out.outputs):
                    lines.append(json.dumps({"ds": name, "i": i, "j": j, "ok": ok[i * k + j], "gold": str(r["gold"]),
                                             "finish": o.finish_reason, "resp": o.text}))
        (od / f"{a.bench}.jsonl").write_text("\n".join(lines) + "\n")
        res["_seconds"] = round(time.time() - t0); res["_model"] = m
        (od / f"{a.bench}.json").write_text(json.dumps(res, indent=1))
        print(tag, a.bench, json.dumps(res), flush=True)
        del llm; gc.collect()
        import torch; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
