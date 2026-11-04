# relex-repro: a critical reproduction of RELEX (arXiv 2605.21468)

Independent, end-to-end reproduction of Wei et al., *"You Only Need Minimal RLVR Training: Extrapolating LLMs via
Rank-1 Trajectories"* (May 2026). RELEX runs a short GRPO prefix, takes the top singular direction of each weight
tensor's trajectory, fits its coefficient linearly in the step, and extrapolates to a later checkpoint.

All code here is my own, except the authors' answer grader (`third_party/`, MIT), which I reuse so scores are
comparable. `PLAN.md` holds the pre-registered predictions and a timestamped log of every result. Status:
**in progress**; the own GRPO run finishes on 2026-11-04.

## Findings so far

| # | Finding | Evidence |
|---|---|---|
| 1 | My implementation reproduces the authors' released RELEX checkpoint exactly. | weight cosine 1.0000, rel. L2 0.18% |
| 2 | Headline MATH numbers reproduce on the released checkpoints. | Qwen2.5-Math-1.5B base / RLVR-500 / RELEX: 46.9 / 71.8 / 71.7 (paper 48.2 / 71.5 / 71.6); Qwen3-4B base / RELEX 64.8 / 84.8 (paper 64.0 / 85.6) |
| 3 | The 1.5B "RELEX step 500" is the line evaluated at checkpoint index **1000**, twice the RLVR run it is compared with. At the matched index 500 RELEX is 2.0 points below RLVR (95% CI [-3.0, -1.1]), and rank-10 and two-point baselines beat it. | authors' code scales the target by 2 for this model only; the full Table 3 row matches under label -> 2x index |
| 4 | The paper's weight-space evidence (rank-1 explained variance ~0.8, R^2 ~0.96) is what a driftless random walk gives, and what bf16 checkpoint rounding produces. | random-walk null 0.81 / 0.96; released bf16 trajectory 0.82 / 0.96 |
| 5 | In fp32 the trajectory really is near rank-1 (0.96-0.98), but its coefficient is curved, not linear (R^2 0.92 over 60 steps). | own GRPO run, same run rounded to bf16 gives 0.82 / 0.96 |
| 6 | Extrapolating from fp32 deltas beats extrapolating from bf16 checkpoints by 2-3 MATH points. | +3.04 [+2.04, +4.02] at step 500 from a 50-step window |
| 7 | On the own math run RELEX tracks real training at moderate horizons (step 200: 69.64 vs real 69.78); rank-1 beats per-weight extrapolation only at long horizons in fp32. | window sweeps 50-125 |
| 8 | Agentic RL: GRPO lifts Qwen3.5-2B from 59.5% to 94.5% on unseen ERP databases (opencode harness), but extrapolating the run collapses the agent (11% at 4x the window; unsafe beyond ~1.2x); a power-law coefficient or a longer window does not fix it. | magnitude saturates: real delta grows 1.5x from step 25 to 100, linear extrapolation predicts 4x |

## Layout

| Path | What |
|---|---|
| `relex.py` | Gram-matrix formulation: every predictor is a weighted sum of observed deltas; RELEX (any rank, linear / poly / power fit), reconstruction, raw per-element least squares, two-point, ExPO; diagnostics (explained variance, R^2, cosine / norm vs real steps) |
| `build.py`, `store.py` | delta store (one safetensors file per step) and streaming checkpoint builder |
| `train_grpo.py` | GRPO on MATH with TRL 1.16 + colocated vLLM, the paper's verl recipe, fp32 delta dumps, length-sort + trim speed patch (gradient-checked against fp32), deadline-guarded grading |
| `agent/` | agentic ERP test: task generator, TRL `environment_factory` environment (6 SQLite tools), trainer, opencode-harness eval (uses `erpagent/make_db.py` and `erpagent/run_eval.py` + harness adapters from the companion repo ehzawad/tau2-agent-posttraining) |
| `eval_math.py`, `paired.py`, `table.py` | vLLM eval with the authors' prompt, decoding and grader; paired bootstrap; tables |
| `analyze_grams.py`, `compare_ckpt.py`, `tests/` | weight-space statistics, weight-level comparison, unit / null / gradient tests |
| `dl_released.py` | converts the authors' released trajectories into exact bf16 deltas |
| `watch_run.sh`, `gpu_janitor.sh`, `eval_worker.sh`, `on_holder.sh` | Slurm orchestration across 12-hour B200 leases |
| `results/` | score summaries, weight-space stats, training logs (synced by `sync_results.sh`) |

## Running

Environment: the `relex` venv layered on a torch 2.13 / vLLM 0.32 / TRL 1.16 / transformers 5.19 env, plus
`latex2sympy2` for the grader. Large artefacts (deltas, checkpoints, generations) live outside the repo.

```bash
python dl_released.py --repo relex-rlvr/RLVR-Qwen2.5-Math-1.5B --base Qwen/Qwen2.5-Math-1.5B --steps 1-125 --out deltas/ --tmp tmp/
python relex.py gram --store deltas/ --out grams.pt
python build.py --grams grams.pt --spec spec.json --outdir ckpts/      # spec: {"name": {"method": "relex", "obs": [1..75], "target": 500}}
python eval_math.py --bench math --models ckpts/name
python train_grpo.py --out runs/q15                                     # own trajectory (fp32 deltas every step to 125)
```
