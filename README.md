# relex-repro: a critical reproduction of RELEX (arXiv 2605.21468)

Independent, end-to-end reproduction of Wei et al., *"You Only Need Minimal RLVR Training: Extrapolating LLMs via
Rank-1 Trajectories"* (May 2026). RELEX runs a short GRPO prefix, takes the top singular direction of each weight
tensor's trajectory, fits its coefficient linearly in the step, and extrapolates to a later checkpoint.

All code here is my own, except the authors' answer grader (`third_party/`, MIT), which I reuse so scores are
comparable. `PLAN.md` holds the pre-registered predictions and a timestamped log of every result. Status: the own
GRPO run (seed 0, 500 steps) is complete and evaluated; a second own math seed is running. Intervals are problem-paired bootstrap (OOD: problem-clustered, stratified by benchmark) and unadjusted for the many exploratory comparisons; pre-registered predictions are scored in PLAN.md. Reviewed by the Codex Council 2026-11-04 (wording corrected per its audit). The findings report is `report/relex_findings.pdf` (built by `report/build_report.py`).

## Findings so far

| # | Finding | Evidence |
|---|---|---|
| 1 | My implementation closely matches the authors' released RELEX checkpoints (1.5B and 4B). | 1.5B: per-tensor cosine > 0.99, rel. L2 0.18%; 4B: rel. L2 0.075% |
| 2 | Headline MATH numbers reproduce on the released checkpoints. | Qwen2.5-Math-1.5B base / RLVR-500 / RELEX: 46.9 / 71.8 / 71.7 (paper 48.2 / 71.5 / 71.6); Qwen3-4B base / RELEX 64.8 / 84.8 (paper 64.0 / 85.6) |
| 3 | The 1.5B "RELEX step 500" is the line evaluated at fit coordinate **1000**, twice the released branch-label coordinate of the RLVR checkpoint it is compared with (the authors' code scales the target by 2 for this model only; how branches map to optimizer steps is not documented). At the matched index 500 RELEX is 2.0 points below RLVR (95% CI [-3.0, -1.1]), and rank-10 and two-point baselines beat it. | authors' code scales the target by 2 for this model only; the full Table 3 row matches under label -> 2x index |
| 4 | Two of the paper's weight-space summaries (median rank-1 explained variance ~0.82, median R^2 ~0.96 over 75 steps) are compatible with a driftless random-walk null (0.811 / 0.960 expected), so on their own they do not establish structure. This does not refute the paper's separate reconstruction evidence. | random-walk null 0.81 / 0.96; released bf16 trajectory 0.82 / 0.96 |
| 5 | In my run (one seed), the fp32 trajectory is strongly concentrated in rank one (median EV 0.958 over 75 steps, 0.942 over 125) while its coefficient fits a line less well (median R^2 0.893 / 0.869); rounding the same trajectory to bf16 gives 0.816 / 0.955. Pre-registered P5's R^2 > 0.95 fails. | own GRPO run, same run rounded to bf16 gives 0.82 / 0.96 |
| 6 | Storage precision matters, but by method and horizon: fp32 vs bf16-rounded deltas of the same run give RELEX +3.04 [+2.04, +4.02] (50 -> 500), +1.70 (75 -> 500), +0.36 (75 -> 1000), and some bf16 baselines beat their fp32 versions at 1000. | one seed; unadjusted intervals |
| 7 | On my run (real step 500 = 71.38 on MATH) RELEX 75 -> 500 from fp32 deltas lands +1.10 [+0.12, +2.10] above real step 500 (pre-registered P5, score part, holds); from bf16-rounded deltas -0.60 [-1.54, +0.32]. It is not specific to rank-1: raw per-weight least squares gives +1.28 [+0.28, +2.32] and two-point (25, 75) +1.64 [+0.62, +2.64]; rank-1 beat raw least squares at 1000 but not at 500. The gain comes from a larger step: the RELEX delta has 3.24x the norm of the real step-500 delta, and the same direction rescaled to the real per-tensor norm scores 68.64, -2.74 [-3.76, -1.74] vs real 500 (pre-registered P10 fails), so per unit norm the rank-1 direction is less effective than real training. On five OOD benchmarks RELEX fp32 is -2.66 [-4.84, -0.67] vs real 500, but real step 400 is also -1.80 [-3.36, -0.27], so there is no robust OOD advantage or deficit. On Qwen3-4B (released, bf16) RELEX beat raw least squares at 500 (84.8 vs 81.7, raw truncates 15% of answers). | one seed; MATH n = 5000 paired; OOD = macro mean of avg@8 over 5 benchmarks (805 problems); unadjusted except P5 and P10 |
| 8 | Agentic RL: GRPO lifts Qwen3.5-2B from 59.5% to 94.5% on unseen ERP databases (opencode harness), but extrapolating the run collapses the agent (11% at 4x the window; the quick 1-trial probes degrade by 1.6x and collapse by 2-4x; 25 -> 30 roughly ties real step 30 under the full protocol); a power-law coefficient or a longer window does not fix it in the quick probes. Replicated on a second seed (RELEX 25 -> 100: 4.5% vs real 94.2%). The collapse is mostly magnitude overshoot: the RELEX direction rescaled to the real step-100 per-tensor norm scores 88.2% (82% of the base -> step-100 gain), but that norm is oracle information. | magnitude saturates: real delta grows 1.5x from step 25 to 100, linear extrapolation predicts 4x |

## Layout

| Path | What |
|---|---|
| `relex.py` | Gram-matrix formulation: every predictor is a weighted sum of observed deltas; RELEX (any rank, linear / poly / power fit), reconstruction, raw per-element least squares, two-point, ExPO; diagnostics (explained variance, R^2, cosine / norm vs real steps) |
| `build.py`, `store.py` | delta store (one safetensors file per step) and streaming checkpoint builder |
| `train_grpo.py` | GRPO on MATH with TRL 1.16 + colocated vLLM, the paper's verl recipe, fp32 delta dumps, length-sort + trim speed patch (gradient-checked against fp32), deadline-guarded grading |
| `agent/` | agentic ERP test: task generator, TRL `environment_factory` environment (6 SQLite tools), trainer, opencode-harness eval (uses `erpagent/make_db.py` and `erpagent/run_eval.py` + harness adapters from the companion repo ehzawad/tau2-agent-posttraining) |
| `eval_math.py`, `paired.py`, `ood_paired.py`, `table.py` | vLLM eval with the authors' prompt, decoding and grader; paired bootstrap; OOD macro-mean paired bootstrap (problem-clustered, stratified by benchmark); tables |
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
