# relex: end-to-end reproduction of RELEX (arXiv 2605.21468)

Wei, Zhu, Chen, Huang, Huang, Meng. "You Only Need Minimal RLVR Training: Extrapolating LLMs via Rank-1 Trajectories" (May 2026).
Authors' code: github.com/weizhepei/RELEX (MIT; grader.py/utils.py vendored in third_party/ for identical scoring).
Released data: hf relex-rlvr/RLVR-Qwen2.5-Math-1.5B (branches step_1..step_500, bf16), RLVR-Qwen3-4B-Base (1..75),
RLVR-Qwen3-8B-Base (1..100), RELEX-* (their predicted checkpoints).
Code: ~/ehz_agent/relex. Data: $DATA/relex.

## Method (what is being reproduced)
Per weight tensor: stack Delta_t = theta_t - theta_0 (t = 1..T_obs) into M [T_obs x d], top right singular vector v1
(Gram trick), coefficients c_t = <Delta_t, v1>, least-squares line c(t) = a t + b, predict theta_T = theta_0 + (aT+b) v1.
Claims: (F1) rank-1 reconstruction keeps the RLVR gain; (F2) c_t is linear in t (R^2 > 0.98 for most tensors);
(C1) T_obs = 75 (15%) matches RLVR step 500 on MATH and OOD; (C2) beats raw-weight / two-point / ExPO / AlphaRL /
logit extrapolation; (C3) rank-1 > rank-5/10, linear > poly-3; (C4) extrapolates to 1000 steps; (C5) "denoising":
the SVD projection removes optimisation noise.

## Implementation (own code, not the authors' scripts)
Every method predicts Delta_T as a weighted sum of observed deltas, Delta_T = sum_t w_t Delta_t (per tensor for the SVD
methods). So relex/ computes one Gram matrix G = M M^T per tensor over all stored steps, derives every method's weights w
(and its R^2, explained variance, cosine/norm against any stored step) from G, and builds checkpoints in one streaming pass.

## Fidelity notes found while reading the code (2026-11-03 07:40 EST)
1. Qwen2.5-Math-1.5B only: svd_extrapolation.py multiplies the target step by 2 ("one saved ckpt per 2 optimizer steps"),
   while the history window is NOT scaled and the released trajectory has exactly 500 checkpoints. So the paper's
   1.5B "RELEX step 500" is the line evaluated at checkpoint index 1000 (2x the run it is compared with). Qwen3 models: x1.
2. Released checkpoints are bf16. With lr 1e-6, one Adam step (~1e-6) is far below bf16 spacing (~1e-4 at |w|~0.02):
   conversion stats (seen BEFORE writing the predictions below): released step_1 delta is exactly zero everywhere, steps
   2/3/4 are 99.5/99.1/98.7% zero. Literature: bf16 hides sub-ulp RL updates (2511.08567, 2602.03839, 2610.41073);
   nobody has measured what this does to the rank-1 fraction or the R^2 of c_t.
3. "MATH" = TianHongZXY/MATH test, 5000 problems (not MATH-500); 1.5B: greedy, 4096 tokens; OOD avg@8, T=0.7, top_p 0.8.

## Phases
R0  Validate my implementation on the released 1.5B trajectory (steps 1..125 every step, 150..500 every 25):
    weight-level match to their released RELEX checkpoint; MATH for base, RLVR {100..500}, RELEX T_obs in {50,75,100,125}
    x targets; ablations at T_obs=75 (raw per-element LS, two-point weight extrapolation, ExPO, rank 5/10, poly-3).
R1  Independent trajectory: my own GRPO run (TRL 1.16 + colocated vLLM 0.32, relex/train_grpo.py), Qwen2.5-Math-1.5B, the
    paper's recipe, 500 steps, fp32 deltas every step to 125 then every 25. Re-test F1, F2, C1, C3 at true step indices.
R2  Precision test of C5: same run, deltas from fp32 master weights vs theta_t rounded to bf16 (what a saved HF checkpoint
    holds). Does rank-1's advantage over raw extrapolation come from optimisation noise or from bf16 rounding noise?

## Pre-registered predictions (2026-11-03 07:40 EST, before any eval)
P1 My RELEX (T_obs=75, target index 1000) matches their released RELEX-Qwen2.5-Math-1.5B weights: per-tensor delta cosine
   > 0.99 for > 95% of tensors and whole-model relative L2 difference < 5%. Target index 500 matches clearly worse.
P2 My MATH-5000 greedy scores reproduce theirs within 1.5 points: base 48.2, RLVR step 500 71.5, RELEX(75->"500") 71.6.
P3 RELEX(75) evaluated at the honest index 500 scores lower than at index 1000 but at least 68.
P4 At T_obs=75, rank-1 beats raw per-element least squares and rank-5 at the far target by >= 0.5 point; poly-3 collapses
   (< 20) by index 400.
P5 Own run: F2 holds with fp32 deltas (median per-tensor R^2 of the c_t line over steps 1..125 > 0.95), and RELEX(75 -> 500)
   is within 2 points of my RLVR step 500 on MATH-5000.
P6 Own run, precision: with bf16-rounded deltas the median rank-1 explained variance over steps 1..75 is lower than with
   fp32 deltas, and the gap (RELEX minus raw per-element LS, MATH at step 500) is at least 1 point smaller with fp32
   deltas than with bf16-rounded deltas (i.e. part of the "denoising" is removing rounding noise).

## Log
- 2026-11-03 07:40 EST project started; released-trajectory download job 22388105; GRPO smoke test on holder 22351370.
