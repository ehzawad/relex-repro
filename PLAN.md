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
- 2026-11-03 08:49 EST P1 HOLDS: my RELEX(c75) at index 1000 vs released RELEX-Qwen2.5-Math-1.5B: per-tensor cos 1.0000, rel L2 0.18%, norm ratio 0.99996; at index 500 the norm ratio is exactly 0.50. So the paper's 1.5B 'RELEX step 500' = index 1000 (2x the 500-checkpoint RLVR run). Its delta is 5.5x the norm of the real RLVR step-500 delta (per-tensor cos 0.37). Released-trajectory Gram stats: window 1-75 median rank-1 EV 0.82, median R^2 of c_1 line 0.963 (17% of tensors > 0.98); whole run (uniform every 25) R^2 median 0.946 (22% > 0.98). App. B: extrapolation c75 cos 0.78->0.41, norm ratio 1.19->2.48 at index 100->500 (paper 0.72->0.35, 1.26->2.70); reconstruction (uniform fit) cos 0.64->0.94 peak at 400 (paper 0.50->0.91 peak 400). Holder 22355168 cancelled by root 08:31. Own GRPO run ~205 s/step after exact sort+trim speedup (verified vs TRL in fp32: max |dlogp| 4e-5).
- 2026-11-03 09:01 EST MATH-5000 (greedy 4K, authors' prompt+grader): base 46.9 (paper 48.2), RLVR step_500 71.78 (71.5), RELEX c75@index1000 71.74 (paper 'step 500' 71.6) -> P2 HOLDS. RELEX c75@index500 (horizon matched to RLVR 500) 69.74 -> P3 HOLDS (2.0 below RLVR 500).
- 2026-11-03 09:26 EST c75 horizon sweep (index axis) MATH: 100 67.44, 200 68.22, 300 69.56, 400 69.74, 500 69.74, 750 70.68, 1000 71.74, 1500 65.58, 2000 68.18. RLVR: 75 67.78, 100 67.52, 200 69.58, 300 70.94, 400 71.40, 500 71.78. Paper Table 3 c75 row under label->2x index: 100->68.2(67.8), 200->69.7(70.0), 500->71.7(71.6), 750->65.6(65.7), 1000->68.2(68.4): whole 1.5B table is on the 2x axis. At matched index RELEX trails RLVR by 0.1/1.4/1.4/1.7/2.0 at 100..500; needs 2x overshoot to reach RLVR-500, collapses at 1500 (truncation 8.9%).
- 2026-11-03 10:18 EST Ablations (released, c75, MATH, index axis). At index 1000 (paper 'step 500'): RELEX 71.74, raw-LS 69.60, rank5 69.70, rank10 69.54, two-point(0,75) 70.94, two-point(25,75) 71.90, poly3 0.02 -> P4 HOLDS at the paper's horizon (paper: 71.6/70.7/70.6/70.5/poly 0.1). At index 500: RELEX 69.74 < raw-LS 70.32, rank5 70.60, rank10 70.76, two-point(25,75) 70.82: ordering reverses at the matched horizon. ExPO (1+a)D75: a=1 68.42, a=3 70.10. Windows @1000: c50 70.76, c100 71.50, c125 71.86; @500: 68.94/70.58/70.50; @2000: 66.70/70.76/71.80. Rank-1 reconstruction (uniform every-25 fit): step 500 71.28, 400 71.18 (RLVR 71.78/71.40) -> F1 holds.
- 2026-11-03 10:19 EST Paired bootstrap (5000 problems, 95% CI): RELEX@1000 - RLVR500 -0.04 [-0.96,+0.90]; RELEX@500 - RLVR500 -2.04 [-3.00,-1.10]; RELEX@1000 - rawLS@1000 +2.14 [+1.16,+3.18]; - rank5@1000 +2.04 [+1.10,+2.96]; - twopoint(25,75)@1000 -0.16 [-1.18,+0.84]; RELEX@500 - rawLS@500 -0.58 [-1.30,+0.12]; - rank10@500 -1.02 [-1.78,-0.28]; - twopoint(25,75)@500 -1.08 [-1.92,-0.20]; recon@500 - RLVR500 -0.50 [-1.24,+0.24].
