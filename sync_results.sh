#!/bin/bash
# Copy small result summaries (scores, weight-space stats, weight-match checks, training logs) from /scratch into
# relex/results/ for the repo. Per-sample generations, deltas and checkpoints stay on /scratch.
D=$HOME/ehz_agent/store/relex; R=$HOME/ehz_agent/relex/results
mkdir -p $R/evals $R/agent_evals $R/weight_stats $R/train_logs
for b in q15_released q15_own q15_own3 q4b_released; do
  for f in $D/evals/$b/*/*.json; do [ -f "$f" ] && { m=$(basename $(dirname $f)); mkdir -p $R/evals/$b/$m; cp $f $R/evals/$b/$m/; }; done
done
cp $D/agent/evals/*.summary.json $R/agent_evals/ 2>/dev/null
for f in released/q15/gram_stats.json released/q15/compare_released_relex.json own/q15_run1/gram_stats_s60.json \
         own/q15_run1/gram_stats_s75.json own/q15_run1/gram_stats_s125.json agent/run1/gram_stats.json released/q4b/gram_stats.json \
         released/q4b/compare.json; do
  [ -f $D/$f ] && cp $D/$f $R/weight_stats/$(echo $f | tr '/' '_')
done
cp $D/own/q15_run1/train_log.jsonl $R/train_logs/math_q15_run1.jsonl 2>/dev/null
cp $D/own/q15_run3/train_log.jsonl $R/train_logs/math_q15_run3_seed1.jsonl 2>/dev/null
cp $D/logs/final_paired.txt $D/logs/final_paired_run3.txt $D/logs/ood_paired.txt $R/ 2>/dev/null
cp $D/agent/run1/train_log.jsonl $R/train_logs/agent_run1.jsonl 2>/dev/null
cp $D/agent/run2/train_log.jsonl $R/train_logs/agent_run2.jsonl 2>/dev/null
cp $D/released/q15/deltas/convert_stats.jsonl $R/weight_stats/released_q15_bf16_zero_fraction.jsonl 2>/dev/null
du -sh $R
