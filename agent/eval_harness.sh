#!/bin/bash
# R3 eval of one checkpoint through the industry harness (erpagent/run_eval.py, opencode), run INSIDE a holder step:
#   bash relex/agent/eval_harness.sh <model path or HF id> <name> [port]
# Starts vLLM (Qwen3.5 tool + reasoning parsers, as in erpagent), waits, runs test 300 x 2 trials and orig37 x 3 trials
# (unguarded tools, same as training), then stops the server. Results: $SCRATCH/.../relex/agent/evals/opencode__<name>[_orig37]/
M=$1; N=$2; P=${3:-8141}; D=$DATA; A=$D/relex/agent; LOG=$D/relex/logs/vllm_$N.log
export PYTHONNOUSERSITE=1 HF_HOME=$D/hf PATH=$D/envs/relex/bin:$D/envs/pt/bin:$PATH
python -m vllm.entrypoints.cli.main serve "$M" --served-model-name "$N" --port $P --host 127.0.0.1 --trust-remote-code \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --max-num-seqs 128 --enable-prefix-caching --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 > $LOG 2>&1 &
S=$!
until grep -qE "startup complete|Traceback|Error:" $LOG 2>/dev/null; do sleep 5; done
grep -q "startup complete" $LOG || { echo "vLLM failed: $LOG"; tail -5 $LOG; kill $S; exit 1; }
cd $HOME/ehz_agent/erpagent
python run_eval.py --harness opencode --model "$N" --url http://127.0.0.1:$P/v1 --trials 2 --conc 16 --tasks $A/tasks_test.jsonl --outroot $A/evals
python run_eval.py --harness opencode --model "$N" --url http://127.0.0.1:$P/v1 --trials 3 --conc 16 --tasks $A/tasks_orig37.jsonl --tag orig37 --outroot $A/evals
kill $S; wait $S 2>/dev/null; echo "done $N"
