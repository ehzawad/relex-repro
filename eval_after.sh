#!/bin/bash
# Wait until no relex_eval* step runs on holder $1 (and an optional file exists), then run eval_math on it.
#   relex/eval_after.sh <holder> <tag> <wait_for_file|-> <eval_math args...>
H=$1; T=$2; F=$3; shift 3
until ! squeue -a -s -h -j "$H" -o "%j" | grep -q "^relex_eval" && { [ "$F" = "-" ] || [ -e "$F" ]; }; do sleep 30; done
RELEX_CPUS=3 $HOME/ehz_agent/relex/on_holder.sh "$H" "$T" "python relex/eval_math.py $*"
