#!/bin/bash
# Work through an eval queue (one line per job: "<bench> <outdir> <comma-separated model dirs>") on B200 holders that the
# GRPO run does not need: a holder is used only if it has no relex_ step and relex_grpo is already running elsewhere
# (or <grpo out>/DONE exists). A line is done when every model has <outdir>/<name>/<bench>.json; unfinished lines are
# retried (eval_math skips finished models), so a holder ending mid-job loses at most one model.
#   nohup relex/eval_worker.sh <queue file> <grpo out dir> > log 2>&1 &
Q=$1; G=$2; R=$HOME/ehz_agent/relex
mins() { awk -F'[-:]' '{ if (NF==4) print $1*1440+$2*60+$3; else if (NF==3) print $1*60+$2; else print $1 }' <<< "$1"; }
done_line() { local b=$1 o=$2 ms=$3; for m in ${ms//,/ }; do [ -e "$o/$(basename $m)/$b.json" ] || return 1; done; }
while true; do
  next=""; while read -r b o ms; do [ -z "$b" ] && continue; done_line "$b" "$o" "$ms" || { next="$b $o $ms"; break; }; done < "$Q"
  [ -z "$next" ] && { sleep 120; continue; }                         # queue drained; keep watching for appended lines
  if squeue -a -s -u "$USER" -h -o "%j" | grep -q "^relex_eval"; then sleep 60; continue; fi
  grpo_ok=0; { [ -f "$G/DONE" ] || squeue -a -s -u "$USER" -h -o "%j" | grep -q "^relex_grpo$"; } && grpo_ok=1
  best=""; bestm=0
  while read -r id left; do
    m=$(mins "$left"); busy=$(squeue -a -s -h -j "$id" -o "%j" | grep -c "^relex_")
    if [ "$busy" -eq 0 ] && [ "$m" -ge 15 ] && [ "$m" -gt "$bestm" ]; then best=$id; bestm=$m; fi
  done < <(squeue -a -u "$USER" -h -t R -o "%i %L %j %b" | awk '/B200_/ && /b200/ {print $1, $2}')
  if [ "$grpo_ok" -eq 1 ] && [ -n "$best" ]; then
    read -r b o ms <<< "$next"
    echo "$(date '+%F %T') eval $b ($(echo $ms | tr ',' '\n' | wc -l) models) on $best"
    RELEX_CPUS=3 "$R/on_holder.sh" "$best" eval_w "python relex/eval_math.py --bench $b --out $o --models $ms"
    sleep 90
  else
    sleep 60
  fi
done
