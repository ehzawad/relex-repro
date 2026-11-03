#!/bin/bash
# GPU janitor (login node, nohup). Every 30 min, and within 2 min of a new B200 holder starting:
#  1. keeps the long-running drivers alive (the GRPO watcher(s) and the eval worker), restarting any that died;
#  2. finds idle holders (no job step besides batch/extern) and starts the next one-shot job from the extra queue
#     (lines "tag<TAB>command", run with relex/on_holder.sh; a started line is moved to extra_queue.started);
#  3. logs one status line per check: holders, time left, steps on each, idle ones.
D=$DATA/relex; R=$HOME/ehz_agent/relex; XQ=$D/extra_queue.txt
alive() { ps -u "$USER" -o cmd | grep -v grep | grep -qF -- "$1"; }
holders() { squeue -a -u "$USER" -h -t R -o "%i %L %j %b" | awk '/B200_/ && /b200/ {print $1, $2}'; }
prev=""; last=0
while true; do
  now=$(date +%s); cur=$(holders | awk '{print $1}' | sort | tr '\n' ' ')
  if [ "$cur" != "$prev" ] || [ $((now - last)) -ge 1800 ]; then
    [ "$cur" != "$prev" ] && [ -n "$prev" ] && echo "$(date '+%F %T') holder set changed: [$prev] -> [$cur]"
    # 1. drivers
    while read -r out tag; do
      [ -z "$out" ] && continue
      if [ ! -f "$out/DONE" ] && ! alive "watch_run.sh $out"; then
        echo "$(date '+%F %T') restarting watcher for $out ($tag)"
        (cd $HOME/ehz_agent && TAG=$tag nohup relex/watch_run.sh "$out" >> $D/logs/watch_$tag.log 2>&1 &)
      fi
    done < <(cat $D/grpo_runs.txt 2>/dev/null)
    if ! alive "eval_worker.sh $D/eval_queue.txt"; then
      echo "$(date '+%F %T') restarting eval worker"
      (cd $HOME/ehz_agent && nohup relex/eval_worker.sh $D/eval_queue.txt $D/own/q15_run1 >> $D/logs/eval_worker.log 2>&1 &)
    fi
    sleep 150                                                  # let drivers claim a newly arrived holder first
    # 2. idle holders get extra one-shot work
    status=""
    while read -r id left; do
      steps=$(squeue -a -s -h -j "$id" -o "%j" | grep -v -E "^(batch|extern)$" | tr '\n' ',')
      status="$status $id(${left}:${steps:-IDLE})"
      if [ -z "$steps" ] && [ -s "$XQ" ]; then
        IFS=$'\t' read -r tag cmd < "$XQ"
        echo "$(date '+%F %T') idle holder $id -> extra job $tag"
        RELEX_CPUS=12 "$R/on_holder.sh" "$id" "$tag" "$cmd"
        head -1 "$XQ" >> "$XQ.started"; sed -i 1d "$XQ"
      fi
    done < <(holders)
    echo "$(date '+%F %T') check:$status | pending B200 requests: $(squeue -a -u "$USER" -h -t PD -o "%j" | grep -c B200_)"
    prev=$cur; last=$now
  fi
  sleep 120
done
