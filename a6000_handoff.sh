#!/bin/bash
# User rule (2026-11-03): the 4x A6000 holder is a stopgap. As soon as two single B200 holders are free (no relex_ step,
# >= MIN_LEFT_MIN minutes left) or a B200 full node is running, release the A6000s: wait (<= 15 min) for a fresh TRL
# checkpoint of the agentic run (or its DONE), then scancel the A6000 job. The B200 watcher for that run (watch_run.sh,
# tag agent) resumes it from the checkpoint; unfinished evals are re-queued by the eval worker / me.
#   nohup relex/a6000_handoff.sh <a6000 job id> <agent run dir> >> logs/a6000_handoff.log 2>&1 &
J=$1; RUN=$2; MIN_LEFT_MIN=${MIN_LEFT_MIN:-90}
mins() { awk -F'[-:]' '{ if (NF==4) print $1*1440+$2*60+$3; else if (NF==3) print $1*60+$2; else print $1 }' <<< "$1"; }
fresh_ckpt() { find "$RUN/trainer" -maxdepth 2 -name trainer_state.json -mmin -3 2>/dev/null | grep -q . ; }
waiting_since=""
while squeue -a -j "$J" -h -o %T 2>/dev/null | grep -qE "RUNNING|PENDING"; do
  if squeue -a -j "$J" -h -o %T | grep -q RUNNING; then
    free=0
    while read -r id left; do
      [ "$(squeue -a -s -h -j "$id" -o "%j" | grep -c '^relex_')" -eq 0 ] && [ "$(mins "$left")" -ge "$MIN_LEFT_MIN" ] && free=$((free + 1))
    done < <(squeue -a -u "$USER" -h -t R -o "%i %L %j %b" | awk '/B200_1gpu/ && /b200/ {print $1, $2}')
    node=$(squeue -a -u "$USER" -h -t R -o "%j" | grep -c "B200_full_node")
    if [ "$free" -ge 2 ] || [ "$node" -ge 1 ]; then
      [ -z "$waiting_since" ] && { waiting_since=$(date +%s); echo "$(date '+%F %T') B200 capacity available (free singles $free, node $node): handing off"; }
      if [ -f "$RUN/DONE" ] || fresh_ckpt || [ $(( $(date +%s) - waiting_since )) -ge 900 ]; then
        echo "$(date '+%F %T') releasing A6000 job $J"; scancel "$J"; exit 0
      fi
    else
      waiting_since=""
    fi
  fi
  sleep 60
done
echo "$(date '+%F %T') A6000 job $J gone; exiting"
