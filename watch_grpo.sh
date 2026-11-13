#!/bin/bash
# Keep the GRPO run alive across B200 holders: if no relex_grpo step is running, start it (it resumes from its last TRL
# checkpoint) on the running B200 holder with the most time left (>= MIN_LEFT_MIN minutes) that has no other relex step.
# Exits when <out>/DONE exists.  Run detached on the login node:
#   nohup relex/watch_grpo.sh $SCRATCH/.../relex/own/q15_run1 > $SCRATCH/.../relex/logs/watch_grpo.log 2>&1 &
OUT=$1; MIN_LEFT_MIN=${MIN_LEFT_MIN:-90}; R=$HOME/ehz_agent/relex
mins() { awk -F'[-:]' '{ if (NF==4) print $1*1440+$2*60+$3; else if (NF==3) print $1*60+$2; else print $1 }' <<< "$1"; }
while [ ! -f "$OUT/DONE" ]; do
  if ! squeue -a -s -u "$USER" -h -o "%j" | grep -q "^relex_grpo$"; then
    best=""; bestm=0
    while read -r id left; do
      m=$(mins "$left")
      busy=$(squeue -a -s -h -j "$id" -o "%j" | grep -c "^relex_")
      if [ "$busy" -eq 0 ] && [ "$m" -ge "$MIN_LEFT_MIN" ] && [ "$m" -gt "$bestm" ]; then best=$id; bestm=$m; fi
    done < <(squeue -a -u "$USER" -h -t R -o "%i %L %j %b" | awk '/B200_/ && /b200/ {print $1, $2}')
    if [ -n "$best" ]; then
      echo "$(date '+%F %T') starting relex_grpo on $best (${bestm} min left)"
      RELEX_CPUS=12 "$R/on_holder.sh" "$best" grpo "python relex/train_grpo.py --out $OUT >> $HOME/ehz_agent/store/relex/logs/grpo_run.log 2>&1"
      sleep 120
    else
      echo "$(date '+%F %T') no free holder with >= $MIN_LEFT_MIN min"
    fi
  fi
  sleep 120
done
echo "$(date '+%F %T') DONE found, exiting"
