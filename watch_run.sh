#!/bin/bash
# Keep one resumable GRPO run alive across B200 holders. Step name relex_$TAG; extra train_grpo.py args come from
# <out>/launch_args (optional). Starts the run (it resumes from its last TRL checkpoint) on the running B200 holder with
# the most time left (>= MIN_LEFT_MIN) that has no relex_ step. Exits when <out>/DONE exists.
#   TAG=grpo nohup relex/watch_run.sh $SCRATCH/.../relex/own/q15_run1 >> log 2>&1 &
OUT=$1; TAG=${TAG:-grpo}; SCRIPT=${SCRIPT:-relex/train_grpo.py}; MIN_LEFT_MIN=${MIN_LEFT_MIN:-90}; R=$HOME/ehz_agent/relex
mins() { awk -F'[-:]' '{ if (NF==4) print $1*1440+$2*60+$3; else if (NF==3) print $1*60+$2; else print $1 }' <<< "$1"; }
while [ ! -f "$OUT/DONE" ]; do
  if ! squeue -a -s -u "$USER" -h -o "%j" | grep -q "^relex_$TAG$"; then
    best=""; bestm=0
    while read -r id left; do
      m=$(mins "$left"); busy=$(squeue -a -s -h -j "$id" -o "%j" | grep -c "^relex_")
      if [ "$busy" -eq 0 ] && [ "$m" -ge "$MIN_LEFT_MIN" ] && [ "$m" -gt "$bestm" ]; then best=$id; bestm=$m; fi
    done < <(squeue -a -u "$USER" -h -t R -o "%i %L %j %b" | awk '/B200_/ && /b200/ {print $1, $2}')
    if [ -n "$best" ]; then
      echo "$(date '+%F %T') starting relex_$TAG on $best (${bestm} min left)"
      RELEX_CPUS=12 "$R/on_holder.sh" "$best" "$TAG" "python $SCRIPT --out $OUT $(cat $OUT/launch_args 2>/dev/null) >> $DATA/relex/logs/${TAG}_run.log 2>&1"
      sleep 120
    fi
  fi
  sleep 120
done
echo "$(date '+%F %T') DONE found, exiting"
