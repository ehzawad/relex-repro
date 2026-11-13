#!/bin/bash
# Run a command on a B200 holder as an overlapping job step (named relex_<tag>), in the relex env, detached.
#   relex/on_holder.sh <holder_job> <tag> <command...>     -> log $HOME/ehz_agent/store/relex/logs/<tag>.log
J=$1; T=$2; shift 2; D=$HOME/ehz_agent/store
for v in $(env | grep -oE '^(SLURM|SBATCH|SQUEUE)_[A-Z_]+'); do unset "$v"; done
nohup srun --jobid="$J" --overlap -N1 -n1 -c ${RELEX_CPUS:-12} --cpu-bind=none -J relex_$T bash -lc "
  export PYTHONNOUSERSITE=1 HF_HOME=$D/hf TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=ERROR PYTHONUNBUFFERED=1
  export PATH=$D/envs/relex/bin:$D/envs/pt/bin:\$PATH
  cd $HOME/ehz_agent; $*" > $D/relex/logs/$T.log 2>&1 &
echo "started relex_$T on $J -> $D/relex/logs/$T.log"
