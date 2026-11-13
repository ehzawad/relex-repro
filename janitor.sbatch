#!/bin/bash
# Run the RELEX GPU janitor (and the watchers it spawns) inside a Slurm job: login-node background processes get killed.
#SBATCH -p standard
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 3-00:00:00
#SBATCH -J relex_janitor
#SBATCH -o $HOME/ehz_agent/store/relex/logs/janitor_job-%j.out
export PYTHONNOUSERSITE=1
cd $HOME/ehz_agent
exec relex/gpu_janitor.sh >> $HOME/ehz_agent/store/relex/logs/janitor.log 2>&1
