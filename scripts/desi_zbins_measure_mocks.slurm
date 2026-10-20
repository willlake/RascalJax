#!/bin/bash
# Array job measuring the two-bin LRG correlations (AA, BB, AB) on GLAM-Uchuu v2 mocks, 4 mocks in flight per node (one
# per GPU; measured on one A100 with DR from 10 and RR from 2 files to 160 Mpc/h: see the timings in the log of mock 150, so 16 mocks
# per 4-GPU node take ~80 min). Submit from ~/Packages/RascalJax with e.g.
#   sbatch --array=0-62 scripts/desi_zbins_measure_mocks.slurm        # 63 x 16 = 1008 >= 1007 good mocks
# Environment overrides: OUT (output dir), NRAN_DR / NRAN_RR (random files for DR / RR, defaults 10 / 2), PER_JOB (mocks per array task, default 16),
# REGIONS (default "SGC NGC"), SMAX (default 160), CORRS (default AA,BB), MOCKLIST (file with one mock index per line).
#SBATCH --account=desi_g
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --time=02:30:00
#SBATCH --license=SCRATCH
#SBATCH --job-name=zbins_xi
#SBATCH --output=/pscratch/sd/w/willlake/rascaljax/logs/zbins_xi_%A_%a.log
set -e
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /global/homes/w/willlake/Packages/RascalJax
export OUT=${OUT:-/pscratch/sd/w/willlake/rascaljax/desi_zbins_LRG_z0.4-0.55_z0.45-0.6}
export NRAN_DR=${NRAN_DR:-10} NRAN_RR=${NRAN_RR:-2}
export REGIONS=${REGIONS:-"SGC NGC"}
PER_JOB=${PER_JOB:-16}
MOCKLIST=${MOCKLIST:-/global/homes/w/willlake/Packages/desi-clustering/clustering_statistics/helper_scripts/glam-uchuu-v2-altmtl_dark-time_imocks_for_covariance_v2.txt}
mapfile -t MOCKS < "$MOCKLIST"
START=$(( ${SLURM_ARRAY_TASK_ID:-0} * PER_JOB ))
export CHUNK="${MOCKS[*]:$START:$PER_JOB}"
echo "array task ${SLURM_ARRAY_TASK_ID:-0}: mocks $CHUNK"
mkdir -p "$OUT/logs"
srun -n 4 bash -c '
  read -a C <<< "$CHUNK"; MY=""
  for ((k=SLURM_PROCID; k<${#C[@]}; k+=4)); do MY="$MY ${C[k]}"; done
  [ -z "$MY" ] && exit 0
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_PYTHON_CLIENT_PREALLOCATE=true
  for REG in $REGIONS; do
    python scripts/desi_zbins_measure_mocks.py "$OUT" $REG 0.4 0.55 0.45 0.6 $MY --nran-dr ${NRAN_DR:-10} --nran-rr ${NRAN_RR:-2} --s-max ${SMAX:-160} --correlations ${CORRS:-AA,BB} --skip-existing \
      > "$OUT/logs/measure_${REG}_task${SLURM_ARRAY_TASK_ID:-0}_rank${SLURM_PROCID}.log" 2>&1
  done'
echo "done"
