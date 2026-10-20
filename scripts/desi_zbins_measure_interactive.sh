#!/bin/bash
# Run the two-bin mock measurements inside an interactive GPU allocation, one mock at a time per GPU over all nodes:
#   salloc -N 2 -C gpu -q interactive -t 03:30:00 --gpus-per-node=4 -A desi_g
#   bash scripts/desi_zbins_measure_interactive.sh [mock list file] [n_mocks]
# 2 nodes x 4 GPUs: 200 NGC mocks (AA, BB to 160 Mpc/h, DR 10 / RR 2 files) in about 2.6 h.
# Environment overrides as in desi_zbins_measure_mocks.slurm: OUT, REGIONS, SMAX, CORRS, NRAN_DR, NRAN_RR.
set -e
cd /global/homes/w/willlake/Packages/RascalJax
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export OUT=${OUT:-/pscratch/sd/w/willlake/rascaljax/desi_zbins_LRG_z0.4-0.55_z0.45-0.6}
export REGIONS=${REGIONS:-NGC} SMAX=${SMAX:-160} CORRS=${CORRS:-AA,BB} NRAN_DR=${NRAN_DR:-10} NRAN_RR=${NRAN_RR:-2}
MOCKLIST=${1:-$OUT/mocklist_200.txt}
N_MOCKS=${2:-200}
mapfile -t MOCKS < <(head -n "$N_MOCKS" "$MOCKLIST")
export CHUNK="${MOCKS[*]}"
NTASKS=$(( ${SLURM_NNODES:-1} * 4 ))
mkdir -p "$OUT/logs"
echo "$(date): ${#MOCKS[@]} mocks over $NTASKS GPUs on ${SLURM_NNODES:-1} node(s); logs in $OUT/logs/measure_<REG>_interactive_rank<k>.log"
srun -N "${SLURM_NNODES:-1}" -n "$NTASKS" --gpus-per-node=4 --cpus-per-task=32 bash -c '
  read -a C <<< "$CHUNK"; MY=""
  for ((k=SLURM_PROCID; k<${#C[@]}; k+=SLURM_NTASKS)); do MY="$MY ${C[k]}"; done
  [ -z "$MY" ] && exit 0
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_PYTHON_CLIENT_PREALLOCATE=true
  for REG in $REGIONS; do
    python scripts/desi_zbins_measure_mocks.py "$OUT" $REG 0.4 0.55 0.45 0.6 $MY --nran-dr $NRAN_DR --nran-rr $NRAN_RR --s-max $SMAX --correlations $CORRS --skip-existing \
      > "$OUT/logs/measure_${REG}_interactive_rank${SLURM_PROCID}.log" 2>&1
  done'
echo "$(date): done; measured mocks: $(ls -d $OUT/mocks/mock*/xi_AA_NGC.npy 2>/dev/null | wc -l)"
