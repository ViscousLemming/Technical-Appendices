#!/bin/zsh
# Run every E4 stage for a plan: train, source, open, target, analysis. Every stage resumes.
#   nohup caffeinate -i e4/run4.sh e4/plan.json runs/e4 > runs/e4/run4.log 2>&1 &
set -e
cd "$(dirname $0)/.."
PLAN=$1; OUT=$2; PROCS=${3:-16}
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.venv/bin/python
mkdir -p $OUT
$PY -m common.drive --exp e4 train  --plan $PLAN --out $OUT --procs $PROCS
$PY -m common.drive --exp e4 source --plan $PLAN --out $OUT --procs $PROCS
$PY -m common.drive --exp e4 open   --plan $PLAN --out $OUT
$PY -m common.drive --exp e4 target --plan $PLAN --out $OUT --procs $PROCS
$PY e4/anal4.py $OUT
echo "run4 finished $(date)"
