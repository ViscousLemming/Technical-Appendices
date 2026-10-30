#!/bin/bash
# Run every E5 stage with the E4 world code: train, source, open, target, then the ranking analysis.
# Every stage resumes.
#   nohup e5/run5.sh e5/plan.json runs/e5 16 > runs/e5/run5.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
PLAN=$1; OUT=$2; PROCS=${3:-16}
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-.venv/bin/python}
mkdir -p "$OUT"
$PY -m common.drive --exp e4 train  --plan "$PLAN" --out "$OUT" --procs "$PROCS"
$PY -m common.drive --exp e4 source --plan "$PLAN" --out "$OUT" --procs "$PROCS"
$PY -m common.drive --exp e4 open   --plan "$PLAN" --out "$OUT"
$PY -m common.drive --exp e4 target --plan "$PLAN" --out "$OUT" --procs "$PROCS"
$PY e5/anal5.py "$OUT"
echo "run5 finished $(date)"
