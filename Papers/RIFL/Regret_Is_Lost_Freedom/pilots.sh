#!/bin/bash
# Rerun every pilot from the package folder. Each command skips work whose record exists.
#   bash pilots.sh            E1, E2 and E3 pilots
# The E3 pilot replication with seed 31 and the E4 pilot have their own commands in README.md.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${PY:-python3}

# E1: the first pilot, then the three worlds of the second pilot, then the spreads
O=runs/e1pilot
$PY -m e1.pilot --out $O --wid 0 --kind main --q 0.6 --seed 1001 --first
$PY -m e1.pilot --out $O --wid 1 --kind ds --q 0.6 --seed 2002
$PY -m e1.pilot --out $O --wid 10 --kind main --q 0.95 --seed 3010
$PY -m e1.pilot --out $O --wid 11 --kind main --q 0.45 --seed 3011
$PY -m e1.pilot --out $O --summary > /dev/null

# E2: three rounds of single agents. The first two rounds ran before the layer-norm fix.
O=runs/e2pilot
while read -r init a s sd u lr en; do
  name=${init}_${a}_${s}_${sd}_${u}${lr:+_$lr}${en:+_$en}
  [ -f "$O/$name.json" ] && continue
  $PY e2/pilot.py $O $([ "$init" = old ] && echo --oldinit) $a $s $sd $u $lr $en
done <<'LIST'
old gru 32 1 150
old lstm 128 2 150
old trxl 8 3 150
old trxl 32 4 150
old trxl 32 4 400 0.0003 0.01
old trxl 32 4 400 0.001 0.003
old trxl 8 3 400 0.0003 0.003
old gru 8 5 400 0.001 0.01
new trxl 32 4 300 0.001 0.01
new trxl 8 3 300 0.001 0.01
new lstm 32 6 300 0.001 0.01
new gru 128 7 300 0.001 0.01
LIST
$PY e2/pilot.py $O --summary > /dev/null

# E3: the RockSample pilot with its 20-second budget, then one agent of each architecture of the
# E3 pool, evaluated after 20, 40, 60 and 100 updates
O=runs/e3rocks
mkdir -p $O
[ -f $O/gru_32_1_b20.txt ] || $PY e3/t_rocks.py gru 32 1 20 > $O/gru_32_1_b20.txt
while read -r a s sd; do
  for u in 20 40 60 100; do
    [ -f $O/${a}_${s}_${sd}_u${u}.txt ] || $PY e3/t_rocks.py $a $s $sd 1000000 0.001 0.01 $u > $O/${a}_${s}_${sd}_u${u}.txt
  done
done <<'LIST'
gru 8 2
gru 32 1
gru 128 3
lstm 8 4
lstm 32 5
lstm 128 6
trxl 8 7
trxl 32 8
LIST

# E3: the noisy T-Maze runs with seed 1, the first with the reward of 4 in both regimes
O=runs/e3pilot1
[ -f $O/good4_gru_32_1_300.json ] || $PY e3/pilot.py $O --good4 gru 32 1 300
[ -f $O/good48_gru_32_1_300_0.001_0.01_0.1.json ] || $PY e3/pilot.py $O gru 32 1 300 0.001 0.01 0.1
[ -f $O/good48_gru_32_1_300_0.001_0.01_0.5.json ] || $PY e3/pilot.py $O gru 32 1 300 0.001 0.01 0.5
