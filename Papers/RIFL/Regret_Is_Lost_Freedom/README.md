# Regret Is Freedom Lost

This folder holds the code, plans fixed in advance and records behind all 312 replications of the five experiments in *Regret Is Freedom Lost* (Michael Timothy Bennett, submitted to the Journal of Machine Learning Research), and the scripts that regenerate the paper's numbers, tables and figures from those records.

## Contents

```
paper/     the numbers (nums.tex), tables (t_*.tex) and figures (fig_*.pdf) of the paper,
           and the scripts that generate them (mknums.py, mkfigs.py)
reqs.txt   Python packages
pilots.sh  reruns the pilots of E1, E2 and E3
common/    freedom lost, decision-rule probes and selectors that use no rewards (fsel.py),
           statistics and selection (anal.py), and the driver for E2 to E5 (drive.py)
e1/        delayed cue (env.py), recurrent actor-critic and PPO (agent.py), teachers (teach.py),
           replication stages (world.py), driver (drive.py), analysis (anal1.py),
           the check of every agent's own memory with lookup tables (kcheck.py), and the pilot runner (pilot.py)
e2/        POBAX T-Maze in NumPy (tmaze.py), GRU, LSTM and gated-transformer agents (agent.py),
           PPO (ppo.py), teacher (teach.py), replication stages (world.py), analysis (anal2.py),
           and the pilot runner (pilot.py)
e3/        noisy T-Maze with a visible regime (nmaze.py), PPO (ppo.py), replication stages (world.py),
           analysis (anal3.py), the pilot runner (pilot.py), and the RockSample environment of the E3 pilot
           in Appendix C (rocks.py) with its PPO (rppo.py) and pilot script (t_rocks.py)
e4/        POPGym Repeat Previous with lags unknown, NumPy port and its check against popgym (rprev.py),
           PPO (ppo4.py), replication stages (world.py), analysis (anal4.py), runner (run4.sh),
           plan (plan.md, plan.json), pilot plan and record (pplan.json, pilot.json)
e5/        E4 rerun on fresh seeds and scored by rankings, plan (plan.md, plan.json),
           analysis (anal5.py), runner (run5.sh)
runs/      plans, selection records, the start time of each evaluation stage, per-replication results
           and hits, summaries, logs and amendments, for all 312 replications and every pilot
```

The scripts `t_train.py` in `e2/` and `e3/` train one agent, and I used them in the pilots.

## Terms in the code and records

The code and the records name some of the paper's terms differently.

| Paper | Code and records |
|---|---|
| replication | world (`wid`) |
| selection stage | source stage (`source`) |
| selection record | lock (`lockNNN.json`) |
| evaluation stage | target stage (`target`) |
| target sample | target panel |
| decision rule | readout |
| freedom lost | deficit |
| task | question |

## Checks that take minutes

These use the records in `runs/` and train nothing.

```
python3 -m venv .venv && .venv/bin/pip install -r reqs.txt
.venv/bin/python paper/mknums.py && .venv/bin/python paper/mkfigs.py  # seconds
```

`mknums.py` writes every number quoted in the paper to `paper/nums.tex`, and `mkfigs.py` writes the generated tables and the figures, all from the summaries, selection records and results in `runs/`.
On the machine that built this package, both scripts reproduced `nums.tex` and the tables byte for byte, and the figures matched pixel for pixel at 150 dots per inch.

## Analyses from the per-replication results

Each analysis loads the selection records, results and hits of one experiment and rewrites its summary.

```
python3 e1/anal1.py runs/e1                # about a minute
python3 e2/anal2.py runs/e2
python3 e3/anal3.py runs/e3dev
python3 e4/anal4.py runs/e4
python3 e5/anal5.py runs/e5
python3 e5/anal5.py runs/e4 --explore      # the same ranking analysis on E4, exploratory (runs/e4/e4_rank.json)
```

On the machine that built this package, the rewritten summaries matched the ones here to within rounding in the fifteenth significant digit, and `mknums.py` then reproduced `nums.tex` byte for byte.

A selector that gives several candidates its best score deploys each of them with equal chance.
`RIWF_TIES=first` reruns an analysis with the lowest-index rule, which deploys the tied candidate listed first, and writes `*_summary_first.json`, which `mkfigs.py` compares in Table 8 of the paper.

```
RIWF_TIES=first python3 e1/anal1.py runs/e1
RIWF_TIES=first python3 e2/anal2.py runs/e2
RIWF_TIES=first python3 e3/anal3.py runs/e3dev
```

## Rerunning the experiments

Every replication runs a selection stage and then an evaluation stage.

1. `source` trains the pool on source episodes, fits the teachers on labelled source episodes, builds the unlabelled target sample, scores every selector and refits every decision rule. It then writes the selection record `lockNNN.json`, and `stateNNN.pt` with the trained agents.
2. `open` runs once all selection records of an experiment exist, and writes `open.json` with the time evaluation was allowed to begin and the replications it covers.
3. `target` runs only on the replications that `open.json` lists. It generates fresh target episodes, and writes `resNNN.json` and `hitsNNN.npz`.

Every stage resumes from the files in its output folder, so a rerun from scratch needs an empty folder, such as `rerun/e1`.

```
# E1, delayed cue, 192 replications
python3 -m e1.drive source --plan e1/plan2.json --out rerun/e1 --procs 4
python3 -m e1.drive open   --plan e1/plan2.json --out rerun/e1
python3 -m e1.drive target --plan e1/plan2.json --out rerun/e1 --procs 4
python3 e1/anal1.py rerun/e1

# E2, T-Maze, 32 replications
python3 -m common.drive --exp e2 source --plan e2/plan.json --out rerun/e2 --procs 4
python3 -m common.drive --exp e2 open   --plan e2/plan.json --out rerun/e2
python3 -m common.drive --exp e2 target --plan e2/plan.json --out rerun/e2 --procs 4
python3 e2/anal2.py rerun/e2

# E3, noisy T-Maze, 24 replications
python3 -m common.drive --exp e3 train  --plan e3/plan.json --out rerun/e3 --procs 4
python3 -m common.drive --exp e3 source --plan e3/plan.json --out rerun/e3 --procs 4
python3 -m common.drive --exp e3 open   --plan e3/plan.json --out rerun/e3
python3 -m common.drive --exp e3 target --plan e3/plan.json --out rerun/e3 --procs 4
python3 e3/anal3.py rerun/e3

# E4 and E5, Repeat Previous, 32 replications each. Each runner calls .venv/bin/python for
# train, source, open, target and the analysis. caffeinate stops a Mac from sleeping. On other
# systems, drop it and start the runners with bash.
nohup caffeinate -i e4/run4.sh e4/plan.json rerun/e4 16 > e4run.log 2>&1 &
nohup caffeinate -i e5/run5.sh e5/plan.json rerun/e5 16 > e5run.log 2>&1 &
```

`--part i/k` splits replications across machines, `--only` names replications, `--budget s` stops starting new agents after `s` seconds, and `RIWF_CACHE` moves the per-agent training cache.

## Pilots

Every pilot ran before its plan was fixed, on random seeds that no confirmatory replication used.
I reran the pilots of E1, E2 and E3 from their seeds and settings with `bash pilots.sh`, which takes about half an hour on one core.

- `runs/e1pilot` holds the first E1 pilot world, with seed 1001, target delays of 12, 20 and 28, and a rare-regime share of 0.10 at source and 0.25 at target. It also holds the three worlds of the second pilot, with seeds 2002, 3010 and 3011. Both pilots trained the main pool with training delays 3 to 7 and 3 to 28 only. `pilots.json` holds the settings of each world, the spread of target success after refitting, and what source validation deployed. The first pilot world used seed 1001, which `e1/plan2.json` lists as 2001.
- `runs/e2pilot` holds the twelve single agents of the E2 pilot in its three rounds. The first two rounds used the initialisation that set every layer-norm gain to zero, which `e2/pilot.py --oldinit` restores, and their transformers stayed at chance. `summary.json` holds the two pilot conditions.
- `runs/e3rocks` holds the RockSample pilot. The pilot trained one GRU of width 32 for 20 seconds (`gru_32_1_b20.txt`). The other files hold one agent of each architecture in the E3 pool, evaluated after 20, 40, 60 and 100 updates.
- `runs/e3pilot1` holds the three single-agent runs of the noisy T-Maze with seed 1. The first paid 4 for a right turn in both regimes, which `e3/pilot.py --good4` restores.
- `runs/e3pilot` holds the pilot replication of the noisy T-Maze in Appendix C, with seed 31, and `runs/e4/pilot` holds the two E4 pilot replications that `e4/pilot.json` summarises.

On the machine that built this package, rerunning E1 replication 0 from its seeds gave its selection record and its result again, identical apart from the recorded timings.

## Records

The trained agents (`stateNNN.pt`) are not included because of their size, and every replication regenerates them from the seeds in its selection record.

All 24 E3 replications ran on the virtual machine, so their records are in `runs/e3dev`.
`runs/e3` holds the plan copy and training log of the partial E3 run on the cloud machine, which was stopped and discarded before any of its selection records were saved.

`runs/amend.json` records where each replication ran, and the rule, written before any E1 or E3 evaluation stage, for which replications would be evaluated after the virtual machine disconnected.
`runs/e5/amend.json` records the first E5 launch, stopped before any E5 selection record was saved, whose logs are in `runs/e5vm`.
`runs/queue.log` is the log of the queue of E1 selection stages on the cloud machine.

`runs/e1/kcheck.json` holds the freedom lost with lookup tables by every agent's own memory on its target sample, which was zero in every E1 replication.
I computed it with `e1/kcheck.py` after the E1 evaluation stage was allowed to begin.
The script loads the trained agents from `runs/e1`, so it runs only after an E1 selection stage has saved them there.
