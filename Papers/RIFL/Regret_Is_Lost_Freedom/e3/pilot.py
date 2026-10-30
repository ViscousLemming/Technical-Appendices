"""Noisy T-Maze pilot runs of E3 with seed 1, rerun. Each call trains one agent with e3/t_train.py
and saves what it prints.

  python3 e3/pilot.py OUT [--good4] ARCH SIZE SEED UPDATES [LR ENT SHARE]

--good4 restores the first version of the environment, which paid 4 for a right turn in both regimes.
The commands, in order:

  python3 e3/pilot.py OUT --good4 gru 32 1 300
  python3 e3/pilot.py OUT gru 32 1 300 0.001 0.01 0.1
  python3 e3/pilot.py OUT gru 32 1 300 0.001 0.01 0.5
"""
import contextlib
import io
import json
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    args = sys.argv[1:]
    out = Path(args.pop(0))
    old = "--good4" in args
    args = [x for x in args if x != "--good4"]
    if old:
        import e3.nmaze
        e3.nmaze.GOOD = (4.0, 4.0)
    out.mkdir(parents=True, exist_ok=True)
    name = "_".join(["good4" if old else "good48"] + args)
    buf = io.StringIO()
    t = time.time()
    sys.argv = [str(ROOT / "e3" / "t_train.py")] + args
    with contextlib.redirect_stdout(buf):
        runpy.run_path(str(ROOT / "e3" / "t_train.py"), run_name="__main__")
    text = buf.getvalue()
    print(text, end="")
    (out / f"{name}.txt").write_text(text)
    (out / f"{name}.json").write_text(json.dumps({"args": args, "reward_right_turn": [4.0, 4.0] if old else [4.0, 8.0],
                                                  "seconds": round(time.time() - t, 1), "printed": text.strip()}, indent=1))


if __name__ == "__main__":
    main()
