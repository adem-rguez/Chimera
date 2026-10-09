"""Step 7: E4B LoRA memory probe (20 steps at recipe batch). OOM-or-not IS the data.

Sizes the Phase 3 run. A nonzero exit is a result, not a failure: report the
log tail + mem_after back.

Usage: python scripts/h100_07_mem_e4b.py [--batch B]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h100_common import E4B, E4B_REV, PY, SESS, SUITE, load_batch, mark, mem_snapshot, run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=None)
    a = ap.parse_args()
    B = a.batch or load_batch()
    print(f"mem before: {mem_snapshot()}", flush=True)
    rc = run([PY, "-m", "kev.train", "--base", E4B, "--base_revision", E4B_REV,
              "--suite", SUITE, "--max_steps", "20", "--batch", str(B),
              "--device", "cuda", "--dtype", "bf16", "--out", "runs/h100-mem-e4b"],
             str(SESS / "mem-e4b.log"))
    print(f"mem after: {mem_snapshot()} (rc={rc})", flush=True)
    mark("mem-e4b", {"rc": rc, "batch": B, "mem_after": mem_snapshot()})


if __name__ == "__main__":
    main()
