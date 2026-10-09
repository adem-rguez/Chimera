"""Step 5: E2B short train bf16 (full suite) + dev read. Clean bf16 verdict on real weights.

Usage: python scripts/h100_05_short_bf16.py [--batch B] [--steps 500]
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h100_common import E2B, E2B_REV, PY, SESS, SUITE, load_batch, mark, mem_snapshot, run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--steps", type=int, default=500)
    a = ap.parse_args()
    B = a.batch or load_batch()
    out = "runs/h100-short-e2b-bf16"
    shutil.rmtree(out, ignore_errors=True)
    shutil.rmtree(out + "-eval", ignore_errors=True)
    rc = run([PY, "-m", "kev.train", "--base", E2B, "--base_revision", E2B_REV,
              "--suite", SUITE, "--max_steps", str(a.steps), "--batch", str(B),
              "--device", "cuda", "--dtype", "bf16", "--attn", "eager", "--out", out],
             str(SESS / "short-bf16-train.log"))
    if rc == 0:
        run([PY, "-m", "kev.benchmark", "--run", out, "--suite", SUITE,
             "--out", "runs/h100-short-e2b-bf16-eval"], str(SESS / "short-bf16-eval.log"),
            env={"KEV_ATTN": "eager"})
    mark("short-bf16", {"train_rc": rc, "mem_after": mem_snapshot()})


if __name__ == "__main__":
    main()
