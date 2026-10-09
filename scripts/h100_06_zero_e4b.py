"""Step 6: E4B zero-shot dev read (thin wrapper over scripts/zero_shot.py).

zero_shot.py uses DecisionModel._load_backbone, NOT kev.anchors (which would
random-init Gemma-4's text weights). Output: runs/h100-zero-e4b.json (with acc).

Usage: python scripts/h100_06_zero_e4b.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h100_common import E4B, E4B_REV, PY, SESS, SUITE, mark, mem_snapshot, run  # noqa: E402


def main():
    rc = run([PY, "scripts/zero_shot.py", "--base", E4B, "--revision", E4B_REV,
              "--suite", SUITE, "--split", "development",
              "--out", "runs/h100-zero-e4b.json", "--device", "cuda", "--dtype", "bf16"],
             str(SESS / "zero-e4b.log"))
    mark("zero-e4b", {"rc": rc, "mem_after": mem_snapshot()})
    if rc != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
