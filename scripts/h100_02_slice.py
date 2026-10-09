"""Step 2: build the 32-record overfit slice -> runs/h100-session/overfit32.jsonl.

Suite records are already in load_records request shape (state str + questions
with labels), so the slice is the first 32 train lines, byte-identical.

Usage: python scripts/h100_02_slice.py [--n 32]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h100_common import SESS, SUITE, mark  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    a = ap.parse_args()
    lines = Path(SUITE, "train.jsonl").read_text(encoding="utf-8").splitlines()
    sl = SESS / "overfit32.jsonl"
    sl.write_text("\n".join(lines[:a.n]) + "\n", encoding="utf-8")
    print(f"slice: {sl} ({min(a.n, len(lines))} records)", flush=True)
    mark("slice", {"n": min(a.n, len(lines))})


if __name__ == "__main__":
    main()
