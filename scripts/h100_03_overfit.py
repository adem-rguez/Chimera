"""Step 3: E2B overfit on the slice (20 epochs, fp32) + benchmark on the slice.

Trainability proof: exit nonzero unless slice acc >= 0.95. Do not proceed on failure.

Usage: python scripts/h100_03_overfit.py [--batch B] [--epochs 20]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h100_common import E2B, E2B_REV, PY, SESS, load_batch, mark, mem_snapshot, run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="recipe lr: hot lrs (1e-3) memorize 1 example but thrash on 32")
    ap.add_argument("--head_lr", type=float, default=0.0,
                    help="0 = same as --lr (recipe); hot head lrs were tried and failed on 32")
    a = ap.parse_args()
    B = a.batch or load_batch()
    sl = str(SESS / "overfit32.jsonl")
    out = "runs/h100-overfit-e2b"
    shutil.rmtree(out, ignore_errors=True)
    shutil.rmtree(out + "-eval", ignore_errors=True)
    rc = run([PY, "-m", "kev.train", "--base", E2B, "--base_revision", E2B_REV,
              "--data", sl, "--epochs", str(a.epochs), "--batch", str(B), "--accum", "1",
              "--lr", str(a.lr), "--head_lr", str(a.head_lr),
              "--device", "cuda", "--dtype", "fp32", "--attn", "eager", "--out", out],
             str(SESS / "overfit-train.log"))
    if rc != 0:
        print("STOP: overfit train crashed. Paste the log back.", flush=True)
        sys.exit(1)
    rc = run([PY, "-m", "kev.benchmark", "--run", out, "--data", sl,
              "--out", "runs/h100-overfit-e2b-eval"], str(SESS / "overfit-eval.log"),
             env={"KEV_ATTN": "eager"})
    rep = json.loads(Path("runs/h100-overfit-e2b-eval/report.json").read_text(encoding="utf-8")) if rc == 0 else {}
    clean = rep.get("clean", {})
    acc = clean.get("acc", clean.get("accuracy", rep.get("accuracy", rep.get("acc"))))
    print(f"OVERFIT acc on slice: {acc}", flush=True)
    mark("overfit", {"acc": acc, "mem_after": mem_snapshot()})
    if acc is None or acc < 0.95:
        print("STOP: overfit did not memorize (acc<0.95). Do not proceed. Paste logs back.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
