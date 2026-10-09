"""Shared constants + helpers for the h100_* step scripts. Not run directly."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

E2B = "google/gemma-4-E2B"
E2B_REV = "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f"
E4B = "google/gemma-4-E4B"
E4B_REV = "411aa17b749aa952df1359d2dcea73917a544d9a"
SUITE = "evals/chimera-v1"
SESS = Path("runs/h100-session")
PY = sys.executable


def run(cmd, log, env=None):
    print(f"$ {' '.join(cmd)}  [log: {log}]", flush=True)
    t0 = time.time()
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                           env={**os.environ, **env} if env else None)
    print(f"exit={r.returncode} ({time.time() - t0:.0f}s)", flush=True)
    return r.returncode


def mem_snapshot():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                               "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi failed: {e}"


def load_batch(default=4):
    """Batch preset from 00_env's env.json (8 H200 / 4 H100 / 2 other)."""
    try:
        return int(json.loads((SESS / "env.json").read_text(encoding="utf-8")).get("preset_batch", default))
    except Exception:  # noqa: BLE001
        return default


def mark(name, info=None):
    SESS.mkdir(parents=True, exist_ok=True)
    (SESS / f"step-{name}.done").write_text(json.dumps({"t": time.time(), **(info or {})}), encoding="utf-8")
