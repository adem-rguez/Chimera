#!/bin/bash
# One chained nohup run for scripts/recursive_probe.py, resident-swap mode: ONE process per organ arm
# (kev, then head), each loading Gemma and its chooser ONCE from disk onto CPU and swapping them on/off
# the GPU every round instead of reloading from disk 24 times -- see recursive_probe.py's resident-arm
# section header for why this needs the kev chooser's device-attribute fix and can't be a bare .to().
# "first"/"gemma_self" piggyback on the kev process's Gemma phases (no extra model, resolved inline).
# Arms run sequentially (kev's process exits -- freeing its ~25GB CPU residency -- before head's starts),
# so only one arm's Gemma+chooser pair is ever resident at once, same RAM footprint as either arm alone.
# If resident-arm reports exit code 2 (psutil found insufficient RAM for Gemma+chooser resident on CPU),
# falls back automatically to the old reload-per-round path for that arm.
set -e
cd "$(dirname "$0")/.."
OUT_DIR=oracle
LOG=oracle/recursive-run.log
: > "$LOG"

echo "=== STEP start $(date)" | tee -a "$LOG"

run_reload_fallback() {
  # old reload-per-round path (24 disk loads), used only if resident-arm exits 2 (insufficient RAM).
  local arms="$1" organ_arm="$2"
  for r in 1 2 3 4 5 6 7 8; do
    echo "=== STEP round $r: gemma-round ($arms) [fallback]" | tee -a "$LOG"
    .venv/bin/python -u scripts/recursive_probe.py gemma-round --round "$r" \
        --arms "$arms" --problems oracle/s1-train-problems.jsonl \
        --out-dir "$OUT_DIR" --batch-size 16 --max-new-tokens 90 2>&1 | tee -a "$LOG"
    if [ -n "$organ_arm" ]; then
      echo "=== STEP round $r: chooser-round $organ_arm [fallback]" | tee -a "$LOG"
      .venv/bin/python -u scripts/recursive_probe.py chooser-round --round "$r" --arm "$organ_arm" \
          --problems oracle/s1-train-problems.jsonl --out-dir "$OUT_DIR" 2>&1 | tee -a "$LOG"
    fi
  done
}

echo "=== STEP arm kev (resident, +first,gemma_self)" | tee -a "$LOG"
set +e
.venv/bin/python -u scripts/recursive_probe.py resident-arm --arms kev,first,gemma_self \
    --problems oracle/s1-train-problems.jsonl --out-dir "$OUT_DIR" --batch-size 16 \
    --max-new-tokens 90 2>&1 | tee -a "$LOG"
kev_status=${PIPESTATUS[0]}
set -e
if [ "$kev_status" -eq 2 ]; then
  echo "=== STEP arm kev: resident-arm reported insufficient RAM, falling back" | tee -a "$LOG"
  run_reload_fallback "kev,first,gemma_self" "kev"
elif [ "$kev_status" -ne 0 ]; then
  exit "$kev_status"
fi

echo "=== STEP arm head (resident)" | tee -a "$LOG"
set +e
.venv/bin/python -u scripts/recursive_probe.py resident-arm --arms head \
    --problems oracle/s1-train-problems.jsonl --out-dir "$OUT_DIR" --batch-size 16 \
    --max-new-tokens 90 2>&1 | tee -a "$LOG"
head_status=${PIPESTATUS[0]}
set -e
if [ "$head_status" -eq 2 ]; then
  echo "=== STEP arm head: resident-arm reported insufficient RAM, falling back" | tee -a "$LOG"
  run_reload_fallback "head" "head"
elif [ "$head_status" -ne 0 ]; then
  exit "$head_status"
fi

echo "=== STEP report" | tee -a "$LOG"
.venv/bin/python -u scripts/recursive_probe.py report \
    --problems oracle/s1-train-problems.jsonl --out-dir "$OUT_DIR" \
    --out-report oracle/recursive-probe.md 2>&1 | tee -a "$LOG"

echo "=== STEP done $(date)" | tee -a "$LOG"
touch oracle/recursive-run.done
