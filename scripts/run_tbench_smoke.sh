#!/usr/bin/env bash
# Terminal-Bench smoke test of base Gemma-4-E4B-it against 5 chosen tasks.
# Waits for GPU to be free (humaneval job done + <1GB VRAM used), then runs
# the real model server (thinking on), runs 5 tasks via tb/terminus-2, stops
# the server, and (if time permits, via --with-thinking-off) repeats with
# thinking off. Writes a summary + .done marker.
#
# Usage:
#   nohup bash ~/Chimera/scripts/run_tbench_smoke.sh [--with-thinking-off] \
#       > ~/Chimera/oracle/tbench-smoke.log 2>&1 &
#
# Do NOT launch automatically; this is meant to be started explicitly.

set -uo pipefail

REPO=~/Chimera
TBENCH_VENV=~/tbench/.venv
SERVER=$REPO/scripts/tb_gemma_server.py
PORT=8099
API_BASE="http://127.0.0.1:${PORT}/v1"
MODEL_NAME="google/gemma-4-E4B-it"   # matches scripts/gen_think_traces.py / humaneval_probe.py load path
LOG=$REPO/oracle/tbench-smoke.log
SUMMARY=$REPO/oracle/tbench-smoke.md
DONE_FLAG=$REPO/oracle/tbench-smoke.done
WAIT_FLAG=$REPO/oracle/humaneval-planned-full.done
RUN_ROOT=/tmp/tbench-smoke-runs
WITH_THINKING_OFF=0

for arg in "$@"; do
  case "$arg" in
    --with-thinking-off) WITH_THINKING_OFF=1 ;;
  esac
done

TASKS=(hello-world fix-permissions grid-pattern-transform extract-safely git-workflow-hack)

log_step() {
  echo "=== STEP: $1 ($(date -u +%FT%TZ)) ===" | tee -a "$LOG"
}

# ---------------------------------------------------------------------------
log_step "waiting for GPU to be free"
while true; do
  if [ -f "$WAIT_FLAG" ]; then
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "${USED:-9999}" -lt 1024 ]; then
      break
    fi
  fi
  sleep 20
done
log_step "GPU free, proceeding"

mkdir -p "$RUN_ROOT"

run_server() {
  local thinking=$1
  local logfile=$2
  source "$REPO/.venv/bin/activate"
  nohup python "$SERVER" \
    --model "$MODEL_NAME" \
    --port "$PORT" \
    --thinking "$thinking" \
    --log-file "$logfile" \
    > "$RUN_ROOT/server-${thinking}.out" 2>&1 &
  echo $!
}

wait_for_server() {
  for i in $(seq 1 120); do
    if curl -s "$API_BASE/models" > /dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  return 1
}

run_tb_suite() {
  local thinking=$1
  local runid="smoke-${thinking}"
  local outdir="$RUN_ROOT/${runid}"
  log_step "starting server (thinking=$thinking)"
  local server_log="$REPO/oracle/tbench-server-${thinking}.jsonl"
  SERVER_PID=$(run_server "$thinking" "$server_log")
  log_step "server pid=$SERVER_PID, waiting for readiness"
  if ! wait_for_server; then
    log_step "server failed to become ready (thinking=$thinking); aborting this arm"
    kill "$SERVER_PID" 2>/dev/null
    return 1
  fi

  log_step "running tb suite (thinking=$thinking)"
  source "$TBENCH_VENV/bin/activate"
  export OPENAI_API_KEY=dummy
  local task_args=()
  for t in "${TASKS[@]}"; do
    task_args+=(-t "$t")
  done
  tb run --agent terminus-2 --model openai/gemma-4-e4b-it \
    -k api_base="$API_BASE" -k api_key=dummy \
    --dataset terminal-bench-core==0.1.1 \
    "${task_args[@]}" --n-concurrent 2 \
    --output-path "$outdir" --run-id "$runid" --no-rebuild \
    >> "$LOG" 2>&1

  log_step "stopping server (thinking=$thinking)"
  kill "$SERVER_PID" 2>/dev/null
  sleep 2

  echo "$outdir/$runid/results.json|$server_log"
}

# ---------------------------------------------------------------------------
log_step "run 1: thinking=on"
RESULT_ON=$(run_tb_suite on)
RESULTS_JSON_ON="${RESULT_ON%%|*}"
SERVER_LOG_ON="${RESULT_ON##*|}"

RESULTS_JSON_OFF=""
SERVER_LOG_OFF=""
if [ "$WITH_THINKING_OFF" -eq 1 ]; then
  log_step "run 2: thinking=off"
  RESULT_OFF=$(run_tb_suite off)
  RESULTS_JSON_OFF="${RESULT_OFF%%|*}"
  SERVER_LOG_OFF="${RESULT_OFF##*|}"
fi

# ---------------------------------------------------------------------------
log_step "writing summary"

{
  echo "# Terminal-Bench smoke test: base Gemma-4-E4B-it"
  echo
  echo "Generated $(date -u +%FT%TZ)"
  echo
  echo "## Tasks"
  for t in "${TASKS[@]}"; do echo "- $t"; done
  echo
  python3 - "$RESULTS_JSON_ON" "$SERVER_LOG_ON" "$RESULTS_JSON_OFF" "$SERVER_LOG_OFF" <<'PYEOF'
import json, sys

def load_results(path):
    if not path:
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

def token_totals(server_log):
    if not server_log:
        return None
    import os
    if not os.path.exists(server_log):
        return None
    prompt = completion = thinking = 0
    n = 0
    for line in open(server_log):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        prompt += rec.get("prompt_tokens", 0)
        completion += rec.get("completion_tokens", 0)
        thinking += rec.get("thinking_tokens", 0)
        n += 1
    return {"requests": n, "prompt_tokens": prompt, "completion_tokens": completion, "thinking_tokens": thinking}

def render(label, results_path, server_log):
    res = load_results(results_path)
    tok = token_totals(server_log)
    print(f"## {label}")
    print()
    if res is None:
        print("(not run)")
        print()
        return
    if "error" in res and "results" not in res:
        print(f"Error loading results: {res['error']}")
        print()
        return
    print("| task | resolved | input_tokens | output_tokens | wall_time_s |")
    print("|---|---|---|---|---|")
    for r in res.get("results", []):
        start = r.get("trial_started_at")
        end = r.get("trial_ended_at")
        wall = ""
        try:
            from datetime import datetime
            s = datetime.fromisoformat(start)
            e = datetime.fromisoformat(end)
            wall = round((e - s).total_seconds(), 1)
        except Exception:
            pass
        print(f"| {r.get('task_id')} | {r.get('is_resolved')} | {r.get('total_input_tokens')} | {r.get('total_output_tokens')} | {wall} |")
    print()
    print(f"Accuracy: {res.get('accuracy')}  (resolved {res.get('n_resolved')}/{res.get('n_resolved',0)+res.get('n_unresolved',0)})")
    print()
    if tok:
        print(f"Server-side token totals across {tok['requests']} requests: "
              f"prompt={tok['prompt_tokens']}, completion={tok['completion_tokens']}, "
              f"thinking={tok['thinking_tokens']}")
        print()

render("thinking=on", sys.argv[1] if len(sys.argv) > 1 else None, sys.argv[2] if len(sys.argv) > 2 else None)
if len(sys.argv) > 3 and sys.argv[3]:
    render("thinking=off", sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
PYEOF
} > "$SUMMARY"

touch "$DONE_FLAG"
log_step "done; summary at $SUMMARY"
