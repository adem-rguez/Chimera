#!/bin/bash
# re-assemble + validate every authored Stage A batch; prints one line per batch
cd "$(dirname "$0")/.."
for f in oracle/phase7r-authored/stageA-[0-9][0-9][0-9].jsonl; do
  n=$(basename "$f" .jsonl | sed 's/stageA-//')
  python scripts/build_phase7r.py assemble --batch oracle/phase7r-batches/stageA-$n.json --authored "$f" --out oracle/phase7r-built/stageA-$n.jsonl >/dev/null 2>&1 || { echo "$n ASSEMBLE-ERROR"; continue; }
  r=$(python scripts/build_phase7r.py validate --rows oracle/phase7r-built/stageA-$n.jsonl 2>&1 | grep -E "FAILED|passed" | head -1)
  echo "$n $r"
done
