"""Laya baseline probe: judged dev rows (or any compatible suite) through
Router(model="typed-decisions").

Reads states/questions/gold from a suite partition (one record per line),
calls Router.predict with return answers + answer_confidence, records per-question
correct/confidence + latency, plus a row-level decide_conf = min over per-question
confidences. CPU laptop, no boxes. Run with .venv-laya python.

Thread-safety: laya.Agent guards every forward pass with an internal read/write
gate (_InferenceGate, laya issue #649) -- "normal inference acquires the read
lock so multiple threads can evaluate the model concurrently without
serialization"; the write lock is only taken for a scoped GPU-OOM-to-CPU
fallback. Router.predict -> Agent.system_one goes through this gate
unconditionally (not just on CUDA), so concurrent predict() calls from a
ThreadPoolExecutor are safe. --workers therefore defaults to 4, matching
scripts/oracle_labels.py's pattern.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from laya import Router

MODEL = "typed-decisions"


def score_row(i, rec, router):
    qs = {qid: {k: v for k, v in q.items() if k not in ("label", "src")}
          for qid, q in rec["questions"].items()}
    t0 = time.perf_counter()
    try:
        res = router.predict(rec["state"], qs, model=MODEL)
        err = None
    except Exception as e:  # noqa: BLE001 - probe must not die on one row
        res, err = {}, f"{type(e).__name__}: {e}"
    ms = (time.perf_counter() - t0) * 1000
    per_q = {}
    ans = res.get("answers", res) if isinstance(res, dict) else {}
    confs = []
    for qid, q in rec["questions"].items():
        y = q["label"]
        a = ans.get(qid)
        if isinstance(a, dict) and "probabilities" in a:
            p = list(a["probabilities"].values())
            conf = max(p) if p else None  # Laya-native answer_confidence = max(p)
            aval = a.get("choice", a.get("noul", a.get("score")))
        elif isinstance(a, dict) and "noul" in a:
            conf = max(a["noul"], 1 - a["noul"])
            aval = a["noul"]
        else:
            conf, aval = None, a
        if q["type"] == "choice":
            ok = aval == y
        elif q["type"] == "noul":
            ok = bool(aval) == bool(y) if not isinstance(aval, float) else (aval >= 0.5) == bool(y)
        else:
            try:
                ok = int(round(float(aval))) == int(y)
            except (TypeError, ValueError):
                ok = False
        per_q[qid] = {"correct": ok, "answer": aval, "confidence": conf}
        confs.append(conf if conf is not None else 0)
    decide_conf = round(min(confs), 4) if confs else None
    return {"id": f"dev-{i}", "per_q": per_q, "decide_conf": decide_conf,
            "ms": round(ms, 1), "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="evals/v7/decision-v7/development.jsonl")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4,
                     help="ThreadPoolExecutor workers; see module docstring for the "
                          "thread-safety note on laya's inference gate")
    a = ap.parse_args()

    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite existing file: {a.out}")

    with open(a.data, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    if a.offset or a.limit:
        recs = recs[a.offset:(a.offset + a.limit) if a.limit else None]
    print(f"pool: {len(recs)} rows from {a.data} (offset={a.offset})")

    router = Router()
    n_q = n_ok = 0

    def task(args):
        j, rec = args
        return score_row(a.offset + j, rec, router)

    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(a.out, "w", encoding="utf-8") as f:
        for row in ex.map(task, enumerate(recs)):
            for q in row["per_q"].values():
                n_q += 1
                n_ok += q["correct"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
    acc = n_ok / n_q if n_q else 0.0
    print(f"rows {len(recs)}, questions {n_q}, correct {n_ok} ({acc:.4f})")


if __name__ == "__main__":
    main()
