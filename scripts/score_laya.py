"""Score convaiinnovations/laya on a frozen Chimera suite via kev.benchmark rows.

Maps each suite record to laya Router.predict(state, questions) and converts
answers back to kev probability dicts keyed by kev.api.question_keys:
  choice: Laya probabilities used directly (keyed by option key).
  noul:   Laya `noul` = P(yes); split over the two noul keys (index 1 = yes).
  score:  Laya probabilities keyed "0".."N-1" mapped positionally to kev keys.
Out-of-box settings (no head-budget tuning); see reports/01-phase1.md.
"""
import sys
import time

from kev.api import question_keys
from kev.benchmark import evaluate_records
from kev.suite import load_split
from laya import Router


def to_probs(answers, record):
    probs = {}
    for qid, q in record["questions"].items():
        keys = question_keys(q["type"], q.get("criteria"))
        a = answers[qid]
        if q["type"] == "choice":
            probs[qid] = {k: float(a["probabilities"][k]) for k in keys}
        elif q["type"] == "noul":
            p = float(a["noul"])
            probs[qid] = {keys[0]: 1.0 - p, keys[1]: p}
        elif q["type"] == "score":
            raw = a["probabilities"]
            probs[qid] = {k: float(raw[str(i)]) for i, k in enumerate(keys)}
        else:
            raise ValueError(f"unknown type {q['type']}")
    return probs


def main():
    suite, out = sys.argv[1], sys.argv[2]
    split = sys.argv[3] if len(sys.argv) > 3 else "development"
    router = Router(device="cuda")
    records = load_split(suite, split)
    print(f"scoring {len(records)} records from {suite}/{split}", flush=True)

    def predictor(record):
        qs = {qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
              for qid, q in record["questions"].items()}
        t = time.perf_counter()
        res = router.predict(record["state"], qs)
        ms = (time.perf_counter() - t) * 1000
        return {"probabilities": to_probs(res["answers"], record), "latency_ms": ms}

    report, _ = evaluate_records(records, predictor, out)
    c = report["clean"]
    print(f"clean acc {c['acc']:.4f} nll {c['nll']:.4f} ece {c['ece']:.4f} brier {c['brier']:.4f} aurc {c['aurc']:.4f}", flush=True)
    print("per-task:", {k: round(v["acc"], 3) for k, v in sorted(report["tasks"].items())}, flush=True)


if __name__ == "__main__":
    main()
