"""Order-robustness probe against a live endpoint, bigger than benchmark's built-in permuted rows (n=60).

Samples choice questions from a suite split, hits POST /v1/systemone/permute with n_perm
orders each (first run unshuffled), and reports flip rate + mean spread. Needs kev.serve up.

Usage:
  python scripts/perm_probe.py --url http://127.0.0.1:8009 --suite evals/v7/decision-v7 \\
      --split development --n_questions 300 --n_perm 16 --out runs/perm-probe.json
"""
import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kev.suite import load_split  # noqa: E402


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--split", default="development")
    ap.add_argument("--n_questions", type=int, default=300)
    ap.add_argument("--n_perm", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pool = [(r["state"], qid, q) for r in load_split(a.suite, a.split)
            for qid, q in (r["questions"].items() if isinstance(r["questions"], dict)
                           else enumerate(r["questions"]))
            if (q.get("type") or q.get("qtype")) == "choice"]
    rng = random.Random(a.seed); rng.shuffle(pool)
    pool = pool[:a.n_questions]
    flips, spreads, lat = 0, [], []
    for state, qid, q in pool:
        qid = str(qid)
        body = {"request": {"state": state, "model": "kev-latest",
                            "questions": {qid: {k: q[k] for k in ("type", "instructions", "criteria") if k in q}}},
                "question": qid, "n_perm": a.n_perm, "seed": a.seed}
        resp = post(a.url.rstrip("/") + "/v1/systemone/permute", body)
        runs = resp["runs"]
        if any(x["choice"] != runs[0]["choice"] for x in runs[1:]): flips += 1
        spreads.append(max(resp["spread"].values()))
        lat.append(sum(x["latency_ms"] for x in runs))
    out = {"url": a.url, "suite": a.suite, "split": a.split, "n_questions": len(pool),
           "n_perm": a.n_perm, "seed": a.seed, "flip_rate": flips / len(pool),
           "mean_spread": sum(spreads) / len(spreads),
           "mean_ms_per_question": sum(lat) / len(lat)}
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
