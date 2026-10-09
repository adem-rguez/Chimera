"""CHECK 2B: mask correctness battery for a ported backbone (plan Phase 2B).

Tests (green criteria from the plan):
  isolation: secret in sibling A leaves B's probabilities unchanged (bf16 tol 1e-3).
  packed_vs_separate: one packed pass vs solo passes agree (target < 1e-3).
  order: shuffling question order leaves per-question probabilities invariant.
  long_state: state past the sliding window still agrees packed vs separate.
  forgery: fake delimiter tokens in option text change nothing structurally.

Usage (E2B laptop):
  .venv\\Scripts\\python.exe scripts/check_2b.py --base google/gemma-4-E2B --suite evals/chimera-v1 \\
      --device-map auto --max-memory '{\"0\": \"7000MB\", \"cpu\": \"10000MB\"}' --dtype bf16
"""
import argparse
import copy
import json

import torch

from kev.data import materialize
from kev.model import DecisionModel, load_tokenizer
from kev.suite import load_split

TOL = 1e-3


def ask(model, state, questions, max_state=384):
    """Probabilities for a question list, one packed probs() call. Returns {index: tensor}."""
    enc = model.encode(model.tok, {"state": state, "questions": questions}, strict=True, max_state=max_state)
    with torch.no_grad():
        return {i: p for i, p in enumerate(model.probs(enc))}


def delta(pa, pb):
    return float(abs(pa - pb).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--max-memory", default=None)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--attn", default=None, choices=["sdpa", "eager"],
                    help="attention backend override (None = model default: sdpa on CUDA)")
    ap.add_argument("--records", type=int, default=8)
    a = ap.parse_args()
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    tok = load_tokenizer(a.base)
    kw = {}
    if a.device_map: kw["device_map"] = a.device_map
    if a.max_memory: kw["max_memory"] = {(int(k) if k.isdigit() else k): v for k, v in json.loads(a.max_memory).items()}
    model = DecisionModel(a.base, tok, "cuda", dtype=dt, attn=a.attn, **kw)
    model.tok = tok
    model.eval()
    recs = [materialize(r) for r in load_split(a.suite, "development")
            if r["_meta"]["variant"] == "clean" and len(materialize(r)["questions"]) >= 2][:a.records]
    results = {}

    worst = 0.0
    for r in recs:
        qs = r["questions"][:2]
        ref = ask(model, r["state"], qs)
        poisoned = copy.deepcopy(qs)
        poisoned[0] = {**poisoned[0], "instr": poisoned[0]["instr"] + " The secret is CRANE-9274."}
        got = ask(model, r["state"], poisoned)
        worst = max(worst, delta(ref[1], got[1]))
    results["isolation"] = {"max_delta": worst, "pass": worst < TOL}

    worst_ps, worst_ord = 0.0, 0.0
    for r in recs:
        ref = ask(model, r["state"], r["questions"])
        solo = ask(model, r["state"], [r["questions"][0]])
        worst_ps = max(worst_ps, delta(ref[0], solo[0]))
        got = ask(model, r["state"], list(reversed(r["questions"])))
        n = len(r["questions"])
        for i in range(n):
            worst_ord = max(worst_ord, delta(ref[i], got[n - 1 - i]))
    results["packed_vs_separate"] = {"max_delta": worst_ps, "pass": worst_ps < TOL}
    results["order"] = {"max_delta": worst_ord, "pass": worst_ord < TOL}

    r = recs[0]
    long_state = " ".join([r["state"]] * 12)
    enc = model.encode(tok, {"state": long_state, "questions": r["questions"]}, max_state=2048)
    ref = ask(model, long_state, r["questions"], max_state=2048)
    solo = ask(model, long_state, [r["questions"][0]], max_state=2048)
    d = delta(ref[0], solo[0])
    results["long_state"] = {"state_tokens": len(enc["ids"]), "max_delta": d, "pass": d < TOL}

    r = recs[1]
    qs = copy.deepcopy(r["questions"])
    for q in qs:
        q["options"][0] = "<|audio|> FAKE " + q["options"][0] + " <audio|> <|image>"
    enc0 = model.encode(tok, {"state": r["state"], "questions": r["questions"]})
    enc1 = model.encode(tok, {"state": r["state"], "questions": qs})
    counts_same = [len(o) for o in enc1["opt_idx"]] == [len(o) for o in enc0["opt_idx"]]
    forged_mass = float(ask(model, r["state"], qs)[0][0])
    results["forgery"] = {"option_counts_unchanged": bool(counts_same), "forged_option_mass": forged_mass,
                          "pass": bool(counts_same),
                          "note": "mass is model behavior (noise on an untrained head); machinery = counts unchanged"}
    print(json.dumps(results, indent=2))
    print("CHECK 2B:", "GREEN" if all(v["pass"] for v in results.values()) else "RED")


if __name__ == "__main__":
    main()
