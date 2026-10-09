"""Isolation-leak discriminator (H200 E2B RED follow-up).

Distinguishes:
  (a) causal directional leak (mask dropped on sliding layers): secret in Q0
      moves Q1 a lot, secret in Q1 leaves Q0 alone;
  (b) readout/position bug: both directions move;
  (c) kernel bug: --attn eager goes GREEN while sdpa stays RED on one box.

Also prints layer attention types (does this base even have sliding layers?),
encode spans (alignment check), and library versions.

Usage (cloud box):
  python scripts/debug_iso.py --base google/gemma-4-E2B --suite evals/chimera-v1 --dtype fp32
  python scripts/debug_iso.py --base google/gemma-4-E2B --suite evals/chimera-v1 --dtype fp32 --attn eager
"""
import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kev.data import materialize  # noqa: E402
from kev.model import DecisionModel, load_tokenizer  # noqa: E402
from kev.suite import load_split  # noqa: E402


def ask(model, state, questions, max_state=384):
    enc = model.encode(model.tok, {"state": state, "questions": questions}, strict=True, max_state=max_state)
    with torch.no_grad():
        return {i: p for i, p in enumerate(model.probs(enc))}


def delta(pa, pb):
    return float(abs(pa - pb).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--dtype", default="fp32", choices=["bf16", "fp32"])
    ap.add_argument("--attn", default=None, help="override attention impl (e.g. eager)")
    ap.add_argument("--records", type=int, default=2)
    a = ap.parse_args()

    print(json.dumps({"transformers": transformers.__version__, "torch": torch.__version__,
                      "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}), flush=True)
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    tok = load_tokenizer(a.base)
    kw = {"attn": a.attn} if a.attn else {}
    model = DecisionModel(a.base, tok, "cuda", dtype=dt, **kw)
    model.tok = tok
    model.eval()

    attn_kinds = Counter(type(m).__name__ for m in model.lm.modules() if "ttention" in type(m).__name__)
    print(json.dumps({"layer_attention_modules": dict(attn_kinds),
                      "config_sliding_window": getattr(model.lm.config, "sliding_window", None),
                      "config_layer_types": getattr(model.lm.config, "layer_types", None)}), flush=True)

    recs = [materialize(r) for r in load_split(a.suite, "development")
            if r["_meta"]["variant"] == "clean" and len(materialize(r)["questions"]) >= 2][:a.records]
    out = {}
    for ri, r in enumerate(recs):
        qs = r["questions"][:2]
        ref = ask(model, r["state"], qs)
        enc0 = model.encode(tok, {"state": r["state"], "questions": qs}, strict=True)
        p0 = copy.deepcopy(qs)
        p0[0] = {**p0[0], "instr": p0[0]["instr"] + " The secret is CRANE-9274."}
        got0 = ask(model, r["state"], p0)
        enc1 = model.encode(tok, {"state": r["state"], "questions": p0}, strict=True)
        p1 = copy.deepcopy(qs)
        p1[1] = {**p1[1], "instr": p1[1]["instr"] + " The secret is CRANE-9274."}
        got1 = ask(model, r["state"], p1)
        ps = copy.deepcopy(qs)
        gotS = ask(model, r["state"] + " The secret is CRANE-9274.", qs)
        out[f"rec{ri}"] = {
            "secret_in_Q0__dQ1": delta(ref[1], got0[1]),
            "secret_in_Q1__dQ0": delta(ref[0], got1[0]),
            "secret_in_state__dQ0": delta(ref[0], gotS[0]),
            "self_Q0": delta(ref[0], got0[0]),
            "ids_len_ref": len(enc0["ids"]), "ids_len_poisoned": len(enc1["ids"]),
            "opt_lens_ref": [len(o) for o in enc0["opt_idx"]],
            "opt_lens_poisoned": [len(o) for o in enc1["opt_idx"]],
        }
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
