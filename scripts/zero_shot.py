"""Anchors-style zero-shot baseline through the CORRECT backbone loader.

kev.anchors uses AutoModelForCausalLM, which silently random-inits Gemma-4's
text weights (all MISSING). This mirrors its letter-logit readout but loads the
text submodel via DecisionModel._load_backbone, so the numbers are real.

Output: {"_meta": {...}, "targets": {record_id: {qid: {option_key: prob}}}}
plus accuracy over questions whose label names an option key.

Usage (H100/H200, full-GPU load, no device map):
  python scripts/zero_shot.py --base google/gemma-4-E4B --revision 411aa17b749aa952df1359d2dcea73917a544d9a \
      --suite evals/chimera-v1 --split development --out runs/h100-zero-e4b.json --device cuda --dtype bf16
"""
import argparse
import json

import torch

from kev.anchors import LETTERS, question_prompt
from kev.model import DecisionModel, load_tokenizer
from kev.suite import load_split, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--split", default="development")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_options", type=int, default=26,
                    help="skip questions with more options (letter readout has 26 letters; same as kev.anchors)")
    a = ap.parse_args()

    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    tok = load_tokenizer(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Full generative model (NOT DecisionModel._load_backbone: that returns the
    # bare text tower without a vocab head, which is all the pointer head needs
    # but useless for letter-logit readout). Strict load still refuses random weights.
    # sdpa is fine here: every question is its own prompt, no packed siblings.
    from transformers import AutoConfig
    if getattr(AutoConfig.from_pretrained(a.base, revision=a.revision), "model_type", "") == "gemma4":
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
        full = DecisionModel._strict_from_pretrained(
            Gemma4ForConditionalGeneration, a.base, a.revision,
            dtype=dt, attn_implementation="sdpa").to(a.device).eval()
    else:
        from transformers import AutoModelForCausalLM
        full = DecisionModel._strict_from_pretrained(
            AutoModelForCausalLM, a.base, a.revision,
            dtype=dt, attn_implementation="sdpa").to(a.device).eval()
    run_logits = lambda enc: full(**enc).logits[:, -1].float()

    letter_ids = [tok.encode(" " + L, add_special_tokens=False)[0] for L in LETTERS]
    records = load_split(a.suite, a.split)
    by_id = {r["_meta"]["id"]: r for r in records}
    jobs, skipped = [], 0
    for r in records:
        for qid, q in r["questions"].items():
            if len(q.get("criteria", [0, 1])) <= a.max_options or q["type"] == "noul":
                jobs.append((r["_meta"]["id"], qid, *question_prompt(r["state"], q)))
            else:
                skipped += 1
    targets = {}  # skipped already counted in the job filter above
    with torch.no_grad():
        for i in range(0, len(jobs), a.batch):
            chunk = jobs[i:i + a.batch]
            enc = tok([p for _, _, p, _ in chunk], return_tensors="pt",
                      padding=True, truncation=True, max_length=3072).to(a.device)
            logits = run_logits(enc)
            for (rid, qid, _, keys), row in zip(chunk, logits):
                if len(keys) > len(letter_ids):
                    skipped += 1
                    continue
                p = torch.softmax(row[letter_ids[:len(keys)]], -1).tolist()
                targets.setdefault(rid, {})[qid] = dict(zip(keys, p))
            if (i // a.batch) % 20 == 0:
                print(f"zero_shot {i}/{len(jobs)}", flush=True)

    # Accuracy where the label names an option key (choice/noul; score skipped).
    hit, scored = 0, 0
    for rid, qs in targets.items():
        for qid, dist in qs.items():
            label = by_id[rid]["questions"][qid]["label"]
            if label in dist:
                scored += 1
                if max(dist, key=dist.get) == label:
                    hit += 1
    meta = {"base": a.base, "revision": a.revision, "suite": a.suite, "split": a.split,
            "readout": "zero-shot next-token letter logits via DecisionModel._load_backbone",
            "dtype": a.dtype, "records": len(targets),
            "questions": sum(len(v) for v in targets.values()),
            "skipped_over_26_options": skipped,
            "acc": hit / scored if scored else None, "acc_n": scored}
    write_json(a.out, {"_meta": meta, "targets": targets})
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
