"""Zero-shot flip baseline: the probe's any-of-N definition against the untrained base.

Same letter-logit readout as scripts/zero_shot.py (strict backbone load, no adapter):
each sampled choice question runs once unshuffled plus n_perm shuffled option orders;
flip = any shuffled argmax (by option key) differs from the unshuffled one. Choice only.

Usage (GPU box):
  python scripts/zero_flip.py --base google/gemma-4-E4B --suite evals/v7/decision-v7 \\
      --n_questions 300 --n_perm 16 --out runs/zero-flip-e4b-dev.json
"""
import argparse
import json
import random

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
    ap.add_argument("--n_questions", type=int, default=300)
    ap.add_argument("--n_perm", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--batch", type=int, default=8,
                    help="prompts per forward pass; the logits tensor scales with batch x length x vocab, "
                         "so long suites OOM at 16 on a 24GB GPU (transfer-v4 did)")
    a = ap.parse_args()

    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    tok = load_tokenizer(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
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
    letter_ids = [tok.encode(" " + L, add_special_tokens=False)[0] for L in LETTERS]

    pool = [(r["state"], q) for r in load_split(a.suite, a.split)
            for q in r["questions"].values() if q.get("type") == "choice"
            and len(q.get("criteria", {})) <= len(letter_ids)]
    rng = random.Random(a.seed); rng.shuffle(pool)
    pool = pool[:a.n_questions]

    prompts = []
    for qi, (state, q) in enumerate(pool):
        keys = list(q["criteria"])
        for i in range(a.n_perm + 1):
            order = keys if i == 0 else rng.sample(keys, len(keys))
            qq = {**q, "criteria": {k: q["criteria"][k] for k in order}}
            prompts.append(question_prompt(state, qq))

    wins, spreads = [], []
    with torch.no_grad():
        probs = []
        for i in range(0, len(prompts), a.batch):
            chunk = prompts[i:i + a.batch]
            enc = tok([p for p, _ in chunk], return_tensors="pt",
                      padding=True, truncation=True, max_length=3072).to(a.device)
            logits = full(**enc).logits[:, -1].float()
            for (p, keys), row in zip(chunk, logits):
                s = torch.softmax(row[letter_ids[:len(keys)]], -1)
                best = keys[int(s.argmax())]
                probs.append((best, {k: float(x) for k, x in zip(keys, s.tolist())}))
            if (i // a.batch) % 20 == 0:
                print(f"zero_flip {i}/{len(prompts)}", flush=True)
    for qi in range(len(pool)):
        runs = probs[qi * (a.n_perm + 1):(qi + 1) * (a.n_perm + 1)]
        base = runs[0][0]
        wins.append(int(any(w != base for w, _ in runs[1:])))
        by_key = {}
        for w, d in runs:
            for k, v in d.items():
                by_key.setdefault(k, []).append(v)
        spreads.append(max(max(v) - min(v) for v in by_key.values()))
    out = {"base": a.base, "revision": a.revision, "suite": a.suite, "split": a.split,
           "n_questions": len(pool), "n_perm": a.n_perm, "seed": a.seed,
           "flip_rate": sum(wins) / len(wins), "mean_spread": sum(spreads) / len(spreads)}
    write_json(a.out, out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
