"""Untreated-base flip baseline on the SAME battery (closes CHECK 8).

GPU-box script, zero_flip.py pattern: strict base load (no adapter), letter-logit
readout. For each attacked state in oracle/flip-battery.jsonl (plus its clean state),
scores every CHOICE question; noul/score questions are excluded (no defined zero-shot
rule — stated, not silently dropped: see `skipped` count).
Output oracle/base-flip-results.jsonl: {id,qid,addition,argmax,clean,gold} per question,
comparable row-for-row with oracle/flip-results.jsonl.

Usage (GPU box):
  python scripts/base_flip.py --base google/gemma-4-E4B \\
      --revision 411aa17b749aa952df1359d2dcea73917a544d9a --out oracle/base-flip-results.jsonl
"""
import argparse
import json

import torch

from kev.anchors import LETTERS, question_prompt
from kev.model import DecisionModel, load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--battery", default="oracle/flip-battery.jsonl")
    ap.add_argument("--suite", default="evals/v7/decision-v7")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--batch", type=int, default=16)
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
            dtype=dt, attn_implementation="eager").to(a.device).eval()
    else:
        from transformers import AutoModelForCausalLM
        full = DecisionModel._strict_from_pretrained(
            AutoModelForCausalLM, a.base, a.revision,
            dtype=dt, attn_implementation="eager").to(a.device).eval()
    letter_ids = [tok.encode(" " + L, add_special_tokens=False)[0] for L in LETTERS]

    gold = {}
    with open(f"{a.suite}/development.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                gold[f"dev-{i}"] = json.loads(line)
    attacks = [json.loads(l) for l in open(a.battery, encoding="utf-8") if l.strip()]

    jobs = []  # (id, qid, addition, prompt, gold_label)
    skipped = 0
    seen_states = {}
    for at in attacks:
        g = gold[at["id"]]
        for qid, q in g["questions"].items():
            if q.get("type") != "choice" or len(q.get("criteria", {})) > len(letter_ids):
                skipped += 1
                continue
            jobs.append((at["id"], qid, at["addition"], at["state"], q, q["label"]))
        key = (at["id"], "__clean__")
        if key not in seen_states:
            seen_states[key] = True
            g = gold[at["id"]]
            for qid, q in g["questions"].items():
                if q.get("type") != "choice" or len(q.get("criteria", {})) > len(letter_ids):
                    continue
                jobs.append((at["id"], qid, "__clean__", g["state"], q, q["label"]))

    prompts = [(question_prompt(st, qq)) for (_, _, _, st, qq, _) in jobs]
    # NOTE: per-job keys differ; the scoring loop below uses each job's own criteria.
    # eager, not sdpa: the sliding-window mask leak (0.78 under sdpa) would contaminate
    # exactly the state-context sensitivity this battery measures.
    out_rows = []
    clean = {}
    with torch.no_grad():
        for i in range(0, len(prompts), a.batch):
            chunk_jobs = jobs[i:i + a.batch]
            chunk = prompts[i:i + a.batch]
            enc = tok([p for p, _ in chunk], return_tensors="pt",
                      padding=True, truncation=True, max_length=3072).to(a.device)
            logits = full(**enc).logits[:, -1].float()
            for (rid, qid, add, st, qq, gl), row in zip(chunk_jobs, logits):
                keys = list(qq["criteria"])
                s = torch.softmax(row[letter_ids[:len(keys)]], -1)
                best = keys[int(s.argmax())]
                if add == "__clean__":
                    clean[(rid, qid)] = best
                else:
                    out_rows.append({"id": rid, "qid": qid, "addition": add,
                                     "argmax": best, "gold": gl})
            print(f"base_flip {min(i + a.batch, len(prompts))}/{len(prompts)}", flush=True)
    for r in out_rows:
        r["clean"] = clean.get((r["id"], r["qid"]))
        r["flip"] = r["argmax"] != r["clean"]
        r["wrong"] = r["argmax"] != r["gold"]
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    n = len(out_rows)
    print(json.dumps({"questions": n, "skipped_non_choice": skipped,
                      "flip_rate": sum(r["flip"] for r in out_rows) / n,
                      "wrong_rate": sum(r["wrong"] for r in out_rows) / n}, indent=2))


if __name__ == "__main__":
    main()
