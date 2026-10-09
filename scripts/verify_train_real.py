"""Real masking test: train_phase7.encode_call against the true E4B-it tokenizer."""
import json
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")

from kev.model import load_tokenizer

tok = load_tokenizer("google/gemma-4-E4B-it")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

import importlib.util
spec = importlib.util.spec_from_file_location("t7", "scripts/train_phase7.py")
t7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t7)

L = [json.loads(l) for l in open("oracle/phase7-calls.jsonl", encoding="utf-8")][:50]
bad = 0
for r in L:
    enc = t7.encode_call(tok, r)
    ids, labs = enc["input_ids"], enc["labels"]
    s, e = r["masked_span"]
    offs = tok(r["target_text"], return_offsets_mapping=True,
                truncation=True, max_length=2048)["offset_mapping"]
    # every token overlapping the label span must be masked
    for i, (a, b) in enumerate(offs):
        if a < e and b > s and labs[i].item() != -100:
            bad += 1
    # no token OUTSIDE prompt+span may be masked (agent format must train)
    agent_at = r["target_text"].index("Agent: ") + len("Agent: ")
    for i, (a, b) in enumerate(offs):
        if b > agent_at and not (a < e and b > s) and labs[i].item() == -100:
            bad += 1
print("rows:", len(L), "masking violations:", bad)
