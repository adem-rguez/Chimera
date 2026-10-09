"""Phase 7 call-example builder (offline: datasets only, no torch).

Banking-draft examples: customer message + fixed reply prefix, one inline intent
call over the full 77-way banking77 list, gold label, template continuation that
names the label (checkable by substring — template-match is code-derived).
Output oracle/phase7-calls.jsonl {id, state, prefix, instructions, criteria,
label, continuation, target_text} + provenance (row, text_sha256).
--n (default 2000), --seed. Replay mix is a manifest note (see MIX); replay rows
assemble at train time from the frozen suites, not here.
"""
import argparse
import hashlib
import json

MIX = {"call_examples": 2000, "chat_replay": 20000, "ratio": "1:10",
       "replay_source": "frozen suites (train partitions) + plain banking77 messages"}

PREFIX = "Thanks for reaching out. "
INSTR = "Which banking intent best describes this customer message?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="oracle/phase7-calls.jsonl")
    a = ap.parse_args()
    from datasets import load_dataset
    import random
    ds = load_dataset("legacy-datasets/banking77", split="train")
    names = ds.features["label"].names
    rng = random.Random(a.seed)
    idx = rng.sample(range(len(ds)), min(a.n, len(ds)))
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for i in idx:
            row = ds[i]
            label = names[row["label"]]
            text = row["text"]
            cont = (f"This looks like {label.replace('_', ' ')}. "
                    f"Let me help with that right away.")
            pre = (f"Customer: {text}\nAgent: {PREFIX}"
                   f"<|decide|>{INSTR} [{', '.join(names)}]<|result|>")
            # The label span is INJECTED by the pointer head at inference; the chat
            # model never predicts it, so training masks exactly this span.
            target = pre + label + f"<|/result|> {cont}"
            mask = [len(pre), len(pre) + len(label)]
            norm = " ".join(text.casefold().split())
            f.write(json.dumps({
                "id": f"p7-{n}", "state": text, "prefix": PREFIX,
                "instructions": INSTR, "criteria": {k: None for k in names},
                "label": label, "continuation": cont, "target_text": target,
                "masked_span": mask,
                "provenance": {"row": i, "text_sha256": hashlib.sha256(
                    norm.encode()).hexdigest()}}) + "\n")
            n += 1
    with open("oracle/phase7-mix.json", "w", encoding="utf-8") as f:
        json.dump({**MIX, "built": n, "seed": a.seed}, f, indent=2)
    print(f"phase7 calls: {n} -> {a.out}")


if __name__ == "__main__":
    main()
