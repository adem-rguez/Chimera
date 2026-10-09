"""Offline verify: masked-span token-masking algorithm (pure python simulation)."""
import json

L = [json.loads(l) for l in open("oracle/phase7-calls.jsonl", encoding="utf-8")]
# simulate tokenization as 4-char chunks with char offsets, apply train_phase7 logic
AGENT = "Agent: "
bad = 0
for r in L:
    text = r["target_text"]
    toks = [(i, min(i + 4, len(text))) for i in range(0, len(text), 4)]
    s, e = r["masked_span"]
    agent_at = text.index(AGENT) + len(AGENT)
    masked = [i for i, (a, b) in enumerate(toks) if b <= agent_at or (a < e and b > s)]
    # every char of the label span must be covered by a masked token
    covered = set()
    for i in masked:
        covered.update(range(toks[i][0], toks[i][1]))
    if not set(range(s, e)) <= covered:
        bad += 1
    # prompt-only tokens (before agent) all masked?
    pre = [i for i, (a, b) in enumerate(toks) if b <= agent_at]
    if not set(pre) <= set(masked):
        bad += 1
print("rows:", len(L), "masking violations:", bad)
x = L[0]
print("agent starts at char:", x["target_text"].index(AGENT), "| span:", x["masked_span"])
