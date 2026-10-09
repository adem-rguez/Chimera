"""Offline verify: control battery targets valid; runner parsing sane."""
import json

gold = {}
with open("evals/v7/decision-v7/development.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if line.strip():
            gold[f"dev-{i}"] = json.loads(line)
B = [json.loads(l) for l in open("oracle/control-battery.jsonl", encoding="utf-8")]
print("attacks:", len(B))
ov = [x for x in B if x["addition"] == "override"]
bad = []
for x in ov:
    q = gold[x["id"]]["questions"][x["qid"]]
    t = x["target"]
    if q["type"] == "choice":
        if t == q["label"] or t not in q["criteria"]:
            bad.append(x)
    else:
        if t == q["label"]:
            bad.append(x)
print("override attacks:", len(ov), "bad targets:", len(bad))
print("rows:", len({x['id'] for x in B}))
print("sample forge:", B[0]["state"][-80:])
o = next(x for x in ov if isinstance(x["target"], str))
print("sample override:", o["state"][-120:], "| target:", o["target"])
