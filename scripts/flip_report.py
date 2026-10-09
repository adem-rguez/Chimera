"""Phase 8 flip-battery read-out (offline)."""
import json

L = [json.loads(l) for l in open("oracle/flip-results.jsonl", encoding="utf-8") if l.strip()]


def norm(v):
    return round(v) if isinstance(v, float) else v


for x in L:
    x["flip"] = norm(x["argmax"]) != norm(x["clean"])
    x["wrong"] = norm(x["argmax"]) != norm(x["gold"])
print("rows:", len({x["id"] for x in L}), "questions:", len(L))
print("BY ADDITION:")
for a in ["meta", "ad", "nudge0", "nudge1"]:
    s = [x for x in L if x["addition"] == a]
    print(f"  {a:7s} n={len(s):4d} flip={sum(x['flip'] for x in s)/len(s):.3f} "
          f"wrong={sum(x['wrong'] for x in s)/len(s):.3f}")
fl = [x for x in L if x["flip"]]
print(f"flips: {len(fl)} ({len(fl)/len(L):.3f})  flip-to-wrong: {sum(x['wrong'] for x in fl)} "
      f"flip-to-right: {sum(not x['wrong'] for x in fl)}")
print(f"clean-wrong rate: {sum(1 for x in L if x['clean'] != x['gold'])/len(L):.3f}")
print("BY FAMILY (from gold):")
gold = [json.loads(l) for l in open("evals/v7/decision-v7/development.jsonl", encoding="utf-8") if l.strip()]
fam = {}
for x in L:
    g = gold[int(x["id"].split("-")[1])]
    fams = "+".join(sorted(g["questions"]))
    fam.setdefault(fams, []).append(x)
for k, s in sorted(fam.items()):
    print(f"  {k:25s} n={len(s):4d} flip={sum(x['flip'] for x in s)/len(s):.3f} "
          f"wrong={sum(x['wrong'] for x in s)/len(s):.3f}")
