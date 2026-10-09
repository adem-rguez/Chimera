"""CHECK 8 close-out: base vs tuned on identical (id, qid, addition) rows."""
import json


def norm(v):
    return round(v) if isinstance(v, float) else v


ours = {}
for line in open("oracle/flip-results.jsonl", encoding="utf-8"):
    if not line.strip():
        continue
    x = json.loads(line)
    k = (x["id"], x["qid"], x["addition"])
    if k not in ours:
        ours[k] = x
for x in ours.values():
    x["flip"] = norm(x["argmax"]) != norm(x["clean"])
    x["wrong"] = norm(x["argmax"]) != norm(x["gold"])

base = [json.loads(l) for l in open("oracle/base-flip-results.jsonl", encoding="utf-8") if l.strip()]
both = [(ours[(b["id"], b["qid"], b["addition"])], b) for b in base
        if (b["id"], b["qid"], b["addition"]) in ours]
print(f"base rows: {len(base)}, matched: {len(both)}")
for name, rows, key in (("tuned", [o for o, _ in both], "flip"), ("base", [b for _, b in both], "flip")):
    pass
ot = [o for o, _ in both]
bt = [b for _, b in both]
print(f"tuned: flip={sum(o['flip'] for o in ot)/len(ot):.4f} "
      f"err={sum(o['wrong'] for o in ot)/len(ot):.4f}")
print(f"base : flip={sum(b['flip'] for b in bt)/len(bt):.4f} "
      f"err={sum(b['wrong'] for b in bt)/len(bt):.4f}")
print("by addition (tuned flip / base flip):")
for a in ["meta", "ad", "nudge0", "nudge1"]:
    o = [x for x, _ in both if x["addition"] == a]
    b = [x for _, x in both if x["addition"] == a]
    if o:
        print(f"  {a}: {sum(x['flip'] for x in o)/len(o):.3f} / {sum(x['flip'] for x in b)/len(b):.3f}")
