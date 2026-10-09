"""Verify the reviewer's four claims on flip-results (offline)."""
import json
from math import comb

L = [json.loads(l) for l in open("oracle/flip-results.jsonl", encoding="utf-8") if l.strip()]


def norm(v):
    return round(v) if isinstance(v, float) else v


seen = set()
U = []
for x in L:
    k = (x["id"], x["qid"], x["addition"])
    if k in seen:
        continue
    seen.add(k)
    U.append(x)
print(f"rows in file: {len(L)}, deduplicated: {len(U)}")

for x in U:
    x["flip"] = norm(x["argmax"]) != norm(x["clean"])
    x["wrong"] = norm(x["argmax"]) != norm(x["gold"])
    x["clean_wrong"] = norm(x["clean"]) != norm(x["gold"])

flips = sum(x["flip"] for x in U)
print(f"dedup flip rate: {flips}/{len(U)} = {flips/len(U):.4f}")

rw = sum(1 for x in U if x["flip"] and not x["clean_wrong"] and x["wrong"])
wr = sum(1 for x in U if x["flip"] and x["clean_wrong"] and not x["wrong"])
ww = sum(1 for x in U if x["flip"] and x["clean_wrong"] and x["wrong"])
rr = sum(1 for x in U if x["flip"] and not x["clean_wrong"] and not x["wrong"])
print(f"right->wrong: {rw}, wrong->right: {wr}, wrong->other-wrong: {ww}, right->other-right: {rr}")
# sign test on decisive flips (rw vs wr), H0 p=0.5
n, k = rw + wr, min(rw, wr)
p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n
print(f"sign test (decisive {n}, minority {k}): p = {p:.4f}")
ce = sum(x["clean_wrong"] for x in U) / len(U)
ae = sum(x["wrong"] for x in U) / len(U)
print(f"clean err: {ce:.4f}, attacked err: {ae:.4f}, delta: {ae-ce:.4f} ({(ae-ce)*100:.1f}pp)")
