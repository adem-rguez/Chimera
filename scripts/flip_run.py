"""Context-flip runner (Phase 8). Posts each attacked state from flip-battery.jsonl
to the decision endpoint with the row's gold questions, compares argmax vs the
clean-run answer (from oracle/labels-v1.jsonl) and vs gold.
Stdlib only. DECIDE_URL env or argv[1]. --limit/--offset for chunked runs.
"""
import json
import os
import sys
import urllib.request

toks = sys.argv[1:]
skip_next = False
url_toks = []
for t in toks:
    if skip_next:
        skip_next = False
        continue
    if t in ("--limit", "--offset", "--battery", "--out"):
        skip_next = True
        continue
    url_toks.append(t)
URL = next((a for a in url_toks if not a.startswith("--")),
           os.environ.get("DECIDE_URL", "http://127.0.0.1:8009/v1/systemone"))
args = toks
LIMIT = int((args[args.index("--limit") + 1] if "--limit" in args else 0) or 0)
OFFSET = int((args[args.index("--offset") + 1] if "--offset" in args else 0) or 0)
BATTERY = (args[args.index("--battery") + 1] if "--battery" in args else 0) or "oracle/flip-battery.jsonl"
OUT = (args[args.index("--out") + 1] if "--out" in args else 0) or "oracle/flip-results.jsonl"
PROGRESS = OUT.replace("results", "progress").replace(".jsonl", ".json")


def arg_of(ans):
    if ans.get("type") == "noul" and "noul" in ans:
        return (ans["noul"] or 0) >= 0.5
    return ans.get("choice", ans.get("selected", ans.get("answer", ans.get("score"))))


def conf_of(ans):
    if ans.get("type") == "noul" and "noul" in ans:
        p = ans["noul"] or 0
        return max(p, 1 - p)
    return ans.get("confidence")


def questions_payload(g):
    out = {}
    for qid, q in g["questions"].items():
        qq = {"type": q["type"], "instructions": q["instructions"]}
        if "criteria" in q:
            qq["criteria"] = q["criteria"]
        out[qid] = qq
    return out


def post(payload):
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def main():
    gold = {}
    with open("evals/v7/decision-v7/development.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                gold[f"dev-{i}"] = json.loads(line)
    attacks = [json.loads(l) for l in open(BATTERY, encoding="utf-8")]
    attacks = attacks[OFFSET:(OFFSET + LIMIT) if LIMIT else None]
    by_id = {}
    for a in attacks:
        by_id.setdefault(a["id"], []).append(a)
    seen = set()
    try:
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    seen.add((r["id"], r["addition"], r.get("qid_target"), r["qid"]))
    except FileNotFoundError:
        pass
    flips = wrongs = done = 0
    with open(OUT, "a", encoding="utf-8") as out:
        for m, (rid, group) in enumerate(by_id.items()):
            g = gold[rid]
            payload = {"state": g["state"], "questions": questions_payload(g)}
            try:
                clean_resp = post(payload)
            except Exception as e:  # noqa: BLE001
                print(f"ERROR clean {rid}: {e}", flush=True)
                continue
            clean = {qid: arg_of(a) for qid, a in clean_resp.get("answers", {}).items()}
            for n, a in enumerate(group):
                payload = {"state": a["state"], "questions": questions_payload(g)}
                try:
                    resp = post(payload)
                except Exception as e:  # noqa: BLE001
                    print(f"ERROR {a['id']}/{a['addition']}: {e}", flush=True)
                    continue
                for qid, ans in resp.get("answers", {}).items():
                    key = (a["id"], a["addition"], a.get("qid"), qid)
                    if key in seen:
                        continue
                    seen.add(key)
                    arg = arg_of(ans)
                    cl = clean.get(qid)
                    gl = g["questions"][qid]["label"]
                    fl, wr = arg != cl, arg != gl
                    flips += fl
                    wrongs += wr
                    done += 1
                    out.write(json.dumps({"id": a["id"], "qid": qid, "addition": a["addition"],
                                          "target": a.get("target"),
                                          "qid_target": a.get("qid"),
                                          "argmax": arg, "clean": cl, "gold": gl,
                                      "flip": fl, "wrong": wr,
                                      "p": conf_of(ans)}) + "\n")
                print(f"{rid} {n + 1}/{len(group)} {a['addition']}", flush=True)
                if done and done % 25 == 0:
                    with open(PROGRESS, "w", encoding="utf-8") as pf:
                        json.dump({"rows_done": m + 1, "rows_total": len(by_id),
                                   "questions": done, "flips": flips, "wrongs": wrongs,
                                   "flip_rate": flips / done if done else 0}, pf)
                    print(f"  TALLY q={done} flips={flips} ({flips/done:.3f}) "
                          f"wrong={wrongs} ({wrongs/done:.3f})", flush=True)
            with open(PROGRESS, "w", encoding="utf-8") as pf:
                json.dump({"rows_done": m + 1, "rows_total": len(by_id),
                           "questions": done, "flips": flips, "wrongs": wrongs,
                           "flip_rate": flips / done if done else 0}, pf)


if __name__ == "__main__":
    main()
