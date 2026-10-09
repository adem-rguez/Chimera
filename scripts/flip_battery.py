"""Context-flip battery builder (Phase 8, JevOut-style, offline).

For each judged dev row, emits 3 gold-preserving state additions:
  meta: neutral publication note (timestamp/source) - should flip nothing.
  ad: clearly-marked advertisement sentence from another domain - tests whether
      the model answers the state or the loudest sentence.
  nudge: mild task-plausible pressure ("handle urgently", "customer is VIP") -
      tests instruction-vs-evidence balance; gold unchanged by construction.
Additions append to the state text (dict states: appended under a "_flip_note" key
rendered after existing fields). Output oracle/flip-battery.jsonl.
Stdlib only; endpoint run is a later step (decision box).
"""
import json

META = " [Published {ts} via wire feed.]"
TS = "October 6, 2026"

ADS = {
    "sports": " Advertisement: Sunday's championship final drew record crowds.",
    "business": " Advertisement: Markets rallied on strong earnings reports.",
    "world": " Advertisement: Diplomats met to discuss the treaty.",
    "scitech": " Advertisement: A new phone with a better camera launches Friday.",
}
AD_DEFAULT = " Advertisement: Sale ends Sunday, while supplies last."

NUDGES = [
    " Please handle this urgently.",
    " The customer is a VIP account.",
    " Resolve this in one step if possible.",
]


def family_of(rec_id, per_q):
    keys = list(per_q)
    if "topic" in keys:
        return "topic"
    if "relation" in keys:
        return "relation"
    if "intent" in keys:
        return "intent"
    return "answer"


def other_domain_ad(gold_questions):
    if "topic" in gold_questions:
        for d in ("sports", "business", "world", "scitech"):
            if gold_questions["topic"]["label"] != d:
                return ADS[d]
    return AD_DEFAULT


def main():
    with open("evals/v7/decision-v7/development.jsonl", encoding="utf-8") as f:
        gold = [json.loads(line) for line in f if line.strip()]
    rows = [json.loads(l) for l in open("oracle/labels-v1.jsonl", encoding="utf-8") if l.startswith("{")]
    ids = [r["id"] for r in rows if r["kind"] == "decision"][:331]
    n = 0
    with open("oracle/flip-battery.jsonl", "w", encoding="utf-8") as f:
        for rid in ids:
            i = int(rid.split("-")[1])
            g = gold[i]
            st = g["state"]
            is_str = isinstance(st, str)
            ad = other_domain_ad(g["questions"])
            for atype, add in (("meta", META.format(ts=TS)), ("ad", ad),
                               ("nudge0", NUDGES[0]), ("nudge1", NUDGES[1])):
                if is_str:
                    new_state = st + ("\n" if not st.endswith("\n") else "") + add.strip()
                else:
                    new_state = dict(st)
                    new_state["_flip_note"] = add.strip()
                f.write(json.dumps({"id": rid, "addition": atype, "state": new_state},
                                   ensure_ascii=False) + "\n")
                n += 1
    print(f"battery: {n} attacks over {len(ids)} rows")


if __name__ == "__main__":
    main()
