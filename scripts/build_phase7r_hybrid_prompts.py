"""Phase 7R Source H, step 1: TRAIN prompt set in the same natural style/distribution as the frozen
eval oracle/phase7r-natural-eval.jsonl, drawn only from items/sources DISJOINT from that eval.

Reuses evals/v7/decision-v7/train.jsonl (never development.jsonl, which the eval is built from) and the
same canonical instruction/option machinery as scripts/build_phase7r_natural_eval.py (load_canonical,
dev_rows_for_type, state_text) so the decision schema is identical, but with:
  - its own entity-name/role/item pools for claim_handling/order_outcome (disjoint from the eval's),
  - its own, larger set of wrapper-sentence phrasings per text-based type (answer_type/kb_category/
    news_topic), reworded relative to the eval's wrappers, for surface-form variety,
  - its own control topic/template pool (disjoint text from the eval's).

Produces oracle/phase7r-hybrid-prompts.jsonl: ~220 decision rows per type (claim_handling, order_outcome,
answer_type, kb_category, news_topic) + ~150 control rows, gold labels balanced within each type, rows
shaped for scripts/gen_think_traces.py input ({id, user, ...passthrough}: kind, type, gold_label,
options_with_descriptions, item_ref).

Disjointness, asserted programmatically against oracle/phase7r-natural-eval.jsonl:
  - zero overlap of item_ref (guaranteed structurally: this script only reads train.jsonl, whose
    `_meta.id` values are disjoint from development.jsonl's -- verified empirically, 0/1204 dev ids
    appear among train's 12576 -- but still checked here rather than assumed)
  - zero overlap of exact `user` text

Usage:
    python scripts/build_phase7r_hybrid_prompts.py
    python scripts/build_phase7r_hybrid_prompts.py --selftest
"""
import argparse
import collections
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.build_phase7r_natural_eval import (  # noqa: E402 -- reused, not copied
    TYPE_SOURCE, dev_rows_for_type, load_canonical, load_jsonl, state_text,
)

TRAIN_SUITE = os.path.join(ROOT, "evals", "v7", "decision-v7", "train.jsonl")
EVAL_FILE = os.path.join(ROOT, "oracle", "phase7r-natural-eval.jsonl")
OUT_PATH = os.path.join(ROOT, "oracle", "phase7r-hybrid-prompts.jsonl")

SEED = 71320261
N_PER_TYPE = 220
N_CONTROL = 150
TEXT_TYPES = ["answer_type", "kb_category", "news_topic"]

AMOUNT_RE = re.compile(r"total is \$([\d,]+)")
QTY_RE = re.compile(r"quantity is (\d+)")

# ---------------------------------------------------------------------------
# entity pools -- disjoint from build_phase7r_natural_eval.py's NAMES/ROLES_CLAIM/ITEMS_CLAIM/
# ITEMS_ORDER/DELIVERY pools (no shared strings).
# ---------------------------------------------------------------------------

NAMES = ["Priya Chandran", "Owen Fitzgerald", "Zineb El Amrani", "Hollis Vance", "Suki Nakamoto",
         "Darius Okafor", "Elin Johansson", "Ravi Subramaniam", "Freya Lindqvist", "Caio Bezerra",
         "Marguerite Dubois", "Tobias Reinholt", "Amara Nwosu", "Lachlan Pryce", "Noor Hassani",
         "Wen Qiu", "Declan Ashworth", "Ingeborg Haugen", "Tariq Farouk", "Clementine Rousseau",
         "Benedek Varga", "Ama Serwaa", "Finnegan Loy", "Soraya Mendez", "Anik Chowdhury",
         "Leila Boutros", "Pernille Dahl", "Osei Agyeman", "Rhiannon Pryce", "Jonas Brandt"]
ROLES_CLAIM = ["procurement officer", "facilities manager", "research assistant", "program coordinator",
               "safety inspector", "field technician", "onboarding specialist", "budget analyst"]
ITEMS_CLAIM = ["IT peripherals", "safety equipment", "offsite workshop costs", "courier fees",
               "software licensing", "temporary staffing", "research consumables",
               "client dinner expenses"]
ITEMS_ORDER = ["ergonomic chair", "monitor arm", "space heater", "water cooler", "whiteboard",
               "shelving unit", "vacuum sealer", "label maker", "first-aid kit", "cable organizer"]
DELIVERY = ["a branch office", "a construction site", "a shared warehouse", "a satellite campus"]

CLAIM_TEMPLATES = [
    "Expense sign-off queue.\n\nPolicy: {policy}\n\n"
    "Claim #{ref} - {name}, {role}. Filed for {item}, claim total ${amount}. "
    "All receipts are on file.\n\nHow should this claim be handled?",
    "Reviewing a submitted claim against policy.\n\nPolicy: {policy}\n\n"
    "{name} ({role}) submitted claim #{ref} for {item}, totaling ${amount}, with full documentation "
    "attached.\n\nWhat's the correct disposition for this claim?",
    "Finance queue, next ticket.\n\nPolicy: {policy}\n\n"
    "Claim #{ref}, submitted by {name} ({role}) for {item}. Claim total: ${amount}. Receipts confirmed.\n\n"
    "How does this one get routed?",
]

ORDER_TEMPLATES = [
    "Order review queue.\n\nPolicy: {policy}\n\n"
    "Order #{ref} - {name} requested {qty} units of the {item}, shipping to {delivery}.\n\n"
    "What happens to this order?",
    "Checking a flagged order against the standing policy.\n\nPolicy: {policy}\n\n"
    "{name} placed order #{ref} for {qty} units of the {item}; delivery address is {delivery}.\n\n"
    "How should this order be resolved?",
    "Fulfillment desk, next case.\n\nPolicy: {policy}\n\n"
    "Order #{ref}: {name} ordered {qty} units of the {item}. Destination: {delivery}.\n\n"
    "What's the outcome for this order?",
]

# ---------------------------------------------------------------------------
# wrapper phrasings for the text-based types -- reworded relative to build_phase7r_natural_eval.py's
# ANSWER_WRAP/KB_WRAP/NEWS_WRAP (different sentences, same task family/voice).
# ---------------------------------------------------------------------------

ANSWER_WRAP = [
    "Support-queue triage pass. A user just submitted this question:\n\n\"{q}\"\n\n"
    "What category of answer is this question actually asking for?",
    "Sorting inbound questions before they hit the answer engine. Here's the next one:\n\n\"{q}\"\n\n"
    "Tag the kind of answer it needs.",
    "Help-desk log, latest entry:\n\n\"{q}\"\n\nWhat type of answer would satisfy this question?",
    "Query router, next item in the batch:\n\n\"{q}\"\n\nWhich answer category does this belong to?",
    "A question just came through the intake form:\n\n\"{q}\"\n\nClassify the answer type it's looking for.",
    "Triaging the question backlog one at a time. Current item:\n\n\"{q}\"\n\n"
    "What's the expected answer type here?",
]

KB_WRAP = [
    "Metadata backfill pass. This entry still needs a category before it ships:\n\n\"{t}\"\n\n"
    "What category should it get?",
    "Catalog ingestion, next record:\n\n\"{t}\"\n\nPick the right category for this one.",
    "This reference snippet is uncategorized:\n\n\"{t}\"\n\nWhich category applies?",
    "Working through the tagging backlog. Next item:\n\n\"{t}\"\n\nWhat's the correct category tag?",
    "A draft entry is waiting on its category field:\n\n\"{t}\"\n\nWhich category fits best?",
    "Index cleanup pass, one record at a time:\n\n\"{t}\"\n\nAssign its category.",
]

NEWS_WRAP = [
    "Morning wire triage. Next snippet:\n\n\"{t}\"\n\nWhich section should it run in?",
    "Routing copy before the editorial meeting. Here's the next piece:\n\n\"{t}\"\n\n"
    "What's the topic of this story?",
    "Wire desk intake, one item at a time:\n\n\"{t}\"\n\nWhich desk owns this?",
    "Sorting the overnight wire feed:\n\n\"{t}\"\n\nWhat topic does this fall under?",
    "Next item in the copy queue:\n\n\"{t}\"\n\nWhich section does this belong to?",
    "Assigning stories before the print deadline:\n\n\"{t}\"\n\nWhat desk should take this one?",
]

WRAP_FOR = {"answer_type": ANSWER_WRAP, "kb_category": KB_WRAP, "news_topic": NEWS_WRAP}

# ---------------------------------------------------------------------------
# controls -- disjoint topics/templates from build_phase7r_natural_eval.py's CONTROL_TOPICS/
# CONTROL_TEMPLATES (no shared strings).
# ---------------------------------------------------------------------------

CONTROL_TOPICS = [
    "a flat tire on the way to work", "a blender that won't turn on", "a goldfish that stopped eating",
    "a delayed train on the morning commute", "a sunburn from the weekend hike", "a printer jam at the office",
    "a lost library book", "a squeaky office chair", "a double-booked meeting room",
    "a overgrown hedge next door", "a missing sock from the laundry", "a stuck zipper on a jacket",
    "a leftover lasagna going bad", "a flickering hallway light", "a forgotten anniversary",
    "a parking ticket downtown", "a tangled set of headphones", "a wobbly kitchen table",
    "a canceled dentist appointment", "a spam call during dinner", "a cracked phone case",
    "a slow Wi-Fi connection at a cafe", "a misplaced car key", "a leaking reusable water bottle",
    "a houseguest staying an extra night", "a recipe missing an ingredient", "a broken shoelace",
    "a late package delivery", "a noisy smoke detector battery", "a overbooked flight",
]
CONTROL_TEMPLATES = [
    "Write a two-line text message to a friend about {t}.",
    "Draft a short, upbeat caption for a photo related to {t}.",
    "What's a polite way to mention {t} in an email to a colleague?",
    "Give a one-paragraph pep talk to someone dealing with {t}.",
    "Suggest three small talk lines about {t}.",
    "Write a quick voicemail greeting that references {t}.",
    "Brainstorm two ways to cheer someone up about {t}.",
    "What would a good friend say about {t}?",
    "Write a one-line fortune-cookie-style quip about {t}.",
    "Draft a brief social media post mentioning {t}.",
    "Write a tiny limerick about {t}.",
    "Suggest a comforting reply to someone complaining about {t}.",
]


def build_claim_rows(rng, base_rows, canonical, used_text):
    instr, opts, desc = canonical["claim_handling"]
    owd = [[o, desc[o]] for o in opts]
    by_label = collections.defaultdict(list)
    for rec in base_rows:
        case = rec["state"]["case"]
        if AMOUNT_RE.search(case):
            by_label[rec["questions"]["decision"]["label"]].append(rec)
    for recs in by_label.values():
        rng.shuffle(recs)
    labels = sorted(by_label)
    cursor = {l: 0 for l in labels}

    rows, i = [], 0
    while i < N_PER_TYPE:
        label = labels[i % len(labels)]
        pool = by_label[label]
        if not pool:
            break
        rec = pool[cursor[label] % len(pool)]
        cursor[label] += 1
        case = rec["state"]["case"]
        policy = rec["state"]["policy"]
        amount = AMOUNT_RE.search(case).group(1)
        name = NAMES[i % len(NAMES)]
        role = ROLES_CLAIM[i % len(ROLES_CLAIM)]
        item = ITEMS_CLAIM[(i * 5 + 2) % len(ITEMS_CLAIM)]
        tpl = CLAIM_TEMPLATES[i % len(CLAIM_TEMPLATES)]
        ref = 51000 + i
        user = tpl.format(policy=policy, ref=ref, name=name, role=role, item=item, amount=amount)
        if user in used_text:
            i += 1
            continue
        used_text.add(user)
        rows.append(dict(
            id=f"p7rh-claim_handling-{i:03d}", user=user, kind="decision", type="claim_handling",
            gold_label=label, options_with_descriptions=owd, item_ref=rec["_meta"]["id"],
            label_surfaces={}))
        i += 1
    return rows


def build_order_rows(rng, base_rows, canonical, used_text):
    instr, opts, desc = canonical["order_outcome"]
    owd = [[o, desc[o]] for o in opts]
    by_label = collections.defaultdict(list)
    for rec in base_rows:
        case = rec["state"]["case"]
        if QTY_RE.search(case):
            by_label[rec["questions"]["decision"]["label"]].append(rec)
    for recs in by_label.values():
        rng.shuffle(recs)
    labels = sorted(by_label)
    cursor = {l: 0 for l in labels}

    rows, i = [], 0
    while i < N_PER_TYPE:
        label = labels[i % len(labels)]
        pool = by_label[label]
        if not pool:
            break
        rec = pool[cursor[label] % len(pool)]
        cursor[label] += 1
        case = rec["state"]["case"]
        policy = rec["state"]["policy"]
        qty = QTY_RE.search(case).group(1)
        name = NAMES[(i + 11) % len(NAMES)]
        item = ITEMS_ORDER[i % len(ITEMS_ORDER)]
        delivery = DELIVERY[i % len(DELIVERY)]
        tpl = ORDER_TEMPLATES[i % len(ORDER_TEMPLATES)]
        ref = 87000 + i
        user = tpl.format(policy=policy, ref=ref, name=name, qty=qty, item=item, delivery=delivery)
        if user in used_text:
            i += 1
            continue
        used_text.add(user)
        rows.append(dict(
            id=f"p7rh-order_outcome-{i:03d}", user=user, kind="decision", type="order_outcome",
            gold_label=label, options_with_descriptions=owd, item_ref=rec["_meta"]["id"],
            label_surfaces={}))
        i += 1
    return rows


def build_text_rows(rng, base_rows, canonical, tname, used_text, used_item_refs):
    instr, opts, desc = canonical[tname]
    owd = [[o, desc[o]] for o in opts]
    qid = TYPE_SOURCE[tname][1]
    wraps = WRAP_FOR[tname]

    by_label = collections.defaultdict(list)
    for rec in base_rows:
        by_label[rec["questions"][qid]["label"]].append(rec)
    for recs in by_label.values():
        rng.shuffle(recs)
    labels = sorted(by_label)
    cursor = {l: 0 for l in labels}

    rows = []
    li = 0
    tries = 0
    max_tries = N_PER_TYPE * 20
    while len(rows) < N_PER_TYPE and tries < max_tries:
        tries += 1
        label = labels[li % len(labels)]
        li += 1
        pool = by_label[label]
        if cursor[label] >= len(pool):
            continue
        rec = pool[cursor[label]]
        cursor[label] += 1
        if rec["_meta"]["id"] in used_item_refs:
            continue
        text = state_text(rec)
        wrap = wraps[len(rows) % len(wraps)]
        user = wrap.format(q=text, t=text)
        if user in used_text:
            continue
        used_text.add(user)
        used_item_refs.add(rec["_meta"]["id"])
        rows.append(dict(
            id=f"p7rh-{tname}-{len(rows):03d}", user=user, kind="decision", type=tname,
            gold_label=label, options_with_descriptions=owd, item_ref=rec["_meta"]["id"],
            label_surfaces={}))
    return rows


def build_control_rows(rng, used_text):
    combos = [(tpl, topic) for tpl in CONTROL_TEMPLATES for topic in CONTROL_TOPICS]
    rng.shuffle(combos)
    rows = []
    for tpl, topic in combos:
        if len(rows) >= N_CONTROL:
            break
        user = tpl.format(t=topic)
        if user in used_text:
            continue
        used_text.add(user)
        rows.append(dict(
            id=f"p7rh-ctrl-{len(rows):03d}", user=user, kind="control", type="control",
            gold_label=None, options_with_descriptions=None, item_ref=None, label_surfaces={}))
    return rows


def build(seed=SEED):
    train_rows = load_jsonl(TRAIN_SUITE)
    canonical = load_canonical(train_rows)
    rng = random.Random(seed)

    used_text = set()
    used_item_refs = set()

    rows = []
    rows += build_claim_rows(rng, dev_rows_for_type(train_rows, "claim_handling"), canonical, used_text)
    rows += build_order_rows(rng, dev_rows_for_type(train_rows, "order_outcome"), canonical, used_text)
    for tname in TEXT_TYPES:
        rows += build_text_rows(rng, dev_rows_for_type(train_rows, tname), canonical, tname,
                                 used_text, used_item_refs)
    rows += build_control_rows(rng, used_text)
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def check_disjoint(rows, eval_rows):
    fails = []
    hybrid_refs = {r["item_ref"] for r in rows if r.get("item_ref")}
    eval_refs = {r["item_ref"] for r in eval_rows if r.get("item_ref")}
    ref_overlap = hybrid_refs & eval_refs
    if ref_overlap:
        fails.append(f"{len(ref_overlap)} item_ref overlap with eval: {sorted(ref_overlap)[:10]}")

    hybrid_text = {r["user"] for r in rows}
    eval_text = {r["user"] for r in eval_rows}
    text_overlap = hybrid_text & eval_text
    if text_overlap:
        fails.append(f"{len(text_overlap)} exact user-text overlap with eval: {list(text_overlap)[:5]}")

    ids = [r["id"] for r in rows]
    dup_ids = [k for k, v in collections.Counter(ids).items() if v > 1]
    if dup_ids:
        fails.append(f"duplicate ids within hybrid set: {dup_ids[:10]}")

    texts = [r["user"] for r in rows]
    dup_text = [k for k, v in collections.Counter(texts).items() if v > 1]
    if dup_text:
        fails.append(f"{len(dup_text)} duplicate user texts within the hybrid set")

    return fails


def report(rows, eval_rows, fails):
    kinds = collections.Counter(r["kind"] for r in rows)
    per_type = collections.Counter(r["type"] for r in rows if r["kind"] == "decision")
    per_type_label = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["kind"] == "decision":
            per_type_label[r["type"]][r["gold_label"]] += 1

    print(f"hybrid prompts: {len(rows)} rows")
    print(f"  kinds: {dict(kinds)}")
    print(f"  decision per type: {dict(per_type)}")
    for t, c in per_type_label.items():
        print(f"    {t} label balance: {dict(c)}")
    print(f"  overlap check vs {EVAL_FILE}: "
          f"{'FAILED' if fails else 'passed, zero overlap'}")
    if fails:
        for f in fails:
            print(f"  - {f}")
    print("  3 sample prompts:")
    for r in rows[:1] + [r for r in rows if r["kind"] == "decision" and r["type"] == "kb_category"][:1] \
              + [r for r in rows if r["kind"] == "control"][:1]:
        print(f"    [{r['id']}] type={r['type']} gold_label={r.get('gold_label')}")
        print(f"      user: {r['user'][:200]!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    rows = build(a.seed)
    eval_rows = load_jsonl(EVAL_FILE)
    fails = check_disjoint(rows, eval_rows)
    report(rows, eval_rows, fails)

    if a.selftest:
        return 1 if fails else 0

    if fails:
        print("NOT writing output: disjointness check failed")
        return 1

    write_jsonl(OUT_PATH, rows)
    print(f"-> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
