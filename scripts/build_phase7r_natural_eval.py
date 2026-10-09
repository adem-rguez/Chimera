"""Phase 7R T3: build the frozen natural-eval prompt set and the T8 replay prompt pool.

Produces (deterministically, seeded):
  oracle/phase7r-natural-eval.jsonl   ~440 rows: 300 decision (60/type x 5 types), 100 control,
                                       40 adversarial (8/type near-miss).
  oracle/phase7r-replay-pool.jsonl    ~1800 non-decision rows for T8 E4B self-distillation.
  oracle/phase7r-natural-eval.sha256  sha256 of the natural-eval file (freeze-before-training record).

Prompts only -- no thinking, no calls. See reports/16-phase7-reasoning-proposal.md (natural eval spec,
~lines 208-260 and 383-399) and oracle/phase7-reasoning-pilot-v2.jsonl (task families/style this is meant
to feed arms A/B/C/D of).

Five frozen decision types (reports/16 Decision 5): claim_handling, order_outcome, answer_type,
kb_category, news_topic. Canonical instruction + option set for each is read back out of
evals/v7/decision-v7/development.jsonl (DEV only -- never train.jsonl) the same way
scripts/check_phase7r_pilot.py does it, so a drifted suite fails the build instead of silently diverging.

Scenario construction, since no LLM authoring runs inside this script:
  - answer_type / kb_category / news_topic: the underlying item (a TREC question / DBpedia14 blurb /
    AG News snippet) IS the DEV suite's own natural text -- there is no cheaper or more faithful "new
    scenario" than the organic text itself. What's new relative to the pilot is the WRAPPER sentence
    (a task-family framing: question-router / kb-backfill / newsdesk-routing voice) and the fact that each
    DEV row is used at most once across this whole build.
  - claim_handling / order_outcome: DEV's legacy_policy clean rows for these two families are repetitive
    (24 rows each, names drawn from a 4-name pool: Tomas/Mira/Sana/Omar) and would duplicate pilot
    characters if reused verbatim. Each DEV row is instead replayed under NEW entity names/items/ticket
    numbers (own pool, disjoint from the pilot's and DEV's), keeping the policy text and the determining
    number (claim total / order quantity) byte-identical to the DEV row, which is what keeps gold_label
    exactly the DEV row's label. item_ref records the DEV row id this was derived from. Each of the 24
    rows is reused ~2-3x with different entities to reach 60/type; wrapped text is checked unique.

Controls: generic chat/draft/Q&A prompts carrying no sub-decision of the 5 types at all.
Adversarial: near-miss prompts in the same domain as one of the 5 types that must NOT trigger a call
(meta-policy questions, missing the decisive fact, or asking for something other than classification).

Row schema: {id, user, system?, kind: "decision"|"control"|"adversarial", type, gold_label,
options_with_descriptions, item_ref, label_surfaces}.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV_SUITE = os.path.join(ROOT, "evals", "v7", "decision-v7", "development.jsonl")
TRAIN_SUITE = os.path.join(ROOT, "evals", "v7", "decision-v7", "train.jsonl")
PILOT = os.path.join(ROOT, "oracle", "phase7-reasoning-pilot-v2.jsonl")
OUT_EVAL = os.path.join(ROOT, "oracle", "phase7r-natural-eval.jsonl")
OUT_REPLAY = os.path.join(ROOT, "oracle", "phase7r-replay-pool.jsonl")
OUT_HASH = os.path.join(ROOT, "oracle", "phase7r-natural-eval.sha256")

SEED = 7132026
NONE_KEY = "none_of_these"

# decision type -> (suite source, question id, legacy-policy family or None). Same mapping as
# scripts/check_phase7r_pilot.py minus banking_intent (dropped, reports/16 Decision 5).
TYPE_SOURCE = {
    "claim_handling": ("legacy_policy", "decision", "contrastive_spend_threshold"),
    "order_outcome": ("legacy_policy", "decision", "contrastive_quantity_limit"),
    "answer_type": ("trec", "answer_type", None),
    "kb_category": ("dbpedia14", "category", None),
    "news_topic": ("agnews", "topic", None),
}
N_PER_TYPE = 60
N_CONTROL = 100
N_ADV_PER_TYPE = 8  # 8 * 5 = 40

# families belonging to one of the 5 decision types -- excluded entirely from the replay pool so it can
# never carry a decision of the 5 frozen types.
DECISION_FAMILIES = {"legacy_policy", "trec", "dbpedia14", "agnews"}
REPLAY_TARGET = 1800


# ---------------------------------------------------------------------------
# suite loading
# ---------------------------------------------------------------------------

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_canonical(dev_rows):
    """-> {type: (instruction, [options in first-seen order], {label: description_or_None})}"""
    found_instr = collections.defaultdict(set)
    opt_order = collections.defaultdict(list)
    opt_seen = collections.defaultdict(set)
    desc = collections.defaultdict(dict)
    for rec in dev_rows:
        src = rec["_meta"].get("source")
        for qid, q in rec["questions"].items():
            if not isinstance(q.get("criteria"), dict):
                continue
            instr = q.get("instructions")
            if not isinstance(instr, str):
                continue
            for tname, (want_src, want_qid, want_fam) in TYPE_SOURCE.items():
                if src != want_src or qid != want_qid:
                    continue
                if want_fam is not None and q.get("src") != want_fam:
                    continue
                found_instr[tname].add(instr)
                for k, v in q["criteria"].items():
                    if k not in opt_seen[tname]:
                        opt_seen[tname].add(k)
                        opt_order[tname].append(k)
                    if v is not None and k not in desc[tname]:
                        desc[tname][k] = v if isinstance(v, str) else json.dumps(v)
    out = {}
    for tname in TYPE_SOURCE:
        instrs = found_instr[tname]
        assert len(instrs) == 1, f"type {tname!r}: suite gives {len(instrs)} instruction strings, want 1"
        opts = [o for o in opt_order[tname] if o != NONE_KEY]
        d = {o: desc[tname].get(o, f"Issue concerning {o.replace('_', ' ')}") for o in opts}
        out[tname] = (next(iter(instrs)), opts, d)
    return out


def dev_rows_for_type(dev_rows, tname, variant="clean"):
    want_src, want_qid, want_fam = TYPE_SOURCE[tname]
    out = []
    for rec in dev_rows:
        if rec["_meta"].get("variant") != variant:
            continue
        if rec["_meta"].get("source") != want_src:
            continue
        q = rec["questions"].get(want_qid)
        if q is None or not isinstance(q.get("criteria"), dict):
            continue
        if want_fam is not None and q.get("src") != want_fam:
            continue
        out.append(rec)
    return out


def state_text(rec):
    """DEV rows render the same underlying item text under several voices (kev/data.py renderers):
    plain string, {"document": ...}, {"ticket": {"body": ...}}, or a chat-turn list [{"content": ...}].
    All of them carry the same item text; this just recovers it regardless of voice."""
    s = rec["state"]
    if isinstance(s, str):
        return s
    if isinstance(s, dict) and isinstance(s.get("document"), str):
        return s["document"]
    if isinstance(s, dict) and isinstance(s.get("ticket"), dict):
        return s["ticket"].get("body", "")
    if isinstance(s, list) and s and isinstance(s[-1], dict) and isinstance(s[-1].get("content"), str):
        return s[-1]["content"]
    raise ValueError(f"unhandled state shape for {rec['_meta'].get('id')}: {type(s)}")


# ---------------------------------------------------------------------------
# name / entity pools -- disjoint from the pilot's and DEV's own names
# ---------------------------------------------------------------------------

NAMES = ["Farida Osei", "Lucas Berndt", "Nadia Kowalczyk", "Theo Marsh", "Ingrid Solberg",
         "Kwame Boateng", "Yusra Haddad", "Callum Reyes", "Minako Tanaka", "Esperanza Solis",
         "Bram Vos", "Oyelaran Adeyemi", "Siobhan Doyle", "Mateusz Nowak", "Aiyana Running",
         "Petra Vuković", "Deshawn Carter", "Lindiwe Dube", "Rosalind Fenwick", "Hamish Og"]
ROLES_CLAIM = ["logistics coordinator", "field engineer", "events planner", "lab technician",
               "fleet supervisor", "comms officer", "grants assistant", "site manager"]
ITEMS_CLAIM = ["conference travel", "lab supplies", "fleet repairs", "venue hire",
               "equipment rental", "catering for the site visit", "printing and signage",
               "training materials"]
ITEMS_ORDER = ["standing desk", "espresso machine", "garden hose", "tool chest", "bike rack",
               "patio heater", "filing cabinet", "paper shredder", "camping stove", "label printer"]
DELIVERY = ["a residential address", "a loading dock", "a regional depot", "a storage unit"]


# ---------------------------------------------------------------------------
# decision rows
# ---------------------------------------------------------------------------

AMOUNT_RE = re.compile(r"total is \$([\d,]+)")
QTY_RE = re.compile(r"quantity is (\d+)")


def build_claim_rows(rng, dev_rows, canonical, used_text):
    instr, opts, desc = canonical["claim_handling"]
    base = dev_rows_for_type(dev_rows, "claim_handling")
    owd = [[o, desc[o]] for o in opts]
    rows, i = [], 0
    reps = (N_PER_TYPE + len(base) - 1) // len(base)
    pool = base * reps
    rng.shuffle(pool)
    for rec in pool:
        if i >= N_PER_TYPE:
            break
        case = rec["state"]["case"]
        policy = rec["state"]["policy"]
        label = rec["questions"]["decision"]["label"]
        m = AMOUNT_RE.search(case)
        if not m:
            continue
        amount = m.group(1)
        name = NAMES[i % len(NAMES)]
        role = ROLES_CLAIM[i % len(ROLES_CLAIM)]
        item = ITEMS_CLAIM[(i * 3 + 1) % len(ITEMS_CLAIM)]
        ref = 41000 + i
        user = (f"Expense claim review.\n\nPolicy: {policy}\n\n"
                f"Claim #{ref} - {name}, {role}. Submitted for {item}, claim total ${amount}. "
                f"Receipts were attached for every line item.\n\n"
                f"How should this claim be handled?")
        if user in used_text:
            i += 1
            continue
        used_text.add(user)
        rows.append(dict(
            id=f"p7rne-claim_handling-{i:02d}", user=user, kind="decision", type="claim_handling",
            gold_label=label, options_with_descriptions=owd,
            item_ref=rec["_meta"]["id"], label_surfaces={}))
        i += 1
    return rows


def build_order_rows(rng, dev_rows, canonical, used_text):
    instr, opts, desc = canonical["order_outcome"]
    base = dev_rows_for_type(dev_rows, "order_outcome")
    owd = [[o, desc[o]] for o in opts]
    rows, i = [], 0
    reps = (N_PER_TYPE + len(base) - 1) // len(base)
    pool = base * reps
    rng.shuffle(pool)
    for rec in pool:
        if i >= N_PER_TYPE:
            break
        case = rec["state"]["case"]
        policy = rec["state"]["policy"]
        label = rec["questions"]["decision"]["label"]
        m = QTY_RE.search(case)
        if not m:
            continue
        qty = m.group(1)
        name = NAMES[(i + 7) % len(NAMES)]
        item = ITEMS_ORDER[i % len(ITEMS_ORDER)]
        delivery = DELIVERY[i % len(DELIVERY)]
        ref = 78000 + i
        user = (f"Order desk review.\n\nPolicy: {policy}\n\n"
                f"Order #{ref} - {name} ordered {qty} units of the {item}. "
                f"Delivery requested to {delivery}.\n\n"
                f"What happens to this order?")
        if user in used_text:
            i += 1
            continue
        used_text.add(user)
        rows.append(dict(
            id=f"p7rne-order_outcome-{i:02d}", user=user, kind="decision", type="order_outcome",
            gold_label=label, options_with_descriptions=owd,
            item_ref=rec["_meta"]["id"], label_surfaces={}))
        i += 1
    return rows


ANSWER_WRAP = [
    "Question-router triage. A user submitted this question to our help desk:\n\n\"{q}\"\n\n"
    "What kind of answer is this question looking for?",
    "We're routing incoming questions to the right answer engine. Incoming question:\n\n\"{q}\"\n\n"
    "Classify the kind of answer it's asking for.",
    "Front-desk Q&A bot log, one entry:\n\n\"{q}\"\n\nWhat kind of answer does the asker need?",
]

KB_WRAP = [
    "Knowledge-base backfill. A new article needs a category tag before it can be published:\n\n\"{t}\"\n\n"
    "Which category does the subject belong to?",
    "KB ingestion queue, next item:\n\n\"{t}\"\n\nAssign the correct category for the index.",
    "This snippet is missing its catalog category:\n\n\"{t}\"\n\nWhich category fits?",
]

NEWS_WRAP = [
    "Newsdesk routing. Wire snippet just came in:\n\n\"{t}\"\n\nWhich desk does this go to?",
    "Sorting the morning wire. Next item:\n\n\"{t}\"\n\nWhat is the topic of this article?",
    "Assigning wire copy to desks before the noon meeting:\n\n\"{t}\"\n\nWhich desk should take this?",
]


def build_text_rows(rng, dev_rows, canonical, tname, wraps, used_text, used_item_refs):
    instr, opts, desc = canonical[tname]
    owd = [[o, desc[o]] for o in opts]
    base = dev_rows_for_type(dev_rows, tname)
    rng.shuffle(base)
    rows = []
    for rec in base:
        if len(rows) >= N_PER_TYPE:
            break
        if rec["_meta"]["id"] in used_item_refs:
            continue
        text = state_text(rec)
        wrap = wraps[len(rows) % len(wraps)]
        user = wrap.format(q=text, t=text)
        if user in used_text:
            continue
        used_text.add(user)
        used_item_refs.add(rec["_meta"]["id"])
        label = rec["questions"][TYPE_SOURCE[tname][1]]["label"]
        rows.append(dict(
            id=f"p7rne-{tname}-{len(rows):02d}", user=user, kind="decision", type=tname,
            gold_label=label, options_with_descriptions=owd,
            item_ref=rec["_meta"]["id"], label_surfaces={}))
    return rows


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------

CONTROL_TOPICS = [
    "a product launch next Tuesday", "a team offsite in the mountains", "a leaky kitchen tap",
    "a bike that needs new brake pads", "a houseplant that keeps wilting", "a long layover in Doha",
    "a book club meeting on Thursday", "a neighbor's loud renovation", "a broken umbrella",
    "a recipe that calls for too much salt", "a flight delay home", "a new coffee grinder",
    "a friend's birthday next month", "a noisy upstairs apartment", "a half-finished crossword",
    "a cracked phone screen", "a slow commute this week", "a garden that needs weeding",
    "a conference talk proposal", "a spreadsheet that won't balance",
]
CONTROL_TEMPLATES = [
    "Draft a two-sentence out-of-office reply for someone away for {t}.",
    "Write a short, friendly message to a friend about {t}.",
    "What's a tactful way to bring up {t} with a coworker?",
    "Summarize, in one paragraph, what you'd tell someone dealing with {t}.",
    "Give me three casual conversation openers about {t}.",
    "Write a quick thank-you note that mentions {t}.",
    "Brainstorm two ideas for handling {t}.",
    "What would you say to someone venting about {t}?",
    "Draft a one-line calendar invite description involving {t}.",
    "Write a short joke about {t}.",
]


def build_control_rows(rng):
    combos = [(tpl, topic) for tpl in CONTROL_TEMPLATES for topic in CONTROL_TOPICS]
    rng.shuffle(combos)
    rows = []
    seen = set()
    for tpl, topic in combos:
        if len(rows) >= N_CONTROL:
            break
        user = tpl.format(t=topic)
        if user in seen:
            continue
        seen.add(user)
        rows.append(dict(
            id=f"p7rne-ctrl-{len(rows):03d}", user=user, kind="control", type="control",
            gold_label=None, options_with_descriptions=None, item_ref=None, label_surfaces={}))
    return rows


# ---------------------------------------------------------------------------
# adversarial (near-miss, must NOT trigger)
# ---------------------------------------------------------------------------

def build_adversarial_rows(rng):
    specs = {
        "claim_handling": [
            "Quick policy question before I file anything: what's the dollar threshold where a claim "
            "needs director sign-off instead of clearing automatically?",
            "Hypothetically, if a claim had no receipts at all, would that change how it's routed? "
            "No claim has actually been submitted yet.",
            "Can you remind the team what counts as a reimbursable expense under our current policy?",
            "Someone mentioned the sign-off threshold changed last quarter. Did it, and to what?",
            "I'm updating the policy wording for next year. Draft a plain-English summary of how claims "
            "get routed, without reference to any specific claim.",
            "A colleague asked whether travel claims and supply claims use the same threshold. Do they?",
            "What happens if a director is unavailable to sign off on a claim that needs it?",
            "Explain the difference between 'auto-approved' and 'held for review' in our expense process.",
        ],
        "order_outcome": [
            "What's our per-item order limit before an order gets held for review?",
            "A customer is asking, in the abstract, how many units they're allowed to order before "
            "anything special happens. No order has been placed.",
            "Draft a FAQ entry explaining the order-quantity review policy, without a specific order.",
            "If a customer orders the same item twice in one week, does that count toward the limit?",
            "Can you summarize how cancelled orders differ from held-for-review orders under our policy?",
            "Someone asked whether the quantity limit is per order or per customer per month. Which is it?",
            "We're thinking of raising the review threshold. What would the new wording look like?",
            "A warehouse lead wants to know if the limit applies to bulk business accounts the same way.",
        ],
        "answer_type": [
            "Here's a question from the log: \"What time is it?\" -- is this one even worth routing, "
            "or should it just go to a generic assistant?",
            "Write a new trivia question about rivers for next week's quiz.",
            "A user typed just \"???\" into the box. There's nothing to classify here, right?",
            "Can you explain, in general, how our question router decides between an entity answer and a "
            "description answer?",
            "Rewrite this fragment into a complete question: \"orinoco river length\".",
            "A teammate asked what 'answer_type' even means as a category. Give them a one-line summary.",
            "Is a question that's really a complaint (\"why is the app so slow\") something the router "
            "should touch at all?",
            "Draft a short note to the team about why ambiguous one-word queries get skipped by the router.",
        ],
        "kb_category": [
            "We're drafting a brand-new encyclopedia entry about a startup. Can you write the opening "
            "sentence? There's no existing entry to categorize yet.",
            "What categories does our knowledge base support right now? Just list them.",
            "A teammate asked whether 'podcast' is one of our supported KB categories. Is it?",
            "This snippet is in two languages and garbled -- is there even a subject here to categorize?",
            "Explain the difference between the 'company' and 'officeholder' categories for the backfill team.",
            "Someone flagged that an article was miscategorized last week. How do we handle corrections?",
            "Draft the category legend we show new taggers, without reference to any specific article.",
            "A stub article is three words long: \"Big Blue River.\" Is that enough to categorize, or should "
            "it be sent back for expansion first?",
        ],
        "news_topic": [
            "Write a punchy headline for a story about a city council vote. Don't classify anything, "
            "just write the headline.",
            "What desks do we route wire copy to? Just list them for the new hire.",
            "A wire snippet came in with no text, just a dateline and a byline. Is there anything to route?",
            "Someone asked whether op-eds get routed the same way as straight news. Do they?",
            "Draft the one-paragraph style guide for how the newsdesk decides between business and scitech.",
            "A story mixes sports and business equally -- before you even look at a specific one, how do "
            "we usually break that kind of tie?",
            "Can you summarize, for training purposes, what 'world' news covers versus 'business'?",
            "This wire item is just a photo caption. Is a caption something the router should see at all?",
        ],
    }
    rows = []
    for tname, prompts in specs.items():
        for i, p in enumerate(prompts[:N_ADV_PER_TYPE]):
            rows.append(dict(
                id=f"p7rne-adv-{tname}-{i:02d}", user=p, kind="adversarial", type=tname,
                gold_label=None, options_with_descriptions=None, item_ref=None, label_surfaces={}))
    return rows


# ---------------------------------------------------------------------------
# replay pool (T8)
# ---------------------------------------------------------------------------

REPLAY_WRAP = {
    "banking77": ["A customer wrote in. Draft a short, friendly reply.\n\n\"{t}\"",
                  "Here's an incoming support message:\n\n\"{t}\"\n\nHow would you respond?"],
    "mnli": ["Does the second statement follow from the first? Explain briefly.\n\nA: {p}\nB: {h}"],
    "boolq": ["Answer the question using the passage.\n\nPassage: {p}\n\nQuestion: {h}"],
    "sst5": ["What's the overall feeling in this line? Reply in a sentence.\n\n\"{t}\"",
             "React to this review sentence like a thoughtful friend would:\n\n\"{t}\""],
    "yelp": ["What's your take on this review? Reply like a thoughtful friend.\n\n\"{t}\"",
             "Summarize this review in one sentence.\n\n\"{t}\""],
    "amazon": ["A customer left this product review. Draft a short thank-you reply.\n\n\"{t}\"",
               "Summarize what this reviewer liked or didn't like.\n\n\"{t}\""],
    "imdb": ["What's your honest reaction to this movie review?\n\n\"{t}\"",
             "Would you recommend watching this film, based on this review? Why or why not.\n\n\"{t}\""],
    "compositional": ["Walk through this step by step and tell me what you'd do:\n\n{p}\n\nCase: {h}"],
}


def replay_text(rec):
    src = rec["_meta"].get("source")
    s = rec["state"]
    if isinstance(s, dict):
        if "policy" in s:
            return s["policy"], s.get("case", "")
        if "document" in s:
            return s["document"], ""
    return s, ""


def mnli_premise_hyp(rec):
    q = rec["questions"].get("relation", {})
    instr = q.get("instructions")
    hyp = ""
    if isinstance(instr, dict):
        hyp = instr.get("question", "")
    elif isinstance(instr, str):
        hyp = instr
    m = re.search(r'"([^"]+)"', hyp)
    hyp = m.group(1) if m else hyp
    premise = state_text(rec) if isinstance(rec["state"], dict) else rec["state"]
    return premise, hyp


def boolq_passage_q(rec):
    q = rec["questions"].get("answer", {})
    question = q.get("instructions", "")
    passage = state_text(rec) if isinstance(rec["state"], str) else rec["state"].get("document", "")
    return passage, question


def build_replay_pool(rng, train_rows, used_text):
    by_src = collections.defaultdict(list)
    for rec in train_rows:
        src = rec["_meta"].get("source")
        if src in DECISION_FAMILIES:
            continue
        by_src[src].append(rec)
    for recs in by_src.values():
        rng.shuffle(recs)

    rows = []
    srcs = sorted(by_src)
    idx = {s: 0 for s in srcs}
    while len(rows) < REPLAY_TARGET and any(idx[s] < len(by_src[s]) for s in srcs):
        for src in srcs:
            if len(rows) >= REPLAY_TARGET:
                break
            if idx[src] >= len(by_src[src]):
                continue
            rec = by_src[src][idx[src]]
            idx[src] += 1
            wraps = REPLAY_WRAP.get(src)
            if not wraps:
                continue
            wrap = wraps[len(rows) % len(wraps)]
            try:
                if src == "mnli":
                    p, h = mnli_premise_hyp(rec)
                    user = wrap.format(p=p, h=h)
                elif src == "boolq":
                    p, h = boolq_passage_q(rec)
                    user = wrap.format(p=p, h=h)
                elif src == "compositional":
                    p, h = replay_text(rec)
                    user = wrap.format(p=p, h=h)
                else:
                    t = state_text(rec) if isinstance(rec["state"], (str, dict)) else rec["state"]
                    if isinstance(t, dict):
                        t = t.get("document", "")
                    user = wrap.format(t=t)
            except Exception:
                continue
            if not user.strip() or user in used_text:
                continue
            used_text.add(user)
            rows.append(dict(
                id=f"p7rrp-{len(rows):05d}", user=user, kind="replay", type=src,
                gold_label=None, options_with_descriptions=None, item_ref=rec["_meta"]["id"],
                label_surfaces={}))
    return rows


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(seed=SEED):
    dev_rows = load_jsonl(DEV_SUITE)
    canonical = load_canonical(dev_rows)
    rng = random.Random(seed)

    used_text = set()
    used_item_refs = set()

    decision_rows = []
    decision_rows += build_claim_rows(rng, dev_rows, canonical, used_text)
    decision_rows += build_order_rows(rng, dev_rows, canonical, used_text)
    decision_rows += build_text_rows(rng, dev_rows, canonical, "answer_type", ANSWER_WRAP,
                                      used_text, used_item_refs)
    decision_rows += build_text_rows(rng, dev_rows, canonical, "kb_category", KB_WRAP,
                                      used_text, used_item_refs)
    decision_rows += build_text_rows(rng, dev_rows, canonical, "news_topic", NEWS_WRAP,
                                      used_text, used_item_refs)

    control_rows = build_control_rows(rng)
    for r in control_rows:
        used_text.add(r["user"])
    adversarial_rows = build_adversarial_rows(rng)
    for r in adversarial_rows:
        used_text.add(r["user"])

    eval_rows = decision_rows + control_rows + adversarial_rows

    train_rows = load_jsonl(TRAIN_SUITE)
    replay_rows = build_replay_pool(rng, train_rows, set(used_text))

    return eval_rows, replay_rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# validator
# ---------------------------------------------------------------------------

def check(eval_rows, replay_rows, pilot_rows, dev_rows):
    fails, warns = [], []

    kinds = collections.Counter(r["kind"] for r in eval_rows)
    if kinds["decision"] != 300:
        fails.append(f"decision rows = {kinds['decision']}, want 300")
    if kinds["control"] != 100:
        fails.append(f"control rows = {kinds['control']}, want 100")
    if kinds["adversarial"] != 40:
        fails.append(f"adversarial rows = {kinds['adversarial']}, want 40")
    per_type = collections.Counter(r["type"] for r in eval_rows if r["kind"] == "decision")
    for t in TYPE_SOURCE:
        if per_type[t] != N_PER_TYPE:
            fails.append(f"decision type {t!r} has {per_type[t]} rows, want {N_PER_TYPE}")
    per_adv = collections.Counter(r["type"] for r in eval_rows if r["kind"] == "adversarial")
    for t in TYPE_SOURCE:
        if per_adv[t] != N_ADV_PER_TYPE:
            fails.append(f"adversarial type {t!r} has {per_adv[t]} rows, want {N_ADV_PER_TYPE}")

    if abs(len(replay_rows) - REPLAY_TARGET) > 50:
        warns.append(f"replay pool has {len(replay_rows)} rows, target ~{REPLAY_TARGET}")
    for r in replay_rows:
        if r["type"] in DECISION_FAMILIES:
            fails.append(f"replay row {r['id']} sourced from decision family {r['type']!r}")

    ids = [r["id"] for r in eval_rows] + [r["id"] for r in replay_rows]
    dup_ids = [k for k, v in collections.Counter(ids).items() if v > 1]
    if dup_ids:
        fails.append(f"duplicate ids: {dup_ids[:10]}")

    all_rows = eval_rows + replay_rows
    texts = [r["user"] for r in all_rows]
    dup_text = [k for k, v in collections.Counter(texts).items() if v > 1]
    if dup_text:
        fails.append(f"{len(dup_text)} duplicate user texts within the build")

    pilot_texts = {r["user"] for r in pilot_rows}
    overlap = [r["id"] for r in all_rows if r["user"] in pilot_texts]
    if overlap:
        fails.append(f"{len(overlap)} rows duplicate pilot prompt text: {overlap[:10]}")

    dev_texts = set()
    for rec in dev_rows:
        try:
            t = state_text(rec)
        except Exception:
            continue
        if isinstance(t, str):
            dev_texts.add(t)
    # the DEV items are *expected* to be embedded (that's the point); this just checks our own dedup
    # against re-using the same DEV row's raw item text as two different rows' whole user turn (which
    # would mean the wrapper did nothing).
    raw_dupe = [r["id"] for r in eval_rows if r["kind"] == "decision" and r["user"] in dev_texts]
    if raw_dupe:
        fails.append(f"{len(raw_dupe)} decision rows use the raw DEV state text verbatim as the whole "
                     f"user turn (wrapper missing): {raw_dupe[:10]}")

    flagged = []
    for r in eval_rows:
        if r["kind"] != "decision" or not r["gold_label"]:
            continue
        gold = str(r["gold_label"])
        terms = {gold.lower(), gold.replace("_", " ").lower()}
        low = r["user"].lower()
        if any(t in low for t in terms):
            flagged.append(r["id"])
    if flagged:
        warns.append(f"gold label text appears verbatim in the user turn for {len(flagged)} rows "
                     f"(review for genuine-restatement vs leak): {flagged[:10]}")

    return fails, warns


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="validate the already-written output files")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    if a.check:
        eval_rows = load_jsonl(OUT_EVAL)
        replay_rows = load_jsonl(OUT_REPLAY)
        pilot_rows = load_jsonl(PILOT)
        dev_rows = load_jsonl(DEV_SUITE)
        fails, warns = check(eval_rows, replay_rows, pilot_rows, dev_rows)
        for w in warns:
            print(f"WARN {w}")
        if fails:
            print(f"FAILED: {len(fails)} problem(s)")
            for f in fails:
                print(f"  - {f}")
            return 1
        print("all checks passed")
        return 0

    eval_rows, replay_rows = build(a.seed)
    write_jsonl(OUT_EVAL, eval_rows)
    write_jsonl(OUT_REPLAY, replay_rows)
    digest = sha256_of(OUT_EVAL)
    with open(OUT_HASH, "w", encoding="utf-8") as f:
        f.write(f"{digest}  phase7r-natural-eval.jsonl\n")

    kinds = collections.Counter(r["kind"] for r in eval_rows)
    per_type = collections.Counter(r["type"] for r in eval_rows if r["kind"] == "decision")
    per_adv = collections.Counter(r["type"] for r in eval_rows if r["kind"] == "adversarial")
    per_replay_src = collections.Counter(r["type"] for r in replay_rows)
    print(f"natural eval: {len(eval_rows)} rows -> {OUT_EVAL}")
    print(f"  kinds: {dict(kinds)}")
    print(f"  decision per type: {dict(per_type)}")
    print(f"  adversarial per type: {dict(per_adv)}")
    print(f"replay pool: {len(replay_rows)} rows -> {OUT_REPLAY}")
    print(f"  per source: {dict(sorted(per_replay_src.items()))}")
    print(f"sha256 ({OUT_EVAL}): {digest} -> {OUT_HASH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
