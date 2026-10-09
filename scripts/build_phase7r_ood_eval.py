"""Phase 7R: out-of-distribution (OOD) decision eval.

oracle/phase7r-natural-eval.jsonl (300 decision rows) is narrow: only 11 distinct wrapper first-lines
and 11 final-instruction lines, and 120/300 share their final instruction line verbatim with the
training prompts (oracle/phase7r-hybrid-prompts.jsonl). The 99% trigger rate measured on it is
in-family. This script builds a second, held-out decision eval meant to measure real generalisation
of the trigger behaviour: new items (never seen by either the natural eval or the hybrid training
prompts) wrapped in >=40 distinct, newly-authored styles that share no verbatim line with either file.

Item source, per decision type:
  - answer_type / kb_category / news_topic: evals/v7/decision-v7/test.jsonl's own trec/dbpedia14/agnews
    rows (source field matches; TEST split only -- neither the natural eval (development.jsonl) nor the
    hybrid prompts (train.jsonl) ever read this split, so item_ref is structurally disjoint from both).
  - claim_handling / order_outcome: test.jsonl carries no `legacy_policy`-source rows at all (that
    family isn't in the test split), but it does carry an equivalent contrastive-pair family under
    `_meta.source == "contrastive"`: `spend_threshold` (same `src: contrastive_spend_threshold`,
    same 2-option criteria dict as legacy_policy's claim_handling) and `quantity_limit` (same
    `src: contrastive_quantity_limit`, same 3-option criteria dict as order_outcome). These are used
    in place of legacy_policy, replayed under new entity names/items exactly like the natural eval and
    hybrid scripts do for their own legacy_policy rows, keeping the policy text and determining number
    byte-identical to the test.jsonl row (what keeps gold_label exactly that row's label).

Canonical instruction/option metadata (the fixed category definitions, not item text) is read from
development.jsonl via the shared load_canonical, same as the other two builders -- this is schema
metadata common to all three builders, not a source of eval/train leakage.

40 wrapper skeletons (20 for the two numeric types sharing a {policy}/{detail}/{question} shape, 20 for
the three text types sharing an {item}/{question} shape), covering: chatty messages, lowercase/typo/
no-punctuation chat, emails, Slack single-turn and multi-turn, support tickets, forms, JSON-ish records,
memos, bullet lists, long buried-in-the-middle narrative context (200-400 tokens, ~20% of rows per
type), instructions-first/last/inline placement, and several distinct final-question phrasings
("which bucket", "should this go through", "what's the right call", etc).

Disjointness, asserted programmatically against both oracle/phase7r-natural-eval.jsonl and
oracle/phase7r-hybrid-prompts.jsonl:
  - zero overlap of item_ref
  - zero overlap of exact `user` text
  - zero overlap of any single line of `user` text (first line / last line / every line), both as
    written and with digit runs masked to `#` (so e.g. two different ticket numbers formatted the same
    way don't count as "different lines")

Row schema matches the natural eval (no `system` key): {id, user, kind: "decision", type, gold_label,
options_with_descriptions, item_ref, label_surfaces}.

Usage:
    python scripts/build_phase7r_ood_eval.py
    python scripts/build_phase7r_ood_eval.py --selftest
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
sys.path.insert(0, ROOT)

from scripts.build_phase7r_natural_eval import (  # noqa: E402 -- reused, not copied
    DEV_SUITE, TYPE_SOURCE, dev_rows_for_type, load_canonical, load_jsonl, state_text,
)

TEST_SUITE = os.path.join(ROOT, "evals", "v7", "decision-v7", "test.jsonl")
NATURAL_EVAL = os.path.join(ROOT, "oracle", "phase7r-natural-eval.jsonl")
HYBRID_PROMPTS = os.path.join(ROOT, "oracle", "phase7r-hybrid-prompts.jsonl")
OUT_EVAL = os.path.join(ROOT, "oracle", "phase7r-ood-eval.jsonl")
OUT_HASH = os.path.join(ROOT, "oracle", "phase7r-ood-eval.sha256")

SEED = 71320269
N_PER_TYPE = 50
TEXT_TYPES = ["answer_type", "kb_category", "news_topic"]
NUMERIC_TYPES = ["claim_handling", "order_outcome"]
LONG_FRACTION = 10  # of 50 rows per type, how many use the long-context skeleton

AMOUNT_RE = re.compile(r"total is \$([\d,]+)")
QTY_RE = re.compile(r"quantity is (\d+)")

# ---------------------------------------------------------------------------
# entity pools -- disjoint from build_phase7r_natural_eval.py's and
# build_phase7r_hybrid_prompts.py's NAMES/ROLES_CLAIM/ITEMS_CLAIM/ITEMS_ORDER/DELIVERY pools.
# ---------------------------------------------------------------------------

NAMES = ["Astrid Kallio", "Renzo Villanueva", "Chidinma Okeke", "Birgit Lindgren", "Tapio Virtanen",
         "Camille Asante", "Hyun-woo Baek", "Zuri Mwangi", "Alistair Hume", "Paloma Reyes",
         "Ingemar Fosse", "Nadir Sultani", "Esme Roch", "Taavi Saar", "Bianca Moretti",
         "Wilhelmina Krog", "Amadou Diallo", "Celestine Okoro", "Rurik Thorvald", "Marisol Pena"]
ROLES_CLAIM = ["dispatch clerk", "compliance aide", "maintenance lead", "survey technician",
               "intake officer", "contracts assistant", "warehouse supervisor", "outreach coordinator"]
ITEMS_CLAIM = ["conference booth setup", "replacement uniforms", "vehicle inspection fees",
               "archival scanning", "translation services", "emergency repairs",
               "volunteer stipends", "signage reprints"]
ITEMS_ORDER = ["drafting table", "industrial fan", "packing crate", "security camera", "floor jack",
               "document scanner", "step stool", "extension cord reel", "pallet jack", "shop vac"]
DELIVERY = ["a remote outpost", "a leased annex", "a partner facility", "a seasonal pop-up site"]

CLAIM_QUESTIONS = [
    "What's the right call on this claim?", "Which bucket does this claim land in?",
    "Should this clear automatically or go to a director?",
    "Route this claim -- what's the disposition?", "Is this one auto-cleared or held for sign-off?",
    "What's the verdict on this claim?",
]
ORDER_QUESTIONS = [
    "What's the outcome here for this order?", "Which bucket does this order fall into?",
    "Should this order go through as-is?", "Route this order -- what's the disposition?",
    "Is this order cleared, held, or cancelled?", "What's the verdict on this order?",
]
ANSWER_QUESTIONS = [
    "What kind of answer is this looking for?", "Which answer bucket does this belong to?",
    "Classify the expected answer type.", "What's the right answer category here?",
    "Should this route to the entity-answer flow or something else?",
    "What type of answer satisfies this question?",
]
KB_QUESTIONS = [
    "Which category does this belong to?", "What's the right category tag?",
    "Classify this entry.", "Which bucket should this go in?",
    "Should this be tagged as-is, or does it need another category?", "What category fits best?",
]
NEWS_QUESTIONS = [
    "Which desk should this go to?", "What's the topic here?", "Classify this wire item.",
    "Which section does this belong in?", "Should this run under its obvious section or somewhere else?",
    "What's the right call on section placement?",
]
QUESTIONS_FOR = {
    "claim_handling": CLAIM_QUESTIONS, "order_outcome": ORDER_QUESTIONS,
    "answer_type": ANSWER_QUESTIONS, "kb_category": KB_QUESTIONS, "news_topic": NEWS_QUESTIONS,
}

# ---------------------------------------------------------------------------
# wrapper skeletons
# ---------------------------------------------------------------------------


def _lower(s):
    return s.lower()


def _nopunct_lower(s):
    return re.sub(r"[?.!,]", "", s).strip().lower()


# Group N (claim_handling / order_outcome): {policy}, {detail}, {question}, {ticket}, plus
# *_lower/_nopunct derived keys. 20 distinct skeletons.
N_SKELETONS = [
    ("chatty_casual",
     "hey -- got a sec? need a quick read on this one.\n\nPolicy: {policy}\n\n{detail}\n\n{question}"),
    ("chatty_lowercase_typos",
     "hey quick one for u -- policy is {policy_lower} and {detail_lower} {question_lower_nopunct}"),
    ("email",
     "Subject: Quick review needed\n\nHi team,\n\nCould someone take a look at the following before "
     "end of day?\n\nPolicy: {policy}\n\n{detail}\n\n{question}\n\nThanks,\nOps"),
    ("email_v2",
     "Subject: Need a ruling on this one\n\nHi,\n\n{detail}\n\nFor reference, the policy is: {policy}\n\n"
     "{question}\n\nBest,\nQueue Bot"),
    ("slack_single",
     "9:42 AM\n@ops-queue: dropping this one in before standup -- {detail} policy says {policy}. "
     "{question}"),
    ("slack_thread",
     "9:10 AM Dana: morning! got a case for review\n9:11 AM Dana: {detail}\n9:11 AM Dana: "
     "policy: {policy}\n9:12 AM Dana: {question}"),
    ("ticket",
     "SUPPORT TICKET\nPriority: Normal\nCategory: Review\n\nDescription:\n{detail}\n"
     "Applicable policy: {policy}\n\n{question}"),
    ("ticket_v2",
     "[Ticket #{ticket}]\nStatus: Open\n\n{detail}\nPolicy on file: {policy}\n\n"
     "Requested action: {question}"),
    ("form",
     "Review form\nPolicy: {policy}\nCase notes: {detail}\nDecision requested: {question}"),
    ("json_ish",
     '{{"policy": "{policy}", "case": "{detail}", "task": "{question}"}}'),
    ("memo",
     "MEMORANDUM\nTo: Review Desk\nFrom: Intake\nRe: pending case\n\n{detail} The governing policy "
     "reads: {policy}\n\n{question}"),
    ("bullets",
     "- Policy: {policy}\n- Case: {detail}\n- Ask: {question}"),
    ("buried_mid",
     "We've been getting a lot of these lately and the team wanted a second pair of eyes before "
     "anything ships. {detail} For context, this all falls under the standing policy: {policy}. "
     "Anyway, once you've had a look -- {question_lower}"),
    ("instructions_first",
     "{question}\n\nPolicy: {policy}\n\n{detail}"),
    ("instructions_inline",
     "Given that {policy_lower}, and that {detail_lower} -- {question_lower}"),
    ("long_context",
     "Quick bit of background before I get to the actual case -- we've had a busier-than-usual week "
     "on the review queue, a couple of the usual reviewers are out, and the backlog's been creeping "
     "up, so I wanted to make sure this one doesn't sit for another day without at least a first "
     "pass. {detail} Just to be thorough about it, here's the policy that governs this particular "
     "kind of case: {policy} I know this probably isn't the most exciting thing to look at this week, and "
     "most of these end up being pretty routine once you actually read them, but it still needs a "
     "clean answer on file before it gets archived, and I'd honestly rather ask now than have it "
     "bounce back later for being incomplete or ambiguous. So, with all of that said: {question}"),
    ("no_punct_lowercase",
     "ok so {detail_lower} and the policy is {policy_lower} so {question_lower_nopunct}"),
    ("which_bucket",
     "{detail}\n\nPolicy: {policy}\n\nWhich bucket does this go in?"),
    ("should_go_through",
     "{detail}\n\nPolicy: {policy}\n\nShould this go through as-is, or does it need to be flagged?"),
    ("whats_the_call",
     "{detail}\n\nPolicy: {policy}\n\nWhat's the right call here?"),
]

# Group T (answer_type / kb_category / news_topic): {item}, {question}, {ticket}, plus *_lower/_nopunct
# derived keys. 20 distinct skeletons.
T_SKELETONS = [
    ("chatty_casual",
     "hey, can you take a look at this one?\n\n\"{item}\"\n\n{question}"),
    ("chatty_lowercase_typos",
     "yo can u sort this real quick\n\n\"{item_lower}\"\n\n{question_lower_nopunct}"),
    ("email",
     "Subject: Tagging needed\n\nHi,\n\nFound this in the queue, needs a tag before it moves on:\n\n"
     "\"{item}\"\n\n{question}\n\nThanks"),
    ("email_v2",
     "Subject: One for you\n\nHi there,\n\n\"{item}\"\n\n{question}\n\nCheers"),
    ("slack_single",
     "10:03 AM @intake: next one -- \"{item}\" -- {question_lower}"),
    ("slack_thread",
     "10:15 AM Sam: ok next\n10:15 AM Sam: \"{item}\"\n10:16 AM Sam: {question}"),
    ("ticket",
     "SUPPORT TICKET\nCategory: Tagging\n\nItem text:\n\"{item}\"\n\n{question}"),
    ("ticket_v2",
     "[Ticket #{ticket}]\nQueue: Backfill\n\n\"{item}\"\n\n{question}"),
    ("form",
     "Entry review form\nText: \"{item}\"\nAction requested: {question}"),
    ("json_ish",
     '{{"text": "{item}", "task": "{question}"}}'),
    ("memo",
     "MEMORANDUM\nTo: Tagging desk\nRe: next item\n\n\"{item}\"\n\n{question}"),
    ("bullets",
     "- Text: \"{item}\"\n- Ask: {question}"),
    ("buried_mid",
     "Been working through the backlog all morning and figured I'd flag this one specially since "
     "it's a little unusual. The text in question reads: \"{item}\" -- nothing too complicated, just "
     "need a clean read on it. {question}"),
    ("instructions_first",
     "{question}\n\n\"{item}\""),
    ("instructions_inline",
     "Looking at the following: \"{item}\" -- {question_lower}"),
    ("long_context",
     "Before I get into the actual item, a bit of context: this queue's been busier than usual this "
     "week, and a couple of the regular taggers are out, so things are moving slower than we'd like "
     "and nothing's had a second pass in days. Here's the next one that needs a decision: \"{item}\". "
     "It's probably routine, and most of the backlog turns out that way once someone actually reads "
     "it, but I'd rather have it double-checked now than let it sit mistagged in the index for "
     "months and have someone else untangle it later. So, bottom line: {question}"),
    ("no_punct_lowercase",
     "ok so heres the next one \"{item_lower}\" and {question_lower_nopunct}"),
    ("which_bucket",
     "\"{item}\"\n\nWhich bucket does this belong to?"),
    ("should_route",
     "\"{item}\"\n\nShould this route to the usual handler, or does it need a second look?"),
    ("whats_the_call",
     "\"{item}\"\n\nWhat's the right call on this one?"),
]


def style_sequence(rng, skeleton_names, n, long_name="long_context", long_count=LONG_FRACTION):
    others = [s for s in skeleton_names if s != long_name]
    seq = [long_name] * long_count
    i = 0
    while len(seq) < n:
        seq.append(others[i % len(others)])
        i += 1
    seq = seq[:n]
    rng.shuffle(seq)
    return seq


# ---------------------------------------------------------------------------
# test.jsonl row filtering
# ---------------------------------------------------------------------------

def contrastive_rows(test_rows, family):
    out = []
    for rec in test_rows:
        meta = rec["_meta"]
        if meta.get("source") != "contrastive" or meta.get("family") != family:
            continue
        if meta.get("variant") != "clean":
            continue
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# row builders
# ---------------------------------------------------------------------------

def build_claim_rows(rng, test_rows, canonical, used_text):
    instr, opts, desc = canonical["claim_handling"]
    owd = [[o, desc[o]] for o in opts]
    base = contrastive_rows(test_rows, "spend_threshold")
    reps = (N_PER_TYPE + len(base) - 1) // len(base)
    pool = base * reps
    rng.shuffle(pool)
    seq = style_sequence(rng, [s for s, _ in N_SKELETONS], N_PER_TYPE)
    skel_map = dict(N_SKELETONS)

    rows, i, pi = [], 0, 0
    while i < N_PER_TYPE and pi < len(pool):
        rec = pool[pi]
        pi += 1
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
        ref = 93000 + i
        detail = (f"{name}, {role}, filed a claim for {item} -- claim total ${amount}, receipts on "
                  f"file (ref #{ref}).")
        question = CLAIM_QUESTIONS[i % len(CLAIM_QUESTIONS)]
        tpl = skel_map[seq[i]]
        user = tpl.format(policy=policy, detail=detail, question=question, ticket=ref,
                           policy_lower=_lower(policy), detail_lower=_lower(detail),
                           question_lower=_lower(question), question_lower_nopunct=_nopunct_lower(question))
        if user in used_text:
            continue
        used_text.add(user)
        rows.append(dict(
            id=f"p7rood-claim_handling-{i:02d}", user=user, kind="decision", type="claim_handling",
            gold_label=label, options_with_descriptions=owd,
            item_ref=rec["_meta"]["id"], label_surfaces={}))
        i += 1
    return rows


def build_order_rows(rng, test_rows, canonical, used_text):
    instr, opts, desc = canonical["order_outcome"]
    owd = [[o, desc[o]] for o in opts]
    base = contrastive_rows(test_rows, "quantity_limit")
    reps = (N_PER_TYPE + len(base) - 1) // len(base)
    pool = base * reps
    rng.shuffle(pool)
    seq = style_sequence(rng, [s for s, _ in N_SKELETONS], N_PER_TYPE)
    skel_map = dict(N_SKELETONS)

    rows, i, pi = [], 0, 0
    while i < N_PER_TYPE and pi < len(pool):
        rec = pool[pi]
        pi += 1
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
        ref = 64000 + i
        detail = f"{name} ordered {qty} units of the {item}, shipping to {delivery} (order #{ref})."
        question = ORDER_QUESTIONS[i % len(ORDER_QUESTIONS)]
        tpl = skel_map[seq[i]]
        user = tpl.format(policy=policy, detail=detail, question=question, ticket=ref,
                           policy_lower=_lower(policy), detail_lower=_lower(detail),
                           question_lower=_lower(question), question_lower_nopunct=_nopunct_lower(question))
        if user in used_text:
            continue
        used_text.add(user)
        rows.append(dict(
            id=f"p7rood-order_outcome-{i:02d}", user=user, kind="decision", type="order_outcome",
            gold_label=label, options_with_descriptions=owd,
            item_ref=rec["_meta"]["id"], label_surfaces={}))
        i += 1
    return rows


def build_text_rows(rng, test_rows, canonical, tname, used_text, used_refs):
    instr, opts, desc = canonical[tname]
    owd = [[o, desc[o]] for o in opts]
    qid = TYPE_SOURCE[tname][1]
    base = dev_rows_for_type(test_rows, tname)
    rng.shuffle(base)
    seq = style_sequence(rng, [s for s, _ in T_SKELETONS], N_PER_TYPE)
    skel_map = dict(T_SKELETONS)
    questions = QUESTIONS_FOR[tname]

    rows = []
    for rec in base:
        if len(rows) >= N_PER_TYPE:
            break
        if rec["_meta"]["id"] in used_refs:
            continue
        text = state_text(rec)
        i = len(rows)
        question = questions[i % len(questions)]
        ref = 55000 + i
        tpl = skel_map[seq[i]]
        user = tpl.format(item=text, question=question, ticket=ref,
                           item_lower=_lower(text), question_lower=_lower(question),
                           question_lower_nopunct=_nopunct_lower(question))
        if user in used_text:
            continue
        used_text.add(user)
        used_refs.add(rec["_meta"]["id"])
        label = rec["questions"][qid]["label"]
        rows.append(dict(
            id=f"p7rood-{tname}-{i:02d}", user=user, kind="decision", type=tname,
            gold_label=label, options_with_descriptions=owd,
            item_ref=rec["_meta"]["id"], label_surfaces={}))
    return rows


def build(seed=SEED):
    dev_rows = load_jsonl(DEV_SUITE)
    canonical = load_canonical(dev_rows)
    test_rows = load_jsonl(TEST_SUITE)
    rng = random.Random(seed)

    used_text, used_refs = set(), set()
    rows = []
    rows += build_claim_rows(rng, test_rows, canonical, used_text)
    rows += build_order_rows(rng, test_rows, canonical, used_text)
    for tname in TEXT_TYPES:
        rows += build_text_rows(rng, test_rows, canonical, tname, used_text, used_refs)
    return rows


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

DIGIT_RE = re.compile(r"\d+")

# The claim_handling/order_outcome families only have a handful of canonical policy-sentence templates
# (same wording in dev/train/test, only the threshold number differs), and the task requires keeping
# the policy statement verbatim in the prompt so the answer stays derivable. That means a "Policy: ..."
# line is EXPECTED to collide with the same line in the natural eval / hybrid prompts (which embed the
# identical template) -- this isn't a wrapper-authoring failure, it's a property of the shared dataset.
# These lines are filtered out of the hard overlap check and reported separately as a WARN.
CLAIM_POLICY_RE = re.compile(
    r"^Expense claims of \$[\d,]+ or less are approved automatically\. "
    r"Claims above \$[\d,]+ require director sign-off\.$")
ORDER_POLICY_RE = re.compile(
    r"^Customers may order at most \d+ units of any single item per order\. "
    r"Orders up to double the limit are held for review; larger orders are cancelled\.$")


def is_known_policy_line(line):
    s = line[len("Policy: "):] if line.startswith("Policy: ") else line
    return bool(CLAIM_POLICY_RE.match(s) or ORDER_POLICY_RE.match(s))


def line_sets(rows, exclude_policy_lines=False):
    firsts, lasts, alllines = set(), set(), set()
    for r in rows:
        lines = [l.strip() for l in r["user"].split("\n") if l.strip()]
        if exclude_policy_lines:
            lines = [l for l in lines if not is_known_policy_line(l)]
        if not lines:
            continue
        firsts.add(lines[0])
        lasts.add(lines[-1])
        alllines.update(lines)
    return firsts, lasts, alllines


def masked(strings):
    return {DIGIT_RE.sub("#", s) for s in strings}


def check(ood_rows, nat_rows, hyb_rows):
    fails, warns = [], []

    kinds = collections.Counter(r["kind"] for r in ood_rows)
    if kinds["decision"] != N_PER_TYPE * 5:
        fails.append(f"decision rows = {kinds['decision']}, want {N_PER_TYPE * 5}")
    per_type = collections.Counter(r["type"] for r in ood_rows)
    for t in TYPE_SOURCE:
        if per_type[t] != N_PER_TYPE:
            fails.append(f"type {t!r} has {per_type[t]} rows, want {N_PER_TYPE}")

    ids = [r["id"] for r in ood_rows]
    dup_ids = [k for k, v in collections.Counter(ids).items() if v > 1]
    if dup_ids:
        fails.append(f"duplicate ids: {dup_ids[:10]}")

    texts = [r["user"] for r in ood_rows]
    dup_text = [k for k, v in collections.Counter(texts).items() if v > 1]
    if dup_text:
        fails.append(f"{len(dup_text)} duplicate user texts within the OOD build")

    other_rows = nat_rows + hyb_rows
    other_refs = {r["item_ref"] for r in other_rows if r.get("item_ref")}
    ref_overlap = {r["item_ref"] for r in ood_rows} & other_refs
    if ref_overlap:
        fails.append(f"{len(ref_overlap)} item_ref overlap with natural/hybrid: {sorted(ref_overlap)[:10]}")

    other_text = {r["user"] for r in other_rows}
    text_overlap = {r["user"] for r in ood_rows} & other_text
    if text_overlap:
        fails.append(f"{len(text_overlap)} exact user-text overlap with natural/hybrid")

    policy_line_rows = sum(1 for r in ood_rows
                            if any(is_known_policy_line(l.strip())
                                   for l in r["user"].split("\n") if l.strip()))
    if policy_line_rows:
        warns.append(f"{policy_line_rows} rows embed one of the dataset's ~8 canonical policy "
                     f"sentences verbatim (digits aside) -- these necessarily share that single line "
                     f"with the natural eval / hybrid prompts, which embed the same canonical policy "
                     f"text; this is a property of the shared decision-v7 dataset (only a handful of "
                     f"fixed policy templates exist for claim_handling/order_outcome), not a wrapper "
                     f"collision, and is excluded from the overlap check below.")

    ood_first, ood_last, ood_all = line_sets(ood_rows, exclude_policy_lines=True)
    oth_first, oth_last, oth_all = line_sets(other_rows, exclude_policy_lines=True)

    first_overlap = ood_first & oth_first
    last_overlap = ood_last & oth_last
    line_overlap = ood_all & oth_all
    if first_overlap:
        fails.append(f"{len(first_overlap)} first-line overlap: {sorted(first_overlap)[:5]}")
    if last_overlap:
        fails.append(f"{len(last_overlap)} last-line overlap: {sorted(last_overlap)[:5]}")
    if line_overlap:
        fails.append(f"{len(line_overlap)} full-line overlap: {sorted(line_overlap)[:5]}")

    m_first_overlap = masked(ood_first) & masked(oth_first)
    m_last_overlap = masked(ood_last) & masked(oth_last)
    m_line_overlap = masked(ood_all) & masked(oth_all)
    if m_first_overlap:
        fails.append(f"{len(m_first_overlap)} digit-masked first-line overlap: {sorted(m_first_overlap)[:5]}")
    if m_last_overlap:
        fails.append(f"{len(m_last_overlap)} digit-masked last-line overlap: {sorted(m_last_overlap)[:5]}")
    if m_line_overlap:
        fails.append(f"{len(m_line_overlap)} digit-masked full-line overlap: {sorted(m_line_overlap)[:5]}")

    n_skeletons = len(N_SKELETONS) + len(T_SKELETONS)
    if n_skeletons < 40:
        fails.append(f"only {n_skeletons} wrapper skeletons, want >=40")

    flagged = []
    for r in ood_rows:
        if not r["gold_label"]:
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


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def token_len_report(rows):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it", local_files_only=True)
    except Exception as e:
        print(f"  (tokenizer unavailable, skipping token-length report: {e})")
        return
    lens = [len(tok(r["user"], add_special_tokens=False)["input_ids"]) for r in rows]
    lens.sort()
    n = len(lens)

    def pct(p):
        return lens[min(n - 1, int(p * n))]
    print(f"  token lengths (user turn, gemma tokenizer): min={lens[0]} p25={pct(.25)} "
          f"median={pct(.5)} p75={pct(.75)} p90={pct(.9)} max={lens[-1]}")
    long_n = sum(1 for l in lens if l >= 200)
    print(f"  rows with >=200 tokens: {long_n}/{n} ({100 * long_n / n:.0f}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="validate the already-written output file")
    ap.add_argument("--selftest", action="store_true", help="build in-memory, validate, don't write")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    nat_rows = load_jsonl(NATURAL_EVAL)
    hyb_rows = load_jsonl(HYBRID_PROMPTS)

    if a.check:
        ood_rows = load_jsonl(OUT_EVAL)
        fails, warns = check(ood_rows, nat_rows, hyb_rows)
        for w in warns:
            print(f"WARN {w}")
        if fails:
            print(f"FAILED: {len(fails)} problem(s)")
            for f in fails:
                print(f"  - {f}")
            return 1
        print("all checks passed")
        return 0

    ood_rows = build(a.seed)
    fails, warns = check(ood_rows, nat_rows, hyb_rows)

    kinds = collections.Counter(r["kind"] for r in ood_rows)
    per_type = collections.Counter(r["type"] for r in ood_rows)
    print(f"OOD eval: {len(ood_rows)} rows" + (f" -> {OUT_EVAL}" if not a.selftest else " (selftest)"))
    print(f"  kinds: {dict(kinds)}")
    print(f"  decision per type: {dict(per_type)}")
    print(f"  wrapper skeletons: {len(N_SKELETONS)} numeric + {len(T_SKELETONS)} text = "
          f"{len(N_SKELETONS) + len(T_SKELETONS)}")
    for w in warns:
        print(f"  WARN {w}")
    if fails:
        print(f"  FAILED: {len(fails)} problem(s)")
        for f in fails:
            print(f"    - {f}")
    else:
        print("  disjointness/format checks: passed")
    token_len_report(ood_rows)
    print("  6 sample prompts:")
    sample_ids = []
    for t in TYPE_SOURCE:
        sample_ids += [r for r in ood_rows if r["type"] == t][:2][:1]
    samples = (sample_ids + ood_rows)[:6]
    seen_ids = set()
    shown = 0
    for r in ood_rows:
        if shown >= 6:
            break
        if r["id"] in seen_ids:
            continue
        seen_ids.add(r["id"])
        print(f"    [{r['id']}] type={r['type']} gold_label={r['gold_label']}")
        print(f"      user: {r['user'][:300]!r}")
        shown += 1

    if a.selftest:
        return 1 if fails else 0
    if fails:
        print("NOT writing output: checks failed")
        return 1

    write_jsonl(OUT_EVAL, ood_rows)
    digest = sha256_of(OUT_EVAL)
    with open(OUT_HASH, "w", encoding="utf-8") as f:
        f.write(f"{digest}  phase7r-ood-eval.jsonl\n")
    print(f"-> {OUT_EVAL}")
    print(f"sha256: {digest} -> {OUT_HASH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
