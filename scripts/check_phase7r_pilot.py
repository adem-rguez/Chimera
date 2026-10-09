"""Validator for the Phase 7R reasoning pilot (oracle/phase7-reasoning-pilot-v2.jsonl).

Phase 7R = inline `<|decide|>` calls emitted INSIDE the Gemma-4 thinking channel
(`<|channel>thought\\n ... \\n<channel|>`, opened by `<|think|>\\n` in the system turn), scored by the Kev
pointer head, with `<|result|>label<|/result|>` injected by the runtime. See reports/16-phase7-reasoning-
proposal.md. This script checks the authored pilot against every constraint that proposal commits to, and
recomputes all token fields from scratch with the real tokenizer instead of trusting the file.

Independence: the canonical instruction text and option list for each decision type are NOT hard-coded
here -- they are read back out of evals/v7/decision-v7/development.jsonl (the suite the decision model's
per-type accuracies were measured on), so a drifted option list or a reworded instruction fails the run.

Usage (laptop, CPU, tokenizer only -- no torch forward passes, no network beyond the HF cache):
    python scripts/check_phase7r_pilot.py
    python scripts/check_phase7r_pilot.py --pilot oracle/phase7-reasoning-pilot-v2.jsonl --verbose
Exit code 0 = all checks passed, 1 = at least one failure.
"""
import argparse
import collections
import json
import os
import re
import sys

MODEL = "google/gemma-4-E4B-it"
REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
SUITE = "evals/v7/decision-v7/development.jsonl"

THINK_OPEN = "<|channel>thought\n"
THINK_CLOSE = "\n<channel|>"
THINK_ENABLE = "<|think|>\n"
TRIGGER = "<|decide|>"
RESULT_OPEN = "<|result|>"
RESULT_CLOSE = "<|/result|>"
NONE_KEY = "none_of_these"          # kev/suite.py contrast_cases' reserved none-of-the-above key

# decision type -> (suite source, question id, legacy-policy family or None). The instruction and the
# option list are pulled from the suite rows this resolves to; see module docstring.
TYPE_SOURCE = {
    "banking_intent": ("banking77", "intent", None),
    "news_topic": ("agnews", "topic", None),
    "kb_category": ("dbpedia14", "category", None),
    "answer_type": ("trec", "answer_type", None),
    "claim_handling": ("legacy_policy", "decision", "contrastive_spend_threshold"),
    "order_outcome": ("legacy_policy", "decision", "contrastive_quantity_limit"),
}

THINK_MIN, THINK_MAX = 150, 600     # decoded thinking tokens, SHORT form (reports/16, data design)
NGRAM_N, NGRAM_MAX_ROWS = 6, 3      # a 6-gram in more than this many rows is authored boilerplate
CONTROL_MIN_FRACTION = 0.18         # "~20% controls"
PER_TYPE_MIN = 5

# Giveaway phrases per gold label, checked (case-insensitively) against the decoded thinking BEFORE each
# call in addition to the label string itself. Deliberately phrase-level: single stopwords that happen to
# live inside a label ("number", "no", "where", "record") produce false positives in any honest reasoning
# trace, so they are not listed and the backstop for them is human review, which is stated in reports/16.
SYNONYMS = {
    # banking77
    "card_payment_not_recognised": ["payment not recognised", "doesn't recognise", "does not recognise",
                                    "don't recognise", "unrecognised charge"],
    "balance_not_updated_after_bank_transfer": ["balance not updated", "balance is unchanged",
                                                "balance has not updated", "balance hasn't updated"],
    "transfer_not_received_by_recipient": ["not received by the recipient", "recipient has not received",
                                           "recipient never received", "transfer not received"],
    "wrong_exchange_rate_for_cash_withdrawal": ["wrong exchange rate", "wrong rate", "rate was wrong",
                                                "bad exchange rate", "mispriced conversion"],
    "pin_blocked": ["pin blocked", "pin is blocked", "blocked the pin", "pin lock"],
    "transaction_charged_twice": ["charged twice", "double charge", "duplicate charge", "charged me twice"],
    "declined_card_payment": ["declined card payment", "card payment declined", "card was declined",
                              "declined payment"],
    "pending_card_payment": ["pending card payment", "payment is pending", "still pending"],
    # agnews topic
    "world": ["world news", "international affairs", "foreign affairs", "foreign desk"],
    "sports": ["sports desk", "a sport ", "sporting"],
    "business": ["business desk", "business story", "a business "],
    "scitech": ["sci-tech", "science and technology", "science desk", "technology desk"],
    # dbpedia14 category
    "meanoftransportation": ["means of transportation", "means of transport", "mode of transport",
                             "a conveyance"],
    "company": ["a company", "the company", "a firm", "a corporation"],
    "naturalplace": ["natural place", "natural feature", "landscape feature"],
    "village": ["a village", "a hamlet", "a settlement"],
    "writtenwork": ["written work", "a treatise", "literary work"],
    "officeholder": ["office holder", "office-holder", "an officeholder", "holder of public office"],
    "album": ["an album", "studio album", "an lp"],
    "animal": ["an animal", "a species", "a creature"],
    # trec answer_type
    "abbreviation": ["an abbreviation", "an acronym", "an initialism", "stands for", "expansion of"],
    "entity": ["an entity", "a named entity", "a named thing"],
    "description": ["a description", "descriptive answer", "a definition", "an explanation of the"],
    "human": ["a human", "a person", "an individual", "someone's name"],
    "location": ["a location", "a place name", "a toponym", "geographic answer"],
    # legacy policy
    "auto_approved": ["auto approved", "auto-approved", "approved automatically",
                      "approved without further review", "clears on its own", "clear automatically"],
    "director_signoff": ["director sign-off", "director signoff", "director sign off",
                         "requires a director", "goes to a director"],
    "rejected": ["rejected outright", "is rejected"],
    "within_limit": ["within limit", "within the limit", "processed normally", "processes normally"],
    "slightly_over": ["slightly over", "held for review"],
    "far_over": ["far over", "is cancelled", "be cancelled"],
    # none-of-the-above
    NONE_KEY: ["none of these", "none of the above", "none of them fit", "no option matches"],
}

REQUIRED_FIELDS = {
    "id": str, "task": str, "kind": str, "decision_types": list, "system": str, "user": str,
    "thinking_short": str, "thinking_long": str, "answer": str, "calls": list,
    "label_surfaces": dict,
    "tokens_decoded_short": int, "tokens_injected_short": int, "tokens_total_short": int,
    "tokens_decoded_long": int, "delta_decoded": int, "delta_total": int,
    "tokens_thinking_decoded_short": int, "tokens_thinking_decoded_long": int, "tokens_answer": int,
}
CALL_FIELDS = {"type": str, "instruction": str, "options": list, "gold_label": str,
               "position_char_start": int}
KINDS = {"call", "control_nocall", "control_none"}


# ---------------------------------------------------------------------------
# canonical types, read back out of the frozen suite
# ---------------------------------------------------------------------------

def load_canonical(suite_path):
    """-> {type: (instruction, frozenset of option keys)} read from the decision-v7 development partition.

    The suite fixes the instruction string and the option SET but NOT the option ORDER (kev/data.py
    shuffles criteria per row, which is also why the suite can measure order sensitivity at all: trained
    E4B flip 0.067 dev vs 0.707 zero-shot). `--force-options` splices ONE fixed order at serve time, so
    the pilot's own order is checked for self-consistency per type instead, further down."""
    found_instr = collections.defaultdict(set)
    found_opts = collections.defaultdict(collections.Counter)
    with open(suite_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            src = rec["_meta"].get("source")
            for qid, q in rec["questions"].items():
                # kev/data.py `_instr` wraps ~15% of instructions as {"question", "focus"}; the inline-call
                # form composes a plain string (scripts/build_phase7.py INSTR), so only the plain rows
                # define what "canonical" means here.
                if not isinstance(q.get("criteria"), dict) or not isinstance(q.get("instructions"), str):
                    continue
                for tname, (want_src, want_qid, want_fam) in TYPE_SOURCE.items():
                    if src != want_src or qid != want_qid:
                        continue
                    if want_fam is not None and q.get("src") != want_fam:
                        continue
                    found_instr[tname].add(q["instructions"])
                    found_opts[tname][frozenset(q["criteria"])] += 1
    out, errs = {}, []
    for tname in TYPE_SOURCE:
        instrs, opts = found_instr.get(tname, set()), found_opts.get(tname, collections.Counter())
        if len(instrs) != 1:
            errs.append(f"type {tname!r}: suite gives {len(instrs)} distinct instruction strings"
                        f" (expected exactly 1)")
        if not opts:
            errs.append(f"type {tname!r}: no rows in the suite")
            continue
        # The suite's clean rows all carry the full option set; the minority sets are its own
        # none-of-the-above augmentation (kev/data.py: adds `none_of_these`, sometimes after dropping the
        # gold). Canonical = the modal set, and every other observed set must be a subset of it plus the
        # none key -- anything else means the suite carries options the pilot does not know about.
        base, _n = opts.most_common(1)[0]
        for other in opts:
            if not other <= (base | {NONE_KEY}):
                errs.append(f"type {tname!r}: suite option set {sorted(other - base)} is not explainable"
                            f" as the modal set plus {NONE_KEY!r}")
        if len(instrs) == 1:
            out[tname] = (next(iter(instrs)), base)
    return out, errs


# ---------------------------------------------------------------------------
# token accounting, recomputed from scratch
# ---------------------------------------------------------------------------

def injected_spans(text):
    """Char spans written by the RUNTIME rather than decoded by the model: from the end of each
    '<|decide|>' trigger through the end of the matching '<|/result|>'. With --force-options the
    instruction + option list is prefilled, the pointer head supplies the label, and the runtime closes
    the span; only the trigger itself is decoded."""
    spans = []
    for m in re.finditer(re.escape(TRIGGER), text):
        close = text.find(RESULT_CLOSE, m.end())
        if close < 0:
            raise ValueError("unterminated call")
        spans.append((m.end(), close + len(RESULT_CLOSE)))
    return spans


def split_tokens(tok, text):
    """-> (decoded, injected) token counts, from ONE tokenization plus offset_mapping (the technique
    scripts/eval_phase7.py's build_decide_and_canonical_ids uses to cut at a char boundary). A token is
    injected iff its START offset falls inside an injected span."""
    spans = injected_spans(text)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    dec = inj = 0
    for a, _b in enc["offset_mapping"]:
        if any(s <= a < e for s, e in spans):
            inj += 1
        else:
            dec += 1
    return dec, inj


# ---------------------------------------------------------------------------
# leak check
# ---------------------------------------------------------------------------

QUOTE_RE = re.compile(r"\"([^\"\n]{3,})\"|'([^'\n]{3,})'")


def strip_runtime_and_quotes(text, user):
    """The text the MODEL decoded, with (a) runtime-injected spans removed (an earlier call's option list
    names every label of its type, and its result span names its own label -- neither was decoded, so
    neither can leak) and (b) quoted spans whose contents appear verbatim in the user turn removed
    (quoting the user's own words back is legitimate evidence, not a leaked answer)."""
    out, cursor = [], 0
    for s, e in injected_spans(text):
        out.append(text[cursor:s])
        cursor = e
    out.append(text[cursor:])
    kept = "".join(out)
    return QUOTE_RE.sub(lambda m: "" if (m.group(1) or m.group(2)) in user else m.group(0), kept)


def leak_terms(label):
    terms = {label.lower(), label.replace("_", " ").lower(), label.replace("_", "-").lower()}
    terms.update(s.lower() for s in SYNONYMS.get(label, []))
    return sorted(t for t in terms if t)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def check(rows, canonical, tok):
    fails, warns = [], []

    def bad(rid, msg):
        fails.append(f"{rid}: {msg}")

    ids = [r.get("id") for r in rows]
    if len(set(ids)) != len(ids):
        fails.append(f"duplicate ids: {[k for k, v in collections.Counter(ids).items() if v > 1]}")

    for r in rows:
        rid = r.get("id", "<no id>")

        # --- schema
        for key, typ in REQUIRED_FIELDS.items():
            if key not in r:
                bad(rid, f"missing field {key!r}")
            elif not isinstance(r[key], typ) or isinstance(r[key], bool):
                bad(rid, f"field {key!r} is {type(r[key]).__name__}, expected {typ.__name__}")
        extra = set(r) - set(REQUIRED_FIELDS)
        if extra:
            bad(rid, f"unexpected fields {sorted(extra)}")
        if any(k not in r for k in REQUIRED_FIELDS):
            continue
        if r["kind"] not in KINDS:
            bad(rid, f"kind {r['kind']!r} not in {sorted(KINDS)}")

        short, long_, answer, user = r["thinking_short"], r["thinking_long"], r["answer"], r["user"]

        # --- exact markers
        if THINK_ENABLE not in r["system"]:
            bad(rid, f"system turn does not contain {THINK_ENABLE!r} (thinking not enabled)")
        for name, txt in (("thinking_short", short), ("thinking_long", long_)):
            if not txt.startswith(THINK_OPEN):
                bad(rid, f"{name} does not start with {THINK_OPEN!r}")
            if not txt.endswith(THINK_CLOSE):
                bad(rid, f"{name} does not end with {THINK_CLOSE!r}")
            if txt.count(THINK_OPEN) != 1 or txt.count(THINK_CLOSE) != 1:
                bad(rid, f"{name} has nested or repeated channel markers")
        for marker in (TRIGGER, RESULT_OPEN, RESULT_CLOSE):
            if marker in long_:
                bad(rid, f"thinking_long contains {marker!r}; the long form must be prose only")
            if marker in answer:
                bad(rid, f"answer contains {marker!r}; calls live in the thinking channel only")

        # --- calls: count, order, exact composition, label span
        n_trigger = short.count(TRIGGER)
        if n_trigger != len(r["calls"]):
            bad(rid, f"{n_trigger} {TRIGGER!r} in thinking_short but {len(r['calls'])} call records")
        if short.count(RESULT_OPEN) != n_trigger or short.count(RESULT_CLOSE) != n_trigger:
            bad(rid, "unbalanced <|result|>/<|/result|> markers in thinking_short")
        last_pos = -1
        for i, c in enumerate(r["calls"]):
            tag = f"call[{i}]"
            for key, typ in CALL_FIELDS.items():
                if key not in c or not isinstance(c[key], typ):
                    bad(rid, f"{tag}: bad/missing field {key!r}")
            if any(k not in c for k in CALL_FIELDS):
                continue
            pos, tname, gold = c["position_char_start"], c["type"], c["gold_label"]
            if pos <= last_pos:
                bad(rid, f"{tag}: position_char_start {pos} is not after the previous call")
            last_pos = pos
            if short[pos:pos + len(TRIGGER)] != TRIGGER:
                bad(rid, f"{tag}: thinking_short[{pos}:] does not start with {TRIGGER!r}")
                continue
            if tname not in canonical:
                bad(rid, f"{tag}: unknown decision type {tname!r}")
                continue
            can_instr, can_opts = canonical[tname]
            if c["instruction"] != can_instr:
                bad(rid, f"{tag}: instruction differs from the suite's\n    got  {c['instruction']!r}\n"
                         f"    want {can_instr!r}")
            opts = c["options"]
            if set(opts) - {NONE_KEY} != set(can_opts) or len(set(opts)) != len(opts):
                bad(rid, f"{tag}: options are not the suite's option set (optionally + {NONE_KEY!r})")
            if any("," in o for o in opts):
                bad(rid, f"{tag}: an option contains a comma; parse_decide_call splits on commas")
            # the call text as it must appear, byte for byte (build_phase7.py's composition)
            want = f"{TRIGGER}{c['instruction']} [{', '.join(opts)}]{RESULT_OPEN}{gold}{RESULT_CLOSE}"
            if short[pos:pos + len(want)] != want:
                bad(rid, f"{tag}: call text does not match the canonical composition"
                         f" '<|decide|>{{INSTR}} [{{names}}]<|result|>{{label}}<|/result|>'")
                continue
            # label span must follow <|result|> immediately, and equal the gold label
            rstart = pos + len(want) - len(RESULT_CLOSE) - len(gold)
            if short[rstart - len(RESULT_OPEN):rstart] != RESULT_OPEN:
                bad(rid, f"{tag}: label span does not immediately follow {RESULT_OPEN!r}")
            if short[rstart:rstart + len(gold)] != gold:
                bad(rid, f"{tag}: result text != gold_label")
            if gold not in opts:
                bad(rid, f"{tag}: gold_label {gold!r} is not in the option list")

            # --- no label/synonym leak before this call
            before = strip_runtime_and_quotes(short[:pos], user).lower()
            for term in leak_terms(gold):
                if term in before:
                    bad(rid, f"{tag}: {term!r} appears in the thinking BEFORE the call (label leak)")

        # --- duplicate calls
        sigs = [(c.get("type"), tuple(c.get("options", [])), c.get("gold_label")) for c in r["calls"]]
        if len(set(sigs)) != len(sigs):
            bad(rid, "duplicate identical calls in one row")
        pairs = [(c.get("type"), c.get("gold_label")) for c in r["calls"]]
        if len(set(pairs)) != len(pairs):
            bad(rid, "two calls of the same type resolve to the same label (re-verification)")
        if sorted({c.get("type") for c in r["calls"]}) != sorted(r["decision_types"]):
            bad(rid, "decision_types does not match the types actually called")

        # --- paired-answer consistency
        golds = [c.get("gold_label") for c in r["calls"]]
        if sorted(r["label_surfaces"]) != sorted(set(golds)):
            bad(rid, f"label_surfaces keys {sorted(r['label_surfaces'])} != gold labels {sorted(set(golds))}")
        for gold, surface in r["label_surfaces"].items():
            if surface.lower() not in answer.lower():
                bad(rid, f"answer does not contain the declared surface form {surface!r} for {gold!r}")
        if r["kind"] == "control_nocall" and long_ != short:
            bad(rid, "control_nocall: thinking_long must equal thinking_short (no call, so no saving)")

        # --- controls
        if r["kind"] == "control_nocall":
            if r["calls"]:
                bad(rid, "control_nocall row has calls")
            if (r["delta_decoded"], r["delta_total"]) != (0, 0):
                bad(rid, "control_nocall row claims a nonzero token delta")
        elif r["kind"] == "control_none":
            if len(r["calls"]) != 1:
                bad(rid, f"control_none row has {len(r['calls'])} calls, expected 1")
            else:
                c = r["calls"][0]
                if c.get("gold_label") != NONE_KEY:
                    bad(rid, f"control_none gold is {c.get('gold_label')!r}, expected {NONE_KEY!r}")
                if NONE_KEY not in c.get("options", []):
                    bad(rid, f"control_none option list does not carry {NONE_KEY!r}")
        else:
            if not r["calls"]:
                bad(rid, "kind 'call' row has no calls")
            if any(c.get("gold_label") == NONE_KEY for c in r["calls"]):
                bad(rid, f"kind 'call' row resolves to {NONE_KEY!r}; that is a control_none row")

        # --- token fields, recomputed
        a_short, a_long = short + "\n" + answer, long_ + "\n" + answer
        d_s, i_s = split_tokens(tok, a_short)
        d_l, i_l = split_tokens(tok, a_long)
        t_s, _ = split_tokens(tok, short)
        t_l, _ = split_tokens(tok, long_)
        if i_l:
            bad(rid, f"thinking_long+answer contains {i_l} injected tokens; it must be prose only")
        want = {
            "tokens_decoded_short": d_s, "tokens_injected_short": i_s, "tokens_total_short": d_s + i_s,
            "tokens_decoded_long": d_l, "delta_decoded": d_l - d_s, "delta_total": d_l - (d_s + i_s),
            "tokens_thinking_decoded_short": t_s, "tokens_thinking_decoded_long": t_l,
            "tokens_answer": d_s - t_s,
        }
        for key, value in want.items():
            if r[key] != value:
                bad(rid, f"{key} = {r[key]}, recomputed {value}")
        if not THINK_MIN <= t_s <= THINK_MAX:
            bad(rid, f"thinking_short decoded tokens {t_s} outside [{THINK_MIN}, {THINK_MAX}]")

    # --- corpus-level: controls, per-type counts, none-key anti-shortcut, n-grams
    kinds = collections.Counter(r.get("kind") for r in rows)
    n_control = kinds["control_nocall"] + kinds["control_none"]
    if n_control / max(len(rows), 1) < CONTROL_MIN_FRACTION:
        fails.append(f"controls are {n_control}/{len(rows)}, below {CONTROL_MIN_FRACTION:.0%}")
    if not kinds["control_nocall"] or not kinds["control_none"]:
        fails.append(f"both control kinds are required, got {dict(kinds)}")

    per_type = collections.Counter(c["type"] for r in rows for c in r["calls"] if "type" in c)
    for tname in TYPE_SOURCE:
        if per_type[tname] < PER_TYPE_MIN:
            fails.append(f"type {tname!r} has {per_type[tname]} calls, below PER_TYPE_MIN={PER_TYPE_MIN}")

    # --force-options splices ONE canonical order per type, so every call of a type must use that order
    # (the none key, when present, is appended; it is never interleaved).
    orders = collections.defaultdict(set)
    for r in rows:
        for c in r["calls"]:
            if "type" in c and "options" in c:
                orders[c["type"]].add(tuple(o for o in c["options"] if o != NONE_KEY))
            if "options" in c and NONE_KEY in c["options"] and c["options"][-1] != NONE_KEY:
                fails.append(f"{r['id']}: {NONE_KEY!r} is not the last option in the list")
    for tname, got in sorted(orders.items()):
        if len(got) != 1:
            fails.append(f"type {tname!r} uses {len(got)} different option ORDERS across the pilot;"
                         f" --force-options splices a single canonical order")

    # kev/data.py: a none-of-the-above option must also appear as a WRONG alternative, or the model learns
    # "this option is present => pick it" (it did, in kev's first training run).
    none_types = {c["type"] for r in rows for c in r["calls"] if NONE_KEY in c.get("options", [])}
    for tname in sorted(none_types):
        non_none = [r["id"] for r in rows for c in r["calls"]
                    if c["type"] == tname and NONE_KEY in c.get("options", [])
                    and c["gold_label"] != NONE_KEY]
        if not non_none:
            fails.append(f"type {tname!r} only ever carries {NONE_KEY!r} as the answer; it must also "
                         f"appear as a wrong alternative (kev/data.py NONE_OPTIONS comment)")

    rows_by_ngram = collections.defaultdict(set)
    for r in rows:
        prose = " ".join([strip_runtime_and_quotes(r["thinking_short"], r["user"]),
                          r["thinking_long"], r["answer"]])
        words = re.findall(r"[a-z0-9']+", prose.lower())
        for i in range(len(words) - NGRAM_N + 1):
            rows_by_ngram[tuple(words[i:i + NGRAM_N])].add(r["id"])
    repeated = {g: sorted(s) for g, s in rows_by_ngram.items() if len(s) > NGRAM_MAX_ROWS}
    for g, s in sorted(repeated.items(), key=lambda kv: -len(kv[1])):
        fails.append(f"{NGRAM_N}-gram {' '.join(g)!r} appears in {len(s)} rows: {s}")

    return fails, warns, per_type, kinds


def summarize(rows, per_type, kinds):
    call_rows = [r for r in rows if r["calls"]]
    n_calls = sum(len(r["calls"]) for r in rows)
    th = [r["tokens_thinking_decoded_short"] for r in rows]
    thl = [r["tokens_thinking_decoded_long"] for r in rows]
    dd = sum(r["delta_decoded"] for r in call_rows)
    inj = sum(r["tokens_injected_short"] for r in call_rows)
    dt = sum(r["delta_total"] for r in call_rows)
    lines = [
        f"rows {len(rows)}; kinds {dict(kinds)}; calls {n_calls}",
        f"calls/row {dict(sorted(collections.Counter(len(r['calls']) for r in rows).items()))}",
        f"per type {dict(per_type)}",
        f"tasks {len(set(r['task'] for r in rows))}: "
        f"{dict(sorted(collections.Counter(r['task'] for r in rows).items()))}",
        f"thinking_short decoded tokens: min {min(th)} mean {sum(th)/len(th):.1f} max {max(th)}",
        f"thinking_long  decoded tokens: min {min(thl)} mean {sum(thl)/len(thl):.1f} max {max(thl)}",
        f"per call-bearing row ({len(call_rows)}): mean delta_decoded {dd/len(call_rows):+.1f}, "
        f"mean injected {inj/len(call_rows):.1f}, mean delta_total {dt/len(call_rows):+.1f}",
        f"per call ({n_calls}): decoded saved {dd/n_calls:+.1f}, injected overhead {inj/n_calls:.1f}",
    ]
    by_type_inj = collections.defaultdict(list)
    for r in rows:
        for c in r["calls"]:
            by_type_inj[c["type"]].append(len(c["options"]))
    lines.append("option-list size by type: " + ", ".join(
        f"{t}={sorted(set(v))}" for t, v in sorted(by_type_inj.items())))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", default="oracle/phase7-reasoning-pilot-v2.jsonl")
    ap.add_argument("--suite", default=SUITE)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--verbose", action="store_true", help="print the per-row token table")
    a = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pilot = a.pilot if os.path.isabs(a.pilot) else os.path.join(root, a.pilot)
    suite = a.suite if os.path.isabs(a.suite) else os.path.join(root, a.suite)

    rows = [json.loads(l) for l in open(pilot, encoding="utf-8") if l.strip()]
    print(f"pilot: {len(rows)} rows from {a.pilot}")

    canonical, suite_errs = load_canonical(suite)
    for tname, (instr, opts) in sorted(canonical.items()):
        print(f"  canonical {tname:16s} K={len(opts):3d}  {instr!r}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)

    fails, warns, per_type, kinds = check(rows, canonical, tok)
    fails = suite_errs + fails

    print()
    print(summarize(rows, per_type, kinds))
    if a.verbose:
        print()
        print(f"{'id':12s} {'kind':14s} {'calls':>5s} {'th_s':>5s} {'th_l':>5s} {'dec_s':>6s} "
              f"{'inj_s':>6s} {'dec_l':>6s} {'d_dec':>6s} {'d_tot':>6s}")
        for r in rows:
            print(f"{r['id']:12s} {r['kind']:14s} {len(r['calls']):5d} "
                  f"{r['tokens_thinking_decoded_short']:5d} {r['tokens_thinking_decoded_long']:5d} "
                  f"{r['tokens_decoded_short']:6d} {r['tokens_injected_short']:6d} "
                  f"{r['tokens_decoded_long']:6d} {r['delta_decoded']:+6d} {r['delta_total']:+6d}")

    print()
    for w in warns:
        print(f"WARN {w}")
    if fails:
        print(f"FAILED: {len(fails)} problem(s)")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
