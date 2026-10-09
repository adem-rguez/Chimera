"""Phase 7R Stage A, task T2 (reports/16-phase7-reasoning-proposal.md, staged plan and decision 3/7):
"T2: Descriptions probe (re-score oracle/decision-v7 dev suite with option descriptions stripped from
`option_text(key, desc)` pairs, keeping bare keys only). Gate order_outcome (5 variants x >=24 instances,
bare/described + numeric restatement; pass >=21/24 under final frozen call format else DROP type; freeze
canonical option expansions at runtime)." Three probes against two sources: Parts A and B score the real
decision-v7 DEV suite (`evals/v7/decision-v7/development.jsonl`); Part A' scores the 30-row phase7r pilot
itself (`oracle/phase7-reasoning-pilot-v2.jsonl`, the file scripts/probe_phase7r_head.py scores). All
three use the shipped E4B decision checkpoint exactly like scripts/eval_phase7.py's load_decision_model +
score_decide_call:

Part A -- order_outcome 5-variant gate, DEV-SET SANITY CHECK ONLY. This is NOT a diagnosis of the pilot's
4/8 failure: all 24 dev rows already score 1.000 with descriptions (reports/16 line ~30), so Part A cannot
reproduce or explain the pilot's 4/8 (reports/16 line ~322-323, ~419) -- see Part A' below for that. Part A
exists to gate the frozen call format (bare vs described option text; raw-number vs worded-comparison state
restatement) against the dev set's own 5-variant ladder, independent of the pilot-reproduction question.
All 24 `contrastive_quantity_limit` dev rows (variant="clean"), scored under 5 option/state formats:
  V1 bare                        -- bare keys (today's pilot format), state unmodified.
  V2 described                   -- "key: description" (the suite's own option_text, descriptions kept).
  V3 bare_numeric                -- bare keys, state + a restated-numbers sentence (quantity/limit/double-limit).
  V4 described_numeric           -- described keys + the same restated-numbers sentence.
  V5 described_numeric_comparison -- described keys + restated numbers AND the worded >/<= comparison
                                      (never names the label) -- the strongest hint, an upper bound.
Gate: >=21/24 correct on whichever variant becomes the frozen call format. If no variant clears 21/24,
order_outcome gets DROPPED from the Stage A type set (reports/16 decision 7).

ASSUMPTIONS (reports/16 does not spell these out; flagged per CLAUDE.md "state assumptions, don't guess"):
  - "numeric restatement" is not defined in reports/16 beyond the phrase itself. Implemented here as a
    regex-extracted sentence restating the case's order quantity, the policy's per-item limit, and double
    that limit -- the exact three numbers order_outcome's label depends on (policy text: "at most N units
    ... Orders up to double the limit are held for review"; case text: "The order quantity is M."). V5 adds
    a worded comparison of those numbers (never the label itself) as a second, stronger rung.
  - "5 variants" count is also not pinned down exactly beyond "bare/described + numeric restatement"; the
    5 above are a monotonic ladder (baseline -> descriptions -> +numbers -> +numbers+comparison) chosen so
    a human reviewing the output can see exactly which ingredient (if any) fixes the type, not just a final
    pass/fail. Re-derive if T2's author intended a different 5.
  - Option order is held at the suite's own per-row criteria order (not shuffled) for all 5 variants --
    order-sensitivity is already a separate, measured axis in scripts/probe_phase7r_head.py's S1-S4b x
    orig/shuffled probe; T2 isolates description/numeric-restatement only, per the report wording.

Part B -- descriptions probe proper (decision 3): all 5 shipped Stage-A types (claim_handling,
order_outcome, answer_type, kb_category, news_topic -- banking_intent dropped per decision 5), dev suite
`variant="clean"` rows only (matches reports/16's own per-type dev-accuracy table methodology), scored
bare keys vs "key: description" to measure the accuracy delta reports/16 line 189 calls "cost unknown."

Part A' -- pilot reproduction (NEW; not in reports/16's T2 text, added because Part A cannot reproduce the
4/8). Takes the real 8 order_outcome calls out of oracle/phase7-reasoning-pilot-v2.jsonl (verified: ids
p7r2-06 x2, p7r2-07 x1, p7r2-17 x1, p7r2-18 x1, p7r2-23 x2, p7r2-24 x1 -- all type="order_outcome",
options always ['within_limit', 'slightly_over', 'far_over']; gold labels printed in reports/18). For each
call, builds the EXACT state scripts/probe_phase7r_head.py uses for its S1 ("best state") definition --
user turn + thinking decoded since the previous call, cut at the call position -- by importing and calling
that script's own build_segments/build_states (not reimplemented here). Scores each call under:
  bare              -- bare option keys (today's pilot format), S1 state unmodified.
  described         -- "key: description", description = order_outcome's own canonical criteria text,
                        read from the dev suite (never hardcoded) via order_outcome_criteria().
  bare_numeric      -- bare keys, S1 state + this module's own numeric_restatement() IF the S1 text
                        contains parseable quantity/limit numbers (reuses _POLICY_LIMIT_RE/_CASE_QUANTITY_RE
                        from Part A, applied to the free-form pilot prose instead of the dev suite's
                        policy/case template fields); rows that don't parse are reported as a parse miss,
                        not silently dropped or guessed at.
  described_numeric -- described keys + the same conditional numeric restatement.
No PASS/FAIL gate here (per task: Part A' reports counts only). The report also names, per row that the
`bare` variant gets wrong, which of the other three variants (if any) first flips it correct -- "which
ingredient fixes this row," not a verdict on the whole type.

ASSUMPTION (Part A'): at drafting time no oracle/phase7r-head-probe-*.{jsonl,md} files exist in this repo
(scripts/probe_phase7r_head.py has not yet been run on a box), so the specific identity of the pilot
probe's 4 wrong-of-8 predictions is NOT recoverable from existing artifacts -- it is unknown until that
probe (or this one) actually runs against the real model. Part A' is a standalone reproduction built
directly from the pilot source + gold labels, not a readout of the head probe's own (not-yet-existing)
results.

Usage (GPU box; loads ONLY the decision checkpoint, bf16, nothing else -- same load path as
scripts/probe_phase7r_head.py):
  .venv/bin/python -u scripts/probe_phase7r_t2.py --tag smoke1 --limit-gate 4 --limit-desc 10 --limit-apos 2
  .venv/bin/python -u scripts/probe_phase7r_t2.py --tag run1

Offline (laptop, no torch/GPU/model -- validates row selection + variant construction + reporting with a
FAKE scorer against the real decision-v7 dev file and the real pilot file):
  python -m py_compile scripts/probe_phase7r_t2.py
  python scripts/probe_phase7r_t2.py --dry-run --tag drytest
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kev.api import option_text, render  # noqa: E402
from scripts.probe_phase7r_head import (  # noqa: E402
    CONF_FLOOR, build_segments, build_states, make_fake_scorer, make_real_scorer,
)

DEV_PATH = "evals/v7/decision-v7/development.jsonl"
PILOT_PATH = "oracle/phase7-reasoning-pilot-v2.jsonl"

# qkey = the row['questions'] dict key this type lives under; src = the question's own 'src' field,
# used only to disambiguate the two families sharing qkey='decision' (claim_handling vs order_outcome).
# All five are reports/16's "five types" (decision 5); instr/criteria/label are read from each row itself,
# never hardcoded, so this probe always reflects whatever evals/v7/decision-v7 actually contains.
TYPES = {
    "claim_handling": {"qkey": "decision", "src": "contrastive_spend_threshold"},
    "order_outcome": {"qkey": "decision", "src": "contrastive_quantity_limit"},
    "answer_type": {"qkey": "answer_type", "src": None},
    "kb_category": {"qkey": "category", "src": None},
    "news_topic": {"qkey": "topic", "src": None},
}

GATE_VARIANTS = ["bare", "described", "bare_numeric", "described_numeric", "described_numeric_comparison"]
GATE_PASS_THRESHOLD = 21  # out of 24; reports/16 T2: "pass >=21/24"
GATE_TOTAL = 24

# order_outcome's two load-bearing numbers, per the contrastive_quantity_limit family's fixed template
# (verified against all 24 dev rows: both regexes match every row -- see this script's docstring).
_POLICY_LIMIT_RE = re.compile(r"at most (\d+) units")
_CASE_QUANTITY_RE = re.compile(r"order quantity is (\d+)")


# ---------------------------------------------------------------------------
# row selection (pure Python; no torch/kev model loading)
# ---------------------------------------------------------------------------

def load_dev_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select_rows(rows, type_id):
    """-> dev rows for `type_id`, variant='clean' only (matches reports/16's own dev-accuracy table:
    'clean variant only'), in file order."""
    spec = TYPES[type_id]
    out = []
    for r in rows:
        if r.get("_meta", {}).get("variant") != "clean":
            continue
        q = r["questions"].get(spec["qkey"])
        if q is None:
            continue
        if spec["src"] is not None and q.get("src") != spec["src"]:
            continue
        out.append(r)
    return out


def bare_options(criteria):
    return [option_text(k, None) for k in criteria]


def described_options(criteria):
    return [option_text(k, v) for k, v in criteria.items()]


def _pred_key(pred):
    """-> the bare option key from a scorer prediction, stripping a described option's ': description'
    suffix if present. NEEDED because gold_label is always the bare key (verified: dev suite and pilot
    both store bare keys) while the scorer's pred is whatever string was IN the options list -- for
    `described`/`described_numeric` that is the full 'key: description' text, so a raw `pred == gold`
    would be False even when the model picked the right option. option_text() never puts ':' in a bare
    key (verified: within_limit/slightly_over/far_over), so split on the first ':' is safe here."""
    return pred.split(":", 1)[0]


# ---------------------------------------------------------------------------
# Part A -- order_outcome 5-variant gate
# ---------------------------------------------------------------------------

def extract_quantity_limit(state):
    """-> (quantity, limit) ints parsed from an order_outcome row's state, or None if either regex misses
    (should not happen on the real dev set -- verified 24/24 -- but a miss must not crash the probe; the
    caller skips that (row, variant) and the gate denominator shrinks, visible in the report)."""
    p = _POLICY_LIMIT_RE.search(state.get("policy", ""))
    c = _CASE_QUANTITY_RE.search(state.get("case", ""))
    if not p or not c:
        return None
    return int(c.group(1)), int(p.group(1))


def numeric_restatement(quantity, limit, comparison):
    double = 2 * limit
    text = (f" The order quantity is {quantity}; the per-item limit is {limit}; double the limit is "
            f"{double}.")
    if not comparison:
        return text
    if quantity <= limit:
        cmp_text = f" {quantity} is at or below the limit of {limit}."
    elif quantity <= double:
        cmp_text = f" {quantity} is above the limit of {limit} but at or below double the limit ({double})."
    else:
        cmp_text = f" {quantity} is above double the limit ({double})."
    return text + cmp_text


def build_gate_case(row, variant):
    """-> (state_text, options) for one order_outcome dev row under `variant`, or None if the row's
    numbers could not be parsed (only possible for the two *_numeric* variants)."""
    q = row["questions"]["decision"]
    criteria = q["criteria"]
    base_text = render(row["state"])
    if variant == "bare":
        return base_text, bare_options(criteria)
    if variant == "described":
        return base_text, described_options(criteria)
    nums = extract_quantity_limit(row["state"])
    if nums is None:
        return None
    quantity, limit = nums
    if variant == "bare_numeric":
        return base_text + numeric_restatement(quantity, limit, comparison=False), bare_options(criteria)
    if variant == "described_numeric":
        return (base_text + numeric_restatement(quantity, limit, comparison=False),
                described_options(criteria))
    if variant == "described_numeric_comparison":
        return (base_text + numeric_restatement(quantity, limit, comparison=True),
                described_options(criteria))
    raise ValueError(f"unknown gate variant {variant!r}")


def order_outcome_criteria(dev_rows):
    """-> order_outcome's canonical {key: description} criteria dict, read from the dev suite (never
    hardcoded) -- asserts every clean dev row agrees (true for all 24, verified) so Part A' can describe
    the pilot's bare option keys with the suite's own wording, not an invented one."""
    rows = select_rows(dev_rows, "order_outcome")
    if not rows:
        raise ValueError("no order_outcome dev rows found; cannot build described options for Part A'")
    criteria = rows[0]["questions"]["decision"]["criteria"]
    for r in rows[1:]:
        if r["questions"]["decision"]["criteria"] != criteria:
            raise ValueError("order_outcome dev rows disagree on criteria text; Part A' needs one set")
    return criteria


def run_gate(rows, scorer, limit=None):
    rows = rows if limit is None else rows[:limit]
    results = []
    for variant in GATE_VARIANTS:
        for row in rows:
            built = build_gate_case(row, variant)
            if built is None:
                continue
            state_text, opts = built
            instr = render(row["questions"]["decision"]["instructions"])
            gold = row["questions"]["decision"]["label"]
            pred, conf, max_prob, probs = scorer(state_text, instr, opts)
            pred_key = _pred_key(pred)
            results.append({
                "part": "A_gate", "variant": variant, "row_id": row["_meta"]["id"],
                "instr": instr, "options": opts, "gold_label": gold,
                "pred": pred, "pred_key": pred_key, "correct": pred_key == gold, "confidence": conf, "max_prob": max_prob,
                "conf_floor_hit": conf >= CONF_FLOOR, "state_preview": state_text[:300],
            })
    return results


# ---------------------------------------------------------------------------
# Part A' -- pilot reproduction (8 order_outcome calls, oracle/phase7-reasoning-pilot-v2.jsonl)
# ---------------------------------------------------------------------------

APOS_VARIANTS = ["bare", "described", "bare_numeric", "described_numeric"]


def load_pilot_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select_pilot_order_outcome_calls(pilot_rows):
    """-> [(row, call_idx), ...] for every pilot call with type=='order_outcome', file order. Expect 8
    (verified against oracle/phase7-reasoning-pilot-v2.jsonl: p7r2-06 x2, -07 x1, -17 x1, -18 x1,
    -23 x2, -24 x1 -- see this module's docstring)."""
    out = []
    for row in pilot_rows:
        for i, call in enumerate(row["calls"]):
            if call["type"] == "order_outcome":
                out.append((row, i))
    return out


def extract_quantity_limit_from_text(text):
    """-> (quantity, limit) ints parsed from a combined state string, or None. Reuses the exact Part A
    regexes (_POLICY_LIMIT_RE / _CASE_QUANTITY_RE, tied to the dev suite's fixed phrasing 'at most N
    units' / 'order quantity is M'), applied directly to the pilot's S1 text instead of a dev row's split
    policy/case dict fields. The pilot's S1 is free-authored reasoning prose, not the suite's template --
    expect this to miss on most/all of the 8 calls (verified: 8/8 miss, case regex never matches since
    the pilot prose computes quantities inline, e.g. '13 lamps', rather than stating 'order quantity is
    13'); a miss is reported as a visible parse-miss row, never silently skipped or guessed."""
    p = _POLICY_LIMIT_RE.search(text)
    c = _CASE_QUANTITY_RE.search(text)
    if not p or not c:
        return None
    return int(c.group(1)), int(p.group(1))


def build_apos_case(state_text, options, criteria, variant):
    """-> (state_text, options) for one pilot order_outcome call under `variant`, or None if a numeric
    variant was requested and extract_quantity_limit_from_text missed."""
    bare = list(options)
    described = [option_text(k, criteria[k]) for k in options]
    if variant == "bare":
        return state_text, bare
    if variant == "described":
        return state_text, described
    nums = extract_quantity_limit_from_text(state_text)
    if nums is None:
        return None
    quantity, limit = nums
    extra = numeric_restatement(quantity, limit, comparison=False)
    if variant == "bare_numeric":
        return state_text + extra, bare
    if variant == "described_numeric":
        return state_text + extra, described
    raise ValueError(f"unknown Part A' variant {variant!r}")


def run_apos(pilot_rows, dev_rows, scorer, limit=None):
    criteria = order_outcome_criteria(dev_rows)
    calls = select_pilot_order_outcome_calls(pilot_rows)
    calls = calls if limit is None else calls[:limit]
    segments_by_row_id = {}
    results = []
    for row, call_idx in calls:
        rid = row["id"]
        if rid not in segments_by_row_id:
            segments_by_row_id[rid] = build_segments(row)
        state_s1 = build_states(row, call_idx, segments_by_row_id[rid])["S1"]
        call = row["calls"][call_idx]
        gold = call["gold_label"]
        for variant in APOS_VARIANTS:
            built = build_apos_case(state_s1, call["options"], criteria, variant)
            base = {"part": "Aprime_pilot", "variant": variant, "pilot_id": rid, "call_index": call_idx,
                    "instr": render(call["instruction"]), "gold_label": gold, "state_preview": state_s1[:300]}
            if built is None:
                results.append({**base, "parse_miss": True})
                continue
            state_text, opts = built
            pred, conf, max_prob, probs = scorer(state_text, render(call["instruction"]), opts)
            results.append({**base, "parse_miss": False, "options": opts,
                             "pred": pred, "correct": _pred_key(pred) == gold, "confidence": conf,
                             "max_prob": max_prob, "conf_floor_hit": conf >= CONF_FLOOR,
                             "state_preview": state_text[:300]})
    return results


# ---------------------------------------------------------------------------
# Part B -- descriptions probe (bare vs described), all 5 shipped types
# ---------------------------------------------------------------------------

def run_descriptions_probe(rows_by_type, scorer, limit=None):
    results = []
    for type_id, rows in rows_by_type.items():
        rows_use = rows if limit is None else rows[:limit]
        for described in (False, True):
            for row in rows_use:
                q = row["questions"][TYPES[type_id]["qkey"]]
                instr = render(q["instructions"])
                opts = described_options(q["criteria"]) if described else bare_options(q["criteria"])
                gold = q["label"]
                state_text = render(row["state"])
                pred, conf, max_prob, probs = scorer(state_text, instr, opts)
                pred_key = _pred_key(pred)
                results.append({
                    "part": "B_descriptions", "type": type_id,
                    "format": "described" if described else "bare",
                    "row_id": row["_meta"]["id"], "instr": instr, "options": opts, "gold_label": gold,
                    "pred": pred, "pred_key": pred_key, "correct": pred_key == gold, "confidence": conf, "max_prob": max_prob,
                    "conf_floor_hit": conf >= CONF_FLOOR, "state_preview": state_text[:300],
                })
    return results


# ---------------------------------------------------------------------------
# reporting (pure Python)
# ---------------------------------------------------------------------------

def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def build_apos_report(apos_results):
    lines = ["## Part A': pilot reproduction (8 order_outcome calls, "
             "oracle/phase7-reasoning-pilot-v2.jsonl, NOT the dev suite)", "",
             "No PASS/FAIL gate here -- counts only, per task. S1 state (user turn + thinking since "
             "previous call) built via scripts/probe_phase7r_head.py's own build_segments/build_states.",
             ""]
    lines.append("| variant | n | parse misses | correct | accuracy |")
    lines.append("|---|---|---|---|---|")
    for variant in APOS_VARIANTS:
        rs = [r for r in apos_results if r["variant"] == variant]
        n = len(rs)
        misses = sum(1 for r in rs if r["parse_miss"])
        scored = [r for r in rs if not r["parse_miss"]]
        correct = sum(1 for r in scored if r["correct"])
        acc = correct / len(scored) if scored else float("nan")
        lines.append(f"| {variant} | {n} | {misses} | {correct}/{len(scored)} | {acc:.3f} |")

    lines.append("")
    lines.append("### Per-row predictions (bare vs described vs numeric variants)")
    lines.append("")
    lines.append("| pilot_id | call | gold | bare pred | described pred | bare_numeric | "
                 "described_numeric | first ingredient that flips bare-wrong -> correct |")
    lines.append("|---|---|---|---|---|---|---|---|")
    by_row = {}
    for r in apos_results:
        by_row.setdefault((r["pilot_id"], r["call_index"]), {})[r["variant"]] = r
    for (pid, ci), by_variant in sorted(by_row.items()):
        gold = next(iter(by_variant.values()))["gold_label"]

        def cell(variant):
            r = by_variant.get(variant)
            if r is None:
                return "n/a"
            if r["parse_miss"]:
                return "PARSE MISS"
            return f"{r['pred']}" + ("" if r["correct"] else " (WRONG)")

        bare_r = by_variant.get("bare")
        flip = "n/a"
        if bare_r and not bare_r["parse_miss"] and not bare_r["correct"]:
            flip = "none"
            for variant in ("described", "bare_numeric", "described_numeric"):
                r = by_variant.get(variant)
                if r and not r["parse_miss"] and r["correct"]:
                    flip = variant
                    break
        elif bare_r and not bare_r["parse_miss"] and bare_r["correct"]:
            flip = "bare already correct"
        lines.append(f"| {pid} | {ci} | {gold} | {cell('bare')} | {cell('described')} | "
                     f"{cell('bare_numeric')} | {cell('described_numeric')} | {flip} |")

    return "\n".join(lines) + "\n"


def build_report(gate_results, desc_results, apos_results):
    lines = ["# Phase 7R T2 probe (descriptions + order_outcome gate + pilot reproduction)", "",
              f"generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}, "
              f"conf_floor={CONF_FLOOR}, dev source={DEV_PATH}, pilot source={PILOT_PATH}", ""]

    lines.append("## Part A: order_outcome 5-variant gate, DEV-SET SANITY CHECK ONLY "
                 f"(pass = >= {GATE_PASS_THRESHOLD}/{GATE_TOTAL} correct)")
    lines.append("")
    lines.append("NOTE: all 24 dev rows already score 1.000 with descriptions (reports/16 line ~30) -- "
                 "this table cannot reproduce or explain the pilot's 4/8 failure. See Part A' below for "
                 "that; this table only gates the frozen call format against the dev set.")
    lines.append("")
    lines.append("| variant | n | correct | accuracy | gate |")
    lines.append("|---|---|---|---|---|")
    for variant in GATE_VARIANTS:
        rs = [r for r in gate_results if r["variant"] == variant]
        n = len(rs)
        correct = sum(1 for r in rs if r["correct"])
        acc = correct / n if n else float("nan")
        gate = "n/a (parse misses)" if n != GATE_TOTAL else ("PASS" if correct >= GATE_PASS_THRESHOLD else "FAIL")
        lines.append(f"| {variant} | {n} | {correct} | {acc:.3f} | {gate} |")
    missing = GATE_TOTAL - len({r["row_id"] for r in gate_results if r["variant"] == "bare"})
    if missing:
        lines.append("")
        lines.append(f"NOTE: {missing}/{GATE_TOTAL} order_outcome dev rows did not parse "
                     "(_POLICY_LIMIT_RE/_CASE_QUANTITY_RE miss) -- inspect before trusting the gate.")

    lines.append("")
    lines.append(build_apos_report(apos_results))

    lines.append("## Part B: descriptions probe, bare vs described, per shipped type")
    lines.append("")
    lines.append("| type | n | bare accuracy | described accuracy | delta (described - bare) |")
    lines.append("|---|---|---|---|---|")
    for type_id in TYPES:
        bare_rs = [r for r in desc_results if r["type"] == type_id and r["format"] == "bare"]
        desc_rs = [r for r in desc_results if r["type"] == type_id and r["format"] == "described"]
        n = len(bare_rs)
        bare_acc = _mean(float(r["correct"]) for r in bare_rs) if bare_rs else float("nan")
        desc_acc = _mean(float(r["correct"]) for r in desc_rs) if desc_rs else float("nan")
        delta = desc_acc - bare_acc if bare_rs and desc_rs else float("nan")
        lines.append(f"| {type_id} | {n} | {bare_acc:.3f} | {desc_acc:.3f} | {delta:+.3f} |")

    return "\n".join(lines) + "\n"


def print_dry_run_examples(gate_rows, rows_by_type, pilot_calls, dev_rows):
    print(f"\n--dry-run example gate cases (row 0 of {len(gate_rows)} order_outcome dev rows):\n")
    row = gate_rows[0]
    for variant in GATE_VARIANTS:
        built = build_gate_case(row, variant)
        if built is None:
            print(f"  {variant}: PARSE MISS")
            continue
        state_text, opts = built
        print(f"  {variant}: options={opts!r}")
        print(f"    state[:300]={state_text[:300]!r}")
    print(f"\n--dry-run example descriptions-probe rows (1 per type, bare vs described options):\n")
    for type_id, rows in rows_by_type.items():
        if not rows:
            print(f"  {type_id}: NO ROWS SELECTED")
            continue
        q = rows[0]["questions"][TYPES[type_id]["qkey"]]
        print(f"  {type_id}: gold={q['label']!r}")
        print(f"    bare:      {bare_options(q['criteria'])!r}")
        print(f"    described: {described_options(q['criteria'])!r}")
    print(f"\n--dry-run example Part A' pilot cases ({len(pilot_calls)} order_outcome pilot calls):\n")
    if pilot_calls:
        criteria = order_outcome_criteria(dev_rows)
        row, call_idx = pilot_calls[0]
        segments = build_segments(row)
        state_s1 = build_states(row, call_idx, segments)["S1"]
        call = row["calls"][call_idx]
        print(f"  {row['id']} call[{call_idx}] gold={call['gold_label']!r}")
        for variant in APOS_VARIANTS:
            built = build_apos_case(state_s1, call["options"], criteria, variant)
            if built is None:
                print(f"    {variant}: PARSE MISS")
                continue
            state_text, opts = built
            print(f"    {variant}: options={opts!r}")
            print(f"      state[:300]={state_text[:300]!r}")
    print()


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dev-path", default=DEV_PATH)
    ap.add_argument("--pilot-path", default=PILOT_PATH)
    ap.add_argument("--decision-run", default="runs/p4-e4b-final")
    ap.add_argument("--tag", required=True, help="output files are "
                    "oracle/phase7r-t2-{tag}.jsonl/.md; never overwritten")
    ap.add_argument("--limit-gate", type=int, default=None,
                    help="score only the first N order_outcome dev rows (per variant) -- smoke run")
    ap.add_argument("--limit-desc", type=int, default=None,
                    help="score only the first N dev rows per type (per bare/described) -- smoke run")
    ap.add_argument("--limit-apos", type=int, default=None,
                    help="score only the first N pilot order_outcome calls (per variant) -- smoke run")
    ap.add_argument("--dry-run", action="store_true", help="fake scorer, no torch/kev/GPU; validates "
                    "row selection + variant construction + reporting against the real dev + pilot files")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dev_path = args.dev_path if os.path.isabs(args.dev_path) else os.path.join(root, args.dev_path)
    pilot_path = args.pilot_path if os.path.isabs(args.pilot_path) else os.path.join(root, args.pilot_path)
    out_jsonl = os.path.join(root, "oracle", f"phase7r-t2-{args.tag}.jsonl")
    out_md = os.path.join(root, "oracle", f"phase7r-t2-{args.tag}.md")
    for p in (out_jsonl, out_md):
        if os.path.exists(p):
            print(f"ERROR: {p} already exists; choose a different --tag (never overwritten)",
                  file=sys.stderr)
            return 1

    rows = load_dev_rows(dev_path)
    gate_rows = select_rows(rows, "order_outcome")
    rows_by_type = {t: select_rows(rows, t) for t in TYPES}
    pilot_rows = load_pilot_rows(pilot_path)
    pilot_calls = select_pilot_order_outcome_calls(pilot_rows)
    print(f"dev suite: {len(rows)} rows from {dev_path}", flush=True)
    print(f"order_outcome gate rows (variant=clean): {len(gate_rows)} (expect {GATE_TOTAL})", flush=True)
    for t, rs in rows_by_type.items():
        print(f"  {t}: {len(rs)} clean dev rows", flush=True)
    print(f"pilot: {len(pilot_rows)} rows from {pilot_path}, "
          f"{len(pilot_calls)} order_outcome calls (expect 8)", flush=True)

    if args.dry_run:
        print_dry_run_examples(gate_rows, rows_by_type, pilot_calls, rows)
        scorer = make_fake_scorer()
    else:
        scorer = make_real_scorer(args.decision_run, args.device)

    gate_results = run_gate(gate_rows, scorer, limit=args.limit_gate)
    desc_results = run_descriptions_probe(rows_by_type, scorer, limit=args.limit_desc)
    apos_results = run_apos(pilot_rows, rows, scorer, limit=args.limit_apos)
    all_results = gate_results + desc_results + apos_results

    write_jsonl(out_jsonl, all_results)
    print(f"wrote {out_jsonl} ({len(all_results)} rows)", flush=True)

    report = build_report(gate_results, desc_results, apos_results)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"wrote {out_md}", flush=True)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
