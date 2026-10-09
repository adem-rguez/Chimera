"""Pre-training head probe for Phase 7R (reports/16-phase7-reasoning-proposal.md, open decision 4 /
"The state-scoping question" section): does the ALREADY-TRAINED Kev decision checkpoint
(runs/p4-e4b-final, loaded exactly like scripts/eval_phase7.py's load_decision_model +
score_decide_call) still work when the "state" it scores is authored reasoning PROSE instead of a
suite-rendered customer message / document?

Per reports/16 (state-scoping section, quoted): "The pilot is authored on the rule state = user turn
+ the thinking decoded since the previous call (which is exactly why the pre-call prose has to restate
the relevant facts, and why it must not name the label)." That rule is this script's S1. S2-S4b are
variants to bound how much of that rule actually matters (user turn alone? thinking alone? the whole
prior trace?) before committing to it in scripts/eval_phase7.py's score_decide_call.

State strategies, per call (see reports/16 and oracle/phase7-reasoning-pilot-v2.jsonl's
thinking_short field, which carries the "<|channel>thought\\n ... <|decide|>{instr} [{opts}]"
"<|result|>{label}<|/result|> ... \\n<channel|>" markers verbatim, scripts/check_phase7r_pilot.py
validates the exact composition):
  S1  = user turn + thinking decoded since the previous call (the pilot's own authoring rule), cut at
        the call position.
  S2  = user turn only.
  S3  = thinking decoded since the previous call only, no user turn.
  S4a = the FULL thinking text before the call (all prior calls' trigger/option/result text kept
        verbatim, i.e. the raw thinking_short slice).
  S4b = same slice as S4a, with every prior call's INJECTED span (the suite instruction + option list
        + result label -- never decoded by the model, same split scripts/check_phase7r_pilot.py's
        split_tokens/injected_spans uses) stripped back out, leaving only the bare "<|decide|>" trigger
        token and the prose in between. Cheap to add once S4a exists, so both are scored.
(S5, "a one-sentence focused evidence extract", is skipped per the task -- not automatable.)

For each call x strategy, scores the pilot's own canonical option order ("orig") AND a seeded shuffle
of the same options ("shuffled") -- reports/16 flags option order as a measured sensitivity axis
(trained E4B flip 0.067 dev / 0.350 transfer on *generation*; this probe checks the *pointer head*'s
own order-sensitivity on these particular calls, nothing to do with that generation number).

Usage (GPU box; loads ONLY the decision checkpoint, bf16, nothing else):
  .venv/bin/python -u scripts/probe_phase7r_head.py --tag smoke1 --limit 3
  .venv/bin/python -u scripts/probe_phase7r_head.py --tag run1

Offline (laptop, no torch/GPU/model -- validates state construction + reporting with a FAKE scorer
against the real pilot file):
  python -m py_compile scripts/probe_phase7r_head.py
  python scripts/probe_phase7r_head.py --dry-run --tag drytest
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.check_phase7r_pilot import (  # noqa: E402
    RESULT_CLOSE, RESULT_OPEN, THINK_OPEN, TRIGGER, injected_spans,
)

CONF_FLOOR = 0.6112  # oracle/gate-threshold.md Phase 5B T; see scripts/eval_phase7.py's DEFAULT_CONF_FLOOR
STRATEGIES = ["S1", "S2", "S3", "S4a", "S4b"]
ORDERS = ["orig", "shuffled"]

# Per-type dev accuracies from reports/16-phase7-reasoning-proposal.md's "Decision types and the
# competence cutoff" table (runs/p4-e4b-final-dev/rows.json, clean variant, argmax vs gold).
DEV_ACCURACY = {
    "claim_handling": 1.000,
    "order_outcome": 1.000,
    "answer_type": 0.963,
    "kb_category": 0.950,
    "news_topic": 0.812,
    "banking_intent": 0.800,
}
FLAG_DROP = 0.15  # absolute accuracy drop below DEV_ACCURACY[type] that gets flagged in the report


# ---------------------------------------------------------------------------
# state construction (pure Python; no torch/kev) -- the part the dry run proves correct
# ---------------------------------------------------------------------------

def call_end_char(call):
    """-> char offset right after a call's '<|/result|>', per the canonical composition
    scripts/check_phase7r_pilot.py's `check()` validates byte-for-byte
    ('<|decide|>{instr} [{opts}]<|result|>{label}<|/result|>')."""
    want = (f"{TRIGGER}{call['instruction']} [{', '.join(call['options'])}]"
            f"{RESULT_OPEN}{call['gold_label']}{RESULT_CLOSE}")
    return call["position_char_start"] + len(want)


def strip_injected(text):
    """-> `text` with every prior call's injected span (instruction+options+result, never decoded by
    the model -- see scripts/check_phase7r_pilot.py's injected_spans/split_tokens) removed, leaving the
    bare '<|decide|>' trigger tokens and the prose around them. Safe to call on a thinking_short slice
    that ends strictly before the current call's own trigger (so the current call contributes no span)."""
    spans = injected_spans(text)
    out, cursor = [], 0
    for s, e in spans:
        out.append(text[cursor:s])
        cursor = e
    out.append(text[cursor:])
    return "".join(out)


def build_segments(row):
    """-> list, one entry per row['calls'], of the thinking text DECODED SINCE THE PREVIOUS CALL (or
    since the start of the thinking channel for the first call), cut at that call's own position --
    the span reports/16's state-scoping rule is authored on. `thinking_short` starts with THINK_OPEN
    (validated), so the first call's segment starts right after that marker."""
    thinking = row["thinking_short"]
    calls = row["calls"]
    segments = []
    prev_end = len(THINK_OPEN)
    for call in calls:
        pos = call["position_char_start"]
        segments.append(thinking[prev_end:pos])
        prev_end = call_end_char(call)
    return segments


def build_states(row, call_idx, segments):
    """-> {strategy: state_text} for row['calls'][call_idx]. See module docstring for S1-S4b."""
    thinking = row["thinking_short"]
    user = row["user"]
    pos = row["calls"][call_idx]["position_char_start"]
    seg = segments[call_idx]
    full_before = thinking[len(THINK_OPEN):pos]  # all prior calls' text, raw
    return {
        "S1": f"{user}\n{seg}",
        "S2": user,
        "S3": seg,
        "S4a": full_before,
        "S4b": strip_injected(full_before),
    }


def seeded_shuffle(options, seed):
    rng = random.Random(seed)
    shuffled = list(options)
    rng.shuffle(shuffled)
    # a shuffle that happens to reproduce the original order tells us nothing about order sensitivity;
    # reroll with a derived seed in that (rare, but possible for small K) case.
    tries = 0
    while shuffled == options and len(options) > 1 and tries < 10:
        seed += 1
        rng = random.Random(seed)
        shuffled = list(options)
        rng.shuffle(shuffled)
        tries += 1
    return shuffled


def call_seed(args_seed, row_id, call_idx):
    h = hashlib.sha256(f"{row_id}:{call_idx}".encode()).hexdigest()
    return args_seed + int(h[:8], 16)


# ---------------------------------------------------------------------------
# scoring (GPU path; lazy torch/kev imports so --dry-run never needs them)
# ---------------------------------------------------------------------------

def make_real_scorer(decision_run, device):
    """-> callable(state_text, instr, options) -> (pred, confidence, max_prob, probs_dict). Loads the
    decision checkpoint exactly like scripts/eval_phase7.py's load_decision_model (bf16, LoadOptions
    fix -- see that function's docstring for why fp32's default would risk an OOM) and mirrors
    score_decide_call's readout, except `instr` is a parameter instead of eval_phase7's module-level
    banking-intent-only INSTR constant: this probe scores five OTHER decision types too, and kev/model.py's
    encode() tokenizes the instruction text into the record (kev/api.py:to_record), so a wrong/fixed
    instruction string would not just be cosmetic -- it would change what the head is scoring."""
    from scripts.eval_phase7 import load_decision_model
    from kev.api import choice_confidence
    dec_tok, dec_model = load_decision_model(decision_run, device)

    def scorer(state_text, instr, options):
        rec = {"state": state_text, "questions": [{"instr": instr, "options": list(options), "label": 0}]}
        enc = dec_model.encode(dec_tok, rec)
        probs = [float(x) for x in dec_model.probs(enc)[0]]
        best_i = max(range(len(probs)), key=lambda k: probs[k])
        return options[best_i], choice_confidence(probs), max(probs), dict(zip(options, probs))

    return scorer


def make_fake_scorer():
    """-> callable with the real scorer's exact signature/return shape, no torch/kev/GPU. Deterministic
    (hash of state+instr+options), so --dry-run output is reproducible; the predicted label and
    confidence are MEANINGLESS -- this only exercises the state-construction/scoring/reporting plumbing
    on the real pilot file."""
    def scorer(state_text, instr, options):
        h = hashlib.sha256(f"{state_text}|{instr}|{options}".encode()).hexdigest()
        best_i = int(h[:8], 16) % len(options)
        # fake probs: a mild peak at best_i, rest uniform-ish, deterministic from the hash
        raw = [int(h[8 + 2 * i:10 + 2 * i] or "0", 16) for i in range(len(options))]
        raw[best_i] += 400
        total = sum(raw) or 1
        probs = [r / total for r in raw]
        best_i = max(range(len(probs)), key=lambda k: probs[k])
        return options[best_i], _fake_choice_confidence(probs), max(probs), dict(zip(options, probs))
    return scorer


def _fake_choice_confidence(p):
    """choice_confidence's own formula ((p_max - 1/K)/(1 - 1/K)), copied so --dry-run never imports
    kev.api (kev/api.py pulls in kev.model, which imports torch at module scope)."""
    k = len(p)
    if k == 1:
        return 1.0
    return (max(p) - 1 / k) / (1 - 1 / k)


# ---------------------------------------------------------------------------
# main scoring loop
# ---------------------------------------------------------------------------

def flatten_calls(rows):
    """-> [(row, call_idx), ...] in file order, across every row with calls (control_nocall rows
    contribute none)."""
    out = []
    for row in rows:
        for i in range(len(row["calls"])):
            out.append((row, i))
    return out


def run_probe(rows, scorer, args):
    flat = flatten_calls(rows)
    if args.limit is not None:
        flat = flat[:args.limit]

    n_forward = len(flat) * len(STRATEGIES) * len(ORDERS)
    print(f"probe: {len(flat)} calls x {len(STRATEGIES)} strategies x {len(ORDERS)} orders = "
          f"{n_forward} forward passes; estimate ~0.5s each -> ~{n_forward * 0.5:.0f}s "
          f"(~{n_forward * 0.5 / 60:.1f} min)", flush=True)

    segments_by_row_id = {}
    results = []
    for row, call_idx in flat:
        rid = row["id"]
        if rid not in segments_by_row_id:
            segments_by_row_id[rid] = build_segments(row)
        segments = segments_by_row_id[rid]
        call = row["calls"][call_idx]
        states = build_states(row, call_idx, segments)
        seed = call_seed(args.seed, rid, call_idx)
        shuffled_opts = seeded_shuffle(call["options"], seed)
        order_opts = {"orig": call["options"], "shuffled": shuffled_opts}

        for strat in STRATEGIES:
            state_text = states[strat]
            for order in ORDERS:
                opts = order_opts[order]
                pred, conf, max_prob, probs = scorer(state_text, call["instruction"], opts)
                results.append({
                    "pilot_id": rid, "task": row["task"], "kind": row["kind"],
                    "call_index": call_idx, "n_calls_in_row": len(row["calls"]),
                    "is_first_call": call_idx == 0,
                    "type": call["type"], "gold_label": call["gold_label"],
                    "strategy": strat, "order": order, "options": opts,
                    "pred": pred, "correct": pred == call["gold_label"],
                    "confidence": conf, "max_prob": max_prob,
                    "conf_floor_hit": conf >= CONF_FLOOR,
                    "state_len_chars": len(state_text),
                    "state_preview": state_text[:300],
                })
    return results


# ---------------------------------------------------------------------------
# reporting (pure Python)
# ---------------------------------------------------------------------------

def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _acc(rows):
    return _mean(float(r["correct"]) for r in rows) if rows else float("nan")


def build_report(results):
    lines = ["# Phase 7R head probe", "",
              f"generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}, "
              f"{len(results)} scored (call, strategy, order) rows, conf_floor={CONF_FLOOR}", ""]

    orig = [r for r in results if r["order"] == "orig"]
    by_strat = {}
    for r in orig:
        by_strat.setdefault(r["strategy"], []).append(r)

    lines.append("## Per-strategy summary (canonical option order only; shuffle-agreement below)")
    lines.append("")
    lines.append("| strategy | n | accuracy | mean confidence | coverage@floor | accuracy@floor |"
                 " shuffle-agreement |")
    lines.append("|---|---|---|---|---|---|---|")
    shuf_by_strat = {}
    for r in results:
        if r["order"] == "shuffled":
            shuf_by_strat.setdefault(r["strategy"], {})[(r["pilot_id"], r["call_index"])] = r
    for strat in STRATEGIES:
        rs = by_strat.get(strat, [])
        n = len(rs)
        acc = _acc(rs)
        mconf = _mean(r["confidence"] for r in rs) if rs else float("nan")
        covered = [r for r in rs if r["conf_floor_hit"]]
        cov = len(covered) / n if n else float("nan")
        acc_floor = _acc(covered) if covered else float("nan")
        shuf_map = shuf_by_strat.get(strat, {})
        agree = [r["pred"] == shuf_map[(r["pilot_id"], r["call_index"])]["pred"]
                 for r in rs if (r["pilot_id"], r["call_index"]) in shuf_map]
        agree_rate = _mean(float(a) for a in agree) if agree else float("nan")
        lines.append(f"| {strat} | {n} | {acc:.3f} | {mconf:.3f} | {cov:.3f} | {acc_floor:.3f} |"
                     f" {agree_rate:.3f} |")

    lines.append("")
    lines.append("## Per-strategy x per-type accuracy vs dev accuracy "
                 f"(flag: drop >= {FLAG_DROP:.2f} absolute below dev)")
    lines.append("")
    lines.append("| strategy | type | n | accuracy | dev accuracy | flag |")
    lines.append("|---|---|---|---|---|---|")
    types_present = sorted({r["type"] for r in orig})
    for strat in STRATEGIES:
        for t in types_present:
            rs = [r for r in by_strat.get(strat, []) if r["type"] == t and r["kind"] != "control_none"]
            if not rs:
                continue
            acc = _acc(rs)
            dev = DEV_ACCURACY.get(t)
            flag = ""
            if dev is not None and acc <= dev - FLAG_DROP:
                flag = f"** FLAG: {acc:.3f} vs dev {dev:.3f} ({acc - dev:+.3f}) **"
            lines.append(f"| {strat} | {t} | {len(rs)} | {acc:.3f} | "
                         f"{'n/a' if dev is None else f'{dev:.3f}'} | {flag} |")

    lines.append("")
    lines.append("## First-call vs later-call accuracy, per strategy")
    lines.append("")
    lines.append("| strategy | first-call n | first-call acc | later-call n | later-call acc |")
    lines.append("|---|---|---|---|---|")
    for strat in STRATEGIES:
        rs = by_strat.get(strat, [])
        first = [r for r in rs if r["is_first_call"]]
        later = [r for r in rs if not r["is_first_call"]]
        lines.append(f"| {strat} | {len(first)} | {_acc(first):.3f} | {len(later)} | {_acc(later):.3f} |")

    lines.append("")
    lines.append("## control_none rows (gold = none_of_these), per strategy")
    lines.append("")
    lines.append("| strategy | n | accuracy (pred == none_of_these) | mean confidence |")
    lines.append("|---|---|---|---|")
    for strat in STRATEGIES:
        rs = [r for r in by_strat.get(strat, []) if r["kind"] == "control_none"]
        if not rs:
            lines.append(f"| {strat} | 0 | n/a | n/a |")
            continue
        acc = _acc(rs)
        mconf = _mean(r["confidence"] for r in rs)
        lines.append(f"| {strat} | {len(rs)} | {acc:.3f} | {mconf:.3f} |")

    return "\n".join(lines) + "\n"


def print_dry_run_examples(rows):
    """Prints 2 example constructed states per strategy (first multi-call row available, else any),
    truncated to 300 chars, per the task's eyeball requirement."""
    multi = [r for r in rows if len(r["calls"]) >= 2]
    row = multi[0] if multi else next((r for r in rows if r["calls"]), rows[0])
    segments = build_segments(row)
    print(f"\n--dry-run example states, row {row['id']!r} ({len(row['calls'])} calls):\n")
    shown = 0
    for call_idx in range(len(row["calls"])):
        if shown >= 2:
            break
        states = build_states(row, call_idx, segments)
        print(f"-- call[{call_idx}] type={row['calls'][call_idx]['type']!r} "
              f"gold={row['calls'][call_idx]['gold_label']!r} --")
        for strat in STRATEGIES:
            print(f"  {strat}: {states[strat][:300]!r}")
        shown += 1
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
    ap.add_argument("--pilot", default="oracle/phase7-reasoning-pilot-v2.jsonl")
    ap.add_argument("--decision-run", default="runs/p4-e4b-final")
    ap.add_argument("--tag", required=True, help="output files are "
                    "oracle/phase7r-head-probe-{tag}.jsonl/.md; never overwritten")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N calls "
                    "(flattened across rows, file order) -- for a smoke run")
    ap.add_argument("--dry-run", action="store_true", help="fake scorer, no torch/kev/GPU; validates "
                    "state construction + reporting against the real pilot file")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0, help="base seed for the per-call option shuffle")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pilot_path = args.pilot if os.path.isabs(args.pilot) else os.path.join(root, args.pilot)
    out_jsonl = os.path.join(root, "oracle", f"phase7r-head-probe-{args.tag}.jsonl")
    out_md = os.path.join(root, "oracle", f"phase7r-head-probe-{args.tag}.md")
    for p in (out_jsonl, out_md):
        if os.path.exists(p):
            print(f"ERROR: {p} already exists; choose a different --tag (never overwritten)",
                  file=sys.stderr)
            return 1

    rows = [json.loads(l) for l in open(pilot_path, encoding="utf-8") if l.strip()]
    n_calls_total = sum(len(r["calls"]) for r in rows)
    print(f"pilot: {len(rows)} rows, {n_calls_total} calls, from {pilot_path}", flush=True)

    if args.dry_run:
        print_dry_run_examples(rows)
        scorer = make_fake_scorer()
    else:
        scorer = make_real_scorer(args.decision_run, args.device)

    results = run_probe(rows, scorer, args)
    write_jsonl(out_jsonl, results)
    print(f"wrote {out_jsonl} ({len(results)} rows)", flush=True)

    report = build_report(results)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"wrote {out_md}", flush=True)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
