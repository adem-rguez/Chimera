"""Score gen_think_traces.py output (run on oracle/phase7r-natural-eval.jsonl, with --adapter pointing at
a trained Phase 7R LoRA) for the natural (organic, un-injected) trigger rate of `<|decide:TYPE|>` inside
the model's own thinking channel.

Input: one JSONL row per line, the gen_think_traces.py output schema -- must have "kind" (decision /
control / adversarial), "type" (one of the 5 valid decision types for decision/adversarial rows, or
"control" for control rows -- passed through from oracle/phase7r-natural-eval.jsonl), "thinking" (the
split-out thinking-channel text, i.e. everything before the `<channel|>` close marker -- see
gen_think_traces.py's split_thinking_answer), "n_thinking_tokens", "finished".

A "trigger" is a regex match of `<|decide[:TYPE]|>` inside "thinking". Bare `<|decide|>` (no type) or a
type outside the 5 valid ones counts as present-but-malformed (metric (d)), not a valid typed trigger.

Metrics (per kind, per type):
  (a) trigger rate -- fraction of rows with >=1 trigger (any form, typed-valid or malformed) in thinking.
  (b) type-correct rate -- decision rows only: fraction with >=1 trigger whose TYPE equals the row's own
      "type".
  (c) false-trigger rate -- control rows only: fraction with >=1 trigger of any form.
  (d) malformed-trigger rate -- fraction of rows with >=1 trigger that is bare `<|decide|>` or has a type
      outside the 5 valid types.
  (e) termination rate -- fraction with finished==True (indicative only: without the runtime injection
      the continuation after a trigger is off-distribution, so the model's own choice to stop here isn't
      representative of real serving behavior).
  (f) mean triggers per decision row, mean thinking tokens.

PASS/FAIL gates (stop criteria from the task):
  - trigger rate on ALL decision rows >= 50%
  - false-trigger rate on ALL control rows <= 5%
  - termination rate on ALL rows >= 95%

Usage:
  .venv/Scripts/python.exe scripts/score_phase7r_triggers.py --input oracle/phase7r-t3-natural.jsonl
  .venv/Scripts/python.exe scripts/score_phase7r_triggers.py --selftest
"""
import argparse
import json
import os
import re
import sys
import tempfile

VALID_TYPES = {"claim_handling", "order_outcome", "answer_type", "kb_category", "news_topic"}
TRIGGER_RE = re.compile(r"<\|decide(?::([^|]*))?\|>")


def find_triggers(thinking):
    """-> list of (type_or_None, valid_bool) for every trigger match in `thinking`. type_or_None is the
    captured TYPE string, or None for a bare '<|decide|>'. valid_bool is True iff type_or_None is one of
    VALID_TYPES (bare triggers and out-of-vocabulary types are both invalid/malformed)."""
    out = []
    for m in TRIGGER_RE.finditer(thinking or ""):
        t = m.group(1)
        out.append((t, t in VALID_TYPES))
    return out


def score_row(row):
    thinking = row.get("thinking", "") or ""
    triggers = find_triggers(thinking)
    has_trigger = len(triggers) > 0
    has_malformed = any(not valid for _t, valid in triggers)
    row_type = row.get("type")
    type_correct = any(valid and t == row_type for t, valid in triggers)
    return {
        "has_trigger": has_trigger,
        "n_triggers": len(triggers),
        "has_malformed": has_malformed,
        "type_correct": type_correct,
        "finished": bool(row.get("finished")),
        "n_thinking_tokens": row.get("n_thinking_tokens", 0) or 0,
    }


def _rate(rows, pred):
    if not rows:
        return None
    return sum(1 for r in rows if pred(r)) / len(rows)


def _mean(rows, key):
    if not rows:
        return None
    return sum(key(r) for r in rows) / len(rows)


def fmt(x, pct=True):
    if x is None:
        return "n/a"
    return f"{x:.1%}" if pct else f"{x:.1f}"


def build_report(rows):
    scored = [(r, score_row(r)) for r in rows]

    def group(pred):
        return [s for r, s in scored if pred(r)]

    kinds = sorted({r.get("kind") for r, _ in scored})
    types = sorted({r.get("type") for r, _ in scored})

    lines = ["# Phase 7R natural trigger-rate scoring", "", f"n rows = {len(rows)}", ""]

    lines.append("## Per kind")
    lines.append("")
    lines.append("| kind | n | trigger rate | type-correct (decision) | false-trigger (control) |"
                 " malformed rate | termination rate | mean triggers/row | mean thinking tok |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    decision_rows = group(lambda r: r.get("kind") == "decision")
    control_rows = group(lambda r: r.get("kind") == "control")
    for k in kinds:
        g = group(lambda r, k=k: r.get("kind") == k)
        trig = _rate(g, lambda s: s["has_trigger"])
        tc = _rate(g, lambda s: s["type_correct"]) if k == "decision" else None
        ft = _rate(g, lambda s: s["has_trigger"]) if k == "control" else None
        mal = _rate(g, lambda s: s["has_malformed"])
        term = _rate(g, lambda s: s["finished"])
        mtrig = _mean(g, lambda s: s["n_triggers"])
        mtok = _mean(g, lambda s: s["n_thinking_tokens"])
        lines.append(f"| {k} | {len(g)} | {fmt(trig)} | {fmt(tc)} | {fmt(ft)} | {fmt(mal)} |"
                     f" {fmt(term)} | {fmt(mtrig, pct=False)} | {fmt(mtok, pct=False)} |")

    lines.append("")
    lines.append("## Per type")
    lines.append("")
    lines.append("| type | kind(s) | n | trigger rate | type-correct (decision) | malformed rate |"
                 " termination rate | mean triggers/row | mean thinking tok |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for t in types:
        g_rows = [r for r, _s in scored if r.get("type") == t]
        g = [s for r, s in scored if r.get("type") == t]
        kset = ",".join(sorted({r.get("kind") for r in g_rows}))
        trig = _rate(g, lambda s: s["has_trigger"])
        dec_g = [s for r, s in zip(g_rows, g) if r.get("kind") == "decision"]
        tc = _rate(dec_g, lambda s: s["type_correct"]) if dec_g else None
        mal = _rate(g, lambda s: s["has_malformed"])
        term = _rate(g, lambda s: s["finished"])
        mtrig = _mean(g, lambda s: s["n_triggers"])
        mtok = _mean(g, lambda s: s["n_thinking_tokens"])
        lines.append(f"| {t} | {kset} | {len(g)} | {fmt(trig)} | {fmt(tc)} | {fmt(mal)} | {fmt(term)} |"
                     f" {fmt(mtrig, pct=False)} | {fmt(mtok, pct=False)} |")

    all_rows_scored = [s for _r, s in scored]
    gate_trigger = _rate(decision_rows, lambda s: s["has_trigger"])
    gate_false = _rate(control_rows, lambda s: s["has_trigger"])
    gate_term = _rate(all_rows_scored, lambda s: s["finished"])

    def gate_line(name, value, cmp, bound):
        if value is None:
            status = "n/a (no rows)"
        else:
            ok = cmp(value, bound)
            status = "PASS" if ok else "FAIL"
        return f"- {name}: {fmt(value)} (threshold {cmp.__name__} {bound:.0%}) -> {status}"

    lines.append("")
    lines.append("## PASS/FAIL gates")
    lines.append("")
    p1 = gate_trigger is not None and gate_trigger >= 0.50
    p2 = gate_false is not None and gate_false <= 0.05
    p3 = gate_term is not None and gate_term >= 0.95
    lines.append(f"- trigger rate on decision rows >= 50%: {fmt(gate_trigger)} -> "
                 f"{'PASS' if p1 else 'FAIL'}")
    lines.append(f"- false-trigger rate on control rows <= 5%: {fmt(gate_false)} -> "
                 f"{'PASS' if p2 else 'FAIL'}")
    lines.append(f"- termination rate (all rows) >= 95%: {fmt(gate_term)} -> "
                 f"{'PASS' if p3 else 'FAIL'} (indicative only -- see module docstring on off-distribution"
                 f" continuation after an un-injected trigger)")
    overall = "PASS" if (p1 and p2 and p3) else "FAIL"
    lines.append(f"- OVERALL: {overall}")

    return "\n".join(lines) + "\n"


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def _selftest_rows():
    return [
        # decision, correct typed trigger
        {"id": "d1", "kind": "decision", "type": "claim_handling",
         "thinking": "Let me think. <|decide:claim_handling|> seems right.",
         "n_thinking_tokens": 10, "finished": True},
        # decision, wrong type trigger
        {"id": "d2", "kind": "decision", "type": "news_topic",
         "thinking": "I'll check <|decide:kb_category|> here.",
         "n_thinking_tokens": 8, "finished": True},
        # decision, no trigger at all
        {"id": "d3", "kind": "decision", "type": "order_outcome",
         "thinking": "Just reasoning in plain text, no trigger.",
         "n_thinking_tokens": 12, "finished": False},
        # decision, malformed bare trigger
        {"id": "d4", "kind": "decision", "type": "answer_type",
         "thinking": "Hmm <|decide|> unclear what type.",
         "n_thinking_tokens": 9, "finished": True},
        # control, clean (no trigger) -- the good case
        {"id": "c1", "kind": "control", "type": "control",
         "thinking": "Nothing special here, just answering.",
         "n_thinking_tokens": 5, "finished": True},
        # control, false trigger (bad case)
        {"id": "c2", "kind": "control", "type": "control",
         "thinking": "Wait <|decide:news_topic|> maybe?",
         "n_thinking_tokens": 7, "finished": True},
        # adversarial, triggers correctly
        {"id": "a1", "kind": "adversarial", "type": "kb_category",
         "thinking": "<|decide:kb_category|> despite the adversarial framing.",
         "n_thinking_tokens": 11, "finished": True},
        # adversarial, unfinished
        {"id": "a2", "kind": "adversarial", "type": "answer_type",
         "thinking": "still reasoning...",
         "n_thinking_tokens": 20, "finished": False},
    ]


def run_selftest():
    scratch = tempfile.gettempdir()
    path = os.path.join(scratch, "score_phase7r_triggers_selftest.jsonl")
    rows = _selftest_rows()
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"selftest: wrote {len(rows)} synthetic rows -> {path}")
    print(build_report(rows))
    # hand-checked expectations against the synthetic rows above:
    #   decision: 3/4 trigger (d1,d2,d4) = 75% >= 50% -> PASS
    #   decision type-correct: only d1 = 1/4 = 25%
    #   control: 1/2 false-trigger (c2) = 50% > 5% -> FAIL
    #   termination (all 8 rows): 6/8 finished = 75% < 95% -> FAIL
    scored = {r["id"]: score_row(r) for r in rows}
    assert scored["d1"]["has_trigger"] and scored["d1"]["type_correct"]
    assert scored["d2"]["has_trigger"] and not scored["d2"]["type_correct"]
    assert not scored["d3"]["has_trigger"]
    assert scored["d4"]["has_trigger"] and scored["d4"]["has_malformed"]
    assert not scored["c1"]["has_trigger"]
    assert scored["c2"]["has_trigger"]
    assert scored["a1"]["has_trigger"] and scored["a1"]["type_correct"]
    assert not scored["a2"]["has_trigger"]
    print("selftest: all per-row assertions OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input")
    ap.add_argument("--selftest", action="store_true",
                    help="run against a tiny synthetic jsonl (written to the system temp dir), no "
                         "--input needed")
    args = ap.parse_args()

    if args.selftest:
        return sys.exit(run_selftest())

    if not args.input:
        ap.error("--input is required unless --selftest is used")

    rows = load_rows(args.input)
    print(f"loaded {len(rows)} rows from {args.input}", flush=True)
    print(build_report(rows))


if __name__ == "__main__":
    main()
