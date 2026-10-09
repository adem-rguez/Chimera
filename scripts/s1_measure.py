"""Stage S1 go/no-go measurement: pure Python, stdlib only (plus an OPTIONAL cached-tokenizer lookup --
no network, no torch, no GPU). Joins organic thinking traces (scripts/gen_think_traces.py's output
format, run over scripts/fetch_s1_problems.py's prompts) against extracted decision points
(scripts/extract_decision_points.py's output) and reports the go/no-go metrics.

Inputs:
  --traces   organic traces jsonl: {"id", "user", "thinking", "n_thinking_tokens", "source", ...}
             ("source" -- one of "gsm8k"/"mmlu"/"arc_challenge"/"advice" -- comes straight through from
             scripts/fetch_s1_problems.py's input rows, per gen_think_traces.py's "meta passthrough".)
  --points   extracted decision points jsonl: {"id", "points": [{"quote_start_offset", "quote_choice_end",
             "chosen_key", ...}, ...], ...} (scripts/extract_decision_points.py's output schema).
  --head-scores (OPTIONAL) head-label jsonl, one row per SCORED POINT (not per trace -- a trace can
             have >1 point): {"id": str, "point_idx": int, "head_chosen_key": str}. "id" matches a
             traces/points row's "id"; "point_idx" is the index into that row's "points" list (0-based,
             in extract_decision_points.py's own point order); "head_chosen_key" is whatever label the
             decision-head scorer picked for that point's (question, options) pair -- same bare-key
             space as the point's own "chosen_key" (scripts/gate_phase7r_stageA.py's option_key
             convention: bare keys, not "key: description" strings). A head-scores row with no matching
             (id, point_idx) in --points, or vice versa, is simply excluded from the agreement
             computation and counted as "unscored" in the report -- never an error.

Metrics (one combined report, printed and optionally written to --out):
  - coverage: fraction of TRACES with >=1 valid point, overall and by "source".
  - replaceable token share: for every point, (quote_choice_end - quote_start_offset) CHARACTERS
    converted to an estimated token count (see `estimate_tokens` below), summed per trace, divided by
    that trace's own "n_thinking_tokens" (from --traces, the real re-tokenization gen_think_traces.py
    already computed) -- i.e. "what share of the trace's thinking tokens fall inside a span this system
    could someday replace with an injected result". Traces with zero points contribute a 0.0 share (not
    excluded) so the overall mean reflects coverage too, not just per-point density. Reported as a mean
    over ALL traces (gsm8k/mmlu/arc_challenge/advice combined) and by source.
  - mean points per trace: mean of len(points) over all traces (zero-point traces count as 0).
  - head agreement: among points with a matching --head-scores row, fraction where
    head_chosen_key == point["chosen_key"]. None/NaN if --head-scores is not given.
  - PASS/FAIL vs the three go/no-go gates: coverage >= 0.40, replaceable share >= 0.20,
    head agreement >= 0.85 (skipped -- "not applicable", not fail -- if --head-scores wasn't given).

Token estimate (`estimate_tokens`): tries `AutoTokenizer.from_pretrained(MODEL, revision=REVISION,
local_files_only=True)` ONCE (no network -- if the tokenizer files are not already in the local HF
cache from an earlier run of e.g. gen_think_traces.py, this raises immediately rather than trying to
download); on any failure, falls back to `len(text) / 4` (a documented, widely-used English-text rough
estimate), and the report's header records WHICH mode was used for every number in it -- this is never
silently inconsistent within one report (one mode is chosen once, at startup, for the whole run).

Usage (laptop, CPU, stdlib -- always runnable, no GPU/model weights required):
  .venv/Scripts/python.exe scripts/s1_measure.py \\
      --traces oracle/s1-problems-organic.jsonl --points oracle/s1-decision-points.jsonl \\
      --out reports/s1-gate.md
  .venv/Scripts/python.exe scripts/s1_measure.py \\
      --traces <t.jsonl> --points <p.jsonl> --head-scores <h.jsonl> --out <report.md>
"""
import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "google/gemma-4-E4B-it"
REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"

GATE_COVERAGE = 0.40
GATE_REPLACEABLE_SHARE = 0.20
GATE_HEAD_AGREEMENT = 0.85

CHAR_PER_TOKEN_ESTIMATE = 4.0


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------------------
# tokenizer-or-estimate (no network; cached-only tokenizer lookup, else char/4)
# ---------------------------------------------------------------------------

def make_token_estimator(model=MODEL, revision=REVISION):
    """-> (estimate_fn(text) -> int, mode: str). mode is "gemma-tokenizer" (cached locally, no network
    attempted) or "char/4-estimate" (documented fallback) -- the report prints this once, up front, and
    every number in the report was computed with the SAME mode (chosen once at startup)."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=True)

        def estimate(text):
            if not text:
                return 0
            return len(tok(text, add_special_tokens=False)["input_ids"])
        return estimate, "gemma-tokenizer (cached, local_files_only=True)"
    except Exception as e:  # noqa: BLE001 -- any cache-miss/import failure falls back, documented
        def estimate(text):
            return len(text) / CHAR_PER_TOKEN_ESTIMATE if text else 0.0
        return estimate, f"char/{CHAR_PER_TOKEN_ESTIMATE:g}-estimate (tokenizer unavailable: {e!r})"


# ---------------------------------------------------------------------------
# pure aggregation logic (CPU-testable independent of the tokenizer mode above)
# ---------------------------------------------------------------------------

def join_rows(traces, points_rows):
    """-> {id: {"trace": trace_row, "points": [point, ...]}} -- every trace id, points=[] if that id has
    no --points row at all (not an error: the extractor may have found nothing, or the id may simply be
    missing from an in-progress --points file)."""
    points_by_id = {r["id"]: r.get("points", []) for r in points_rows}
    out = {}
    for t in traces:
        out[t["id"]] = {"trace": t, "points": points_by_id.get(t["id"], [])}
    return out


def trace_replaceable_share(trace_row, points, estimate_fn):
    """-> float in [0, 1] (can exceed 1 only if points overlap AND together exceed n_thinking_tokens --
    not clamped, so a pathological overlap is visible rather than hidden). 0.0 for a trace with no
    points or n_thinking_tokens == 0 (reported separately via coverage, not as a divide-by-zero)."""
    n_thinking = trace_row.get("n_thinking_tokens") or 0
    if not points or n_thinking <= 0:
        return 0.0
    thinking_text = trace_row.get("thinking") or ""
    span_tokens_total = 0.0
    for p in points:
        start, end = p.get("quote_start_offset"), p.get("quote_choice_end")
        if start is None or end is None or end <= start:
            continue
        span_tokens_total += estimate_fn(thinking_text[start:end])
    return span_tokens_total / n_thinking


def compute_report(joined, head_scores_rows, estimate_fn):
    """-> dict with every metric this module reports (see module docstring); pure, no I/O."""
    by_source = collections.defaultdict(list)  # source -> list of (has_point, n_points, share)
    for row_id, entry in joined.items():
        trace, points = entry["trace"], entry["points"]
        source = trace.get("source", "unknown")
        share = trace_replaceable_share(trace, points, estimate_fn)
        by_source[source].append((len(points) > 0, len(points), share))

    def summarize(records):
        n = len(records)
        if n == 0:
            return dict(n=0, coverage=float("nan"), mean_points=float("nan"),
                        mean_replaceable_share=float("nan"))
        n_covered = sum(1 for has_point, _n, _s in records if has_point)
        return dict(
            n=n, coverage=n_covered / n,
            mean_points=sum(n_pts for _h, n_pts, _s in records) / n,
            mean_replaceable_share=sum(s for _h, _n, s in records) / n,
        )

    overall = summarize([rec for recs in by_source.values() for rec in recs])
    per_source = {source: summarize(recs) for source, recs in sorted(by_source.items())}

    # head agreement: index head_scores_rows by (id, point_idx)
    head_by_key = {}
    for r in (head_scores_rows or []):
        head_by_key[(r["id"], r["point_idx"])] = r.get("head_chosen_key", r.get("head_choice"))
    n_scored = n_agree = n_unmatched_head_rows = 0
    matched_keys = set()
    for row_id, entry in joined.items():
        for idx, p in enumerate(entry["points"]):
            head_key = head_by_key.get((row_id, idx))
            if head_key is None:
                continue
            matched_keys.add((row_id, idx))
            n_scored += 1
            n_agree += int(head_key == p.get("chosen_key"))
    n_unmatched_head_rows = len(head_by_key) - len(matched_keys)
    head_agreement = (n_agree / n_scored) if n_scored else None

    return dict(
        overall=overall, per_source=per_source,
        head_agreement=head_agreement, n_head_scored=n_scored,
        n_head_rows_unmatched=n_unmatched_head_rows,
        n_traces=len(joined),
    )


def gate_status(report):
    """-> list of (name, requirement_str, value, status_str) for the three go/no-go gates. head
    agreement gate is "N/A (no --head-scores)" rather than fail/pass when head_agreement is None."""
    rows = []
    cov = report["overall"]["coverage"]
    rows.append(("coverage >= 0.40", f"{cov:.3f}" if cov == cov else "nan",
                 "PASS" if cov >= GATE_COVERAGE else "FAIL"))
    share = report["overall"]["mean_replaceable_share"]
    rows.append(("replaceable token share >= 0.20", f"{share:.3f}" if share == share else "nan",
                 "PASS" if share >= GATE_REPLACEABLE_SHARE else "FAIL"))
    agree = report["head_agreement"]
    if agree is None:
        rows.append(("head agreement >= 0.85", "n/a", "N/A (no --head-scores given)"))
    else:
        rows.append(("head agreement >= 0.85", f"{agree:.3f}",
                     "PASS" if agree >= GATE_HEAD_AGREEMENT else "FAIL"))
    return rows


# ---------------------------------------------------------------------------
# report rendering
# ---------------------------------------------------------------------------

def render_report(report, estimator_mode, traces_path, points_path, head_scores_path):
    lines = ["# Stage S1 go/no-go measurement", "",
              f"traces: {traces_path} ({report['n_traces']} rows)",
              f"points: {points_path}",
              f"head-scores: {head_scores_path or '(none given)'}",
              f"token-count mode: {estimator_mode}", ""]

    lines.append("## Overall")
    lines.append("")
    o = report["overall"]
    lines.append(f"- n traces: {o['n']}")
    lines.append(f"- coverage (>=1 valid point): {o['coverage']:.3f}")
    lines.append(f"- mean points per trace: {o['mean_points']:.3f}")
    lines.append(f"- mean replaceable token share: {o['mean_replaceable_share']:.3f}")
    lines.append("")

    lines.append("## By source")
    lines.append("")
    lines.append("| source | n | coverage | mean points/trace | mean replaceable share |")
    lines.append("|---|---|---|---|---|")
    for source, s in report["per_source"].items():
        lines.append(f"| {source} | {s['n']} | {s['coverage']:.3f} | {s['mean_points']:.3f} | "
                      f"{s['mean_replaceable_share']:.3f} |")
    lines.append("")

    lines.append("## Head agreement")
    lines.append("")
    if report["head_agreement"] is None:
        lines.append("(no --head-scores given)")
    else:
        lines.append(f"- n points scored by head: {report['n_head_scored']}")
        lines.append(f"- agreement with trace's own chosen_key: {report['head_agreement']:.3f}")
        lines.append(f"- head-score rows with no matching (id, point_idx) in --points: "
                      f"{report['n_head_rows_unmatched']}")
    lines.append("")

    lines.append("## Gate status")
    lines.append("")
    lines.append("| gate | value | status |")
    lines.append("|---|---|---|")
    for name, value, status in gate_status(report):
        lines.append(f"| {name} | {value} | {status} |")
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", required=True)
    ap.add_argument("--points", required=True)
    ap.add_argument("--head-scores", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    args = ap.parse_args()

    traces = load_jsonl(resolve(args.traces))
    points_rows = load_jsonl(resolve(args.points))
    head_scores_rows = load_jsonl(resolve(args.head_scores)) if args.head_scores else None

    estimate_fn, mode = make_token_estimator(args.model, args.revision)
    print(f"token-count mode: {mode}", flush=True)

    joined = join_rows(traces, points_rows)
    report = compute_report(joined, head_scores_rows, estimate_fn)
    text = render_report(report, mode, args.traces, args.points, args.head_scores)
    print(text)
    if args.out:
        out_path = resolve(args.out)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
