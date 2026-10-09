"""Stage S1 combined go/no-go report: all FOUR gates from reports/25-phase7g-general-decisions-plan.md's
S1 row in one command -- (a) coverage >= 40%, (b) replaceable token share >= 20%, (c) head agreement
with the trace's own chosen_key >= 85% (scripts/s1_measure.py's three gates, unchanged, called as-is),
plus (d) head accuracy on outcome-verified choices >= 70% (scripts/s1_head_and_outcomes.py's
head_accuracy_on_outcome_verified, new).

This is a thin wrapper -- scripts/s1_measure.py is NOT edited (its own CLI/report is reused verbatim for
(a)-(c)); this script only ADDS gate (d) and prints one combined table. All four inputs are the same
files scripts/s1_measure.py / scripts/s1_head_and_outcomes.py already read:
  --traces, --points        (a)/(b)/(c) -- scripts/s1_measure.py
  --head-scores              (c) head agreement (vs the trace's OWN chosen_key, scripts/s1_measure.py)
  --problems, --headscores   (d) head accuracy on OUTCOME-verified (gold-correct) choices,
                              scripts/s1_head_and_outcomes.py:head_accuracy_on_outcome_verified
Note --head-scores (s1_measure.py's flag, hyphenated) and --headscores (s1_head_and_outcomes.py's flag)
are the SAME file in practice (one scored-points jsonl) but are accepted as two flags here so each
underlying module's own loader/schema expectations stay exactly as documented there; pass the same path
to both unless you have a reason not to.

Usage (CPU, stdlib + the two modules above; no GPU/model):
  .venv/Scripts/python.exe scripts/s1_report.py \\
      --traces oracle/s1-traces.jsonl --points oracle/s1-points.jsonl \\
      --problems oracle/s1-train-problems.jsonl --headscores oracle/s1-headscores.jsonl \\
      --out reports/s1-gate.md
  (--head-scores defaults to --headscores's path if not given separately)
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.s1_measure import (  # noqa: E402
    compute_report, gate_status, join_rows, load_jsonl as s1_load_jsonl,
    make_token_estimator, resolve,
)
from scripts.s1_head_and_outcomes import (  # noqa: E402
    head_accuracy_on_outcome_verified, load_jsonl as outcomes_load_jsonl,
)

GATE_HEAD_ACCURACY = 0.70


def render_combined_report(s1_report, estimator_mode, head_acc_result, paths):
    lines = ["# Stage S1 combined go/no-go report (all four gates)", "",
             f"traces: {paths['traces']} ({s1_report['n_traces']} rows)",
             f"points: {paths['points']}",
             f"head-scores (agreement, c): {paths['head_scores'] or '(none given)'}",
             f"problems (outcomes, d): {paths['problems'] or '(none given)'}",
             f"headscores (outcome accuracy, d): {paths['headscores'] or '(none given)'}",
             f"token-count mode: {estimator_mode}", ""]

    lines.append("## Gate status (all four)")
    lines.append("")
    lines.append("| gate | value | status |")
    lines.append("|---|---|---|")
    for name, value, status in gate_status(s1_report):
        lines.append(f"| {name} | {value} | {status} |")
    if head_acc_result is None:
        lines.append(f"| head accuracy on outcome-verified choices >= {GATE_HEAD_ACCURACY:.2f} | n/a | "
                      "N/A (no --problems/--headscores given) |")
    else:
        acc = head_acc_result["accuracy"]
        if acc is None:
            lines.append(f"| head accuracy on outcome-verified choices >= {GATE_HEAD_ACCURACY:.2f} | "
                          f"n/a (0 scored, {head_acc_result['n_determinable']} determinable) | "
                          "N/A (0 scored) |")
        else:
            status = "PASS" if acc >= GATE_HEAD_ACCURACY else "FAIL"
            lines.append(f"| head accuracy on outcome-verified choices >= {GATE_HEAD_ACCURACY:.2f} | "
                          f"{acc:.3f} (n={head_acc_result['n_scored']}) | {status} |")
    lines.append("")

    lines.append("## Outcome-verified head accuracy detail")
    lines.append("")
    if head_acc_result is None:
        lines.append("(no --problems/--headscores given)")
    else:
        lines.append(f"- checkable points: {head_acc_result['n_checkable']}")
        lines.append(f"- determinable (MC, point IS the final-answer decision): "
                      f"{head_acc_result['n_determinable']}")
        lines.append(f"- scored by head: {head_acc_result['n_scored']}")
        lines.append(f"- correct: {head_acc_result['n_correct']}")
    lines.append("")

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", required=True)
    ap.add_argument("--points", required=True)
    ap.add_argument("--head-scores", default=None, help="for gate (c); see module docstring")
    ap.add_argument("--problems", default=None, help="for gate (d)")
    ap.add_argument("--headscores", default=None, help="for gate (d); defaults --head-scores if unset")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    head_scores_path = args.head_scores or args.headscores
    headscores_path = args.headscores or args.head_scores

    traces = s1_load_jsonl(resolve(args.traces))
    points_rows = s1_load_jsonl(resolve(args.points))
    head_scores_rows = s1_load_jsonl(resolve(head_scores_path)) if head_scores_path else None

    estimate_fn, mode = make_token_estimator()
    joined = join_rows(traces, points_rows)
    s1_report = compute_report(joined, head_scores_rows, estimate_fn)

    head_acc_result = None
    if args.problems and headscores_path:
        problems = outcomes_load_jsonl(resolve(args.problems))
        problems_by_id = {p["id"]: p for p in problems}
        headscores_rows = outcomes_load_jsonl(resolve(headscores_path))
        head_acc_result = head_accuracy_on_outcome_verified(traces, points_rows, problems_by_id,
                                                              headscores_rows)

    paths = dict(traces=args.traces, points=args.points, head_scores=head_scores_path,
                 problems=args.problems, headscores=headscores_path)
    text = render_combined_report(s1_report, mode, head_acc_result, paths)
    print(text)
    if args.out:
        out_path = resolve(args.out)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
