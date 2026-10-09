"""Phase 7R T6 (reports/16 line ~390, "head gate"): run the decision head on every Stage-A call,
using the authored pre-call prose as state, and flag calls the head gets wrong WITH confidence --
those are rejection candidates. Scores every call in oracle/phase7r-built/stageA-*.jsonl with the
same checkpoint/readout as scripts/probe_phase7r_head.py's S1 strategy (the pilot's own authoring
rule: state = user turn + thinking decoded since the previous call), reusing that script's
make_real_scorer/make_fake_scorer verbatim -- this script does not reimplement scoring.

Stage-A format differs from the pilot file probe_phase7r_head.py was built against: there is no
`position_char_start`/char-offset bookkeeping to recover per-call thinking spans from a single
concatenated string. scripts/build_phase7r.py's assemble step already hands us that split directly:
row["thinking_segments"] is a list of length len(calls)+1, where thinking_segments[i] is exactly the
thinking text decoded since the previous call (or since the start, for i==0) up to call i -- the same
span probe_phase7r_head.py's build_segments/build_states compute from position_char_start in the
pilot. So S1's state here is simply f"{row['user']}\\n{row['thinking_segments'][call_idx]}", with no
char-offset helper needed (verified: oracle/phase7r-built's rows have thinking_segments length ==
len(calls)+1 for all 104 rows across stageA-000..012).

Stage-A's call["options"] are already kev.api.option_text-described strings ("key: description"),
frozen by build_phase7r.py (see its make_call_spec/load_type_items) -- passed straight through as the
`options` argument, exactly like probe_phase7r_head.py passes call["options"].

Usage (GPU box; loads ONLY the decision checkpoint, bf16, nothing else):
  .venv/bin/python -u scripts/gate_phase7r_stageA.py --limit 10   # smoke
  .venv/bin/python -u scripts/gate_phase7r_stageA.py              # full

Offline (laptop, no torch/GPU/model -- validates state construction + reporting with a FAKE scorer):
  python -m py_compile scripts/gate_phase7r_stageA.py
  python scripts/gate_phase7r_stageA.py --dry-run
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.probe_phase7r_head import (  # noqa: E402
    CONF_FLOOR, make_fake_scorer, make_real_scorer,
)

CONF_THRESHOLD = 0.50  # reports/16 line ~390: reject only wrong-and-confident rows, threshold ~0.50
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def option_key(opt_text):
    """-> the bare key of a kev.api.option_text-described option string ("key: description" -> "key";
    "key" -> "key", maxsplit=1 so a colon inside the description text doesn't matter). Stage-A's
    call["gold_label"] is always the bare key (build_phase7r.py's make_call_spec); call["options"] are
    the described strings the scorer sees, so predictions need this to compare against gold."""
    return opt_text.split(": ", 1)[0]


def load_rows(pattern):
    paths = sorted(glob.glob(pattern))
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return paths, rows


def build_state(row, call_idx):
    """S1, per probe_phase7r_head.py's build_states: user turn + thinking decoded since the previous
    call. Stage-A's row["thinking_segments"][call_idx] already IS that span (see module docstring)."""
    return f"{row['user']}\n{row['thinking_segments'][call_idx]}"


def flatten_calls(rows):
    """-> [(row, call_idx), ...] in file order; control_nocall rows (calls == []) contribute none."""
    out = []
    for row in rows:
        for i in range(len(row["calls"])):
            out.append((row, i))
    return out


def run_gate(rows, scorer, limit=None):
    flat = flatten_calls(rows)
    if limit is not None:
        flat = flat[:limit]
    results = []
    for row, call_idx in flat:
        call = row["calls"][call_idx]
        state_text = build_state(row, call_idx)
        pred_text, conf, max_prob, probs = scorer(state_text, call["instruction"], call["options"])
        pred = option_key(pred_text)
        gold = call["gold_label"]
        correct = pred == gold
        results.append({
            "row_id": row["id"], "call_id": call.get("call_id", call_idx), "type": call["type"],
            "gold": gold, "pred": pred, "conf": conf, "max_prob": max_prob,
            "correct": correct,
            "wrong_and_confident": (not correct) and conf >= CONF_THRESHOLD,
            "conf_floor_hit": conf >= CONF_FLOOR,
        })
    return results


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _acc(rs):
    return _mean(float(r["correct"]) for r in rs) if rs else float("nan")


def build_report(results, paths):
    lines = ["# Phase 7R Stage-A head gate", "",
              f"generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}, "
              f"{len(results)} calls scored from {len(paths)} batch file(s), "
              f"strategy=S1, conf_threshold={CONF_THRESHOLD}, conf_floor={CONF_FLOOR}", ""]

    n = len(results)
    overall_acc = _acc(results)
    n_wac = sum(1 for r in results if r["wrong_and_confident"])
    lines.append(f"Overall: n={n}, accuracy={overall_acc:.3f}, "
                 f"wrong_and_confident={n_wac} ({n_wac / n:.3f} of all calls)" if n else
                 "Overall: n=0")
    lines.append("")

    types = sorted({r["type"] for r in results})
    lines.append("## Per-type accuracy and mean confidence")
    lines.append("")
    lines.append("| type | n | accuracy | mean confidence | wrong_and_confident |")
    lines.append("|---|---|---|---|---|")
    for t in types:
        rs = [r for r in results if r["type"] == t]
        lines.append(f"| {t} | {len(rs)} | {_acc(rs):.3f} | {_mean(r['conf'] for r in rs):.3f} | "
                     f"{sum(1 for r in rs if r['wrong_and_confident'])} |")
    lines.append("")

    oo = [r for r in results if r["type"] == "order_outcome"]
    lines.append("## order_outcome accuracy")
    lines.append("")
    if oo:
        lines.append(f"n={len(oo)}, accuracy={_acc(oo):.3f}")
    else:
        lines.append("n=0 (no order_outcome calls in the scored batches)")
    lines.append("")

    n_floor = sum(1 for r in results if r["conf_floor_hit"])
    lines.append("## Confidence floor coverage")
    lines.append("")
    lines.append(f"{n_floor}/{n} calls ({n_floor / n:.3f})" if n else "n=0"
                 f" at/above the production floor {CONF_FLOOR}")
    lines.append("")

    wac_rows = [r for r in results if r["wrong_and_confident"]]
    lines.append(f"## Wrong-and-confident rows (conf >= {CONF_THRESHOLD}, pred != gold) "
                 f"-- rejection candidates ({len(wac_rows)})")
    lines.append("")
    if wac_rows:
        lines.append("| row_id | call_id | type | gold | pred | conf |")
        lines.append("|---|---|---|---|---|---|")
        for r in wac_rows:
            lines.append(f"| {r['row_id']} | {r['call_id']} | {r['type']} | {r['gold']} | "
                         f"{r['pred']} | {r['conf']:.3f} |")
    else:
        lines.append("(none)")
    lines.append("")

    return "\n".join(lines) + "\n"


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--built", default="oracle/phase7r-built/stageA-*.jsonl",
                     help="glob (relative to repo root unless absolute) matching built Stage-A batch files")
    ap.add_argument("--decision-run", default="runs/p4-e4b-final")
    ap.add_argument("--out-jsonl", default="oracle/phase7r-stageA-headgate.jsonl")
    ap.add_argument("--out-md", default="reports/20-phase7r-stageA-headgate.md")
    ap.add_argument("--filter-out", default=None,
                     help="write the list of rejected row ids (one per line) to this file")
    ap.add_argument("--limit", type=int, default=None,
                     help="score only the first N calls (flattened across rows, file order) -- smoke run")
    ap.add_argument("--dry-run", action="store_true",
                     help="fake scorer, no torch/kev/GPU; validates state construction + reporting")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    built_pattern = args.built if os.path.isabs(args.built) else os.path.join(ROOT, args.built)
    out_jsonl = args.out_jsonl if os.path.isabs(args.out_jsonl) else os.path.join(ROOT, args.out_jsonl)
    out_md = args.out_md if os.path.isabs(args.out_md) else os.path.join(ROOT, args.out_md)

    paths, rows = load_rows(built_pattern)
    n_calls_total = sum(len(r["calls"]) for r in rows)
    print(f"built: {len(rows)} rows, {n_calls_total} calls, from {len(paths)} file(s) "
          f"matching {built_pattern}", flush=True)

    scorer = make_fake_scorer() if args.dry_run else make_real_scorer(args.decision_run, args.device)

    results = run_gate(rows, scorer, limit=args.limit)
    write_jsonl(out_jsonl, results)
    print(f"wrote {out_jsonl} ({len(results)} rows)", flush=True)

    report = build_report(results, paths)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"wrote {out_md}", flush=True)
    print(report)

    if args.filter_out:
        reject_ids = sorted({r["row_id"] for r in results if r["wrong_and_confident"]})
        out_filter = args.filter_out if os.path.isabs(args.filter_out) else os.path.join(ROOT, args.filter_out)
        with open(out_filter, "w", encoding="utf-8") as f:
            for rid in reject_ids:
                f.write(rid + "\n")
        print(f"wrote {out_filter} ({len(reject_ids)} row ids)", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
