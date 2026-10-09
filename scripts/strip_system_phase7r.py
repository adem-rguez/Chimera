"""Phase 7R: strip the system-priming message from already-built rows (oracle/phase7r-built/*.jsonl),
re-rendering each row's prompt/target text through the real chat template with NO system turn, so the
retrain prompt format matches the natural-eval/inference prompts, which carry no system message at all
(scripts/gen_think_traces.py's render_prompt only appends a system turn `iff the input row supplies one`;
with none supplied the template still synthesizes an empty system turn and injects `<|think|>\\n` into
it on its own -- see that function's docstring and module docstring). Content (thinking segments,
trigger, injected span, answer) is kept byte-identical to the original built row; only the system message
is removed and every char-offset-derived field (target_text_short/long, masked_spans_short, tokens_*,
delta_*) is recomputed from scratch via scripts/build_phase7r.py's own helpers (build_reasoning,
injected_spans_typed, token_accounting) -- never hand-crafted strings.

Usage (laptop, CPU, tokenizer only):
    python scripts/strip_system_phase7r.py --in-dir oracle/phase7r-built --out-dir oracle/phase7r-built-nosys
    python scripts/strip_system_phase7r.py --in-dir oracle/phase7r-built --out-dir oracle/phase7r-built-nosys \\
        --single-only
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.build_phase7r import (  # noqa: E402  -- reused, not copied
    MODEL, REVISION, build_reasoning, get_tokenizer, injected_spans_typed, token_accounting,
)


def render_row_nosys(tok, user, reasoning, answer):
    """Same render path as build_phase7r.render_row, but with NO system message in the input messages
    list at all -- matches scripts/gen_think_traces.py's render_prompt(row, thinking_on=True) when
    row.get("system") is falsy. Returns (full_text, prompt_len), same contract as render_row."""
    messages = [{"role": "user", "content": user}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                      enable_thinking=True)
    full = tok.apply_chat_template(
        messages + [{"role": "assistant", "content": answer, "reasoning": reasoning}],
        tokenize=False, add_generation_prompt=False, enable_thinking=True)
    if not full.startswith(prompt):
        raise ValueError("chat template render did not produce prompt as an exact prefix (no-system path)")
    return full, len(prompt)


def restrip_row(tok, rec):
    calls = rec["calls"]
    segs = rec["thinking_segments"]
    answer = rec["answer"]
    user = rec["user"]
    if calls:
        reasoning_short = build_reasoning(calls, segs)
    else:
        reasoning_short = segs[0]
    reasoning_long = rec["thinking_long"]

    full_short, plen_short = render_row_nosys(tok, user, reasoning_short, answer)
    full_long, plen_long = render_row_nosys(tok, user, reasoning_long, answer)

    injected = injected_spans_typed(full_short)
    if injected_spans_typed(full_long):
        raise ValueError(f"{rec['id']}: thinking_long contains trigger/result markers")

    acct_short = token_accounting(tok, full_short, plen_short, injected)
    acct_long = token_accounting(tok, full_long, plen_long, [])

    masked_spans_short = [[0, plen_short]] + [[s, e] for s, e in injected]

    out = dict(rec)
    out["system"] = ""
    out["target_text_short"] = full_short
    out["target_text_long"] = full_long
    out["masked_spans_short"] = masked_spans_short
    out["tokens_short"] = acct_short
    out["tokens_long"] = acct_long
    out["delta_supervised"] = acct_long["tokens_supervised"] - acct_short["tokens_supervised"]
    out["delta_total"] = ((acct_long["tokens_total"] - acct_long["tokens_prompt_masked"]) -
                           (acct_short["tokens_total"] - acct_short["tokens_prompt_masked"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in-dir", default="oracle/phase7r-built")
    ap.add_argument("--out-dir", default="oracle/phase7r-built-nosys")
    ap.add_argument("--single-only", action="store_true",
                     help="keep only rows with <=1 call (eval/inference prompts are single-item)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    args = ap.parse_args()

    in_dir = args.in_dir if os.path.isabs(args.in_dir) else os.path.join(ROOT, args.in_dir)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    tok = get_tokenizer(args.model, args.revision)

    paths = sorted(glob.glob(os.path.join(in_dir, "stageA-[0-9][0-9][0-9].jsonl")))
    n_total = n_kept = n_dropped_multi = 0
    for path in paths:
        out_rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                n_total += 1
                if args.single_only and len(rec["calls"]) > 1:
                    n_dropped_multi += 1
                    continue
                out_rows.append(restrip_row(tok, rec))
                n_kept += 1
        out_path = os.path.join(out_dir, os.path.basename(path))
        with open(out_path, "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")
        print(f"{os.path.basename(path)}: {len(out_rows)} rows -> {out_path}")

    print(f"total: {n_total} rows read, {n_kept} written, {n_dropped_multi} dropped "
          f"(--single-only, >1 call)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
