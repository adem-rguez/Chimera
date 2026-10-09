"""Phase 7R "Source H" step B: continue generation from a spliced prefill on the base model.

See scripts/build_phase7r_hybrid.py `prefix` mode (step A) for how the prefill is built: for each decision
row with a valid splice, `prompt_text` is the real chat-template render of the user turn (no assistant
content, add_generation_prompt=True, enable_thinking=True -- same as gen_think_traces.render_prompt), and
`prefix_text` is `<|channel>thought\\n` + the kept organic thinking prefix + the typed `<|decide:TYPE|>`
trigger + the runtime-injected `{instruction} [{options}]<|result|>{gold}<|/result|>` span (GOLD, correct
by construction). This script feeds `prompt_text + prefix_text` to the base model as a PREFILL (not a
prompt it renders itself) and lets the model continue generating from exactly that point, so its own
continuation naturally reads the injected result rather than ignoring it (the bug this whole step exists
to fix -- see scripts/build_phase7r_hybrid.py module docstring).

Modeled directly on scripts/gen_think_traces.py (batched HF generate, left padding, same sampling
defaults, --resume/--limit/--dry-run) -- reused, not re-implemented, wherever the two scripts need the
same thing: EOS_IDS, load_model_and_tokenizer, batched_generate. No --adapter here: this step is base-model
organic continuation, there is never a LoRA adapter in the loop for Source H.

CRITICAL: the prefill's tokenization must match how train_phase7.encode_think will tokenize the final
assembled text later (scripts/build_phase7r_hybrid.py build2 mode). The injected `<|decide:TYPE|>` /
`<|result|>` / `<|/result|>` strings are plain multi-piece text on this tokenizer, not atomic tokens
(reports/17), so `prompt_text + prefix_text` is tokenized as ONE concatenated string (add_special_tokens=
False, no separate tokenization of the two pieces) -- exactly what encode_think does to the final
target_text_short, and exactly what --dry-run below prints the token count of.

Output (one row per line): {"id": str, "continuation": str, "finished": bool, "n_cont_tokens": int}.
  - "continuation": the newly generated text only (skip_special_tokens=False, so the channel-close marker
    and <turn|> byte layout survive for build2 mode to split on), with any leading/trailing "<pad>" runs
    and a trailing "<turn|>" NOT stripped here -- build2 mode needs the raw bytes to find the channel-close
    marker and the turn boundary itself (same reasoning as gen_think_traces.py keeping "raw_text" verbatim).
  - "finished": True iff any of generation_config.json's 3-way eos ids [1, 106, 50] appears anywhere in
    this row's own newly generated tokens (same definition as gen_think_traces.py).
  - "n_cont_tokens": non-pad new-token count, for throughput reporting.

Usage (GPU box):
  smoke run (5 rows):
    .venv/bin/python -u scripts/continue_phase7r_hybrid.py \\
        --prefixes oracle/phase7r-hybrid-prefixes.jsonl --output oracle/phase7r-hybrid-continuations.jsonl \\
        --limit 5 --batch-size 4 --seed 0
  full run:
    .venv/bin/python -u scripts/continue_phase7r_hybrid.py \\
        --prefixes oracle/phase7r-hybrid-prefixes.jsonl --output oracle/phase7r-hybrid-continuations.jsonl \\
        --max-new-tokens 500 --batch-size 8 --seed 0 --resume

Offline (laptop, CPU, tokenizer only -- no model weights, no GPU):
  .venv/Scripts/python.exe -m py_compile scripts/continue_phase7r_hybrid.py
  .venv/Scripts/python.exe scripts/continue_phase7r_hybrid.py \\
      --prefixes oracle/phase7r-hybrid-prefixes.jsonl --output <scratch-out.jsonl> --dry-run --limit 3
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.gen_think_traces import (  # noqa: E402 -- reused, not copied
    DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P, MODEL, REVISION, batched_generate,
    load_model_and_tokenizer,
)

DEFAULT_MAX_NEW_TOKENS = 500


def load_prefixes(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("id", "prompt_text", "prefix_text"):
                if key not in row:
                    raise ValueError(f"line {i} (id={row.get('id')!r}): missing required field {key!r}")
            rows.append(row)
    return rows


def load_done_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(l)["id"] for l in f if l.strip()}


def write_jsonl_append(f, rows):
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.flush()
    os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# dry run (CPU, tokenizer only) -- prints the first fully-assembled prefill and its token count, from ONE
# tokenization of the concatenated string (see module docstring's CRITICAL note).
# ---------------------------------------------------------------------------

def run_dry(rows, args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    prefills = [r["prompt_text"] + r["prefix_text"] for r in rows]
    lens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prefills]
    print(f"dry-run: {len(rows)} prefix rows")
    print(f"prefill token lengths: min={min(lens)} mean={sum(lens) / len(lens):.1f} max={max(lens)}")
    print(f"first fully-assembled prefill (id={rows[0]['id']!r}), {lens[0]} tokens:")
    print(repr(prefills[0]))
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prefixes", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="skip ids already present in --output; appends instead of overwriting")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N remaining rows")
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU only: validate prefill tokenization, no model load, no output written")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-impl", default="sdpa")
    args = ap.parse_args()

    prefixes_path = args.prefixes if os.path.isabs(args.prefixes) else os.path.join(ROOT, args.prefixes)
    out_path = args.output if os.path.isabs(args.output) else os.path.join(ROOT, args.output)

    rows = load_prefixes(prefixes_path)
    print(f"loaded {len(rows)} prefix rows from {args.prefixes}", flush=True)

    if args.dry_run:
        if args.limit:
            rows = rows[:args.limit]
        return sys.exit(run_dry(rows, args))

    done = load_done_ids(out_path) if args.resume else set()
    if done:
        before = len(rows)
        rows = [r for r in rows if r["id"] not in done]
        print(f"--resume: {before - len(rows)}/{before} rows already in {args.output}, "
              f"{len(rows)} remaining", flush=True)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print("nothing to do", flush=True)
        return

    import torch
    torch.manual_seed(args.seed)

    tok, model = load_model_and_tokenizer(args.model, args.revision, args.device, args.attn_impl)

    prefills = [r["prompt_text"] + r["prefix_text"] for r in rows]
    prefill_lens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prefills]
    order = sorted(range(len(rows)), key=lambda i: prefill_lens[i])
    rows_sorted = [rows[i] for i in order]
    prefills_sorted = [prefills[i] for i in order]

    print(f"generating {len(rows_sorted)} rows, max_new_tokens={args.max_new_tokens}, "
          f"batch_size={args.batch_size}, temperature={args.temperature}, top_p={args.top_p}, "
          f"top_k={args.top_k}", flush=True)

    mode = "a" if args.resume else "w"
    t0 = time.time()
    total_new_tokens = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for i in range(0, len(rows_sorted), args.batch_size):
            batch_rows = rows_sorted[i:i + args.batch_size]
            batch_prefills = prefills_sorted[i:i + args.batch_size]
            results = batched_generate(tok, model, batch_prefills, args, args.device)
            out_rows = []
            batch_gen_tokens = 0
            for row, (raw_text, finished, n_new) in zip(batch_rows, results):
                batch_gen_tokens += n_new
                out_rows.append({"id": row["id"], "continuation": raw_text, "finished": finished,
                                  "n_cont_tokens": n_new})
            write_jsonl_append(f, out_rows)
            total_new_tokens += batch_gen_tokens
            elapsed = time.time() - t0
            print(f"wrote {len(out_rows)} rows ({i + len(out_rows)}/{len(rows_sorted)} total); "
                  f"batch generated {batch_gen_tokens} tokens; running throughput "
                  f"{total_new_tokens / max(elapsed, 1e-9):.1f} tok/s", flush=True)

    elapsed = time.time() - t0
    print(f"done: {len(rows_sorted)} rows, {total_new_tokens} continuation tokens, "
          f"{elapsed:.1f}s, {total_new_tokens / max(elapsed, 1e-9):.1f} tok/s", flush=True)


if __name__ == "__main__":
    main()
