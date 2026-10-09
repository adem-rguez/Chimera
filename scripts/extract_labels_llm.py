"""LLM-based forced-choice label extraction for Phase 7R arm-B (think-only) baseline.

Replaces the regex/substring heuristic in `scripts/eval_phase7r_armB.py` (37-58% extraction rate,
reports/19) with a forced-choice *reader*: the base instruction model itself is shown the original
user prompt plus arm B's own VISIBLE ANSWER (not its thinking) and the lettered option list (with
descriptions, from `options_with_descriptions`), plus one extra "none of these / does not commit"
letter, and asked "Which option does the assistant's answer commit to?". No text is generated --
a single forward pass reads the next-token logits at the position right after the rendered chat
prompt (`add_generation_prompt=True`, `enable_thinking=False`, same model and loading approach as
scripts/gen_think_traces.py: google/gemma-4-E4B-it @ ee0ef6023621cff504d758262d4e04895a5af4a2, bf16,
no adapter), and argmaxes over just the candidate letters' first-token ids.

Per-letter token ids are resolved PER ROW (not hardcoded), by tokenizing the rendered prompt both
with and without the bare letter appended and diffing the ids -- this is robust to any
tokenizer-boundary effects where the letter's own tokenization right after the chat template's
"model" turn marker merges differently than a bare standalone letter would. If the prefix doesn't
match (rare BPE merge across the boundary), falls back to the standalone-letter tokenization and
counts it so the run log reports how often this happened.

Inputs: `oracle/t3-armB-think.fixed.jsonl` (arm B: --thinking on, no calls; has "answer"/"thinking"
per scripts/gen_think_traces.py) joined by "id" with `oracle/phase7r-natural-eval.jsonl` (source of
truth for "options_with_descriptions"/"gold_label"/"type"), restricted to `kind == "decision"`.

--use-thinking-tail: for rows whose arm-B "answer" is empty/blank (channel opened but never closed
within the token budget -- see gen_think_traces.py), the prompt's "ASSISTANT'S ANSWER" section is
augmented with an extra "ASSISTANT'S REASONING (end of thinking, answer was empty)" section holding
the last 400 chars of "thinking", before the single forward pass -- simplest reading of the spec's
"second pass...when the answer is not conclusive" that avoids a second model invocation: the
augmentation happens at prompt-construction time, per row, and every row still gets exactly one
forward pass. Without this flag, an empty answer is sent through with its (empty) ASSISTANT'S ANSWER
section, which the model will normally read as "does not commit" (the extra none-of-these letter).

Output (`oracle/t3-armB-extract.jsonl`, one row per line, append-and-flush, resume-safe by "id"):
  {"id", "type", "gold_label", "chosen", "p_top", "margin", "committed"}
  - "chosen": the option key the model argmaxed to, or the sentinel "__none__" if it picked the
    extra "none of these / does not commit" letter.
  - "p_top"/"margin": softmax probability mass is computed over the FULL vocab, then renormalized
    over just the candidate letters' first-token ids (so these sum to 1 across all candidates
    including "__none__"); "p_top" is the top candidate's renormalized probability, "margin" is
    top-1 minus top-2 among that same renormalized set.
  - "committed": chosen != "__none__".

Usage:
  Offline (laptop, CPU, tokenizer only -- no model weights, no GPU):
    .venv/Scripts/python.exe -m py_compile scripts/extract_labels_llm.py
    .venv/Scripts/python.exe scripts/extract_labels_llm.py --dry-run --limit 5

  GPU box, smoke (8 rows):
    .venv/bin/python -u scripts/extract_labels_llm.py \\
        --armb oracle/t3-armB-think.fixed.jsonl --natural oracle/phase7r-natural-eval.jsonl \\
        --output oracle/t3-armB-extract-smoke.jsonl --limit 8 --batch-size 8

  GPU box, full run (resume-safe):
    .venv/bin/python -u scripts/extract_labels_llm.py \\
        --armb oracle/t3-armB-think.fixed.jsonl --natural oracle/phase7r-natural-eval.jsonl \\
        --output oracle/t3-armB-extract.jsonl --use-thinking-tail --batch-size 8 --resume
"""
import argparse
import json
import os
import sys
import time

MODEL = "google/gemma-4-E4B-it"
REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
THINKING_TAIL_CHARS = 400
NONE_KEY = "__none__"


# ---------------------------------------------------------------------------
# data loading / join
# ---------------------------------------------------------------------------

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def join_rows(armb_path, natural_path):
    """-> list of dicts, one per decision row present in both files: {id, type, gold_label,
    options_with_descriptions, user, answer, thinking}. Order follows natural eval's decision rows."""
    armb_by_id = {r["id"]: r for r in load_jsonl(armb_path)}
    natural_rows = load_jsonl(natural_path)
    decision_rows = [r for r in natural_rows if r["kind"] == "decision"]
    out = []
    missing = 0
    for nr in decision_rows:
        b = armb_by_id.get(nr["id"])
        if b is None:
            missing += 1
            continue
        out.append(dict(
            id=nr["id"], type=nr["type"], gold_label=nr["gold_label"],
            options_with_descriptions=nr["options_with_descriptions"],
            user=nr["user"], answer=b.get("answer") or "", thinking=b.get("thinking") or "",
        ))
    if missing:
        print(f"WARNING: {missing} decision rows have no arm-B match", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# prompt construction (pure string logic -- safe for --dry-run)
# ---------------------------------------------------------------------------

def letters_for(n_options):
    """-> list of n_options+1 single-letter strings 'A', 'B', ... -- the extra trailing letter is the
    'none of these' choice."""
    return [chr(ord("A") + i) for i in range(n_options + 1)]


def build_prompt_body(row, use_thinking_tail):
    """-> (user_content str, letter_to_key dict incl. NONE_KEY, used_tail bool)."""
    options = row["options_with_descriptions"]
    letters = letters_for(len(options))
    letter_to_key = {letters[i]: options[i][0] for i in range(len(options))}
    letter_to_key[letters[-1]] = NONE_KEY

    used_tail = False
    answer_section = row["answer"].strip() or "(no visible answer)"
    extra_section = ""
    if use_thinking_tail and not row["answer"].strip():
        tail = row["thinking"][-THINKING_TAIL_CHARS:].strip()
        if tail:
            extra_section = f"\n\nASSISTANT'S REASONING (end of thinking, answer was empty):\n{tail}"
            used_tail = True

    option_lines = "\n".join(
        f"{letters[i]}. {key}: {desc}" for i, (key, desc) in enumerate(options)
    )
    option_lines += f"\n{letters[-1]}. None of these / the answer does not clearly commit to any option above."

    user_content = (
        f"USER REQUEST:\n{row['user']}\n\n"
        f"ASSISTANT'S ANSWER:\n{answer_section}{extra_section}\n\n"
        f"OPTIONS:\n{option_lines}\n\n"
        "Which option does the assistant's answer commit to? Respond with exactly one letter."
    )
    return user_content, letter_to_key, used_tail


SYSTEM_PROMPT = (
    "You are a careful grader. You will see a user request and an AI assistant's answer to it. "
    "Decide which single option from the list the assistant's answer commits to, based only on "
    "what the assistant actually says. Respond with exactly one letter and nothing else."
)


def render_prompt(tok, user_content):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


# ---------------------------------------------------------------------------
# per-row letter -> token id resolution (needs a tokenizer; CPU-only, safe for --dry-run)
# ---------------------------------------------------------------------------

def resolve_letter_token_ids(tok, prompt_text, letters):
    """-> (dict letter -> token id, n_fallback int). For each letter, tokenizes `prompt_text` and
    `prompt_text + letter` and diffs the ids; if the prompt's own tokenization isn't a strict prefix
    of the combined one (BPE merge across the boundary), falls back to the standalone-letter token id
    and counts it in n_fallback (see module docstring)."""
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    out = {}
    n_fallback = 0
    for letter in letters:
        combined_ids = tok(prompt_text + letter, add_special_tokens=False)["input_ids"]
        if combined_ids[:len(prompt_ids)] == prompt_ids and len(combined_ids) > len(prompt_ids):
            out[letter] = combined_ids[len(prompt_ids)]
        else:
            n_fallback += 1
            standalone = tok(letter, add_special_tokens=False)["input_ids"]
            out[letter] = standalone[0]
    return out, n_fallback


# ---------------------------------------------------------------------------
# dry run (CPU, tokenizer only)
# ---------------------------------------------------------------------------

def run_dry(rows, args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    print(f"dry-run: {len(rows)} rows, use_thinking_tail={args.use_thinking_tail}")
    row = rows[0]
    user_content, letter_to_key, used_tail = build_prompt_body(row, args.use_thinking_tail)
    prompt = render_prompt(tok, user_content)
    letters = list(letter_to_key.keys())
    letter_ids, n_fallback = resolve_letter_token_ids(tok, prompt, letters)
    n_tok = len(tok(prompt, add_special_tokens=False)["input_ids"])
    print(f"first row id={row['id']!r} type={row['type']!r} gold={row['gold_label']!r} "
          f"used_thinking_tail={used_tail} prompt_tokens={n_tok}")
    print(f"letters -> option keys: {letter_to_key}")
    print(f"letters -> token ids: {letter_ids} (fallback count={n_fallback})")
    print("rendered prompt:")
    safe = prompt.encode("ascii", "backslashreplace").decode("ascii")
    print(repr(safe))
    return 0


# ---------------------------------------------------------------------------
# GPU-dependent scoring (imported/used lazily so --dry-run never needs torch)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name, revision, device, attn_impl="sdpa"):
    import torch
    from transformers import AutoModelForCausalLM
    from kev.model import load_tokenizer
    tok = load_tokenizer(model_name, revision)
    tok.padding_side = "right"  # forward-pass-only scoring: no generation, causal attention is
                                 # unaffected by trailing pad tokens, and right-padding lets us index
                                 # each row's last real position as attention_mask.sum() - 1 directly.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, dtype=torch.bfloat16, attn_implementation=attn_impl
    ).to(device).eval()
    return tok, model


def score_batch(tok, model, rows, args, device):
    """-> list of dicts {id, type, gold_label, chosen, p_top, margin, committed}, one per row in
    `rows` (a batch). One forward pass for the whole batch; per-row letter->token-id maps are
    resolved first (CPU, cheap) since each row's option set/prompt differs."""
    import torch

    prompts = []
    per_row_letter_to_id = []
    per_row_letter_to_key = []
    used_tail_count = 0
    fallback_count = 0
    for row in rows:
        user_content, letter_to_key, used_tail = build_prompt_body(row, args.use_thinking_tail)
        if used_tail:
            used_tail_count += 1
        prompt = render_prompt(tok, user_content)
        letters = list(letter_to_key.keys())
        letter_ids, n_fallback = resolve_letter_token_ids(tok, prompt, letters)
        fallback_count += n_fallback
        prompts.append(prompt)
        per_row_letter_to_id.append(letter_ids)
        per_row_letter_to_key.append(letter_to_key)

    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    with torch.no_grad():
        logits = model(**enc).logits  # (batch, seq_len, vocab)
    real_lens = enc["attention_mask"].sum(dim=1)

    out = []
    for i, row in enumerate(rows):
        pos = int(real_lens[i].item()) - 1
        row_logits = logits[i, pos, :].to(torch.float32)
        probs_full = torch.softmax(row_logits, dim=-1)
        letter_to_key = per_row_letter_to_key[i]
        letter_to_id = per_row_letter_to_id[i]
        cand_probs = {letter: probs_full[tok_id].item() for letter, tok_id in letter_to_id.items()}
        total = sum(cand_probs.values()) or 1e-12
        renorm = {letter: p / total for letter, p in cand_probs.items()}
        ranked = sorted(renorm.items(), key=lambda kv: kv[1], reverse=True)
        top_letter, p_top = ranked[0]
        margin = p_top - (ranked[1][1] if len(ranked) > 1 else 0.0)
        chosen = letter_to_key[top_letter]
        out.append(dict(
            id=row["id"], type=row["type"], gold_label=row["gold_label"], chosen=chosen,
            p_top=p_top, margin=margin, committed=(chosen != NONE_KEY),
        ))
    return out, used_tail_count, fallback_count


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def write_jsonl_append(f, rows):
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.flush()
    os.fsync(f.fileno())


def load_done_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(l)["id"] for l in f if l.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--armb", default="oracle/t3-armB-think.fixed.jsonl")
    ap.add_argument("--natural", default="oracle/phase7r-natural-eval.jsonl")
    ap.add_argument("--output", default="oracle/t3-armB-extract.jsonl")
    ap.add_argument("--use-thinking-tail", action="store_true",
                    help="augment the prompt with the last 400 chars of thinking when answer is "
                         "empty/blank (see module docstring)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--resume", action="store_true",
                    help="skip ids already present in --output; appends instead of overwriting")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N remaining rows")
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU only: validate prompt rendering + letter token ids, no model load, "
                         "no output written")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-impl", default="sdpa")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    armb_path = args.armb if os.path.isabs(args.armb) else os.path.join(root, args.armb)
    natural_path = args.natural if os.path.isabs(args.natural) else os.path.join(root, args.natural)
    out_path = args.output if os.path.isabs(args.output) else os.path.join(root, args.output)

    rows = join_rows(armb_path, natural_path)
    print(f"joined {len(rows)} decision rows", flush=True)

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

    tok, model = load_model_and_tokenizer(args.model, args.revision, args.device, args.attn_impl)

    # sort by rendered prompt token length ascending, same reasoning as gen_think_traces.py (reduce
    # padding waste); recomputed here since prompt length depends on use_thinking_tail augmentation.
    def _plen(row):
        user_content, _, _ = build_prompt_body(row, args.use_thinking_tail)
        return len(tok(render_prompt(tok, user_content), add_special_tokens=False)["input_ids"])
    rows_sorted = sorted(rows, key=_plen)

    print(f"scoring {len(rows_sorted)} rows, use_thinking_tail={args.use_thinking_tail}, "
          f"batch_size={args.batch_size}", flush=True)

    mode = "a" if args.resume else "w"
    t0 = time.time()
    total_tail = total_fallback = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for i in range(0, len(rows_sorted), args.batch_size):
            batch = rows_sorted[i:i + args.batch_size]
            out_rows, used_tail_count, fallback_count = score_batch(tok, model, batch, args, args.device)
            write_jsonl_append(f, out_rows)
            total_tail += used_tail_count
            total_fallback += fallback_count
            elapsed = time.time() - t0
            print(f"wrote {len(out_rows)} rows ({i + len(out_rows)}/{len(rows_sorted)} total); "
                  f"{elapsed:.1f}s elapsed", flush=True)

    print(f"done: {len(rows_sorted)} rows, {total_tail} used thinking-tail augmentation, "
          f"{total_fallback} letter-token fallbacks, {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
