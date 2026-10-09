"""Batched organic-thinking generation on google/gemma-4-E4B-it, for Phase 7R Stage A/B tasks T3
(natural eval set: arms A/no-think and B/think-only baselines) and T8 (replay distillation: frozen
dev rows, `enable_thinking=True`, organic thinking+answer kept as-is, no calls) -- see
reports/16-phase7-reasoning-proposal.md lines ~212-300, 383-399 and reports/17-phase7r-template-dump.md
(the tokenizer-level ground truth this script is written against).

No LoRA adapter is loaded here by default -- this is the BASE instruction model's own chat/thinking
behavior, captured once and frozen, per reports/16's "Natural E4B-thinking evaluation set, pre-training"
(T3) and "E4B self-distillation" replay source (T8). Both call sites just need: render each prompt
through the real chat template with `enable_thinking` on or off, sample, split the decoded continuation
into the thinking channel and the visible answer, and write it out. `--adapter PATH` is an optional
escape hatch (added for measuring a trained LoRA's natural trigger rate) that loads a PEFT adapter on
top of the base model, same pattern as scripts/eval_phase7.py's load_lora_model (PeftModel.from_pretrained,
no merge); omitted, this script's behavior is unchanged.

Model: google/gemma-4-E4B-it @ revision ee0ef6023621cff504d758262d4e04895a5af4a2, bf16 (the trained/served
dtype per other Phase 7 scripts), transformers>=5.17,<6, torch>=2.6,<2.9 (pyproject.toml pins).
No vLLM available on this box -- plain `model.generate`, batched by hand (left padding), same pattern as
scripts/eval_phase7.py's batched_generate.

Chat template facts this script relies on (reports/17, re-verified directly against the live tokenizer
with transformers 5.17.0 while writing this script -- see the two `tok.apply_chat_template` calls below
and their printed output, included here as the record of that check):
  - `enable_thinking=True` makes the template inject `<|think|>\\n` into the (possibly auto-created, empty)
    system turn by itself. The caller's own `system`/`user` content must NOT contain `<|think|>\\n` --
    doing so duplicates it (reports/17 Test A'). This script never writes that marker into prompt text.
  - With no `system` message in the input at all, `apply_chat_template` still synthesizes an empty system
    turn and puts `<|think|>\\n` in it when `enable_thinking=True` (confirmed directly, see module
    docstring note above); `--thinking off` renders no system turn at all when none is supplied. Either
    way, a bare `{"role": "user", ...}` turn works -- no caller-side system scaffolding is required.
  - The thinking channel in the model's OWN completion is delimited by the literal strings
    `<|channel>thought\\n` (open) and `\\n<channel|>` (close) -- plain multi-piece text on this tokenizer,
    not single special tokens (reports/17's byte-level table), so splitting is done on decoded text
    (skip_special_tokens=False, so the markers survive decoding) rather than on token ids.
  - generation_config.json's `eos_token_id` is the 3-way list `[1, 106, 50]` = `<eos>`, `<turn|>`,
    `<|tool_response>` (reports/17). Unlike scripts/eval_phase7.py (which overrides this down to plain
    `<eos>` because its SFT targets are plain text that never uses the other two), this script generates
    real multi-turn-chat-shaped completions and KEEPS the full 3-way list, so a model that cleanly ends its
    turn at `<turn|>` is correctly treated as finished rather than running to `--max-new-tokens`.

Input (one prompt per line): {"id": str, "system": str (optional), "user": str, ...any other keys}.
Any keys beyond "id"/"system"/"user" (and "system"/"user" themselves) are passed through to the output
row verbatim ("meta passthrough" -- simplest reading of the spec: the whole input row survives, with the
fields below added/overwritten).

Output (one row per line, written append-and-flush per batch so a box sleep/crash loses at most one
in-flight batch):
  {..every input key.., "thinking": str, "answer": str, "raw_text": str, "n_thinking_tokens": int,
   "n_answer_tokens": int, "finished": bool}
  - "raw_text": the verbatim decoded continuation (just the newly generated tokens, none of the prompt),
    skip_special_tokens=False -- the unprocessed record, markers and all.
  - "thinking"/"answer": "raw_text" split on the channel markers above, markers themselves stripped,
    each .strip()-ed. "--thinking off" (no channel ever opens) or a row that never opens/never closes the
    channel within the token budget both degrade gracefully: "thinking" is "" (or everything after the
    open marker, if opened-but-not-closed) and "answer" is the remainder (or "", respectively) -- see
    split_thinking_answer's docstring for the exact cases. No row is dropped or raises for this.
  - "n_thinking_tokens"/"n_answer_tokens": re-tokenizations (add_special_tokens=False) of the split
    "thinking"/"answer" strings, independently -- NOT an offset-mapping split of one tokenization (unlike
    scripts/check_phase7r_pilot.py's split_tokens, which has to do that because it also accounts for
    runtime-injected call spans; nothing is injected into these rows, so the simpler independent
    re-tokenization is equivalent and this script does not import that machinery).
  - "finished": True iff any of the three generation_config.json eos ids [1, 106, 50] appears anywhere in
    this row's own newly-generated tokens (i.e. the row's own sequence reached a real stop, not a
    batch-wide stall at --max-new-tokens while other rows in the same batch were still going).

Batching: prompts are tokenized once (chat-template text -> ids), then sorted by prompt TOKEN length
ascending before batching (reduces padding waste -- a long batchmate forces every other row in its batch
to pad up to it) and left-padded by hand within each batch. Consequence: output row ORDER is by prompt
length, not input order; harmless for JSONL (every row is self-identified by "id"), noted here so a
consumer that expects input order is not surprised.

Sampling: defaults come from the model's own generation_config.json (temperature=1.0, top_p=0.95,
top_k=64, do_sample=true) -- this script sets the same three and samples by default, since the whole
point of T3/T8 is the model's OWN organic register, not a canonical greedy answer. `--temperature 0`
switches to greedy (do_sample=False, top_p/top_k ignored) as an explicit escape hatch; not the default.

Usage (GPU box):
  smoke run (5 prompts):
    .venv/bin/python -u scripts/gen_think_traces.py \\
        --input oracle/phase7r-t3-natural-prompts.jsonl --output oracle/phase7r-t3-natural-smoke.jsonl \\
        --thinking on --limit 5 --batch-size 4 --seed 0
  full run:
    .venv/bin/python -u scripts/gen_think_traces.py \\
        --input oracle/phase7r-t3-natural-prompts.jsonl --output oracle/phase7r-t3-natural.jsonl \\
        --thinking on --max-new-tokens 1200 --batch-size 8 --seed 0 --resume

Offline (laptop, CPU, tokenizer only -- no model weights, no GPU):
  .venv/Scripts/python.exe -m py_compile scripts/gen_think_traces.py
  .venv/Scripts/python.exe scripts/gen_think_traces.py --input <small-jsonl-in-scratchpad> \\
      --output <scratch-out.jsonl> --thinking on --dry-run
"""
import argparse
import json
import os
import re
import sys
import time

MODEL = "google/gemma-4-E4B-it"
REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
EOS_IDS = [1, 106, 50]          # generation_config.json: <eos>, <turn|>, <|tool_response> (reports/17)
THINK_OPEN = "<|channel>thought\n"
THINK_CLOSE = "<channel|>"            # NOTE: no preceding "\n" -- the model does not reliably emit one
                                       # before the close marker (observed: "...)<channel|>**answer**"),
                                       # so matching on "\n<channel|>" silently failed to find the close
                                       # and glued the answer into "thinking" on ~most rows. Fixed.
PAD_TOKEN_TEXT = "<pad>"
TURN_TOKEN_TEXT = "<turn|>"
DEFAULT_MAX_NEW_TOKENS = 1200
DEFAULT_TEMPERATURE = 1.0       # generation_config.json
DEFAULT_TOP_P = 0.95            # generation_config.json
DEFAULT_TOP_K = 64              # generation_config.json


# ---------------------------------------------------------------------------
# prompt rendering (tokenizer only -- safe for --dry-run)
# ---------------------------------------------------------------------------

def render_prompt(tok, row, thinking_on):
    """-> prompt text, via the real chat template. One user turn, plus a system turn iff the input row
    supplies one -- never synthesizes '<|think|>\\n' ourselves; the template does that on its own when
    enable_thinking=True (see module docstring)."""
    messages = []
    if row.get("system"):
        messages.append({"role": "system", "content": row["system"]})
    messages.append({"role": "user", "content": row["user"]})
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking_on)


def validate_row(row, line_no):
    if "id" not in row:
        raise ValueError(f"line {line_no}: missing required field 'id'")
    if "user" not in row or not isinstance(row["user"], str) or not row["user"]:
        raise ValueError(f"line {line_no} (id={row.get('id')!r}): missing/empty required field 'user'")
    if "system" in row and row["system"] is not None and not isinstance(row["system"], str):
        raise ValueError(f"line {line_no} (id={row.get('id')!r}): 'system' must be a string if present")


def load_prompts(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            validate_row(row, i)
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# thinking/answer split (pure string logic -- unit-testable without a model)
# ---------------------------------------------------------------------------

def _strip_pad_and_turn(text):
    """Strip leading/trailing runs of the literal "<pad>" marker and a trailing "<turn|>" marker, then
    .strip() whitespace. Left-padding/eos-padding and a clean end-of-turn both show up as literal decoded
    text here (skip_special_tokens=False upstream), e.g. "...a dramatic prop.<turn|><pad><pad>...<pad>".
    Order matters: pads usually trail a <turn|>, so pads are stripped first, then <turn|>, repeated until
    stable (handles any pad/turn stacked in either order)."""
    while True:
        new = re.sub(r"^(?:" + re.escape(PAD_TOKEN_TEXT) + r")+", "", text)
        new = re.sub(r"(?:" + re.escape(PAD_TOKEN_TEXT) + r")+$", "", new)
        if new.endswith(TURN_TOKEN_TEXT):
            new = new[:-len(TURN_TOKEN_TEXT)]
        if new == text:
            return new.strip()
        text = new


def split_thinking_answer(raw_text, thinking_on):
    """-> (thinking, answer), both .strip()-ed, markers removed, and with any leading/trailing "<pad>"
    runs and a trailing "<turn|>" stripped (see _strip_pad_and_turn) -- raw_text itself is left untouched
    by the caller; only these derived strings are cleaned. Cases:
      - thinking disabled, or the channel never opens at all (model ignored it): thinking="", answer=raw.
      - channel opens but never closes within the token budget (ran out of --max-new-tokens mid-thought):
        thinking = everything after the open marker, answer = "".
      - normal case: thinking = text between the markers, answer = everything after the close marker.
    THINK_OPEN/THINK_CLOSE are plain multi-piece text on this tokenizer (reports/17), not special tokens,
    so this is a decoded-text search (skip_special_tokens=False decode upstream keeps them present), same
    reasoning as scripts/check_phase7r_pilot.py's injected_spans but there is no runtime-injected span to
    account for here since nothing is spliced into organic generation. NOTE: THINK_CLOSE has no required
    preceding "\\n" -- the model does not reliably emit one before closing the channel."""
    if not thinking_on:
        return "", _strip_pad_and_turn(raw_text)
    start = raw_text.find(THINK_OPEN)
    if start < 0:
        return "", _strip_pad_and_turn(raw_text)
    body_start = start + len(THINK_OPEN)
    end = raw_text.find(THINK_CLOSE, body_start)
    if end < 0:
        return _strip_pad_and_turn(raw_text[body_start:]), ""
    return (_strip_pad_and_turn(raw_text[body_start:end]),
            _strip_pad_and_turn(raw_text[end + len(THINK_CLOSE):]))


# ---------------------------------------------------------------------------
# GPU-dependent generation (imported/used lazily so --dry-run never needs torch)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name, revision, device, attn_impl="sdpa", adapter_dir=None):
    """adapter_dir=None (default): base model only, unchanged. adapter_dir set: loads a PEFT adapter on
    top of the base model, same pattern as scripts/eval_phase7.py's load_lora_model (PeftModel.from_pretrained,
    no merge_and_unload -- eval_phase7.py doesn't merge either, so this matches it)."""
    import torch
    from transformers import AutoModelForCausalLM
    from kev.model import load_tokenizer
    tok = load_tokenizer(model_name, revision)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, dtype=torch.bfloat16, attn_implementation=attn_impl
    ).to(device)
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir).to(device)
    model = model.eval()
    return tok, model


def batched_generate(tok, model, prompt_texts, args, device):
    """-> list of (raw_text, finished, n_new_tokens) for ONE batch (caller does the chunking -- see
    main()). raw_text is the newly generated continuation only, decoded with skip_special_tokens=False
    (so the thinking-channel markers survive); finished is whether this row's OWN sequence produced one
    of EOS_IDS anywhere in its new tokens (see module docstring); n_new_tokens is the non-pad count, for
    throughput reporting. Left-pads by hand; `prompt_texts` is assumed already sorted by the caller
    across the whole run for padding efficiency, not just within this batch."""
    import torch
    eos_set = set(EOS_IDS)
    greedy = args.temperature <= 0
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id,
                      eos_token_id=EOS_IDS, do_sample=not greedy)
    if not greedy:
        gen_kwargs.update(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    enc = tok(prompt_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    prompt_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        ids = model.generate(**enc, **gen_kwargs)
    out = []
    for j in range(len(prompt_texts)):
        new_ids = ids[j, prompt_len:]
        new_list = new_ids.tolist()
        raw_text = tok.decode(new_ids, skip_special_tokens=False)
        finished = any(t in eos_set for t in new_list)
        nonpad = sum(1 for t in new_list if t != tok.pad_token_id)
        out.append((raw_text, finished, nonpad))
    return out


# ---------------------------------------------------------------------------
# dry run (CPU, tokenizer only)
# ---------------------------------------------------------------------------

def run_dry(rows, args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    thinking_on = args.thinking == "on"
    prompts = [render_prompt(tok, r, thinking_on) for r in rows]
    lens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    print(f"dry-run: {len(rows)} prompts, thinking={args.thinking}")
    print(f"prompt token lengths: min={min(lens)} mean={sum(lens)/len(lens):.1f} max={max(lens)}")
    print(f"first rendered prompt (id={rows[0]['id']!r}):")
    print(repr(prompts[0]))
    think_marker_ok = ("<|think|>\n" in prompts[0]) == thinking_on
    print(f"'<|think|>\\n' present in rendered prompt: {'<|think|>' + chr(10) in prompts[0]} "
          f"(expected {thinking_on}) -- {'OK' if think_marker_ok else 'MISMATCH'}")
    return 0 if think_marker_ok else 1


# ---------------------------------------------------------------------------
# offline resplit (CPU, tokenizer only -- no model load; re-derives thinking/answer/token counts from an
# existing output file's "raw_text", e.g. after a split_thinking_answer bug fix)
# ---------------------------------------------------------------------------

def run_resplit(in_path, out_path, model_name, revision):
    """Re-derive "thinking"/"answer"/"n_thinking_tokens"/"n_answer_tokens" from each row's existing
    "raw_text" (left verbatim) using the current split_thinking_answer, overwriting those four fields and
    passing every other field through unchanged. thinking_on=True is used unconditionally: when no
    thinking channel is actually present in raw_text (arm A / thinking disabled), split_thinking_answer's
    "channel never opens" branch is identical to its thinking_on=False branch, so this is safe for either
    arm without needing to know which arm a row came from."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    n_total = n_nonempty_answer = n_finished = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            n_total += 1
            thinking, answer = split_thinking_answer(row["raw_text"], True)
            n_think = len(tok(thinking, add_special_tokens=False)["input_ids"]) if thinking else 0
            n_ans = len(tok(answer, add_special_tokens=False)["input_ids"]) if answer else 0
            row["thinking"] = thinking
            row["answer"] = answer
            row["n_thinking_tokens"] = n_think
            row["n_answer_tokens"] = n_ans
            if answer:
                n_nonempty_answer += 1
            if row.get("finished"):
                n_finished += 1
            fout.write(json.dumps(row) + "\n")
    print(f"resplit {n_total} rows -> {out_path}: "
          f"{n_nonempty_answer}/{n_total} non-empty answer, {n_finished}/{n_total} finished=True",
          flush=True)
    return 0


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
    ap.add_argument("--resplit", nargs=2, metavar=("IN", "OUT"),
                    help="offline mode: no model load, re-derive thinking/answer/token counts from an "
                         "existing output JSONL's raw_text into a new file; all other flags except "
                         "--model/--revision are ignored")
    ap.add_argument("--input")
    ap.add_argument("--output")
    ap.add_argument("--thinking", choices=["on", "off"])
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                    help="generation_config.json default 1.0; <=0 switches to greedy (do_sample=False)")
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="skip ids already present in --output; appends instead of overwriting")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N remaining rows")
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU only: validate prompt rendering, no model load, no output written")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--adapter", default=None,
                    help="optional PEFT adapter dir (e.g. runs/p7r-smoke/step-NNNN) to load on top of"
                         " the base model, same pattern as scripts/eval_phase7.py's load_lora_model;"
                         " omitted (default), this script's behavior is unchanged")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.resplit:
        rsplit_in, rsplit_out = args.resplit
        rsplit_in = rsplit_in if os.path.isabs(rsplit_in) else os.path.join(root, rsplit_in)
        rsplit_out = rsplit_out if os.path.isabs(rsplit_out) else os.path.join(root, rsplit_out)
        return sys.exit(run_resplit(rsplit_in, rsplit_out, args.model, args.revision))

    if not (args.input and args.output and args.thinking):
        ap.error("--input/--output/--thinking are required unless --resplit is used")

    in_path = args.input if os.path.isabs(args.input) else os.path.join(root, args.input)
    out_path = args.output if os.path.isabs(args.output) else os.path.join(root, args.output)

    rows = load_prompts(in_path)
    print(f"loaded {len(rows)} prompts from {args.input}", flush=True)

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

    thinking_on = args.thinking == "on"
    tok, model = load_model_and_tokenizer(
        args.model, args.revision, args.device, args.attn_impl, adapter_dir=args.adapter)

    prompt_texts = [render_prompt(tok, r, thinking_on) for r in rows]
    prompt_lens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompt_texts]
    order = sorted(range(len(rows)), key=lambda i: prompt_lens[i])
    rows_sorted = [rows[i] for i in order]
    prompts_sorted = [prompt_texts[i] for i in order]

    print(f"generating {len(rows_sorted)} rows, thinking={args.thinking}, "
          f"max_new_tokens={args.max_new_tokens}, batch_size={args.batch_size}, "
          f"temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}", flush=True)

    mode = "a" if args.resume else "w"
    t0 = time.time()
    total_new_tokens = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for i in range(0, len(rows_sorted), args.batch_size):
            batch_rows = rows_sorted[i:i + args.batch_size]
            batch_prompts = prompts_sorted[i:i + args.batch_size]
            results = batched_generate(tok, model, batch_prompts, args, args.device)
            out_rows = []
            batch_gen_tokens = 0
            for row, (raw_text, finished, n_new) in zip(batch_rows, results):
                thinking, answer = split_thinking_answer(raw_text, thinking_on)
                n_think = len(tok(thinking, add_special_tokens=False)["input_ids"]) if thinking else 0
                n_ans = len(tok(answer, add_special_tokens=False)["input_ids"]) if answer else 0
                batch_gen_tokens += n_new
                out = dict(row)
                out.update(thinking=thinking, answer=answer, raw_text=raw_text,
                           n_thinking_tokens=n_think, n_answer_tokens=n_ans, finished=finished)
                out_rows.append(out)
            write_jsonl_append(f, out_rows)
            total_new_tokens += batch_gen_tokens
            elapsed = time.time() - t0
            print(f"wrote {len(out_rows)} rows ({i + len(out_rows)}/{len(rows_sorted)} total); "
                  f"batch generated {batch_gen_tokens} tokens; running throughput "
                  f"{total_new_tokens / max(elapsed, 1e-9):.1f} tok/s", flush=True)

    elapsed = time.time() - t0
    print(f"done: {len(rows_sorted)} rows, {total_new_tokens} thinking+answer tokens, "
          f"{elapsed:.1f}s, {total_new_tokens / max(elapsed, 1e-9):.1f} tok/s", flush=True)


if __name__ == "__main__":
    main()
