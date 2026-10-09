"""Stage S1 decision-point extractor: reads organic thinking traces (scripts/gen_think_traces.py's
output format, over scripts/fetch_s1_problems.py's prompts) and, using the BASE instruction model
(google/gemma-4-E4B-it @ ee0ef6023621cff504d758262d4e04895a5af4a2, NO adapter, forced short generation,
`enable_thinking=False` -- this is an ANALYSIS pass over the trace's own already-decoded text, not a
continuation of its thinking), asks the model to point at where in its own reasoning it weighed
concrete alternatives and then committed to one.

Output (`--output`, one row per line, append-and-flush, resume-safe by "id"):
  {"id": str, "points": [{"quote_start": str, "quote_choice": str, "quote_start_offset": int,
                           "quote_start_end": int, "quote_choice_offset": int, "quote_choice_end": int,
                           "question": str, "options": [{"key": str, "desc": str}, ...],
                           "chosen_key": str}, ...],
   "n_candidate_points": int, "n_dropped_points": int, "drop_reasons": [str, ...],
   "raw_model_output": str}
  - "points": at most 3, ALWAYS validated (see `validate_points` below) -- no unverified point is ever
    written here. A trace with no valid point gets "points": [].
  - "quote_start"/"quote_choice": VERBATIM substrings of the input row's "thinking" field (char-exact,
    confirmed by `str.find`, offsets recorded) -- "where the model starts weighing alternatives" and
    "where it commits", per the spec this script is written against.
  - "chosen_key" is guaranteed to be one of "options"' keys (validate_points enforces this) AND to pass
    a commit-check against the trace's OWN later text (see `verify_chosen_key_string_evidence` /
    `verify_chosen_key_forced_choice` below) -- a candidate point whose chosen_key has no evidence in
    the trace is dropped, not just flagged.
  - "raw_model_output": the analysis model's own raw decoded JSON text (or best-effort salvage), for
    debugging a bad extraction without re-running the model.

Two independent pieces, split so the quote/offset/cap/chosen-key logic is fully CPU-testable without a
model (`validate_points`, `verify_chosen_key_string_evidence`, `parse_model_json_points`) while only the
JSON-producing generation itself (`generate_points_batch`) needs the GPU:

  1. GPU-dependent: render an analysis prompt (EXTRACTION_SYSTEM_PROMPT + the trace's user+thinking
     text) through the real chat template with `enable_thinking=False`, batch-generate a short JSON
     response (same batching/left-padding/EOS pattern as scripts/gen_think_traces.py's batched_generate,
     reused directly -- not reimplemented), and best-effort parse it as a JSON array of candidate points
     via `parse_model_json_points` (tolerant of a model that wraps the array in prose or a code fence).

  2. CPU-only: `validate_points(trace, candidates, verify_fn=verify_chosen_key_string_evidence)` --
     for each candidate point (in the order the model gave them), drop it (recording why) unless ALL of:
       - quote_start/quote_choice are both non-empty strings and both occur VERBATIM in `trace`
         (`trace.find`), quote_choice found starting no earlier than quote_start's own offset (commit
         cannot precede the start of weighing);
       - options is a non-empty list of {"key","desc"} objects;
       - chosen_key is one of options' keys;
       - `verify_fn(trace, candidate)` is True -- by default `verify_chosen_key_string_evidence`, a
         pure-Python heuristic: the CHOSEN option's key or description words must appear (case folded)
         somewhere in `trace` at or after quote_choice's offset, i.e. the trace's own later text backs
         up the claimed commitment. `--verify forced-choice` (GPU, optional, see below) swaps in a
         second model forward pass instead (reusing extract_labels_llm.py's per-row letter->token-id
         resolution trick, imported, not reimplemented, since it is pure-tokenizer logic until actually
         scored).
     Keeps at most 3 valid points, in the model's own order; every rejection (verbatim-quote failure,
     chosen_key not in options, ordering violation, failed commit-check, >3rd valid point) is recorded
     in `drop_reasons` as a short human-readable string, never silently.

`--verify forced-choice` (GPU, optional; default is the string-evidence heuristic above, which needs no
model call beyond the main extraction pass): for each candidate point that otherwise passes, builds a
forced-choice prompt (trace's quote_choice text as "the assistant's commitment", the candidate's own
options lettered, plus a trailing "none of these" letter -- exactly extract_labels_llm.py's pattern) and
does ONE extra forward pass (no generation) per point, argmaxing over the candidate letters' first-token
ids (`extract_labels_llm.resolve_letter_token_ids`, imported); the point passes iff the argmax letter
maps back to chosen_key.

Usage:
  Offline (laptop, CPU, tokenizer only -- no model weights, no GPU):
    .venv/Scripts/python.exe -m py_compile scripts/extract_decision_points.py
    .venv/Scripts/python.exe scripts/extract_decision_points.py --selftest
    .venv/Scripts/python.exe scripts/extract_decision_points.py --input <organic-traces.jsonl> \\
        --output <scratch-out.jsonl> --dry-run --limit 2

  GPU box, smoke (8 rows):
    .venv/bin/python -u scripts/extract_decision_points.py \\
        --input oracle/s1-problems-organic.jsonl --output oracle/s1-decision-points-smoke.jsonl \\
        --limit 8 --batch-size 8

  GPU box, full run (resume-safe), with the stronger forced-choice commit check:
    .venv/bin/python -u scripts/extract_decision_points.py \\
        --input oracle/s1-problems-organic.jsonl --output oracle/s1-decision-points.jsonl \\
        --verify forced-choice --batch-size 8 --resume
"""
import argparse
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL = "google/gemma-4-E4B-it"
REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
MAX_POINTS_PER_TRACE = 3
DEFAULT_MAX_NEW_TOKENS = 600

EXTRACTION_SYSTEM_PROMPT = (
    "You will see a USER REQUEST and an AI assistant's private REASONING (its internal thinking, not "
    "shown to the user) while answering that request. Find up to 3 places in the REASONING where the "
    "assistant clearly weighs concrete alternatives and then commits to one of them. Do NOT invent "
    "alternatives that are not actually discussed in the REASONING.\n\n"
    "Respond with a JSON array only (no prose, no code fence). Each element:\n"
    '{"quote_start": "<verbatim substring of REASONING, copied EXACTLY character-for-character, where '
    'the weighing begins>", "quote_choice": "<verbatim substring of REASONING, copied EXACTLY, where it '
    'commits to one alternative>", "question": "<short question the alternatives answer>", '
    '"options": [{"key": "<short_slug>", "desc": "<short description>"}, ...], '
    '"chosen_key": "<the key of the option actually chosen>"}\n\n'
    "If there is no such point, respond with []. Both quote fields MUST be copied verbatim from "
    "REASONING -- do not paraphrase, summarize, or fix typos in them."
)


def build_extraction_user_content(row):
    """-> the analysis prompt's user-turn text (pure string logic, no model/tokenizer needed)."""
    return (
        f"USER REQUEST:\n{row['user']}\n\n"
        f"REASONING (the assistant's own private thinking):\n{row['thinking']}"
    )


# ---------------------------------------------------------------------------
# JSON parsing of the model's own output (pure string/json logic)
# ---------------------------------------------------------------------------

def _extract_bracketed_array(text):
    """-> the substring of `text` spanning the first balanced '[' ... ']' (bracket-depth counted,
    ignoring brackets inside JSON string literals), or None if no complete balanced array is found.
    Tolerates a model that wraps its JSON array in prose or a ```json ... ``` fence."""
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_model_json_points(raw_text):
    """-> (points: list[dict], error: str | None). Tries json.loads on the raw text first; on failure,
    salvages the first balanced bracketed array (_extract_bracketed_array) and retries. `points` is []
    (with no error) for a clean "no decision points" response. Never raises -- a malformed response
    yields ([], "<reason>") so the caller can record it rather than crash a batch run."""
    text = raw_text.strip()
    for candidate in (text, _extract_bracketed_array(text)):
        if candidate is None:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed, None
        return [], f"parsed JSON is not a list (got {type(parsed).__name__})"
    return [], "no valid JSON array found in model output"


# ---------------------------------------------------------------------------
# validate_points -- CPU-only, no model needed (verify_fn defaults to the string-evidence heuristic)
# ---------------------------------------------------------------------------

def verify_chosen_key_string_evidence(trace, point):
    """-> bool. Pure-Python commit check: True iff the trace's OWN text at-or-after quote_choice's
    offset contains, case-folded, either the chosen option's key (with underscores turned to spaces) or
    at least one content word (len >= 4) of its description. This is the default, no-model verification
    mode; `--verify forced-choice` (main()) swaps in a model-scored check instead (see module docstring).
    Caller is expected to have already confirmed quote_choice occurs verbatim in `trace` and to pass in
    the SAME point dict `validate_points` is currently checking (needs "chosen_key"/"options" and the
    offset `validate_points` itself just computed -- see call site)."""
    choice_offset = point.get("_quote_choice_offset")
    if choice_offset is None:
        return False
    tail = trace[choice_offset:].casefold()
    options_by_key = {o["key"]: o["desc"] for o in point["options"]}
    desc = options_by_key.get(point["chosen_key"], "")
    key_phrase = point["chosen_key"].replace("_", " ").casefold()
    if key_phrase and key_phrase in tail:
        return True
    words = [w for w in re.findall(r"[a-zA-Z]+", desc) if len(w) >= 4]
    return any(w.casefold() in tail for w in words)


def _valid_options(options):
    if not isinstance(options, list) or not options:
        return False
    for o in options:
        if not isinstance(o, dict):
            return False
        if not isinstance(o.get("key"), str) or not o["key"]:
            return False
        if not isinstance(o.get("desc"), str) or not o["desc"]:
            return False
    return True


def validate_points(trace, candidates, verify_fn=verify_chosen_key_string_evidence):
    """-> (valid_points: list[dict], drop_reasons: list[str]). See module docstring for the exact
    checks. `valid_points` is capped at MAX_POINTS_PER_TRACE, in the candidates' own order; each kept
    point has "quote_start_offset"/"quote_start_end"/"quote_choice_offset"/"quote_choice_end" (char
    offsets into `trace`) added, and its internal "_quote_choice_offset" scratch key removed before
    returning (verify_fn's own bookkeeping, not part of the public schema)."""
    valid = []
    drop_reasons = []
    for idx, cand in enumerate(candidates if isinstance(candidates, list) else []):
        if not isinstance(cand, dict):
            drop_reasons.append(f"point {idx}: not an object")
            continue
        quote_start = cand.get("quote_start")
        quote_choice = cand.get("quote_choice")
        if not isinstance(quote_start, str) or not quote_start:
            drop_reasons.append(f"point {idx}: empty/missing quote_start")
            continue
        if not isinstance(quote_choice, str) or not quote_choice:
            drop_reasons.append(f"point {idx}: empty/missing quote_choice")
            continue
        start_offset = trace.find(quote_start)
        if start_offset < 0:
            drop_reasons.append(f"point {idx}: quote_start not found verbatim in trace")
            continue
        choice_offset = trace.find(quote_choice, start_offset)
        if choice_offset < 0:
            drop_reasons.append(f"point {idx}: quote_choice not found verbatim at/after quote_start")
            continue
        if not _valid_options(cand.get("options")):
            drop_reasons.append(f"point {idx}: options missing/empty/malformed")
            continue
        chosen_key = cand.get("chosen_key")
        option_keys = {o["key"] for o in cand["options"]}
        if chosen_key not in option_keys:
            drop_reasons.append(f"point {idx}: chosen_key {chosen_key!r} not among option keys")
            continue
        cand_with_offset = dict(cand, _quote_choice_offset=choice_offset)
        if not verify_fn(trace, cand_with_offset):
            drop_reasons.append(f"point {idx}: chosen_key {chosen_key!r} has no evidence in trace "
                                 "after quote_choice (failed commit check)")
            continue
        if len(valid) >= MAX_POINTS_PER_TRACE:
            drop_reasons.append(f"point {idx}: dropped, already have {MAX_POINTS_PER_TRACE} valid points")
            continue
        kept = dict(cand)
        kept["quote_start_offset"] = start_offset
        kept["quote_start_end"] = start_offset + len(quote_start)
        kept["quote_choice_offset"] = choice_offset
        kept["quote_choice_end"] = choice_offset + len(quote_choice)
        valid.append(kept)
    return valid, drop_reasons


# ---------------------------------------------------------------------------
# optional stronger commit check: one extra forced-choice forward pass per point (GPU)
# ---------------------------------------------------------------------------

def make_forced_choice_verifier(tok, model, device):
    """-> a verify_fn(trace, point) -> bool with the same signature as
    verify_chosen_key_string_evidence, but backed by ONE extra forward pass per point: builds a
    forced-choice prompt (point's quote_choice text as "the assistant's commitment", its own lettered
    options + a trailing 'none of these'), argmaxes over the candidate letters' first-token ids
    (extract_labels_llm.resolve_letter_token_ids, imported, not reimplemented), and returns True iff the
    argmax letter maps to point['chosen_key']. Lazy torch import -- only called from main() when
    `--verify forced-choice` is passed, never from --selftest/--dry-run."""
    import torch
    from scripts.extract_labels_llm import NONE_KEY, letters_for, resolve_letter_token_ids

    def verify(trace, point):
        options = [(o["key"], o["desc"]) for o in point["options"]]
        letters = letters_for(len(options))
        letter_to_key = {letters[i]: options[i][0] for i in range(len(options))}
        letter_to_key[letters[-1]] = NONE_KEY
        option_lines = "\n".join(f"{letters[i]}. {k}: {d}" for i, (k, d) in enumerate(options))
        option_lines += f"\n{letters[-1]}. None of these."
        user_content = (
            f"ASSISTANT'S COMMITMENT (excerpt of its own reasoning):\n{point['quote_choice']}\n\n"
            f"OPTIONS:\n{option_lines}\n\n"
            "Which option does this commitment pick? Respond with exactly one letter."
        )
        messages = [
            {"role": "system", "content": "You are a careful grader. Respond with exactly one letter."},
            {"role": "user", "content": user_content},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)
        letter_ids, _n_fallback = resolve_letter_token_ids(tok, prompt, letters)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[0, -1, :]
        best_letter = max(letters, key=lambda l: logits[letter_ids[l]].item())
        return letter_to_key[best_letter] == point["chosen_key"]

    return verify


# ---------------------------------------------------------------------------
# dry run (CPU, tokenizer only)
# ---------------------------------------------------------------------------

def run_dry(rows, args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    print(f"dry-run: {len(rows)} traces, verify={args.verify}")
    for row in rows[:2]:
        user_content = build_extraction_user_content(row)
        messages = [{"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)
        n_tok = len(tok(prompt, add_special_tokens=False)["input_ids"])
        print(f"  id={row['id']!r} prompt_tokens={n_tok} thinking_chars={len(row.get('thinking', ''))}")
    think_marker_absent = "<|think|>" not in tok.apply_chat_template(
        [{"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
         {"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    print(f"'<|think|>' absent with enable_thinking=False: {think_marker_absent} "
          f"({'OK' if think_marker_absent else 'MISMATCH'})")
    return 0 if think_marker_absent else 1


# ---------------------------------------------------------------------------
# GPU-dependent generation (lazy imports -- --dry-run/--selftest never need torch)
# ---------------------------------------------------------------------------

def generate_points_batch(tok, model, rows, args, device):
    """-> list of (raw_model_output: str) for one batch, via scripts/gen_think_traces.py's own
    batched_generate (reused, not reimplemented) with enable_thinking=False prompts built from
    EXTRACTION_SYSTEM_PROMPT + build_extraction_user_content. "Forced short generation" per the spec
    this script targets: args.max_new_tokens defaults small (DEFAULT_MAX_NEW_TOKENS) and args.temperature
    defaults to 0 (greedy) so repeated runs on the same trace are reproducible."""
    from scripts.gen_think_traces import _strip_pad_and_turn, batched_generate
    prompts = []
    for row in rows:
        messages = [{"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": build_extraction_user_content(row)}]
        prompts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False))
    results = batched_generate(tok, model, prompts, args, device)
    return [_strip_pad_and_turn(raw_text) for raw_text, _finished, _n_new in results]


def load_model_and_tokenizer(model_name, revision, device, attn_impl="sdpa"):
    from scripts.gen_think_traces import load_model_and_tokenizer as _load
    return _load(model_name, revision, device, attn_impl, adapter_dir=None)


# ---------------------------------------------------------------------------
# shared I/O helpers (same pattern as gen_think_traces.py)
# ---------------------------------------------------------------------------

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


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


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", default="oracle/s1-problems-organic.jsonl")
    ap.add_argument("--output", default="oracle/s1-decision-points.jsonl")
    ap.add_argument("--verify", choices=["string-evidence", "forced-choice"], default="string-evidence")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="default 0 (greedy) -- this is a deterministic analysis pass, not organic "
                         "sampling, unlike gen_think_traces.py")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-impl", default="sdpa")
    args = ap.parse_args()

    if args.selftest:
        return sys.exit(selftest())

    in_path, out_path = resolve(args.input), resolve(args.output)
    rows = load_jsonl(in_path)
    print(f"loaded {len(rows)} traces from {args.input}", flush=True)

    if args.dry_run:
        if args.limit:
            rows = rows[:args.limit]
        return sys.exit(run_dry(rows, args))

    done = load_done_ids(out_path) if args.resume else set()
    if done:
        before = len(rows)
        rows = [r for r in rows if r["id"] not in done]
        print(f"--resume: {before - len(rows)}/{before} traces already in {args.output}, "
              f"{len(rows)} remaining", flush=True)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print("nothing to do", flush=True)
        return

    import torch
    torch.manual_seed(args.seed)
    tok, model = load_model_and_tokenizer(args.model, args.revision, args.device, args.attn_impl)
    verify_fn = (make_forced_choice_verifier(tok, model, args.device)
                 if args.verify == "forced-choice" else verify_chosen_key_string_evidence)

    prompt_lens = [len(tok(build_extraction_user_content(r), add_special_tokens=False)["input_ids"])
                   for r in rows]
    order = sorted(range(len(rows)), key=lambda i: prompt_lens[i])
    rows_sorted = [rows[i] for i in order]

    mode = "a" if args.resume else "w"
    t0 = time.time()
    n_with_point = n_total_points = n_total_dropped = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for i in range(0, len(rows_sorted), args.batch_size):
            batch = rows_sorted[i:i + args.batch_size]
            raw_outputs = generate_points_batch(tok, model, batch, args, args.device)
            out_rows = []
            for row, raw in zip(batch, raw_outputs):
                candidates, parse_err = parse_model_json_points(raw)
                drop_reasons = [parse_err] if parse_err else []
                valid_points = []
                if not parse_err:
                    valid_points, reasons = validate_points(row["thinking"], candidates, verify_fn)
                    drop_reasons.extend(reasons)
                n_with_point += int(bool(valid_points))
                n_total_points += len(valid_points)
                n_total_dropped += len(drop_reasons)
                out_rows.append(dict(
                    id=row["id"], points=valid_points, n_candidate_points=len(candidates),
                    n_dropped_points=len(drop_reasons), drop_reasons=drop_reasons, raw_model_output=raw,
                ))
            write_jsonl_append(f, out_rows)
            elapsed = time.time() - t0
            print(f"wrote {len(out_rows)} rows ({i + len(out_rows)}/{len(rows_sorted)} total); "
                  f"{n_with_point} traces with >=1 valid point, {n_total_points} total valid points, "
                  f"{n_total_dropped} dropped so far; {elapsed:.1f}s", flush=True)
    print(f"done: {len(rows_sorted)} traces, {n_with_point} with >=1 valid point, "
          f"{n_total_points} total valid points, {n_total_dropped} dropped, "
          f"{time.time() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# selftest: pure-Python checks on fabricated, hand-labelled traces -- no torch/GPU
# ---------------------------------------------------------------------------

def selftest():
    _selftest_parse_model_json_points()
    _selftest_validate_points_happy_path()
    _selftest_validate_points_drops()
    _selftest_validate_points_cap_at_three()
    _selftest_string_evidence_verifier()
    print("selftest: all checks passed")
    return 0


def _selftest_parse_model_json_points():
    clean = '[{"quote_start": "a", "quote_choice": "b"}]'
    pts, err = parse_model_json_points(clean)
    assert err is None and pts == [{"quote_start": "a", "quote_choice": "b"}], (pts, err)

    fenced = "Sure, here you go:\n```json\n[{\"quote_start\": \"a\"}]\n```\nHope that helps!"
    pts, err = parse_model_json_points(fenced)
    assert err is None and pts == [{"quote_start": "a"}], (pts, err)

    empty = "  []  "
    pts, err = parse_model_json_points(empty)
    assert err is None and pts == [], (pts, err)

    garbage = "I cannot find any such points in this reasoning."
    pts, err = parse_model_json_points(garbage)
    assert pts == [] and err is not None, (pts, err)

    not_a_list = '{"foo": "bar"}'
    pts, err = parse_model_json_points(not_a_list)
    assert pts == [] and err is not None, (pts, err)
    print("selftest: parse_model_json_points OK (5 cases)")


FABRICATED_TRACE = (
    "The user wants to know if it's better to take the highway or the back road. "
    "Let me think about the highway option: it's faster but has more traffic. "
    "The back road is slower but more scenic and predictable. "
    "On balance, I will recommend the back road because predictability matters more here for this "
    "user's stated goal of arriving on time. So the back road is the better choice."
)


def _selftest_validate_points_happy_path():
    candidates = [{
        "quote_start": "Let me think about the highway option",
        "quote_choice": "I will recommend the back road",
        "question": "highway or back road?",
        "options": [{"key": "highway", "desc": "faster but more traffic"},
                    {"key": "back_road", "desc": "slower but more scenic and predictable"}],
        "chosen_key": "back_road",
    }]
    valid, drops = validate_points(FABRICATED_TRACE, candidates)
    assert drops == [], drops
    assert len(valid) == 1, valid
    p = valid[0]
    assert FABRICATED_TRACE[p["quote_start_offset"]:p["quote_start_end"]] == candidates[0]["quote_start"]
    assert FABRICATED_TRACE[p["quote_choice_offset"]:p["quote_choice_end"]] == candidates[0]["quote_choice"]
    assert "_quote_choice_offset" not in p, "scratch key must not leak into the kept point"
    print("selftest: validate_points happy path OK (1 valid point, correct offsets)")


def _selftest_validate_points_drops():
    base_options = [{"key": "highway", "desc": "faster but more traffic"},
                     {"key": "back_road", "desc": "slower but more scenic and predictable"}]

    not_verbatim = [{"quote_start": "this text is not in the trace at all",
                      "quote_choice": "I will recommend the back road",
                      "options": base_options, "chosen_key": "back_road"}]
    valid, drops = validate_points(FABRICATED_TRACE, not_verbatim)
    assert valid == [] and "not found verbatim" in drops[0], drops

    bad_key = [{"quote_start": "Let me think about the highway option",
                "quote_choice": "I will recommend the back road",
                "options": base_options, "chosen_key": "scenic_route"}]
    valid, drops = validate_points(FABRICATED_TRACE, bad_key)
    assert valid == [] and "not among option keys" in drops[0], drops

    choice_before_start = [{"quote_start": "I will recommend the back road",
                             "quote_choice": "Let me think about the highway option",
                             "options": base_options, "chosen_key": "highway"}]
    valid, drops = validate_points(FABRICATED_TRACE, choice_before_start)
    assert valid == [] and "quote_choice not found verbatim at/after quote_start" in drops[0], drops

    no_evidence = [{"quote_start": "Let me think about the highway option",
                     "quote_choice": "On balance, I will recommend the back road",
                     "options": [{"key": "highway", "desc": "faster but more traffic"},
                                 {"key": "zzz_unrelated_option", "desc": "qqqqq wwwww xxxxx"}],
                     "chosen_key": "zzz_unrelated_option"}]
    valid, drops = validate_points(FABRICATED_TRACE, no_evidence)
    assert valid == [] and "failed commit check" in drops[0], drops
    print("selftest: validate_points drops OK (not-verbatim, bad chosen_key, ordering, no-evidence)")


def _selftest_validate_points_cap_at_three():
    options = [{"key": "a", "desc": "the back road option here"},
               {"key": "b", "desc": "the highway option here"}]
    # 4 structurally-valid candidates (same quotes repeated is fine for this cap test -- validate_points
    # does not dedupe identical quotes, it only checks each candidate independently).
    one = {"quote_start": "Let me think about the highway option",
           "quote_choice": "I will recommend the back road", "options": options, "chosen_key": "a"}
    candidates = [one, one, one, one]
    valid, drops = validate_points(FABRICATED_TRACE, candidates)
    assert len(valid) == 3, valid
    assert len(drops) == 1 and "already have 3 valid points" in drops[0], drops
    print("selftest: validate_points caps at 3, extra point recorded in drop_reasons OK")


def _selftest_string_evidence_verifier():
    point_hit = dict(options=[{"key": "back_road", "desc": "slower but more scenic and predictable"}],
                      chosen_key="back_road", _quote_choice_offset=FABRICATED_TRACE.find("I will"))
    assert verify_chosen_key_string_evidence(FABRICATED_TRACE, point_hit) is True

    point_miss = dict(options=[{"key": "zzz", "desc": "qqqqq wwwww xxxxx"}],
                       chosen_key="zzz", _quote_choice_offset=FABRICATED_TRACE.find("I will"))
    assert verify_chosen_key_string_evidence(FABRICATED_TRACE, point_miss) is False
    print("selftest: verify_chosen_key_string_evidence OK (hit + miss)")


if __name__ == "__main__":
    main()
