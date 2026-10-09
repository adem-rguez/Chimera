"""Phase 7R inference RUNTIME ("arm C"): a 3-stage OFFLINE pipeline that stands in for a single live
serving loop, split into stages because the trigger-LoRA (base google/gemma-4-E4B-it @
ee0ef6023621cff504d758262d4e04895a5af4a2 + a PEFT adapter, e.g. runs/p7r-run4-mix) and the pointer-head
decision model (kev.serve --run runs/p4-e4b-final, built on the DIFFERENT base google/gemma-4-E4B @
411aa17b749aa952df1359d2dcea73917a544d9a) cannot both fit in VRAM on a 23GB L4 at once (reports/16's
runtime design, confidence floor T=0.6112). Each stage loads ONLY its own model, writes a JSONL, and
frees its model before the next stage runs -- a real server would interleave these per-request; this
pipeline fakes that by doing stage1 for every row, then stage2 for every row's trigger, then stage3.

Frozen formats this script reconstructs byte-for-byte (does not re-derive, imports/reuses):
  - typed trigger `<|decide:TYPE|>`, SHIPPED_TYPES, TRIGGER_FOR, build_reasoning -- scripts/build_phase7r.py
  - RESULT_OPEN/RESULT_CLOSE (`<|result|>`/`<|/result|>`), NONE_KEY, load_canonical (per-type canonical
    instruction string, read off evals/v7/decision-v7/development.jsonl) -- scripts/check_phase7r_pilot.py
  - THINK_OPEN/THINK_CLOSE channel markers, EOS_IDS, render_prompt, load_model_and_tokenizer,
    batched_generate, _strip_pad_and_turn -- scripts/gen_think_traces.py
  - the prefill-rebuild shape (`prompt_text + THINK_OPEN + kept_thinking + TRIGGER + instr + " ["
    + options + "]" + RESULT_OPEN + label + RESULT_CLOSE`) -- scripts/build_phase7r_hybrid.py's
    build_prefix_row/build_reasoning, verified byte-identical against oracle/phase7r-hybrid-prefixes-tail.jsonl
    by `--selftest` below (no torch needed: string-only check).
  - the decision-head readout (state -> (label, confidence, probs)) -- scripts/probe_phase7r_head.py's
    make_real_scorer/make_fake_scorer, used AS IS (same S1 state rule: user turn + thinking decoded since
    the previous call -- scripts/gate_phase7r_stageA.py's build_state, same rule, reused here too) and the
    same CONF_FLOOR=0.6112 (oracle/gate-threshold.md Phase 5B T).
  - bare-key extraction from a described "key: description" option string -- scripts/gate_phase7r_stageA.py
    option_key.
  - label-from-prose extraction (regex/substring heuristic) -- scripts/eval_phase7r_armB.py's
    extract_label/surface_set, reused for stage3's "final answer agrees with the injected label" check.

Three model-loading subcommands + one pure-Python one:

  stage1 --input EVAL --adapter DIR --out S1.jsonl
    Batched generation with the trigger-LoRA loaded on the base model (gen_think_traces.load_model_and_
    tokenizer(..., adapter_dir=DIR)), enable_thinking=True always (render_prompt with no `system`,
    matching EVAL's schema). A custom StoppingCriteria (TriggerStopper, below) decodes each row's own new
    tokens every generation step and marks that row done the instant its tail contains a COMPLETE,
    VALID `<|decide:TYPE|>` (TYPE in SHIPPED_TYPES). Rows whose tail never shows a valid trigger (no
    trigger at all, or a trigger with an invalid TYPE string) are NOT stopped early -- they are "treated
    as plain text" by simply not special-casing them: generation runs to EOS/max-new-tokens exactly like
    gen_think_traces.py's organic path, and since an invalid-type match does not qualify as this script's
    trigger, those rows are already fully answered by the time stage1 finishes (stage3 skips them; see
    stage3's docstring).
    Output per row (passthrough of every EVAL field, plus): prompt_text, thinking_so_far (decoded text
    BEFORE the trigger match, or the whole decoded text if no valid trigger), trigger_type (the matched
    TYPE string, valid or not, or null), valid (bool), finished_without_trigger (== not valid),
    finished (this row's own sequence hit one of gen_think_traces.EOS_IDS, same definition as there),
    n_thinking_so_far_tokens (re-tokenization of thinking_so_far), n_decoded_tokens (non-pad new-token
    count for the whole stage1 call), raw_tail (verbatim decoded continuation, skip_special_tokens=False,
    for debugging/resumption).

  stage2 --in S1.jsonl --out S2.jsonl
    Loads ONLY the decision head (runs/p4-e4b-final, via probe_phase7r_head.make_real_scorer -- same
    loader as scripts/eval_phase7.py's load_decision_model). For every row with valid==true: state =
    f"{user}\n{thinking_so_far}" (S1 strategy, gate_phase7r_stageA.build_state's exact rule), instr =
    load_canonical(check_phase7r_pilot.SUITE)[type][0] (the frozen canonical instruction string for that
    type -- EVAL rows do not carry their own instruction text), options = [kev.api.option_text(k, d) for
    k, d in options_with_descriptions] (described options, frozen format). Confidence floor 0.6112
    (probe_phase7r_head.CONF_FLOOR) decides abstain. Rows with valid==false pass through unchanged (no
    head call -- there is no call to score).
    Output: passthrough of every S1 field, plus (for valid rows): chosen_label (bare key, via
    gate_phase7r_stageA.option_key), confidence, max_prob, probs (dict, described-option -> prob),
    abstain (bool, confidence < CONF_FLOOR). Non-valid rows get these four fields set to null.

  stage3 --in S2.jsonl --adapter DIR --out S3.jsonl
    Loads ONLY the trigger-LoRA again (same adapter as stage1 -- a real server would never have unloaded
    it; this script re-loads it because stage2 needed the VRAM in between). For every row with
    valid==true: rebuilds the prefill prompt_text + THINK_OPEN + thinking_so_far + TRIGGER_FOR[type] +
    (if not abstain) `{instr} [{options}]` + RESULT_OPEN + chosen_label + RESULT_CLOSE -- exactly
    build_phase7r.build_reasoning's single-call shape, byte-for-byte (see --selftest). Abstained rows
    get the trigger with NO injected span -- "continue without injection": the prefill is just
    prompt_text + THINK_OPEN + thinking_so_far + TRIGGER_FOR[type], i.e. resume right where the model's
    own generation left off, nothing stripped. Prefill-continues with gen_think_traces.batched_generate
    (reused, not reimplemented) up to --max-new-tokens (default 600) or EOS.
    Multi-call limitation (documented, not implemented): the continuation is scanned once more for a
    second valid trigger (find_trigger); if one is found this is recorded (re_triggered=true,
    second_trigger_type) but NOT acted on -- this script does not loop back to stage2/stage3 for a
    second call (that would mean unloading the adapter and reloading the head mid-stage3, repeatedly,
    for a single row; out of scope here). `score` reports a re-trigger rate so this is visible, not
    silent.
    Rows with valid==false are NOT touched here: they were already run to completion in stage1 (nothing
    stopped them early), so they are copied straight through with stage3_skipped=true and the final
    answer read directly off stage1's own raw_tail (via split_continuation, which also handles the
    ordinary case of finding THINK_CLOSE in a mid-channel continuation).
    Output: passthrough of every S2 field, plus (for valid rows): continuation (raw decoded text, the
    newly generated tokens only), finished, n_cont_tokens, n_injected_tokens (0 if abstain),
    thinking_tail / final_answer (continuation split on THINK_CLOSE, via split_continuation),
    re_triggered, second_trigger_type, stage3_skipped (always false here). Non-valid rows get
    continuation=null, n_cont_tokens=0, n_injected_tokens=0, thinking_tail="", final_answer (read off
    raw_tail via split_continuation... see caveat in that row's own comment below), re_triggered=false,
    second_trigger_type=null, stage3_skipped=true.

  score --in S3.jsonl --baseline-armB oracle/t3-armB-think.fixed.jsonl [--extract FILE]
    Pure Python, no model, no GPU -- just joins/aggregates existing JSONL files (same spirit as
    scripts/eval_phase7r_armB.py). Per decision type: head-label-vs-gold accuracy on triggered rows;
    final-answer agreement with the injected label (non-abstain triggered rows only -- abstain rows have
    no injected label to agree with); decoded tokens (thinking_so_far + continuation, WITH and WITHOUT
    the injected span, injected counted separately as it is never produced by the model) vs arm B's
    (oracle/t3-armB-think.fixed.jsonl) n_thinking_tokens+n_answer_tokens for the same id, paired mean
    diff + bootstrap 95% CI; trigger rate; re-trigger rate; abstain rate; false triggers on
    control/adversarial rows (valid-type-only, and any-type [valid or invalid]).
    --extract FILE: optional, same schema as scripts/eval_phase7r_armB.py's --llm-extract (one row per
    id: chosen/committed) -- if given, used for the final-answer-agreement check instead of the
    regex/substring heuristic (extract_label/surface_set, imported from eval_phase7r_armB, unchanged).

Resumability / memory safety: every stage supports --resume (skip ids already in --out, append) exactly
like gen_think_traces.py/continue_phase7r_hybrid.py's load_done_ids/write_jsonl_append. Every stage loads
only its own model and does not import the others' heavy deps at module scope (torch/peft/kev.model are
all lazy, inside functions) -- `score` and every stage's --dry-run need no model at all.

Usage (GPU box):
  .venv/bin/python -u scripts/run_phase7r_runtime.py stage1 \\
      --input oracle/phase7r-natural-eval.jsonl --adapter runs/p7r-run4-mix --out oracle/p7rc-s1.jsonl \\
      --batch-size 8 --max-new-tokens 400 --resume
  .venv/bin/python -u scripts/run_phase7r_runtime.py stage2 \\
      --in oracle/p7rc-s1.jsonl --out oracle/p7rc-s2.jsonl --resume
  .venv/bin/python -u scripts/run_phase7r_runtime.py stage3 \\
      --in oracle/p7rc-s2.jsonl --adapter runs/p7r-run4-mix --out oracle/p7rc-s3.jsonl \\
      --batch-size 8 --max-new-tokens 600 --resume
  .venv/bin/python -u scripts/run_phase7r_runtime.py score \\
      --in oracle/p7rc-s3.jsonl --baseline-armB oracle/t3-armB-think.fixed.jsonl \\
      --out reports/22-phase7r-armC-runtime.md

Offline (laptop, CPU, no torch/GPU/model weights):
  .venv/Scripts/python.exe -m py_compile scripts/run_phase7r_runtime.py
  .venv/Scripts/python.exe scripts/run_phase7r_runtime.py --selftest
  .venv/Scripts/python.exe scripts/run_phase7r_runtime.py stage1 --input <eval.jsonl> --adapter X \\
      --out <scratch> --dry-run
  .venv/Scripts/python.exe scripts/run_phase7r_runtime.py score --in <fabricated-S3.jsonl> \\
      --baseline-armB oracle/t3-armB-think.fixed.jsonl --out <scratch.md>
"""
import argparse
import collections
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kev.api import option_text  # noqa: E402 -- no torch at kev.api module scope (verified)

from scripts.gen_think_traces import (  # noqa: E402
    EOS_IDS, MODEL, REVISION, THINK_CLOSE, THINK_OPEN, _strip_pad_and_turn, batched_generate,
    load_model_and_tokenizer, render_prompt,
)
from scripts.build_phase7r import SHIPPED_TYPES, TRIGGER_FOR, TRIGGER_RE, build_reasoning  # noqa: E402
from scripts.check_phase7r_pilot import (  # noqa: E402
    NONE_KEY, RESULT_CLOSE, RESULT_OPEN, SUITE, load_canonical,
)
from scripts.probe_phase7r_head import CONF_FLOOR, make_fake_scorer, make_real_scorer  # noqa: E402
from scripts.gate_phase7r_stageA import option_key  # noqa: E402
from scripts.eval_phase7r_armB import extract_label, surface_set  # noqa: E402 -- unused import kept

DEFAULT_MAX_NEW_TOKENS_S1 = 400
DEFAULT_MAX_NEW_TOKENS_S3 = 600


# ---------------------------------------------------------------------------
# pure string/regex logic -- CPU-testable, no torch
# ---------------------------------------------------------------------------

def find_trigger(text):
    """-> {"type", "valid", "start", "end"} for the FIRST complete '<|decide:TYPE|>' in `text`, or None.
    `start`/`end` are char offsets of the trigger match itself (end == char right after the closing
    '|>'). `valid` iff TYPE is one of SHIPPED_TYPES (build_phase7r.py's typed-trigger regex already
    restricts TYPE to [a-z_]+, so an unshipped/garbled type string still matches the regex but is
    flagged invalid here -- that is the "treated as plain text" case stage1 does not stop on)."""
    m = TRIGGER_RE.search(text)
    if m is None:
        return None
    return {"type": m.group(1), "valid": m.group(1) in SHIPPED_TYPES, "start": m.start(), "end": m.end()}


def split_continuation(raw_text):
    """-> (thinking_tail, answer). For text that resumes MID-thinking-channel (no THINK_OPEN in it --
    that was already consumed by an earlier stage): split on THINK_CLOSE only. If THINK_CLOSE is not
    present (ran out of tokens still inside the thinking channel), thinking_tail is the raw text
    (.strip()-ed, no pad/turn stripping -- there is no trailing pad/turn mid-channel) and answer is "".
    If THINK_CLOSE is found, thinking_tail is the text before it (.strip()-ed) and answer is the text
    after it, through _strip_pad_and_turn (gen_think_traces.py's helper, reused -- the answer portion CAN
    end in pad/<turn|> bytes, same as organic generation)."""
    idx = raw_text.find(THINK_CLOSE)
    if idx < 0:
        return raw_text.strip(), ""
    return raw_text[:idx].strip(), _strip_pad_and_turn(raw_text[idx + len(THINK_CLOSE):])


def build_injected_span(call_type, instr, options, label):
    """-> the runtime-injected text for one call, TRIGGER included: '<|decide:TYPE|>{instr} [{options}]
    <|result|>{label}<|/result|>' -- exactly build_phase7r.build_reasoning's single-call composition
    (segments=[kept, ""]), byte for byte (see --selftest)."""
    return (f"{TRIGGER_FOR[call_type]}{instr} [{', '.join(options)}]"
            f"{RESULT_OPEN}{label}{RESULT_CLOSE}")


def rebuild_prefill(prompt_text, thinking_so_far, call_type, instr, options, label, abstain):
    """-> the full prefill text to resume generation from. `abstain=True` resumes right after the bare
    trigger, no injected span at all ("continue without injection: strip nothing; just resume after the
    trigger")."""
    head = prompt_text + THINK_OPEN + thinking_so_far
    if abstain:
        return head + TRIGGER_FOR[call_type]
    return head + build_injected_span(call_type, instr, options, label)


_TYPE_OPTIONS = None


def load_type_options():
    """-> {type: options_with_descriptions} for SHIPPED_TYPES, lazily built once from the first decision
    row of each type in oracle/phase7r-natural-eval.jsonl (keeps that row's option ORDER as canonical),
    falling back to oracle/phase7r-hybrid-prompts.jsonl for any type missing there. This is the single
    source of truth for a type's options AT RUNTIME -- the eval row's own `options_with_descriptions` is
    the GOLD row's options and must never be used once a row has been re-typed to its MODEL-EMITTED
    trigger (see normalize_s1_row): adversarial/near-miss eval rows can have a gold `type` the model never
    emitted and `options_with_descriptions=None` entirely."""
    global _TYPE_OPTIONS
    if _TYPE_OPTIONS is not None:
        return _TYPE_OPTIONS
    table = {}
    for path in ("oracle/phase7r-natural-eval.jsonl", "oracle/phase7r-hybrid-prompts.jsonl"):
        full = resolve(path)
        if not os.path.exists(full):
            continue
        for row in load_jsonl(full):
            t = row.get("type")
            opts = row.get("options_with_descriptions")
            if t not in SHIPPED_TYPES or not opts:
                continue
            if t in table:
                existing_set = {k for k, _d in table[t]}
                new_set = {k for k, _d in opts}
                assert existing_set == new_set, (
                    f"type {t!r}: option set mismatch between rows "
                    f"({existing_set} vs {new_set} from {path})")
                continue
            table[t] = opts
    missing = [t for t in SHIPPED_TYPES if t not in table]
    if missing:
        raise SystemExit(f"load_type_options is missing options for: {missing}")
    _TYPE_OPTIONS = table
    return table


def normalize_s1_row(row):
    """Idempotent: re-types a stage1 row from its GOLD fields to its MODEL-EMITTED trigger fields, so
    every downstream stage (2, 3, score) reads the type/options the model actually committed to, never
    the eval row's gold answer (that would both leak the gold type into the head/injection and crash on
    adversarial/near-miss rows whose gold `options_with_descriptions` is None and whose gold `type` has
    nothing to do with what the model emitted). No-op for non-valid rows (nothing to re-type) and for rows
    already normalized."""
    if not (row.get("valid") and not row.get("_normalized")):
        return row
    trigger_type = row["trigger_type"]
    row["gold_type"] = row.get("type")
    row["type"] = trigger_type
    row["gold_options_with_descriptions"] = row.get("options_with_descriptions")
    row["options_with_descriptions"] = load_type_options()[trigger_type]
    row["_normalized"] = True
    return row


def load_canonical_instructions():
    """-> {type: instruction} for SHIPPED_TYPES, via check_phase7r_pilot.load_canonical(SUITE) -- the
    frozen canonical instruction string per type (EVAL rows do not carry their own instruction text)."""
    canonical, errs = load_canonical(os.path.join(ROOT, SUITE))
    if errs:
        raise SystemExit(f"load_canonical errors: {errs}")
    missing = [t for t in SHIPPED_TYPES if t not in canonical]
    if missing:
        raise SystemExit(f"load_canonical is missing instruction(s) for: {missing}")
    return {t: canonical[t][0] for t in SHIPPED_TYPES}


# ---------------------------------------------------------------------------
# shared I/O helpers (same pattern as gen_think_traces.py / continue_phase7r_hybrid.py)
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


def apply_resume_limit(rows, out_path, resume, limit):
    done = load_done_ids(out_path) if resume else set()
    if done:
        before = len(rows)
        rows = [r for r in rows if r["id"] not in done]
        print(f"--resume: {before - len(rows)}/{before} rows already in {out_path}, "
              f"{len(rows)} remaining", flush=True)
    if limit:
        rows = rows[:limit]
    return rows


def free_model(model):
    """Best-effort VRAM release between stages -- lazy torch import so this file stays importable
    without torch installed (CPU laptop)."""
    del model
    try:
        import torch
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# stage1: trigger-LoRA generation, stop at the first valid typed trigger
# ---------------------------------------------------------------------------

class TriggerStopper:
    """Duck-typed `transformers.StoppingCriteria`: callable(input_ids, scores, **kwargs) -> a
    torch.BoolTensor of shape (batch,), one bool per row, True once that row is done. transformers'
    generate() loop (>=4.41-ish, pinned here at >=5.17 per gen_think_traces.py) treats any True entry as
    "this sequence is finished" for the purposes of `unfinished_sequences`, independently per row within
    a batch -- rows that return True early stop advancing (padded from then on) while the rest of the
    batch keeps generating. A row is marked done the first time ITS OWN new-token tail decodes to contain
    a complete, VALID '<|decide:TYPE|>' (find_trigger, above); a complete INVALID-type trigger is left
    running (never marked done by this criterion alone -- it still stops normally at EOS/max-new-tokens,
    same as a row with no trigger at all). Re-decodes the whole per-row tail every step (not just the
    newest token) since the trigger string can span multiple tokens and a token-boundary check alone
    could miss a trigger that completes understanding only once decoded; this is the same text-level
    reasoning gen_think_traces.py's split_thinking_answer uses for the (also multi-piece-token) channel
    markers. Costlier than an id-level check but correctness-first, matching this whole script's stance
    (GPU-only code here could not be run/tested on this box; see module docstring)."""

    def __init__(self, tok, prompt_len, batch_size):
        self.tok = tok
        self.prompt_len = prompt_len
        self._done = [False] * batch_size

    def __call__(self, input_ids, scores, **kwargs):
        import torch
        for i in range(input_ids.shape[0]):
            if self._done[i]:
                continue
            new_ids = input_ids[i, self.prompt_len:].tolist()
            text = self.tok.decode(new_ids, skip_special_tokens=False)
            hit = find_trigger(text)
            if hit is not None and hit["valid"]:
                self._done[i] = True
        return torch.tensor(self._done, dtype=torch.bool, device=input_ids.device)


def generate_stage1_batch(tok, model, prompt_texts, args, device):
    """-> list of (raw_text, hit_or_None, eos_hit, n_new_tokens) for one batch. `raw_text` is the row's
    full newly-generated continuation (skip_special_tokens=False) at whatever point it stopped -- either
    the trigger-stop (valid trigger) or the ordinary EOS/max-new-tokens stop (no trigger, or an invalid
    one). Mirrors gen_think_traces.batched_generate's shape/left-padding/eos bookkeeping, with a
    StoppingCriteriaList added."""
    import torch
    from transformers import StoppingCriteriaList
    eos_set = set(EOS_IDS)
    greedy = args.temperature <= 0
    enc = tok(prompt_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    prompt_len = enc["input_ids"].shape[1]
    stopper = TriggerStopper(tok, prompt_len, len(prompt_texts))
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id,
                      eos_token_id=EOS_IDS, do_sample=not greedy,
                      stopping_criteria=StoppingCriteriaList([stopper]))
    if not greedy:
        gen_kwargs.update(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    with torch.no_grad():
        ids = model.generate(**enc, **gen_kwargs)
    out = []
    for j in range(len(prompt_texts)):
        new_ids = ids[j, prompt_len:].tolist()
        raw_text = tok.decode(new_ids, skip_special_tokens=False)
        hit = find_trigger(raw_text)
        eos_hit = any(t in eos_set for t in new_ids)
        nonpad = sum(1 for t in new_ids if t != tok.pad_token_id)
        out.append((raw_text, hit, eos_hit, nonpad))
    return out


def stage1_row_from_result(tok, row, raw_text, hit, eos_hit, n_new):
    valid = hit is not None and hit["valid"]
    if hit is not None:
        thinking_so_far = raw_text[:hit["start"]].strip()
        trigger_type = hit["type"]
    else:
        thinking_so_far = _strip_pad_and_turn(raw_text)
        trigger_type = None
    n_think = len(tok(thinking_so_far, add_special_tokens=False)["input_ids"]) if thinking_so_far else 0
    out = dict(row)
    out.update(
        prompt_text=render_prompt(tok, row, True), thinking_so_far=thinking_so_far,
        trigger_type=trigger_type, valid=valid, finished_without_trigger=not valid,
        finished=eos_hit, n_thinking_so_far_tokens=n_think, n_decoded_tokens=n_new, raw_tail=raw_text,
    )
    return normalize_s1_row(out)


def run_dry_stage1(rows, args):
    print(f"stage1 --dry-run: {len(rows)} rows, adapter={args.adapter!r}, "
          f"max_new_tokens={args.max_new_tokens}")
    for row in rows[:2]:
        print(f"  id={row['id']!r} kind={row.get('kind')!r} type={row.get('type')!r}")
        print(f"    user[:150]={row['user'][:150]!r}")
        print("    plan: render_prompt(enable_thinking=True) -> generate with the trigger-LoRA, "
              "stopping the instant a valid <|decide:TYPE|> trigger decodes; else run to EOS/"
              f"{args.max_new_tokens} tokens.")
    return 0


def cmd_stage1(args):
    in_path, out_path = resolve(args.input), resolve(args.out)
    rows = load_jsonl(in_path)
    print(f"loaded {len(rows)} eval rows from {args.input}", flush=True)

    if args.dry_run:
        if args.limit:
            rows = rows[:args.limit]
        return sys.exit(run_dry_stage1(rows, args))

    rows = apply_resume_limit(rows, out_path, args.resume, args.limit)
    if not rows:
        print("nothing to do", flush=True)
        return

    import torch
    torch.manual_seed(args.seed)
    tok, model = load_model_and_tokenizer(args.model, args.revision, args.device, args.attn_impl,
                                          adapter_dir=args.adapter)

    prompts = [render_prompt(tok, r, True) for r in rows]
    lens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    order = sorted(range(len(rows)), key=lambda i: lens[i])
    rows_sorted = [rows[i] for i in order]
    prompts_sorted = [prompts[i] for i in order]

    mode = "a" if args.resume else "w"
    t0 = time.time()
    n_valid = n_invalid = n_none = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for i in range(0, len(rows_sorted), args.batch_size):
            batch_rows = rows_sorted[i:i + args.batch_size]
            batch_prompts = prompts_sorted[i:i + args.batch_size]
            results = generate_stage1_batch(tok, model, batch_prompts, args, args.device)
            out_rows = []
            for row, (raw_text, hit, eos_hit, n_new) in zip(batch_rows, results):
                out = stage1_row_from_result(tok, row, raw_text, hit, eos_hit, n_new)
                out_rows.append(out)
                if hit is None:
                    n_none += 1
                elif hit["valid"]:
                    n_valid += 1
                else:
                    n_invalid += 1
            write_jsonl_append(f, out_rows)
            print(f"wrote {len(out_rows)} rows ({i + len(out_rows)}/{len(rows_sorted)} total); "
                  f"valid={n_valid} invalid_trigger={n_invalid} no_trigger={n_none}", flush=True)
    print(f"done: {len(rows_sorted)} rows in {time.time() - t0:.1f}s; "
          f"valid={n_valid} invalid_trigger={n_invalid} no_trigger={n_none}", flush=True)
    free_model(model)


# ---------------------------------------------------------------------------
# stage2: decision head readout on each valid trigger
# ---------------------------------------------------------------------------

def stage2_score_row(scorer, instructions, row):
    instr = instructions[row["type"]]
    options = [option_text(k, d) for k, d in row["options_with_descriptions"]]
    state = f"{row['user']}\n{row['thinking_so_far']}"
    pred_text, conf, max_prob, probs = scorer(state, instr, options)
    out = dict(row)
    out.update(chosen_label=option_key(pred_text), confidence=conf, max_prob=max_prob,
               probs=probs, abstain=conf < CONF_FLOOR)
    return out


def run_dry_stage2(rows, args):
    instructions = load_canonical_instructions()
    scorer = make_fake_scorer()
    valid_rows = [r for r in rows if r.get("valid")][:2]
    print(f"stage2 --dry-run: {len(rows)} rows total, {sum(1 for r in rows if r.get('valid'))} valid")
    for row in valid_rows:
        out = stage2_score_row(scorer, instructions, row)
        print(f"  id={row['id']!r} type={row['type']!r} instr={instructions[row['type']][:80]!r}")
        print(f"    state[:150]={(row['user'] + chr(10) + row['thinking_so_far'])[:150]!r}")
        print(f"    (fake) chosen_label={out['chosen_label']!r} confidence={out['confidence']:.3f} "
              f"abstain={out['abstain']}")
    return 0


def cmd_stage2(args):
    in_path, out_path = resolve(args.in_), resolve(args.out)
    rows = [normalize_s1_row(r) for r in load_jsonl(in_path)]
    print(f"loaded {len(rows)} rows from {args.in_}", flush=True)

    if args.dry_run:
        if args.limit:
            rows = rows[:args.limit]
        return sys.exit(run_dry_stage2(rows, args))

    rows = apply_resume_limit(rows, out_path, args.resume, args.limit)
    if not rows:
        print("nothing to do", flush=True)
        return

    instructions = load_canonical_instructions()
    scorer = make_real_scorer(args.decision_run, args.device)

    mode = "a" if args.resume else "w"
    n_scored = n_abstain = 0
    with open(out_path, mode, encoding="utf-8") as f:
        out_rows = []
        for row in rows:
            if row.get("valid"):
                out = stage2_score_row(scorer, instructions, row)
                n_scored += 1
                n_abstain += int(out["abstain"])
            else:
                out = dict(row)
                out.update(chosen_label=None, confidence=None, max_prob=None, probs=None, abstain=None)
            out_rows.append(out)
            if len(out_rows) >= 50:
                write_jsonl_append(f, out_rows)
                print(f"scored {n_scored} ({n_abstain} abstain) of {len(out_rows)} rows so far "
                      f"this flush", flush=True)
                out_rows = []
        if out_rows:
            write_jsonl_append(f, out_rows)
    print(f"done: {len(rows)} rows, {n_scored} scored, {n_abstain} abstain", flush=True)


# ---------------------------------------------------------------------------
# stage3: rebuild prefix with the head's label, prefill-continue with the trigger-LoRA
# ---------------------------------------------------------------------------

def run_dry_stage3(rows, args):
    valid_rows = [r for r in rows if r.get("valid")][:2]
    print(f"stage3 --dry-run: {len(rows)} rows total, {sum(1 for r in rows if r.get('valid'))} valid")
    instructions = load_canonical_instructions()
    for row in valid_rows:
        options = [option_text(k, d) for k, d in row["options_with_descriptions"]]
        label = row["chosen_label"] if not row["abstain"] else None
        prefill = rebuild_prefill(row["prompt_text"], row["thinking_so_far"], row["type"],
                                   instructions[row["type"]], options, label, row["abstain"])
        print(f"  id={row['id']!r} type={row['type']!r} abstain={row['abstain']} "
              f"chosen_label={row['chosen_label']!r}")
        print(f"    prefill tail[-200:]={prefill[-200:]!r}")
        print(f"    plan: prefill-continue with the trigger-LoRA up to {args.max_new_tokens} new tokens")
    skipped = [r for r in rows if not r.get("valid")][:1]
    for row in skipped:
        print(f"  id={row['id']!r}: valid=False -> stage3_skipped, final answer read off raw_tail")
    return 0


def cmd_stage3(args):
    in_path, out_path = resolve(args.in_), resolve(args.out)
    rows = [normalize_s1_row(r) for r in load_jsonl(in_path)]
    print(f"loaded {len(rows)} rows from {args.in_}", flush=True)

    if args.dry_run:
        if args.limit:
            rows = rows[:args.limit]
        return sys.exit(run_dry_stage3(rows, args))

    rows = apply_resume_limit(rows, out_path, args.resume, args.limit)
    if not rows:
        print("nothing to do", flush=True)
        return

    valid_rows = [r for r in rows if r.get("valid")]
    skip_rows = [r for r in rows if not r.get("valid")]
    print(f"{len(valid_rows)} valid rows go through stage3 generation; "
          f"{len(skip_rows)} non-valid rows pass through unchanged", flush=True)

    mode = "a" if args.resume else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        # non-valid rows first (cheap, no model needed) so a crash mid-generation still leaves them done
        skip_out = []
        for row in skip_rows:
            thinking_tail, answer = split_continuation(row["raw_tail"])
            out = dict(row)
            out.update(continuation=None, finished=row.get("finished"), n_cont_tokens=0,
                       n_injected_tokens=0, thinking_tail=thinking_tail, final_answer=answer,
                       re_triggered=False, second_trigger_type=None, stage3_skipped=True)
            skip_out.append(out)
        if skip_out:
            write_jsonl_append(f, skip_out)
            print(f"wrote {len(skip_out)} non-valid (stage3_skipped) rows", flush=True)

        if not valid_rows:
            return

        import torch
        torch.manual_seed(args.seed)
        tok, model = load_model_and_tokenizer(args.model, args.revision, args.device, args.attn_impl,
                                              adapter_dir=args.adapter)
        instructions = load_canonical_instructions()

        prefills = []
        for row in valid_rows:
            options = [option_text(k, d) for k, d in row["options_with_descriptions"]]
            label = row["chosen_label"] if not row["abstain"] else None
            prefills.append(rebuild_prefill(row["prompt_text"], row["thinking_so_far"], row["type"],
                                             instructions[row["type"]], options, label, row["abstain"]))
        lens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prefills]
        order = sorted(range(len(valid_rows)), key=lambda i: lens[i])
        rows_sorted = [valid_rows[i] for i in order]
        prefills_sorted = [prefills[i] for i in order]

        n_retrigger = 0
        t0 = time.time()
        for i in range(0, len(rows_sorted), args.batch_size):
            batch_rows = rows_sorted[i:i + args.batch_size]
            batch_prefills = prefills_sorted[i:i + args.batch_size]
            results = batched_generate(tok, model, batch_prefills, args, args.device)
            out_rows = []
            for row, (raw_text, finished, n_new) in zip(batch_rows, results):
                hit2 = find_trigger(raw_text)
                re_triggered = hit2 is not None and hit2["valid"]
                n_retrigger += int(re_triggered)
                thinking_tail, answer = split_continuation(raw_text)
                options = [option_text(k, d) for k, d in row["options_with_descriptions"]]
                n_inj = 0
                if not row["abstain"]:
                    injected = build_injected_span(row["type"], instructions[row["type"]], options,
                                                    row["chosen_label"])
                    n_inj = len(tok(injected, add_special_tokens=False)["input_ids"])
                out = dict(row)
                out.update(continuation=raw_text, finished=finished, n_cont_tokens=n_new,
                           n_injected_tokens=n_inj, thinking_tail=thinking_tail, final_answer=answer,
                           re_triggered=re_triggered,
                           second_trigger_type=hit2["type"] if hit2 is not None else None,
                           stage3_skipped=False)
                out_rows.append(out)
            write_jsonl_append(f, out_rows)
            elapsed = time.time() - t0
            print(f"wrote {len(out_rows)} rows ({i + len(out_rows)}/{len(rows_sorted)} total); "
                  f"re_triggered so far={n_retrigger}; {elapsed:.1f}s", flush=True)
    print(f"done: {len(valid_rows)} valid rows processed, {n_retrigger} re-triggered", flush=True)
    free_model(model)


# ---------------------------------------------------------------------------
# score: pure Python, joins S3 output against the arm-B baseline
# ---------------------------------------------------------------------------

def total_decoded_tokens(row):
    """-> (without_injected, with_injected) decoded-token totals for one S3 row. For stage3_skipped
    (non-valid-trigger) rows, n_thinking_so_far_tokens already covers the ENTIRE decoded text (stage1
    never stopped them early), so both totals equal that one count (n_cont_tokens/n_injected_tokens are
    0 for those rows by construction)."""
    base = row["n_thinking_so_far_tokens"] + row["n_cont_tokens"]
    return base, base + row["n_injected_tokens"]


def bootstrap_mean_ci(diffs, n_resamples=2000, seed=0):
    """-> (mean, lo95, hi95) via simple percentile bootstrap, no scipy."""
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(diffs)
    mean = sum(diffs) / n
    if n == 1:
        return mean, mean, mean
    means = []
    for _ in range(n_resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[min(n_resamples - 1, int(0.975 * n_resamples))]
    return mean, lo, hi


def extracted_label(row, llm_rows):
    """-> label extracted from the FINAL answer text, or None if uncommitted/unextracted. Uses --extract
    output if given, else the regex/substring heuristic (eval_phase7r_armB.extract_label/surface_set,
    reused verbatim)."""
    if llm_rows is not None:
        l = llm_rows.get(row["id"])
        if l is None or not l.get("committed"):
            return None
        return l["chosen"]
    option_keys = [k for k, _d in row["options_with_descriptions"]]
    return extract_label(row.get("final_answer") or "", option_keys)


def cmd_score(args):
    in_path = resolve(args.in_)
    armb_path = resolve(args.baseline_armb)
    rows = [normalize_s1_row(r) for r in load_jsonl(in_path)]
    armb_by_id = {r["id"]: r for r in load_jsonl(armb_path)}
    llm_rows = {r["id"]: r for r in load_jsonl(resolve(args.extract))} if args.extract else None

    decision_rows = [r for r in rows if r.get("gold_label") is not None]
    triggered = [r for r in decision_rows if r.get("valid")]
    non_abstain_triggered = [r for r in triggered if r.get("abstain") is False]

    lines = ["# Phase 7R arm-C runtime score", "",
              f"generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}, {len(rows)} S3 rows "
              f"from {args.in_}, {len(decision_rows)} decision rows, {len(triggered)} triggered "
              f"(valid typed trigger)", ""]

    # per-type head accuracy + final-answer agreement
    # `type` (post-normalize_s1_row) is the MODEL-EMITTED trigger type; `gold_type` is the eval row's own
    # gold type. A row only counts toward head_correct_overall if BOTH the trigger type matches gold_type
    # AND the chosen label matches gold_label -- a wrong-type trigger is always scored as an error there,
    # never silently compared against a mismatched gold label. head_correct_given_type_correct restricts
    # to the type-correct subset only (the head-accuracy-if-routed-right view).
    lines.append("## Per-type head accuracy and final-answer agreement (triggered rows)")
    lines.append("")
    lines.append("| type | n triggered | type correct rate | head correct (overall) | head correct "
                 "(given type correct) | n non-abstain | final answer agrees w/ injected label |")
    lines.append("|---|---|---|---|---|---|---|")
    for t in sorted(SHIPPED_TYPES):
        rs = [r for r in triggered if r["type"] == t]
        if not rs:
            continue
        type_correct_rows = [r for r in rs if r.get("gold_type") == r["type"]]
        type_correct_rate = len(type_correct_rows) / len(rs)
        n_overall_correct = sum(1 for r in rs
                                 if r.get("gold_type") == r["type"] and r["chosen_label"] == r["gold_label"])
        head_correct_overall = n_overall_correct / len(rs)
        head_correct_given_type_correct = (
            sum(1 for r in type_correct_rows if r["chosen_label"] == r["gold_label"]) / len(type_correct_rows)
        ) if type_correct_rows else float("nan")
        na = [r for r in rs if r.get("abstain") is False]
        agree = [extracted_label(r, llm_rows) == r["chosen_label"] for r in na]
        agree_rate = (sum(agree) / len(agree)) if agree else float("nan")
        lines.append(f"| {t} | {len(rs)} | {type_correct_rate:.3f} | {head_correct_overall:.3f} | "
                     f"{head_correct_given_type_correct:.3f} | {len(na)} | {agree_rate:.3f} |")
    lines.append("")

    # token accounting vs arm B, paired
    lines.append("## Decoded tokens vs arm B (think-only baseline), paired by id")
    lines.append("")
    paired = []
    for r in decision_rows:
        b = armb_by_id.get(r["id"])
        if b is None:
            continue
        without_inj, with_inj = total_decoded_tokens(r)
        armb_tok = b.get("n_thinking_tokens", 0) + b.get("n_answer_tokens", 0)
        paired.append((without_inj, with_inj, armb_tok))
    if paired:
        diffs_wo = [wo - ab for wo, _wi, ab in paired]
        diffs_wi = [wi - ab for _wo, wi, ab in paired]
        m_wo, lo_wo, hi_wo = bootstrap_mean_ci(diffs_wo)
        m_wi, lo_wi, hi_wi = bootstrap_mean_ci(diffs_wi)
        lines.append(f"n paired = {len(paired)}")
        lines.append(f"- without injected span: mean diff (armC - armB) = {m_wo:+.1f} "
                     f"tokens, 95% CI [{lo_wo:+.1f}, {hi_wo:+.1f}]")
        lines.append(f"- with injected span: mean diff (armC - armB) = {m_wi:+.1f} "
                     f"tokens, 95% CI [{lo_wi:+.1f}, {hi_wi:+.1f}]")
    else:
        lines.append("(no paired ids found against the arm-B baseline)")
    lines.append("")

    # rates
    n_dec = len(decision_rows)
    trigger_rate = len(triggered) / n_dec if n_dec else float("nan")
    re_trig = [r for r in triggered if r.get("re_triggered")]
    re_trigger_rate = len(re_trig) / len(triggered) if triggered else float("nan")
    abstain = [r for r in triggered if r.get("abstain")]
    abstain_rate = len(abstain) / len(triggered) if triggered else float("nan")
    lines.append("## Rates")
    lines.append("")
    lines.append(f"- trigger rate (decision rows): {trigger_rate:.3f} ({len(triggered)}/{n_dec})")
    lines.append(f"- re-trigger rate (among triggered): {re_trigger_rate:.3f} "
                 f"({len(re_trig)}/{len(triggered)})")
    lines.append(f"- abstain rate (among triggered, conf < {CONF_FLOOR}): {abstain_rate:.3f} "
                 f"({len(abstain)}/{len(triggered)})")
    lines.append("")

    # false triggers on controls/adversarial
    lines.append("## False triggers on control/adversarial rows")
    lines.append("")
    lines.append("| kind | n | valid-trigger rate | any-trigger rate (valid or invalid) |")
    lines.append("|---|---|---|---|")
    for kind in ("control", "adversarial"):
        rs = [r for r in rows if r.get("kind") == kind]
        if not rs:
            lines.append(f"| {kind} | 0 | n/a | n/a |")
            continue
        n_valid = sum(1 for r in rs if r.get("valid"))
        n_any = sum(1 for r in rs if r.get("trigger_type") is not None)
        lines.append(f"| {kind} | {len(rs)} | {n_valid/len(rs):.3f} | {n_any/len(rs):.3f} |")
    lines.append("")

    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        out_path = resolve(args.out)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"wrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
# selftest: pure-Python checks, no torch/GPU -- prefix-rebuild byte check against the frozen hybrid
# fixture, plus find_trigger/split_continuation unit checks and a score() smoke test on a FABRICATED
# S3 file.
# ---------------------------------------------------------------------------

def selftest_find_trigger():
    cases = [
        ("no trigger here at all", None),
        ("some thinking <|decide:claim_handling|>rest", ("claim_handling", True)),
        ("some thinking <|decide:not_a_real_type|>rest", ("not_a_real_type", False)),
        ("<|decide:order_outcome|> first one <|decide:news_topic|> second one",
         ("order_outcome", True)),  # regex.search finds the FIRST match only
        ("an unterminated <|decide:claim_handling", None),  # no closing |> -- not a complete trigger
    ]
    for text, want in cases:
        hit = find_trigger(text)
        if want is None:
            assert hit is None, f"expected no trigger in {text!r}, got {hit}"
        else:
            wtype, wvalid = want
            assert hit is not None and hit["type"] == wtype and hit["valid"] == wvalid, \
                f"{text!r}: expected ({wtype}, {wvalid}), got {hit}"
    print("selftest: find_trigger OK (%d cases)" % len(cases))


def selftest_split_continuation():
    cases = [
        ("more thought.<channel|>The answer is X.<turn|><pad>", ("more thought.", "The answer is X.")),
        ("still thinking, no close yet", ("still thinking, no close yet", "")),
        ("<channel|>only answer, nothing before", ("", "only answer, nothing before")),
    ]
    for raw, (want_think, want_ans) in cases:
        think, ans = split_continuation(raw)
        assert think == want_think, f"{raw!r}: thinking_tail {think!r} != {want_think!r}"
        assert ans == want_ans, f"{raw!r}: answer {ans!r} != {want_ans!r}"
    print("selftest: split_continuation OK (%d cases)" % len(cases))


def selftest_prefix_rebuild(fixture_path):
    """Asserts rebuild_prefill(..., abstain=False) reproduces oracle/phase7r-hybrid-prefixes-tail.jsonl's
    own prefix_text up through its '<|/result|>' close (i.e. before that fixture's own appended tail +
    THINK_CLOSE, which this script's stage3 does NOT bake into the prefill -- the tail there is base-model
    organic CONTINUATION text, generated by a separate step, not part of the frozen injected-span shape
    this function reconstructs). Uses check_phase7r_pilot.load_canonical for the per-type instruction
    string, same source build_phase7r_hybrid.py's build_prefix_row itself reads from."""
    rows = load_jsonl(fixture_path)[:5]
    instructions = load_canonical_instructions()
    n = 0
    for row in rows:
        options = [option_text(k, d) for k, d in row["options_with_descriptions"]]
        rebuilt = rebuild_prefill(row["prompt_text"], row["organic_kept"], row["type"],
                                   instructions[row["type"]], options, row["gold_label"], False)
        # row["prefix_text"] = THINK_OPEN + organic_kept + TRIGGER + injected + tail + THINK_CLOSE
        # (continue_phase7r_hybrid tail mode); cut it at the end of the FIRST RESULT_CLOSE to get the
        # no-tail prefix this function reproduces.
        close_idx = row["prefix_text"].find(RESULT_CLOSE)
        assert close_idx >= 0, f"{row['id']}: no {RESULT_CLOSE!r} in fixture prefix_text"
        notail = row["prefix_text"][:close_idx + len(RESULT_CLOSE)]
        want = row["prompt_text"] + notail
        assert rebuilt == want, (
            f"{row['id']}: prefix rebuild mismatch\n  got : {rebuilt[-200:]!r}\n  want: {want[-200:]!r}")
        n += 1
    print(f"selftest: prefix_rebuild OK ({n} rows vs {fixture_path})")


def _fabricate_s3_rows():
    """Small, clearly-labelled FABRICATED S3.jsonl content (not real model output) for a CPU smoke test
    of cmd_score's aggregation logic. 3 rows: one correct non-abstain trigger, one wrong-but-triggered,
    one abstain, shaped like real stage1->stage2->stage3 output."""
    common = dict(kind="decision", type="claim_handling",
                  options_with_descriptions=[["auto_approved", "Approved without further review"],
                                              ["director_signoff", "Requires a director"],
                                              ["rejected", "Rejected outright"]])
    rows = [
        dict(common, id="fab-001", user="u1", gold_label="auto_approved", valid=True, abstain=False,
             chosen_label="auto_approved", confidence=0.9, trigger_type="claim_handling",
             n_thinking_so_far_tokens=50, n_cont_tokens=20, n_injected_tokens=15,
             final_answer="This claim is auto approved.", re_triggered=False),
        dict(common, id="fab-002", user="u2", gold_label="rejected", valid=True, abstain=False,
             chosen_label="auto_approved", confidence=0.7, trigger_type="claim_handling",
             n_thinking_so_far_tokens=60, n_cont_tokens=25, n_injected_tokens=15,
             final_answer="This claim is auto approved.", re_triggered=False),
        dict(common, id="fab-003", user="u3", gold_label="director_signoff", valid=True, abstain=True,
             chosen_label="rejected", confidence=0.3, trigger_type="claim_handling",
             n_thinking_so_far_tokens=40, n_cont_tokens=30, n_injected_tokens=0,
             final_answer="This needs a director.", re_triggered=False),
    ]
    return rows


def selftest_score(scratch_dir):
    os.makedirs(scratch_dir, exist_ok=True)
    s3_path = os.path.join(scratch_dir, "fabricated-s3.jsonl")
    armb_path = os.path.join(scratch_dir, "fabricated-armb.jsonl")
    with open(s3_path, "w", encoding="utf-8") as f:
        for r in _fabricate_s3_rows():
            f.write(json.dumps(r) + "\n")
    with open(armb_path, "w", encoding="utf-8") as f:
        for i in range(1, 4):
            f.write(json.dumps({"id": f"fab-{i:03d}", "n_thinking_tokens": 100,
                                 "n_answer_tokens": 10}) + "\n")

    class _Args:
        pass
    a = _Args()
    a.in_, a.baseline_armb, a.extract, a.out = s3_path, armb_path, None, None
    rc = cmd_score(a)
    assert rc == 0
    print(f"selftest: score() smoke test OK on FABRICATED data ({s3_path}, {armb_path})")


def selftest_normalize_adversarial():
    """A fabricated S1 row mimicking a real adversarial/near-miss row: gold `type`='claim_handling' with
    `options_with_descriptions`=None (adversarial rows carry no gold options), but the model actually
    EMITTED a 'kb_category' trigger. normalize_s1_row must re-type the row to kb_category (never touch
    claim_handling's -- nonexistent -- options) and stage2_score_row must then score it against
    kb_category's instruction/options without crashing. Also checks normalize_s1_row is idempotent (a
    second call is a no-op) and leaves non-valid rows untouched."""
    row = dict(id="fab-adv-001", user="some adversarial user turn", kind="adversarial",
               type="claim_handling", options_with_descriptions=None, valid=True,
               trigger_type="kb_category", thinking_so_far="some thinking before the trigger")
    out = normalize_s1_row(dict(row))
    assert out["type"] == "kb_category", out["type"]
    assert out["gold_type"] == "claim_handling", out["gold_type"]
    assert out["gold_options_with_descriptions"] is None
    assert out["options_with_descriptions"] == load_type_options()["kb_category"]
    assert out["_normalized"] is True

    # idempotence: re-running on the already-normalized row is a no-op
    out2 = normalize_s1_row(dict(out))
    assert out2 == out, "normalize_s1_row is not idempotent"

    # non-valid rows pass through untouched
    nonvalid = dict(row, valid=False)
    out3 = normalize_s1_row(dict(nonvalid))
    assert out3 == nonvalid

    # stage2_score_row must use kb_category's instruction/options, not claim_handling's (which would be
    # None here and crash on `for k, d in row["options_with_descriptions"]`)
    instructions = load_canonical_instructions()
    scorer = make_fake_scorer()
    scored = stage2_score_row(scorer, instructions, out)
    assert scored["chosen_label"] in {k for k, _d in load_type_options()["kb_category"]}, \
        scored["chosen_label"]
    print("selftest: normalize_s1_row (adversarial-mimic, idempotence, non-valid passthrough) OK")


def selftest():
    selftest_find_trigger()
    selftest_split_continuation()
    selftest_normalize_adversarial()
    fixture = resolve("oracle/phase7r-hybrid-prefixes-tail.jsonl")
    if os.path.exists(fixture):
        selftest_prefix_rebuild(fixture)
    else:
        print(f"SKIP prefix_rebuild selftest: {fixture} not found")
    scratch = os.environ.get("TMPDIR") or os.path.join(ROOT, "_selftest_scratch")
    selftest_score(scratch)
    print("selftest: all checks passed")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common_gen_args(p, default_max_new):
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=default_max_new)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--device", default="cuda")
    p.add_argument("--attn-impl", default="sdpa")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="mode")

    p1 = sub.add_parser("stage1")
    p1.add_argument("--input", required=True)
    p1.add_argument("--adapter", required=True)
    p1.add_argument("--out", required=True)
    add_common_gen_args(p1, DEFAULT_MAX_NEW_TOKENS_S1)

    p2 = sub.add_parser("stage2")
    p2.add_argument("--in", dest="in_", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument("--decision-run", default="runs/p4-e4b-final")
    p2.add_argument("--resume", action="store_true")
    p2.add_argument("--limit", type=int, default=None)
    p2.add_argument("--dry-run", action="store_true")
    p2.add_argument("--device", default="cuda")

    p3 = sub.add_parser("stage3")
    p3.add_argument("--in", dest="in_", required=True)
    p3.add_argument("--adapter", required=True)
    p3.add_argument("--out", required=True)
    add_common_gen_args(p3, DEFAULT_MAX_NEW_TOKENS_S3)

    ps = sub.add_parser("score")
    ps.add_argument("--in", dest="in_", required=True)
    ps.add_argument("--baseline-armB", dest="baseline_armb", required=True)
    ps.add_argument("--extract", default=None)
    ps.add_argument("--out", default=None)

    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.mode == "stage1":
        return cmd_stage1(args)
    if args.mode == "stage2":
        return cmd_stage2(args)
    if args.mode == "stage3":
        return cmd_stage3(args)
    if args.mode == "score":
        return cmd_score(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
