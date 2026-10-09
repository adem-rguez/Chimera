"""CHECK 7 eval harness for Phase 7(a) inline decision calls.

Reads: reports/14-phase7-proposal.md, hybrid-llm-decision-model-plan.md (Phase 7 / CHECK 7),
scripts/build_phase7.py, scripts/train_phase7.py, kev/model.py (probs_one, DecisionModel), kev/api.py
(SystemOneRequest shape, choice_confidence), kev/checkpoint.py (Checkpoint.load).

Mechanism (per the proposal, not re-derived here): the Phase 7 LoRA on google/gemma-4-E4B-it emits
"<|decide|>{instructions} [{opts}]<|result|>" verbatim (trained, unmasked) and then the pointer head
of the ALREADY-TRAINED Kev decision model (the one that scores banking77 at 0.80+; shipped checkpoint
runs/p4-e4b-final per progress.md's E4B validation entry) reads the 77-way criteria out of that call
and produces the winning key -- the chat LoRA never predicts the label token span itself (train_phase7.py
masks exactly that span). This script re-implements the serving-time half of that: parse the emitted
call, build a kev.api.SystemOneRequest-shaped record from its options, score it on the decision
checkpoint via probs_one, inject "<label><|/result|>" as literal text, and resume generation on the
LoRA model.

Held-out set: Banking77's HF "test" split (scripts/build_phase7.py drew only from "train", and so does
train_phase7.py's replay pool; "test" is a different split of the same dataset, so it cannot overlap by
construction, no hash-filtering needed). Built once, cached at --heldout; reuse across runs by pointing
--heldout at the same file.

Baselines: "adapter" = the trained LoRA as above. "free" = base E4B-it (no adapter), prompted to name the
intent in its draft in plain text (BASELINE_PROMPT below), scored with the same extract_named_intent
extractor used for the adapter path's fallback case -- UNFAIR and not comparable to the adapter (kept
only for continuity with earlier runs): BASELINE_PROMPT never shows the model the 77 label names, so it
almost never names one in-vocabulary (run1combined: 5/200 rows, 2.5%), which inflates the adapter-minus-
baseline gap as an artifact of the baseline never having a chance, not of the adapter being better.
"options" = base E4B-it (no adapter), given the SAME information the decision model's pointer head sees
-- the customer message plus the full 77-label list, via tokenizer.apply_chat_template (E4B-it is an
instruction model) -- and asked to answer with exactly one label (run_options_baseline /
build_options_baseline_prompt below; wording mirrors scripts/oracle_labels.py's build_options_prompt, so
the resulting number is comparable to the Phase 5/6 73-75% chat-with-options figures). This is the fair
baseline: "does the inline <|decide|> call + pointer-head readout beat the base model answering with the
same label list, chat-style?" See reports/15-phase7-eval-plan.md for further gaps.

Usage (GPU box, after scripts/train_phase7.py has written an adapter dir):
  .venv/bin/python -u scripts/eval_phase7.py \\
      --model google/gemma-4-E4B-it --revision ee0ef6023621cff504d758262d4e04895a5af4a2 \\
      --adapter runs/p7-it-lora/final --decision-run runs/p4-e4b-final \\
      --tag run1 --n-heldout 200

On a RAM-limited box, run the three phases as separate processes (each loads exactly one model,
never two at once, and the host never has to hold both the ~16GB bf16 chat model and the decision
model's backbone at the same time):
  .venv/bin/python -u scripts/eval_phase7.py --adapter ... --decision-run ... --tag run1 --phase 1
  .venv/bin/python -u scripts/eval_phase7.py --adapter ... --decision-run ... --tag run1 --phase 2
  .venv/bin/python -u scripts/eval_phase7.py --adapter ... --decision-run ... --tag run1 --phase 3
--phase defaults to "all" (the three steps in one process, in sequence); prefer the stepwise form above
whenever host RAM is uncertain. --baseline must be "adapter" (or "none") when --phase isn't "all": the
free baseline only ever loads one model and has no phase split -- run it in its own --phase all invocation.

--force-options (default OFF): the 77-name option list is fixed/identical across every training row, so
instead of letting the model type it out in phase 1 (and risk a malformed list), generation stops at the
'<|decide|>' trigger and the harness splices in the canonical call as training-identical token ids (see
build_decide_and_canonical_ids) -- call validity is then 100% by construction; only the trigger itself
(whether the model opens the call at all, "trigger rate") is still measured. Same three-phase/--tag flow
as above, e.g.:
  .venv/bin/python -u scripts/eval_phase7.py --adapter ... --decision-run ... --tag run1f --force-options --phase 1
  .venv/bin/python -u scripts/eval_phase7.py --adapter ... --decision-run ... --tag run1f --force-options --phase 2
  .venv/bin/python -u scripts/eval_phase7.py --adapter ... --decision-run ... --tag run1f --force-options --phase 3
(--force-options must be passed on every phase for a given --tag; output files are tagged '-forced' so
they never collide with a non-forced run under the same --tag.)

Offline (laptop, no GPU, no model/dataset loads -- verifies the pure-Python parts only):
  .venv/bin/python -m py_compile scripts/eval_phase7.py scripts/eval_phase7_metrics.py
  .venv/bin/python scripts/eval_phase7.py --dry-parse
  .venv/bin/python scripts/eval_phase7.py --test-prompt-boundary  (needs the tokenizer only, not the model)
  EVAL_PHASE7_TEST_TORCH=1 .venv/bin/python scripts/eval_phase7.py --dry-parse  (adds the torch-dependent
      choice_confidence/write_jsonl checks when torch is importable)
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_phase7_metrics import (  # noqa: E402
    bootstrap_ci, build_options_baseline_prompt, classify_corrupted_row, classify_miss_row,
    conditional_rate, continuation_clean, extract_named_intent, first_sentence, first_sentence_phrase,
    first_sentence_consistent_with_injected_label,
    first_sentence_names_other_label, has_loop, has_stray_tags, off_template_first_sentence, clean_stop,
    is_corrupted, label_phrase, paired_bootstrap_diff, parse_decide_call, parse_injected_result,
    parse_options_answer, score_first_sentence, template_match, template_phrase,
)
from scripts.build_phase7 import PREFIX as _BUILD_PREFIX, INSTR as _BUILD_INSTR  # noqa: E402

PREFIX = "Thanks for reaching out. "          # must match scripts/build_phase7.py exactly
INSTR = "Which banking intent best describes this customer message?"
assert PREFIX == _BUILD_PREFIX and INSTR == _BUILD_INSTR, (
    "eval_phase7's PREFIX/INSTR have drifted from scripts/build_phase7.py's -- --force-options splices"
    " in a canonical call built from these constants, so a mismatch here would make every forced call"
    " well-formed-but-wrong rather than training-identical")
BASELINE_PROMPT = (
    "You are a banking support agent. In your reply, explicitly state which banking issue category "
    "this message falls into, then help with it.\nCustomer: {text}\nAgent: "
)
DEFAULT_CONF_FLOOR = 0.6112   # oracle/gate-threshold.md Phase 5B T (accuracy-maximising; see caveats there:
                              # fit+eval on the same 331 dev rows, confidences likely uncalibrated raw softmax)


# ---------------------------------------------------------------------------
# Held-out set (needs `datasets`, no torch; safe on the laptop)
# ---------------------------------------------------------------------------

def _calls_text_hashes(calls_path):
    """text_sha256 set from oracle/phase7-calls.jsonl's provenance, so the held-out set can exclude them.
    Banking77's train/test splits have a handful (7, measured) of near-duplicate texts across the split
    boundary; the split alone already prevents any *row* overlap, this closes the remaining text-level
    gap. Missing file -> empty set (first run, before scripts/build_phase7.py); not an error."""
    if not os.path.exists(calls_path):
        return set()
    return {json.loads(l)["provenance"]["text_sha256"]
            for l in open(calls_path, encoding="utf-8") if l.strip()}


def build_heldout(out_path, n, seed, calls_path="oracle/phase7-calls.jsonl"):
    from datasets import load_dataset
    import hashlib
    import random
    ds = load_dataset("legacy-datasets/banking77", split="test")
    names = ds.features["label"].names
    excluded = _calls_text_hashes(calls_path)
    rng = random.Random(seed)
    pool = list(range(len(ds)))
    rng.shuffle(pool)
    rows, skipped = [], 0
    for i in pool:
        if len(rows) >= n:
            break
        row = ds[i]
        text = row["text"]
        norm = " ".join(text.casefold().split())
        h = hashlib.sha256(norm.encode()).hexdigest()
        if h in excluded:
            skipped += 1
            continue
        rows.append({
            "id": f"p7-held-{i}", "state": text, "label": names[row["label"]], "names": names,
            "provenance": {"split": "test", "row": i, "text_sha256": h},
        })
    if len(rows) < n:
        raise ValueError(f"only found {len(rows)}/{n} held-out rows after excluding {skipped}"
                          f" train-overlapping texts; lower --n-heldout or widen the pool")
    if skipped:
        print(f"heldout: skipped {skipped} banking77 test rows whose text also appears in"
              f" {calls_path} (train/test near-duplicates)", flush=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def load_or_build_heldout(path, n, seed):
    if os.path.exists(path):
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        print(f"heldout: loaded {len(rows)} rows from {path} (ignoring --n-heldout/--seed; delete the"
              f" file to rebuild)", flush=True)
        return rows
    rows = build_heldout(path, n, seed)
    print(f"heldout: built {len(rows)} rows (banking77 test split, seed {seed}) -> {path}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# GPU-dependent pieces (imported lazily so --dry-parse never needs torch)
# ---------------------------------------------------------------------------

def build_decide_and_canonical_ids(tok, state, names, prefix=PREFIX, instr=INSTR):
    """-> (prompt_ids, trigger_ids, canonical_ids): three slices of ONE tokenization of the full pre-call
    template (see build_decide_prompt_ids's docstring for why it must be one tokenization, not several
    tokenized separately). prompt_ids ends right before '<|decide|>' (== build_decide_prompt_ids's
    return); trigger_ids is exactly the '<|decide|>' piece (what the model itself would have to generate
    to open the call -- plain multi-piece text on google/gemma-4-E4B-it, e.g. '▁<','|','dec','ide','|>',
    not a single special token); canonical_ids is '{instr} [{', '.join(names)}]<|result|>', the fixed
    per-row suffix build_phase7.py trains on (the 77-name option list is identical across rows, so this
    never varies at eval time). --force-options mode generates only up to the '<|decide|>' trigger, then
    splices trigger_ids + canonical_ids in verbatim instead of letting the model type the options list,
    so the resulting call is well-formed by construction on every triggered row."""
    pre = f"Customer: {state}\nAgent: {prefix}<|decide|>{instr} [{', '.join(names)}]<|result|>"
    decide_char = pre.index("<|decide|>")
    decide_end_char = decide_char + len("<|decide|>")
    enc = tok(pre, return_offsets_mapping=True, add_special_tokens=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    cut = next(k for k, (a, b) in enumerate(offs) if a >= decide_char)
    trigger_end = next(k for k, (a, b) in enumerate(offs) if a >= decide_end_char)
    return ids[:cut], ids[cut:trigger_end], ids[trigger_end:]


def build_decide_prompt_ids(tok, state, names, prefix=PREFIX, instr=INSTR):
    """Phase-1 generation prompt as TOKEN IDS, cut at the exact token boundary train_phase7.encode_call
    sees for '<|decide|>' in the real target_text (scripts/build_phase7.py). Tokenizing just the prompt
    STRING in isolation (ending right after `prefix`, nothing after it) is NOT equivalent: a lone trailing
    space at end-of-string tokenizes as its own token ('▁', id 236743 on google/gemma-4-E4B-it), while
    in training that same space is immediately followed by '<|decide|>' and merges into a single '▁<'
    token (id 655) instead -- a token sequence the model never saw ending any other way during training,
    which is why it falls back to a free draft instead of ever emitting '<|decide|>' (verified against 50
    real oracle/phase7-calls.jsonl rows; see scripts/eval_phase7.py's module docstring / CHECK 7 notes).
    Ending the string right at '<|decide|>' (nothing after it) does NOT fix this either -- '<|decide|>'
    itself tokenizes differently at end-of-string than when followed by more text. The only string that
    reproduces training's tokens exactly is the FULL pre-call template (instr + the complete options
    list + '<|result|>'), identical to build_phase7.py's `pre`; this function builds exactly that, then
    slices the ids back to where '<|decide|>' begins via offset_mapping, discarding the fixed-template
    tail that merely comes along for tokenization purposes. See build_decide_and_canonical_ids for the
    --force-options counterpart that also keeps the discarded tail."""
    prompt_ids, _trigger_ids, _canonical_ids = build_decide_and_canonical_ids(tok, state, names, prefix, instr)
    return prompt_ids


def batched_generate_from_ids(tok, model, id_lists, max_new_tokens, device, stop_strings=None, batch_size=4):
    """batched_generate, but the prompt is already tokenized (see build_decide_prompt_ids): left-pads the
    given id lists by hand instead of re-tokenizing a string (re-tokenizing would reintroduce exactly the
    boundary mismatch this is meant to avoid). -> (texts, n_new_tokens, gen_texts): texts/n_new_tokens are
    the same contract as batched_generate; gen_texts is each row's generated-continuation-ONLY text
    (decoded from just the new tokens, skip_special_tokens=False, no prompt/padding) -- --force-options
    needs this to check for the '<|decide|>' trigger without the prompt text (cut before '<|decide|>' by
    construction, so it can never contain the trigger) or left-padding getting in the way."""
    import torch
    pad_id = tok.pad_token_id
    out, counts, gen_texts = [], [], []
    for i in range(0, len(id_lists), batch_size):
        chunk = id_lists[i:i + batch_size]
        L = max(len(x) for x in chunk)
        input_ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, ids in enumerate(chunk):
            input_ids[j, L - len(ids):] = torch.tensor(ids, dtype=torch.long)
            attn[j, L - len(ids):] = 1
        input_ids, attn = input_ids.to(device), attn.to(device)
        prompt_len = L
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        if stop_strings:
            gen_kwargs.update(stop_strings=stop_strings, tokenizer=tok)
        with torch.no_grad():
            ids_out = model.generate(input_ids=input_ids, attention_mask=attn, **gen_kwargs)
        for j in range(len(chunk)):
            out.append(tok.decode(ids_out[j], skip_special_tokens=False))
            new_ids = ids_out[j, prompt_len:]
            counts.append(int((new_ids != tok.pad_token_id).sum()))
            gen_texts.append(tok.decode(new_ids, skip_special_tokens=False))
    return out, counts, gen_texts


def _check_prompt_boundary(calls_path="oracle/phase7-calls.jsonl", n=50, model=None, revision=None):
    """Tokenizer-only unit test (no torch, no GPU): for `n` real training rows, asserts (a)
    build_decide_prompt_ids's output ids are an exact token prefix of the real training-format
    tokenization (train_phase7.encode_call's tokenization of rec['target_text']) up to '<|decide|>', (b)
    the per-row INSTR/names/format build_decide_and_canonical_ids assumes matches the real
    target_text's suffix verbatim, and (c) --force-options's constructed call --
    prompt_ids + trigger_ids + canonical_ids -- is EXACTLY the training ids through '<|result|>', built
    by slicing training ids at the '<|decide|>' token boundary (offset_mapping) rather than by
    re-tokenizing the canonical string on its own (which may merge differently at the boundary; see
    build_decide_prompt_ids's docstring for why that distinction matters). -> True/False (prints per-row
    failures)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    rows = [json.loads(l) for l in open(calls_path, encoding="utf-8") if l.strip()][:n]
    failures = 0
    forced_failures = 0
    for i, rec in enumerate(rows):
        text = rec["target_text"]
        names = list(rec["criteria"].keys())
        decide_char = text.index("<|decide|>")
        decide_end_char = decide_char + len("<|decide|>")
        call_end_char = text.index("<|result|>") + len("<|result|>")
        full = tok(text, return_offsets_mapping=True, add_special_tokens=True)
        cut = next(k for k, (a, b) in enumerate(full["offset_mapping"]) if a >= decide_char)
        call_end_cut = next(k for k, (a, b) in enumerate(full["offset_mapping"]) if a >= call_end_char)
        train_prefix_ids = full["input_ids"][:cut]
        train_call_ids = full["input_ids"][:call_end_cut]

        prompt_ids = build_decide_prompt_ids(tok, rec["state"], names)
        if prompt_ids != train_prefix_ids:
            failures += 1
            print(f"  FAIL row {i} ({rec['id']}): prompt_ids len={len(prompt_ids)} != "
                  f"train_prefix_ids len={len(train_prefix_ids)}")

        canon_string = f"{INSTR} [{', '.join(names)}]<|result|>"
        real_suffix = text[decide_end_char:call_end_char]
        if real_suffix != canon_string:
            forced_failures += 1
            print(f"  FAIL(canon) row {i} ({rec['id']}): real target_text suffix != canonical "
                  f"INSTR/names/format -- {real_suffix!r} != {canon_string!r}")

        p_ids, trig_ids, canon_ids = build_decide_and_canonical_ids(tok, rec["state"], names)
        forced_call_ids = p_ids + trig_ids + canon_ids
        if forced_call_ids != train_call_ids:
            forced_failures += 1
            print(f"  FAIL(forced) row {i} ({rec['id']}): prompt+trigger+canonical ids len="
                  f"{len(forced_call_ids)} != train_call_ids len={len(train_call_ids)}")

    if failures or forced_failures:
        print(f"{failures}/{len(rows)} rows FAILED the token-prefix boundary check; "
              f"{forced_failures}/{len(rows)} rows FAILED the --force-options canonical-call check")
    else:
        print(f"all {len(rows)} rows: build_decide_prompt_ids ids are an exact token prefix of training "
              f"ids; --force-options's prompt+trigger+canonical ids exactly reproduce training ids "
              f"through '<|result|>'")
    return failures == 0 and forced_failures == 0


def load_lora_model(model_name, revision, adapter_dir, device, attn_impl="sdpa"):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from kev.model import load_tokenizer
    tok = load_tokenizer(model_name, revision)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, dtype=torch.bfloat16, attn_implementation=attn_impl
    ).to(device)
    model = PeftModel.from_pretrained(base, adapter_dir).to(device).eval()
    return tok, model


def load_decision_model(run, device):
    """bf16, straight to `device`, low_cpu_mem_usage (via accelerate's default for from_pretrained with a
    dtype set): the default LoadOptions() loads fp32 (DecisionModel's own default dtype), i.e. ~2x the
    backbone's trained bf16 size staged in host RAM while the LoRA chat model's memory may still be alive
    (see unload()) -- on a host with limited RAM this is what killed phase 2 in the CHECK 7 smoke test
    (progress stalled mid-shard-load, then SIGKILL). bf16 is also the dtype the checkpoint was actually
    trained/served in (progress.md's E4B validation number), so this is not a precision tradeoff here."""
    import torch
    from kev.checkpoint import Checkpoint, LoadOptions
    tok, model = Checkpoint(run).load(device, LoadOptions(dtype=torch.bfloat16))
    return tok, model


def unload(model):
    """-> None (always reassign the caller's variable to this, e.g. `model = unload(model)`): a bare
    `del model` inside this function only drops ITS OWN local reference. If the caller still has its own
    variable bound to the same object (e.g. run_adapter_eval's `lora_model`, left bound between the
    unload(lora_model) call and the next load_lora_model(...) reassignment), the object's refcount never
    hits zero, gc.collect() has nothing to collect, and the memory (host RAM for the staged weights, GPU
    memory for the resident copy) stays held exactly while the next model is being loaded -- this is the
    other half of the phase-2 OOM: the LoRA model was still alive, in full, while the decision model's
    shards were loading."""
    import gc
    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return None


def batched_generate(tok, model, prompts, max_new_tokens, device, stop_strings=None, batch_size=4):
    """-> (texts, n_new_tokens, gen_texts): texts is the full decoded text (prompt + continuation) for each
    prompt, in order, decoded with skip_special_tokens=False (verified necessary: round-tripping a real
    oracle/phase7-calls.jsonl target_text through this tokenizer shows <|decide|>/<|result|>/<|/result|>
    are NOT registered special/added tokens on google/gemma-4-E4B-it -- they tokenize to ordinary
    multi-piece text ('<','|','dec','ide','|>', ...) -- so skip_special_tokens made no difference in that
    test; kept False anyway since it is still the correct setting and costs nothing). n_new_tokens is each
    row's own count of non-pad tokens generated after its prompt (approximate: counts up to the first
    pad_token_id run, so an early stop_strings/eos stop does not get inflated by the batch's right-padding
    to the longest row) -- for --dump-raw's diagnostics. gen_texts is each row's generated-CONTINUATION-
    ONLY text, decoded from just the newly generated token ids (skip_special_tokens=False, no prompt, no
    padding) -- callers that need "what did the model add after the prompt" MUST use this, never
    `texts[j][len(prompt):]`: re-tokenizing `prompt` here and decoding it back (`texts[j]`'s prompt
    portion) does not always round-trip to exactly `prompt` character-for-character, so a character-
    offset slice drifts and can grab a trailing fragment of the prompt's own call-marker text into the
    "continuation" (confirmed root cause of run_adapter_phase3's run1combined3 first-sentence-metrics
    bug: stored `continuation` rows frequently began mid-call-text, e.g. 'ult|> This looks like ...',
    instead of cleanly at the model's actual first generated token).

    eos_token_id is pinned to the tokenizer's own <eos> id, overriding the base model's generation_config
    default (google/gemma-4-E4B-it's generation_config.json: eos_token_id=[<eos>, <turn|>, <|tool_response>]
    -- chat-turn-boundary ids from its instruction-tuning that never occur in train_phase7.py's plain-text
    SFT targets; left at the model default, greedy decoding can emit one of them mid-call and end
    generation long before <|result|>, which would show up exactly as an unparseable, truncated row)."""
    import torch
    out, counts, gen_texts = [], [], []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        prompt_len = enc["input_ids"].shape[1]
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        if stop_strings:
            gen_kwargs.update(stop_strings=stop_strings, tokenizer=tok)
        with torch.no_grad():
            ids = model.generate(**enc, **gen_kwargs)
        for j in range(len(chunk)):
            out.append(tok.decode(ids[j], skip_special_tokens=False))
            new_ids = ids[j, prompt_len:]
            counts.append(int((new_ids != tok.pad_token_id).sum()))
            gen_texts.append(tok.decode(new_ids, skip_special_tokens=False))
    return out, counts, gen_texts


def score_decide_call(dec_tok, dec_model, customer_text, options):
    """Pointer-head readout for one emitted <|decide|> call: options in call order -> (resolved_label,
    confidence, probs_by_option). Mirrors kev.api.to_record's choice-question internal record shape and
    kev.model.probs_one's non-batched path (model.probs(enc)); no new readout machinery, per the proposal.
    dec_model.probs returns a torch Tensor (model.py:540's F.softmax(...).cpu()); pull it off-device into
    plain Python floats here, at the source, so every downstream consumer (choice_confidence, the phase2
    row dict, write_jsonl) only ever sees JSON-serializable values."""
    from kev.api import choice_confidence
    rec = {"state": customer_text, "questions": [{"instr": INSTR, "options": list(options), "label": 0}]}
    enc = dec_model.encode(dec_tok, rec)
    probs = [float(x) for x in dec_model.probs(enc)[0]]
    best_i = max(range(len(probs)), key=lambda k: probs[k])
    return options[best_i], choice_confidence(probs), dict(zip(options, probs))


# ---------------------------------------------------------------------------
# Adapter-path pipeline: 2 LoRA loads + 1 decision load total (never both
# models resident at once -- the proposal's box is the same ~22GB L4 as
# training, and bf16 E4B-it + bf16 E4B-decision together do not comfortably fit).
# ---------------------------------------------------------------------------

def _json_default(obj):
    """json.dumps(default=...) fallback: torch Tensors / numpy scalars and arrays are not JSON-serializable
    on their own (score_decide_call converts at the source, but this is the backstop so no later phase can
    crash the same way on a value that slipped through)."""
    tolist = getattr(obj, "tolist", None)
    if tolist is not None:
        return tolist()
    item = getattr(obj, "item", None)
    if item is not None:
        return item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_jsonl(path, rows):
    """Atomic: writes to a tmp file and renames into place, so a crash mid-write never leaves a
    half-written file that the next phase would load as complete."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=_json_default) + "\n")
    os.replace(tmp_path, path)


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _backfill_names(adapter_rows, heldout_path):
    """In-place: for any row missing 'names' (adapter-rows files written before run_adapter_phase3
    started storing it), fill it in from the heldout file's label list -- the full 77-name option list is
    identical across every row (see build_phase7.py), so any one heldout row's 'names' works for all.
    No-op if every row already has 'names', or if the heldout file is missing/empty."""
    if all("names" in r for r in adapter_rows):
        return
    if not os.path.exists(heldout_path):
        return
    heldout_rows = read_jsonl(heldout_path)
    if not heldout_rows:
        return
    default_names = heldout_rows[0]["names"]
    for r in adapter_rows:
        r.setdefault("names", default_names)


def run_adapter_phase1(rows, args, device):
    """-> list of {id, gold, state, call_found, raw_text, call}. Loads only the LoRA chat model.

    --force-options: generation stops at the '<|decide|>' TRIGGER, not at '<|result|>' -- the only part
    still being measured is whether the model emits '<|decide|>' at all ("trigger rate"). Once it does,
    the harness splices in trigger_ids + canonical_ids (build_decide_and_canonical_ids: the training ids
    of '<|decide|>{INSTR} [{', '.join(names)}]<|result|>') instead of letting the model type the 77-name
    options list itself, so call_found/call are well-formed BY CONSTRUCTION for every triggered row --
    call_found here means "triggered", not "parsed" (no parse_decide_call is ever run on these rows).
    Rows that never trigger are reported as call_found=False with raw_text = the raw generation, and
    also written to args.misses_out for inspection."""
    lora_tok, lora_model = load_lora_model(args.model, args.revision, args.adapter, device, args.attn_impl)
    print(f"phase 1/3: generating decide calls for {len(rows)} rows "
          f"(max_new_tokens={args.max_new_before}, prompt_mode={args.prompt_mode}, "
          f"force_options={args.force_options})", flush=True)

    if args.force_options:
        id_lists = [build_decide_prompt_ids(lora_tok, r["state"], r["names"]) for r in rows]
        _texts, counts, gen_texts = batched_generate_from_ids(
            lora_tok, lora_model, id_lists, args.max_new_before, device,
            stop_strings=["<|decide|>"], batch_size=args.batch_size)
        lora_model = unload(lora_model)

        triggered = ["<|decide|>" in g for g in gen_texts]
        n_triggered = sum(triggered)
        print(f"force-options: {n_triggered}/{len(rows)} rows emitted the <|decide|> trigger within "
              f"{args.max_new_before} tokens", flush=True)

        out_rows, misses = [], []
        for r, prompt_ids, g, n, trig in zip(rows, id_lists, gen_texts, counts, triggered):
            if not trig:
                out_rows.append({"id": r["id"], "gold": r["label"], "state": r["state"],
                                  "call_found": False, "raw_text": g, "call": None})
                misses.append({"id": r["id"], "n_new_tokens": n, "raw_generation_200": g[:200]})
                continue
            _p, trig_ids, canon_ids = build_decide_and_canonical_ids(lora_tok, r["state"], r["names"])
            full_ids = prompt_ids + trig_ids + canon_ids
            raw_text = lora_tok.decode(full_ids, skip_special_tokens=False)
            out_rows.append({"id": r["id"], "gold": r["label"], "state": r["state"],
                             "call_found": True, "raw_text": raw_text,
                             "call": {"options": r["names"], "call_end": len(raw_text)}})

        if args.dump_raw:
            write_jsonl(args.dump_raw, [
                {"id": r["id"], "raw_text": g, "n_new_tokens": n, "triggered": t}
                for r, g, n, t in zip(rows, gen_texts, counts, triggered)])
            print(f"wrote {args.dump_raw}", flush=True)
        if misses:
            write_jsonl(args.misses_out, misses)
            print(f"wrote {args.misses_out} ({len(misses)} rows where the <|decide|> trigger did not"
                  f" fire)", flush=True)
        return out_rows

    if args.prompt_mode == "token":
        id_lists = [build_decide_prompt_ids(lora_tok, r["state"], r["names"]) for r in rows]
        texts, counts, _gen_texts = batched_generate_from_ids(
            lora_tok, lora_model, id_lists, args.max_new_before, device,
            stop_strings=["<|result|>"], batch_size=args.batch_size)
    else:
        prompts = [f"Customer: {r['state']}\nAgent: {PREFIX}" for r in rows]
        texts, counts, _gen_texts = batched_generate(lora_tok, lora_model, prompts, args.max_new_before, device,
                                         stop_strings=["<|result|>"], batch_size=args.batch_size)
    lora_model = unload(lora_model)

    calls = [parse_decide_call(t) for t in texts]
    n_missing = sum(c is None for c in calls)
    if n_missing:
        print(f"WARNING: {n_missing}/{len(calls)} rows produced no parseable <|decide|> call within"
              f" {args.max_new_before} tokens", flush=True)
        shown = 0
        for t, c in zip(texts, calls):
            if c is None and shown < 2:
                print(f"  raw generation (first 300 chars): {t[:300]!r}", flush=True)
                shown += 1

    if args.dump_raw:
        write_jsonl(args.dump_raw, [
            {"id": r["id"], "raw_text": t, "n_new_tokens": n}
            for r, t, n in zip(rows, texts, counts)])
        print(f"wrote {args.dump_raw}", flush=True)

    return [
        {"id": r["id"], "gold": r["label"], "state": r["state"], "call_found": c is not None,
         "raw_text": t, "call": {"options": c["options"], "call_end": c["call_end"]} if c else None}
        for r, t, c in zip(rows, texts, calls)
    ]


def run_adapter_phase2(phase1_rows, args, device):
    """-> list of {id, resolved}. Loads only the decision checkpoint."""
    print("phase 2/3: scoring decide calls on the decision checkpoint", flush=True)
    dec_tok, dec_model = load_decision_model(args.decision_run, device)
    out = []
    for r in phase1_rows:
        if not r["call_found"]:
            out.append({"id": r["id"], "resolved": None})
            continue
        label, conf, _probs = score_decide_call(dec_tok, dec_model, r["state"], r["call"]["options"])
        out.append({"id": r["id"], "resolved": {"label": label, "confidence": conf}})
    dec_model = unload(dec_model)
    return out


def run_adapter_phase3(phase1_rows, phase2_rows, args, device):
    """-> list of final adapter-path result rows (the shape summarize()/the old run_adapter_eval wrote).
    Loads only the LoRA chat model. Joins phase1_rows and phase2_rows by id."""
    print("phase 3/3: injecting results and resuming generation", flush=True)
    resolved_by_id = {r["id"]: r["resolved"] for r in phase2_rows}
    resume_prompts, resume_idx = [], []
    for i, r in enumerate(phase1_rows):
        res = resolved_by_id.get(r["id"])
        if not r["call_found"] or res is None:
            continue
        text = r["raw_text"][:r["call"]["call_end"]] + f"{res['label']}<|/result|>"
        resume_prompts.append(text)
        resume_idx.append(i)
    lora_tok, lora_model = load_lora_model(args.model, args.revision, args.adapter, device, args.attn_impl)
    _final_texts, _counts, gen_texts = batched_generate(
        lora_tok, lora_model, resume_prompts, args.max_new_after, device,
        batch_size=args.batch_size) if resume_prompts else ([], [], [])
    lora_model = unload(lora_model)

    out = []
    for i, r in enumerate(phase1_rows):
        res = resolved_by_id.get(r["id"])
        rec = {"id": r["id"], "gold": r["gold"], "call_found": r["call_found"]}
        if not r["call_found"] or res is None:
            out.append(rec)
            continue
        j = resume_idx.index(i)
        # continuation == the NEWLY GENERATED token ids, decoded directly (gen_texts) -- NOT a character
        # offset into a re-decoded full string (that slice drifts: re-tokenizing resume_prompts[j] and
        # decoding it back does not always round-trip to the same string character-for-character, so the
        # old `full[len(resume_prompts[j]):]` frequently grabbed a trailing fragment of the prompt's own
        # call-marker text into `continuation`; see batched_generate's docstring and
        # scripts/eval_phase7_metrics.py's continuation_clean for the matching robust-scoring fix applied
        # to rows stored by earlier runs before this fix).
        continuation = gen_texts[j]
        rec.update(resolved_label=res["label"], confidence=res["confidence"],
                   continuation=continuation,
                   template_match=template_match(continuation, res["label"]),
                   label_correct=res["label"] == r["gold"],
                   corrupted=is_corrupted(continuation, res["label"]),
                   # full 77-name option list (identical across rows, see build_phase7.py), stored so
                   # --report-only/--inspect can recompute first_sentence_names_other_label from the
                   # continuation text alone, without re-deriving it from the heldout file.
                   names=r["call"]["options"])
        out.append(rec)
    return out


def run_adapter_eval(rows, args, device):
    """--phase all: the three phases in one process, in sequence (never two models resident at once --
    see run_adapter_phase{1,2,3}'s docstrings). Equivalent to running --phase 1, 2, 3 as separate
    processes with the same --tag, minus the file round-trip."""
    phase1_rows = run_adapter_phase1(rows, args, device)
    phase2_rows = run_adapter_phase2(phase1_rows, args, device)
    return run_adapter_phase3(phase1_rows, phase2_rows, args, device)


def run_free_baseline(rows, args, device):
    tok, model = _load_base_only(args.model, args.revision, device, args.attn_impl)
    prompts = [BASELINE_PROMPT.format(text=r["state"]) for r in rows]
    print(f"baseline: free-drafting {len(prompts)} rows (no adapter, max_new_tokens="
          f"{args.max_new_before + args.max_new_after})", flush=True)
    _texts, _counts, gen_texts = batched_generate(
        tok, model, prompts, args.max_new_before + args.max_new_after, device, batch_size=args.batch_size)
    model = unload(model)
    out = []
    for r, draft in zip(rows, gen_texts):
        named = extract_named_intent(draft, rows[0]["names"])
        out.append({"id": r["id"], "gold": r["label"], "named_intent": named,
                   "label_correct": named == r["label"], "draft": draft})
    return out


def run_options_baseline(rows, args, device):
    """The fair baseline (see module docstring): base E4B-it (no adapter), given the full 77-label list
    via a chat-templated prompt, asked to answer with exactly one label. Greedy, small max_new_tokens
    (answers are one short line). -> list of {id, gold, answer_text, pred_label, label_correct, parsed}."""
    tok, model = _load_base_only(args.model, args.revision, device, args.attn_impl)
    print(f"baseline(options): prompting {len(rows)} rows (no adapter, chat template, "
          f"max_new_tokens={args.options_max_new_tokens})", flush=True)
    # tokenize=True would return a BatchEncoding/dict on transformers>=5 (return_dict default changed),
    # so id_lists would hold dict-like objects instead of plain int lists -- iterating one yields its
    # string keys ('input_ids', 'attention_mask'), which is exactly what trips batched_generate_from_ids's
    # torch.tensor(ids, ...) with "'str' object cannot be interpreted as an integer". Build the prompt
    # TEXT instead (tokenize=False) and tokenize it ourselves so id_lists is always plain python ints,
    # on any transformers version; the chat template text already starts with <bos>, so
    # add_special_tokens=False avoids doubling it.
    id_lists = [
        tok(
            tok.apply_chat_template(
                [{"role": "user", "content": build_options_baseline_prompt(r["state"], r["names"], INSTR)}],
                add_generation_prompt=True, enable_thinking=False, tokenize=False),
            add_special_tokens=False)["input_ids"]
        for r in rows
    ]
    for ids in id_lists:
        assert isinstance(ids, list) and len(ids) > 0 and all(isinstance(x, int) for x in ids), (
            "run_options_baseline: expected a non-empty list of int token ids, got "
            f"{type(ids).__name__} ({ids!r:.200})")
    _texts, _counts, gen_texts = batched_generate_from_ids(
        tok, model, id_lists, args.options_max_new_tokens, device, batch_size=args.batch_size)
    model = unload(model)
    out = []
    for r, g in zip(rows, gen_texts):
        pred, parsed = parse_options_answer(g, r["names"])
        out.append({"id": r["id"], "gold": r["label"], "answer_text": g,
                   "pred_label": pred, "label_correct": pred == r["label"], "parsed": parsed})
    return out


def _load_base_only(model_name, revision, device, attn_impl):
    import torch
    from transformers import AutoModelForCausalLM
    from kev.model import load_tokenizer
    tok = load_tokenizer(model_name, revision)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, dtype=torch.bfloat16, attn_implementation=attn_impl
    ).to(device).eval()
    return tok, model


# ---------------------------------------------------------------------------
# Metrics report
# ---------------------------------------------------------------------------

def summarize(adapter_rows, baseline_rows, conf_floor, forced=False, misses_path=None, options_rows=None):
    lines = ["# Phase 7 CHECK 7 eval results", "",
             f"generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}, n={len(adapter_rows)},"
             f" conf_floor={conf_floor}, force_options={forced}", ""]

    scored = [r for r in adapter_rows if r.get("call_found") and "resolved_label" in r]
    if not adapter_rows:
        lines.append("## Adapter path: n=0 rows -- section skipped (no adapter rows to report on)")
        scored = []
    else:
        n_call = sum(r["call_found"] for r in adapter_rows)
        mode_label = " (forced-options mode)" if forced else ""
        lines.append(f"## Adapter path{mode_label} (n={len(adapter_rows)})")
        if forced:
            lines.append(f"- trigger rate (model emitted <|decide|> before max tokens): "
                         f"{n_call}/{len(adapter_rows)} = {n_call/len(adapter_rows):.4f} -- the only part"
                         f" of a forced call that is still learned/measured")
            lines.append("- call validity: 1.0000 -- by construction (forced mode splices the canonical"
                         " 77-name options list in as token ids once the trigger fires, instead of letting"
                         " the model type it); not a measured rate")
            if misses_path:
                lines.append(f"- rows where the trigger did not fire: see {misses_path}")
            if scored:
                lines.append("")
                lines.append(f"### Metrics below computed over the {len(scored)} TRIGGERED rows only")
        else:
            lines.append(f"- call-emission rate: {n_call}/{len(adapter_rows)} = {n_call/len(adapter_rows):.4f}")
    if scored:
        tm, tm_lo, tm_hi = bootstrap_ci([float(r["template_match"]) for r in scored])
        lc, lc_lo, lc_hi = bootstrap_ci([float(r["label_correct"]) for r in scored])
        corr, corr_lo, corr_hi = bootstrap_ci([float(r["corrupted"]) for r in scored])
        lines.append(f"- template-match rate: {tm:.4f} [{tm_lo:.4f}, {tm_hi:.4f}] (n={len(scored)})")
        lines.append(f"- label-correct rate: {lc:.4f} [{lc_lo:.4f}, {lc_hi:.4f}]")
        lines.append(f"- corrupted-continuation rate (overall): {corr:.4f} [{corr_lo:.4f}, {corr_hi:.4f}]")

        confident = [r["confidence"] >= conf_floor for r in scored]
        wrong = [not r["label_correct"] for r in scored]
        corrupted = [r["corrupted"] for r in scored]
        cw = [c and w for c, w in zip(confident, wrong)]
        cc = [c and not w for c, w in zip(confident, wrong)]
        m_cw, lo_cw, hi_cw, n_cw = conditional_rate(corrupted, cw)
        m_cc, lo_cc, hi_cc, n_cc = conditional_rate(corrupted, cc)
        lines.append("")
        lines.append("### Error propagation (P(corrupted | ...)) -- OLD definition, DEGENERATE: see"
                     " 'Corrupted-continuation definition' below (corrupted == not template_match by"
                     " construction, so this restates template_match, not an independent signal)")
        lines.append(f"- confident + wrong label (conf>={conf_floor}, n={n_cw}): "
                     f"{'n/a' if m_cw is None else f'{m_cw:.4f} [{lo_cw:.4f}, {hi_cw:.4f}]'}")
        lines.append(f"- confident + correct label (conf>={conf_floor}, n={n_cc}): "
                     f"{'n/a' if m_cc is None else f'{m_cc:.4f} [{lo_cc:.4f}, {hi_cc:.4f}]'}")

        # Fixed first-sentence metrics, recomputed from each row's stored continuation text (never from
        # stored booleans) -- see scripts/eval_phase7_metrics.py's score_first_sentence/
        # first_sentence_consistent_with_injected_label docstrings for why the old template_match flags
        # so many genuinely-consistent continuations as corrupted (underscore/hyphen spelling, trailing
        # '?' labels, truncated/extended phrasing, and matching only the whole continuation instead of
        # just the first sentence).
        new_scores = [score_first_sentence(r["continuation"], r["resolved_label"], r.get("names"))
                      for r in scored]
        fsc = [s["first_sentence_consistent_with_injected_label"] for s in new_scores]
        off_t = [s["off_template_first_sentence"] for s in new_scores]
        cs = [s["clean_stop"] for s in new_scores]
        stray = [s["stray_tags"] for s in new_scores]
        loop = [s["loop"] for s in new_scores]
        otl_vals = [s["first_sentence_names_other_label"] for s in new_scores
                   if "first_sentence_names_other_label" in s]
        n_names_avail = len(otl_vals)

        fsc_m, fsc_lo, fsc_hi = bootstrap_ci([float(x) for x in fsc])
        off_m, off_lo, off_hi = bootstrap_ci([float(x) for x in off_t])
        cs_m, cs_lo, cs_hi = bootstrap_ci([float(x) for x in cs])
        stray_m, stray_lo, stray_hi = bootstrap_ci([float(x) for x in stray])
        loop_m, loop_lo, loop_hi = bootstrap_ci([float(x) for x in loop])
        lines.append("")
        lines.append("### Fixed first-sentence metrics (new -- old vs new side by side)")
        lines.append(f"- OLD template-match rate (whole continuation, exact string match, no"
                     f" underscore/hyphen normalization): {tm:.4f} [{tm_lo:.4f}, {tm_hi:.4f}]"
                     f" (n={len(scored)})")
        lines.append(f"- NEW first_sentence_consistent_with_injected_label rate (first sentence only,"
                     f" normalized phrase, word-level prefix tolerant either direction):"
                     f" {fsc_m:.4f} [{fsc_lo:.4f}, {fsc_hi:.4f}]")
        lines.append(f"- off_template_first_sentence rate (first sentence has no 'This looks like X.'"
                     f" shape at all): {off_m:.4f} [{off_lo:.4f}, {off_hi:.4f}]")
        if n_names_avail:
            otl_m, otl_lo, otl_hi = bootstrap_ci([float(x) for x in otl_vals])
            lines.append(f"- first_sentence_names_other_label rate (on-template but names a DIFFERENT"
                         f" known banking label than the one injected -- the genuine override/ignore"
                         f" failure; n={n_names_avail}/{len(scored)} rows with a stored/derivable label"
                         f" list): {otl_m:.4f} [{otl_lo:.4f}, {otl_hi:.4f}]")
        else:
            lines.append("- first_sentence_names_other_label: unavailable (no row has a stored 'names'"
                         " field or a matching heldout entry -- re-run phase 3, or pass --heldout to"
                         " --report-only/--inspect so it can be backfilled)")
        lines.append(f"- clean_stop rate (no repeated-sentence loop / stray control tag after the first"
                     f" sentence; n={len(scored)}): {cs_m:.4f} [{cs_lo:.4f}, {cs_hi:.4f}] -- NOTE: training"
                     f" targets carried no end-of-text marker after the template continuation, so plain"
                     f" rambling after the first sentence is EXPECTED generation behavior, not a"
                     f" label-propagation error; this only flags loops/stray markup.")
        lines.append(f"  - of which stray_tags rate (a leaked '<|...|>'-style control tag after the"
                     f" first sentence): {stray_m:.4f} [{stray_lo:.4f}, {stray_hi:.4f}]")
        lines.append(f"  - of which loop rate (a repeated sentence, case-folded, appearing >=2 times):"
                     f" {loop_m:.4f} [{loop_lo:.4f}, {loop_hi:.4f}] -- reported separately from stray_tags"
                     f" so it's visible which failure mode dominates (clean_stop fails if either does).")

        inconsistent = [not s["first_sentence_consistent_with_injected_label"] for s in new_scores]
        m_iw, lo_iw, hi_iw, n_iw = conditional_rate(inconsistent, wrong)
        m_ic, lo_ic, hi_ic, n_ic = conditional_rate(inconsistent, [not w for w in wrong])
        lines.append("")
        lines.append("### Error propagation (fixed, non-degenerate): P(first sentence"
                     " ignores/overrides the injected label | injected label wrong vs right)")
        lines.append(f"- injected label WRONG (n={n_iw}): "
                     f"{'n/a' if m_iw is None else f'{m_iw:.4f} [{lo_iw:.4f}, {hi_iw:.4f}]'}")
        lines.append(f"- injected label RIGHT (n={n_ic}): "
                     f"{'n/a' if m_ic is None else f'{m_ic:.4f} [{lo_ic:.4f}, {hi_ic:.4f}]'}")

        covered = [r for r in scored if r["confidence"] >= conf_floor]
        if covered:
            cov_acc, cov_lo, cov_hi = bootstrap_ci([float(r["label_correct"]) for r in covered])
            lines.append("")
            lines.append(f"### Confidence-floor coverage: {len(covered)}/{len(scored)} = "
                         f"{len(covered)/len(scored):.4f}; accuracy on covered: {cov_acc:.4f} "
                         f"[{cov_lo:.4f}, {cov_hi:.4f}]")

    if baseline_rows:
        named = sum(r["named_intent"] is not None for r in baseline_rows)
        correct_among_named = [r["label_correct"] for r in baseline_rows if r["named_intent"] is not None]
        lines.append("")
        lines.append(f"## Free-draft naming baseline (n={len(baseline_rows)}, base E4B-it, no adapter) --"
                     " NOT COMPARABLE: the model is never shown the 77 label names, so it almost never"
                     " names one in-vocabulary; see module docstring")
        lines.append(f"- named-some-intent rate: {named}/{len(baseline_rows)} = {named/len(baseline_rows):.4f}")
        if correct_among_named:
            bc, bc_lo, bc_hi = bootstrap_ci([float(x) for x in correct_among_named])
            lines.append(f"- label-correct rate among those that named an intent: {bc:.4f} "
                         f"[{bc_lo:.4f}, {bc_hi:.4f}] (n={len(correct_among_named)})")
        all_correct = [float(r["label_correct"]) for r in baseline_rows]
        bc_all, bc_all_lo, bc_all_hi = bootstrap_ci(all_correct)
        lines.append(f"- label-correct rate over all rows (unnamed counts as wrong): {bc_all:.4f} "
                     f"[{bc_all_lo:.4f}, {bc_all_hi:.4f}]")
        lines.append("")
        lines.append("NOTE: the baseline has no fixed template, so it is not scored for template-match;"
                     " see reports/15-phase7-eval-plan.md for why this makes the two paths only "
                     "partially comparable.")

    if options_rows is not None:
        n_opt = len(options_rows)
        lines.append("")
        lines.append(f"## Chat-with-options baseline (n={n_opt}, base E4B-it, no adapter, shown the full"
                     " 77-label list -- the fair, decide-vs-chat comparison; see module docstring)")
        if n_opt:
            n_parsed = sum(r["parsed"] for r in options_rows)
            lines.append(f"- parsed rate: {n_parsed}/{n_opt} = {n_parsed/n_opt:.4f} (unparseable rows"
                         " count as wrong in the rate below, not excluded)")
            oc, oc_lo, oc_hi = bootstrap_ci([float(r["label_correct"]) for r in options_rows])
            lines.append(f"- label-correct rate: {oc:.4f} [{oc_lo:.4f}, {oc_hi:.4f}]")

    return "\n".join(lines) + "\n"


def _paired_label_correct_lines(adapter_rows, other_rows, other_name):
    """Paired label-correct comparison (adapter minus `other_rows`, 2000-resample paired bootstrap, seed
    0) over ids present in both -- shared by the free-draft and options-baseline comparisons in
    build_combined_report. -> list of report lines."""
    lines = []
    a_by_id = {r["id"]: r for r in adapter_rows}
    b_by_id = {r["id"]: r for r in other_rows}
    paired_ids = sorted(set(a_by_id) & set(b_by_id))
    if not adapter_rows or not other_rows:
        lines.append(f"- paired comparison vs {other_name} skipped: need both adapter rows and"
                     f" {other_name} rows non-empty.")
        return lines
    if not paired_ids:
        lines.append(f"- paired comparison vs {other_name} skipped: 0 ids in common between the"
                     f" {len(adapter_rows)} adapter rows and {len(other_rows)} {other_name} rows"
                     " (different held-out sets?).")
        return lines
    a_vals = [float(a_by_id[i].get("label_correct", False)) for i in paired_ids]
    b_vals = [float(b_by_id[i].get("label_correct", False)) for i in paired_ids]
    diff, lo, hi = paired_bootstrap_diff(a_vals, b_vals, n_boot=2000, seed=0)
    lines.append(f"- paired label-correct rate, adapter minus {other_name} (n={len(paired_ids)} rows"
                 f" present in both files; 2000-resample paired bootstrap, seed 0): "
                 f"{diff:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    return lines


def build_combined_report(adapter_rows, baseline_rows, conf_floor, forced=False, options_rows=None):
    """--report-only: combined CHECK 7 report over two already-generated row files, no model/GPU/dataset
    load (reads adapter_rows/baseline_rows straight off disk). Calls summarize() on both lists TOGETHER
    for the per-path sections (template-match, confidence-floor coverage, etc. -- all adapter-only, per
    summarize's own NOTE), then appends the CHECK 7 side-by-side comparison: a paired bootstrap CI for the
    label-correct-rate difference (adapter minus baseline) over ids present in both inputs, the baseline's
    unscored-row count, and the corrupted-continuation definition (plus one additional, non-degenerate
    propagation breakdown -- see is_corrupted's docstring for why plain 'corrupted' is degenerate here)."""
    report = summarize(adapter_rows, baseline_rows, conf_floor, forced=forced, options_rows=options_rows)
    lines = [report.rstrip("\n"), "",
             "## CHECK 7 comparison: adapter vs free-draft naming baseline (NOT COMPARABLE -- see above)",
             ""]
    lines += _paired_label_correct_lines(adapter_rows, baseline_rows, "free-draft baseline")
    if adapter_rows and baseline_rows:
        lines.append("- only label-correctness is paired/compared here. The adapter is additionally"
                     " scored against a fixed call+template (template-match, confidence-floor coverage,"
                     " above); the free baseline has no fixed template -- run_free_baseline scores it via"
                     " extract_named_intent, which just substring-matches the longest banking-intent"
                     " phrase (case-insensitive) anywhere in the free-drafted text -- so template-match /"
                     " confidence-floor coverage are adapter-only and are NOT comparable to the baseline."
                     " This baseline never sees the 77 label names, so this diff is dominated by that"
                     " unfairness, not by the adapter+pointer-head being better -- use the options-"
                     "baseline comparison below for the honest number.")

    if baseline_rows:
        n_unscored = sum(r.get("named_intent") is None for r in baseline_rows)
        lines.append(f"- unscored baseline rows (extract_named_intent found no intent phrase in the"
                     f" draft; already counted as incorrect in the baseline's label-correct rate above,"
                     f" reported separately here): {n_unscored}/{len(baseline_rows)}")

    if options_rows is not None:
        lines.append("")
        lines.append("## CHECK 7 comparison: adapter vs chat-with-options baseline (the fair comparison)")
        lines.append("")
        lines += _paired_label_correct_lines(adapter_rows, options_rows, "options-baseline")
        lines.append("- decide-vs-chat gap: both paths are given exactly the same information (the"
                     " customer message + the full 77-label list); the adapter resolves it via the"
                     " inline <|decide|> call + the trained decision model's pointer-head readout, the"
                     " options-baseline resolves it by having the base chat model answer directly. This"
                     " diff is the answer to 'does the decide-call mechanism beat just asking the base"
                     " model, chat-style, with the same options' -- unlike the free-draft comparison"
                     " above, neither side has an information advantage here.")
        if options_rows:
            n_unparsed = sum(not r["parsed"] for r in options_rows)
            lines.append(f"- unparseable options-baseline rows (no listed label found in the answer;"
                         f" already counted as incorrect above, reported separately here):"
                         f" {n_unparsed}/{len(options_rows)}")

    scored = [r for r in adapter_rows if r.get("call_found") and "resolved_label" in r]
    lines.append("")
    lines.append("### Corrupted-continuation definition")
    lines.append("- `corrupted` is defined as `not template_match` (scripts/eval_phase7_metrics.py:"
                 " is_corrupted/template_match) -- 'corrupted' and 'not template-matched' are IDENTICAL"
                 " by construction, not two independent checks (run1: corrupted 8.42% =="
                 " 1 - template_match 91.58%, confirming this). The 'P(corrupted | ...)' error-propagation"
                 " numbers above are therefore a restatement of template-match conditioned on"
                 " confidence/correctness, not an independent propagation signal -- degenerate as"
                 " originally specified.")
    if scored:
        corrupted_rows = [r for r in scored if r.get("corrupted")]
        off_template = sum(template_phrase(r["continuation"]) is None for r in corrupted_rows)
        mentions_diff = len(corrupted_rows) - off_template
        lines.append(f"- additional, non-degenerate breakdown of those {len(corrupted_rows)} corrupted"
                     " rows, using only the continuation text already in each row (no new data): rows"
                     " that are off-template entirely (no 'This looks like X. Let me help...' shape at"
                     f" all): {off_template}/{len(corrupted_rows)}; rows that ARE on-template but name a"
                     f" DIFFERENT intent than the one injected: {mentions_diff}/{len(corrupted_rows)}.")
    else:
        lines.append("- no scored adapter rows available to break down further.")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# --inspect: no model/GPU -- row-level qualitative breakdown of an already-generated
# adapter-rows file (misses + corrupted rows), reading from disk only.
# ---------------------------------------------------------------------------

def _flag_reason(continuation, injected_label):
    """-> (detail, category) short auto-explanation for why `continuation` is NOT
    first_sentence_consistent_with_injected_label. `detail` is the per-row message; `category` is the
    coarser bucket used for the (d) breakdown table (there are only two ways this metric can fail: the
    first sentence never had the "This looks like X." shape at all, or it had the shape but named a
    phrase inconsistent with the injected label)."""
    phrase = first_sentence_phrase(continuation)
    if phrase is None:
        fs = first_sentence(continuation_clean(continuation))
        return (f"no 'This looks like' within 200 chars; first sentence is {fs[:120]!r}",
                "off-template (no 'This looks like' match)")
    expected = label_phrase(injected_label)
    return f"phrase differs: {phrase!r} vs {expected!r}", "on-template, phrase mismatch"


def build_inspect_report(adapter_rows, raw_by_id, heldout_by_id):
    """-> markdown string. `raw_by_id`: id -> {"raw_text":..., "triggered": bool|None} from a
    --phase1-raw / dump_raw file, or {} if none was given (every row then classifies as 'unknown', see
    classify_miss_row). `heldout_by_id`: id -> {"state":..., "label":...} from the heldout file, or {} if
    not found (state then reported as 'n/a')."""
    lines = ["# Phase 7 CHECK 7 inspect report", "",
             f"generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}, n={len(adapter_rows)}",
             ""]

    misses = [r for r in adapter_rows if not r.get("call_found")]
    lines.append(f"## (a) Rows where no call was emitted (n={len(misses)})")
    lines.append("")
    miss_categories = {}
    for r in misses:
        raw = raw_by_id.get(r["id"])
        raw_text = raw.get("raw_text") if raw else None
        triggered = raw.get("triggered") if raw else None
        category = classify_miss_row(raw_text, triggered)
        miss_categories[category] = miss_categories.get(category, 0) + 1
        preview = raw_text[:300] if raw_text is not None else "n/a (no --phase1-raw row for this id)"
        lines.append(f"- id={r['id']} category={category} raw(first 300 chars)={preview!r}")
    if not misses:
        lines.append("(none)")

    scored = [r for r in adapter_rows if r.get("call_found") and "corrupted" in r]
    corrupted_rows = [r for r in scored if r["corrupted"]]
    lines.append("")
    lines.append(f"## (b) Corrupted rows (n={len(corrupted_rows)})")
    lines.append("")
    corrupted_categories = {}
    for r in corrupted_rows:
        hr = heldout_by_id.get(r["id"], {})
        state = hr.get("state", "n/a (id not found in heldout file)")
        gold = r.get("gold", hr.get("label", "n/a"))
        injected = r.get("resolved_label")
        injected_correct = r.get("label_correct")
        continuation = r.get("continuation", "")
        phrase = template_phrase(continuation)
        category = classify_corrupted_row(continuation, injected_correct)
        corrupted_categories[category] = corrupted_categories.get(category, 0) + 1
        new_metrics = score_first_sentence(continuation, injected, r.get("names") or hr.get("names"))
        lines.append(f"- id={r['id']} user_message={state!r} injected_label={injected!r} gold={gold!r}"
                     f" injected_label_correct={injected_correct} confidence={r.get('confidence')}"
                     f" continuation(first 250 chars)={continuation[:250]!r}"
                     f" continuation_names={phrase!r} category={category}"
                     f" | NEW: first_sentence_consistent="
                     f"{new_metrics['first_sentence_consistent_with_injected_label']}"
                     f" off_template_first_sentence={new_metrics['off_template_first_sentence']}"
                     f" first_sentence_names_other_label="
                     f"{new_metrics.get('first_sentence_names_other_label', 'n/a')}"
                     f" clean_stop={new_metrics['clean_stop']}")
    if not corrupted_rows:
        lines.append("(none)")

    lines.append("")
    lines.append("## (c) Category count summaries")
    lines.append("")
    lines.append("### (a) miss categories")
    lines.append("")
    lines.append("| category | n |")
    lines.append("|---|---|")
    for cat, n in sorted(miss_categories.items()):
        lines.append(f"| {cat} | {n} |")
    if not miss_categories:
        lines.append("| (none) | 0 |")
    lines.append("")
    lines.append("### (b) corrupted-row categories")
    lines.append("")
    lines.append("| category | n |")
    lines.append("|---|---|")
    for cat, n in sorted(corrupted_categories.items()):
        lines.append(f"| {cat} | {n} |")
    if not corrupted_categories:
        lines.append("| (none) | 0 |")

    # (d) every row flagged by the NEW first-sentence metric, not just the OLD-corrupted rows listed in
    # (b) above -- the two metrics disagree in both directions (see first_sentence_consistent_with_
    # injected_label's docstring), so a row can be flagged by one and not the other.
    new_flagged = []
    for r in scored:
        hr = heldout_by_id.get(r["id"], {})
        names = r.get("names") or hr.get("names")
        continuation = r.get("continuation", "")
        injected = r.get("resolved_label")
        nm = score_first_sentence(continuation, injected, names)
        if not nm["first_sentence_consistent_with_injected_label"]:
            new_flagged.append((r, hr, nm))

    old_flagged_ids = {r["id"] for r in corrupted_rows}
    new_flagged_ids = {r["id"] for r, _hr, _nm in new_flagged}
    both_ids = old_flagged_ids & new_flagged_ids
    only_new_ids = new_flagged_ids - old_flagged_ids
    only_old_ids = old_flagged_ids - new_flagged_ids

    lines.append("")
    lines.append(f"## (d) Rows flagged by the NEW first-sentence metric (n={len(new_flagged)})")
    lines.append("")
    lines.append("Every scored row where first_sentence_consistent_with_injected_label is False -- not"
                 " just the OLD-corrupted rows in (b) above. Ordered: rows NOT in (b)'s old-corrupted"
                 " list first, then rows also flagged by the old metric.")
    lines.append("")
    reason_counts = {}
    for r, hr, nm in sorted(new_flagged, key=lambda t: t[0]["id"] in old_flagged_ids):
        state = hr.get("state", "n/a (id not found in heldout file)")
        gold = r.get("gold", hr.get("label", "n/a"))
        injected = r.get("resolved_label")
        injected_correct = r.get("label_correct")
        continuation = r.get("continuation", "")
        cleaned = continuation_clean(continuation)
        fsent_phrase = first_sentence_phrase(continuation)
        old_flag = r["id"] in old_flagged_ids
        detail, category = _flag_reason(continuation, injected)
        if r["id"] in only_new_ids:
            reason_counts[category] = reason_counts.get(category, 0) + 1
        lines.append(f"- id={r['id']} user_message={state!r} injected_label={injected!r} gold={gold!r}"
                     f" injected_label_correct={injected_correct}"
                     f" continuation_clean(first 300 chars)={cleaned[:300]!r}"
                     f" first_sentence_phrase={fsent_phrase!r}"
                     f" off_template_first_sentence={nm['off_template_first_sentence']}"
                     f" first_sentence_names_other_label="
                     f"{nm.get('first_sentence_names_other_label', 'n/a')}"
                     f" old_metric_flagged={'yes' if old_flag else 'no'}"
                     f" reason={detail}")
    if not new_flagged:
        lines.append("(none)")

    lines.append("")
    lines.append("### (d) summary: new vs old metric agreement")
    lines.append("")
    lines.append("| category | n |")
    lines.append("|---|---|")
    lines.append(f"| flagged by both | {len(both_ids)} |")
    lines.append(f"| flagged only by new | {len(only_new_ids)} |")
    lines.append(f"| flagged only by old | {len(only_old_ids)} |")
    lines.append("")
    lines.append("### (d) only-by-new rows: breakdown by auto-reason")
    lines.append("")
    lines.append("| reason | n |")
    lines.append("|---|---|")
    for cat, n in sorted(reason_counts.items()):
        lines.append(f"| {cat} | {n} |")
    if not reason_counts:
        lines.append("| (none) | 0 |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# --dry-parse: offline unit tests on canned strings (no model/dataset/torch)
# ---------------------------------------------------------------------------

def _dry_parse_tests():
    failures = []

    def check(name, cond):
        if cond:
            print(f"  ok   {name}")
        else:
            failures.append(name)
            print(f"  FAIL {name}")

    # --baseline options parsing (parse_options_answer) and --inspect categorization (classify_miss_row /
    # classify_corrupted_row): pure-Python, no torch/model needed.
    opt_names = ["card_not_working", "age_limit", "card_arrival"]
    check("parse_options_answer: [answer] marker",
         parse_options_answer("I think [answer] card_not_working", opt_names) == ("card_not_working", True))
    check("parse_options_answer: bracket format `[card_arrival]`",
         parse_options_answer("[card_arrival]", opt_names) == ("card_arrival", True))
    check("parse_options_answer: bare label with trailing punctuation",
         parse_options_answer("age_limit.", opt_names) == ("age_limit", True))
    check("parse_options_answer: quoted label on its own line",
         parse_options_answer("Sure.\n\"card_not_working\"\n", opt_names) == ("card_not_working", True))
    check("parse_options_answer: label embedded in a sentence (substring fallback)",
         parse_options_answer("This is clearly a card_arrival issue, let me check.", opt_names)
         == ("card_arrival", True))
    check("parse_options_answer: garbage -> unparseable",
         parse_options_answer("I'm not sure, could you clarify?", opt_names) == (None, False))

    check("classify_miss_row: triggered=True -> trigger-fired-call-collapsed",
         classify_miss_row("anything", triggered=True) == "trigger-fired-call-collapsed")
    check("classify_miss_row: triggered=False -> trigger-missed",
         classify_miss_row("anything", triggered=False) == "trigger-missed")
    check("classify_miss_row: raw_text has <|decide|> -> trigger-fired-call-collapsed",
         classify_miss_row("foo <|decide|> bar") == "trigger-fired-call-collapsed")
    check("classify_miss_row: raw_text without <|decide|> -> trigger-missed",
         classify_miss_row("I think this is about cards.") == "trigger-missed")
    check("classify_miss_row: raw_text None -> unknown", classify_miss_row(None) == "unknown")

    check("classify_corrupted_row: off-template",
         classify_corrupted_row("I see, let me look into your account.", True) == "off-template")
    check("classify_corrupted_row: on-template, injected label correct",
         classify_corrupted_row("This looks like age limit. Let me help with that right away.", True)
         == "names-different-intent, injected label RIGHT")
    check("classify_corrupted_row: on-template, injected label wrong",
         classify_corrupted_row("This looks like age limit. Let me help with that right away.", False)
         == "names-different-intent, injected label WRONG")

    names = ["card_not_working", "age_limit", "card_arrival"]
    call_text = ("Customer: My card isn't working\nAgent: Thanks for reaching out. "
                "<|decide|>Which banking intent best describes this customer message? "
                "[card_not_working, age_limit, card_arrival]<|result|>")
    call = parse_decide_call(call_text)
    check("parse_decide_call finds options", call is not None and call["options"] == names)
    check("parse_decide_call returns None on absent call", parse_decide_call("no call here") is None)
    check("parse_decide_call returns None on empty options", parse_decide_call("<|decide|>x [ ]<|result|>") is None)

    injected = call_text + "card_not_working<|/result|> This looks like card not working. Let me help with that right away."
    check("parse_injected_result extracts label", parse_injected_result(injected) == "card_not_working")
    check("parse_injected_result None when absent", parse_injected_result(call_text) is None)

    good_cont = "This looks like card not working. Let me help with that right away."
    bad_label_cont = "This looks like age limit. Let me help with that right away."
    off_template_cont = "I see, let me look into your account."
    check("template_match true on exact shape+label", template_match(good_cont, "card_not_working"))
    check("template_match false on right shape wrong label", not template_match(bad_label_cont, "card_not_working"))
    check("template_match false off-template", not template_match(off_template_cont, "card_not_working"))

    check("is_corrupted false when template matches", not is_corrupted(good_cont, "card_not_working"))
    check("is_corrupted true when label mismatches", is_corrupted(bad_label_cont, "card_not_working"))
    check("is_corrupted true when off-template", is_corrupted(off_template_cont, "card_not_working"))

    check("extract_named_intent exact phrase",
         extract_named_intent("this is about card not working today", names) == "card_not_working")
    check("extract_named_intent prefers longer/specific over shorter substring",
         extract_named_intent("your card arrival is delayed", names) == "card_arrival")
    check("extract_named_intent None when nothing matches",
         extract_named_intent("totally unrelated text", names) is None)

    # build_inspect_report end-to-end on synthetic rows (no model/dataset/torch).
    synth_adapter_rows = [
        {"id": "r1", "gold": "age_limit", "call_found": False},
        {"id": "r2", "gold": "card_arrival", "call_found": True, "resolved_label": "card_arrival",
         "confidence": 0.9, "label_correct": True, "continuation": "I see, let me look into your account.",
         "corrupted": True, "template_match": False},
        {"id": "r3", "gold": "age_limit", "call_found": True, "resolved_label": "card_arrival",
         "confidence": 0.7, "label_correct": False,
         "continuation": "This looks like card arrival. Let me help with that right away.",
         "corrupted": False, "template_match": True},
        # r4: only-NEW-flagged. A long (>200-char) filler prefix before the template means
        # TEMPLATE_RE.search (unbounded, old template_match) still finds the exact-match template anywhere
        # in the continuation -> old says template_match True / not corrupted; but continuation_clean only
        # looks within the first 200 chars for "this looks like", so it can't strip the filler -> the
        # NEW metric's first_sentence_phrase sees the filler as the "first sentence" (no "This looks
        # like" shape) -> off_template_first_sentence True -> flagged, even though old passed it.
        {"id": "r4", "gold": "card_arrival", "call_found": True, "resolved_label": "card_arrival",
         "confidence": 0.8, "label_correct": True,
         "continuation": "x" * 210 + " This looks like card arrival. Let me help with that right away.",
         "corrupted": False, "template_match": True},
        # r5: only-OLD-flagged. A boundary-drifted stored continuation (leading 'ult|> ' fragment, see
        # continuation_clean's docstring) with a hyphenated phrase ('verify top-up') that old
        # template_match's exact string compare rejects against label_phrase's underscore-to-space form
        # ('verify top up') -> old corrupted True; the NEW metric's continuation_clean/normalization
        # strips the fragment and treats the hyphen as a word boundary, so it's consistent -> not flagged.
        {"id": "r5", "gold": "verify_top_up", "call_found": True, "resolved_label": "verify_top_up",
         "confidence": 0.75, "label_correct": True,
         "continuation": "ult|> This looks like verify top-up. Let me help with that right away.",
         "corrupted": True, "template_match": False},
    ]
    synth_raw_by_id = {"r1": {"raw_text": "I'm not sure what this is about.", "triggered": None}}
    synth_heldout_by_id = {"r1": {"state": "my card never arrived", "label": "age_limit"},
                           "r2": {"state": "how old do I need to be", "label": "card_arrival"}}
    inspect_md = build_inspect_report(synth_adapter_rows, synth_raw_by_id, synth_heldout_by_id)
    check("build_inspect_report: miss row classified trigger-missed",
         "id=r1 category=trigger-missed" in inspect_md)
    check("build_inspect_report: corrupted row (off-template) listed",
         "id=r2" in inspect_md and "category=off-template" in inspect_md)
    check("build_inspect_report: non-corrupted scored row excluded from (b)",
         "id=r3" not in inspect_md)
    check("build_inspect_report: summary table counts miss category",
         "| trigger-missed | 1 |" in inspect_md)
    check("build_inspect_report: summary table counts corrupted category",
         "| off-template | 1 |" in inspect_md)

    # (d) new-metric section: r2 is flagged by both (old-corrupted AND new-inconsistent), r4 is
    # only-new (old template_match passed, new metric doesn't), r5 is only-old (old template_match
    # failed, new metric is fine with it), r3 is flagged by neither.
    check("build_inspect_report (d): both-flagged row listed with old_metric_flagged=yes",
         "id=r2" in inspect_md and "old_metric_flagged=yes" in inspect_md)
    check("build_inspect_report (d): only-new row listed with old_metric_flagged=no",
         re.search(r"id=r4\b.*old_metric_flagged=no", inspect_md) is not None)
    check("build_inspect_report (d): only-new row's reason is off-template",
         re.search(r"id=r4\b.*reason=no 'This looks like'", inspect_md) is not None)
    check("build_inspect_report (d): only-old row NOT in the (d) listing (new metric is fine with it)",
         not re.search(r"id=r5\b.*old_metric_flagged", inspect_md))
    check("build_inspect_report (d): neither-flagged row (r3) excluded from the (d) listing",
         not re.search(r"id=r3\b.*old_metric_flagged", inspect_md))
    section_d = inspect_md[inspect_md.index("## (d)"):]
    check("build_inspect_report (d): r4 (not old-flagged) ordered before r2 (old-flagged)",
         section_d.index("id=r4 ") < section_d.index("id=r2 "))
    check("build_inspect_report (d): summary table counts (both=1, only-new=1, only-old=1)",
         "| flagged by both | 1 |" in inspect_md and "| flagged only by new | 1 |" in inspect_md
         and "| flagged only by old | 1 |" in inspect_md)
    check("build_inspect_report (d): only-by-new reason breakdown counts the off-template category",
         "| off-template (no 'This looks like' match) | 1 |" in inspect_md)

    mean, lo, hi = bootstrap_ci([1.0] * 10, n_boot=200, seed=0)
    check("bootstrap_ci degenerate all-ones -> mean 1, CI [1,1]", mean == 1.0 and lo == 1.0 and hi == 1.0)
    mean0, lo0, hi0 = bootstrap_ci([], n_boot=200, seed=0)
    check("bootstrap_ci empty -> None triple", (mean0, lo0, hi0) == (None, None, None))
    mean2, lo2, hi2 = bootstrap_ci([1.0, 0.0, 1.0, 0.0], n_boot=500, seed=1)
    check("bootstrap_ci mixed mean is 0.5", mean2 == 0.5)
    check("bootstrap_ci mixed CI brackets 0.5", lo2 <= 0.5 <= hi2)

    m, lo3, hi3, n = conditional_rate([True, False, True, True], [True, True, False, False], n_boot=200, seed=0)
    check("conditional_rate restricts to condition subset (n=2)", n == 2)
    check("conditional_rate mean over subset", m == 0.5)
    m_none, lo_none, hi_none, n_none = conditional_rate([True], [False], n_boot=10, seed=0)
    check("conditional_rate n=0 -> None triple", (m_none, lo_none, hi_none) == (None, None, None) and n_none == 0)

    check("label_phrase underscores to spaces", label_phrase("card_not_working") == "card not working")

    # first_sentence_consistent_with_injected_label (the fixed template_match): must-pass examples (a)-(e)
    # from run1's inspect report, each a genuine false positive under the OLD template_match, and the
    # must-fail "genuinely bad" examples that must stay flagged under the fix.
    banking_names = opt_names + [
        "verify_top_up", "activate_my_card", "reverted_card_payment?", "lost_or_stolen_card",
        "supported_cards_and_currencies",
    ]
    cont_a = ("This looks like verify top-up. Let me help with that right away. <|/verify_top-up|> This"
              " looks like verify top-up. Let me help with that right away.")
    check("(a) hyphenated template phrase consistent with underscored injected label",
         first_sentence_consistent_with_injected_label(cont_a, "verify_top_up"))

    cont_b = (" This looks like activate_my_card. Let me help with that right away.\nAgent: Let me know"
              " if you need anything else. <|/agent|>\n<|card_api|>...This looks like everything(|card")
    check("(b) literal underscored label in first sentence consistent (not the trailing junk match)",
         first_sentence_consistent_with_injected_label(cont_b, "activate_my_card"))

    cont_c = "This looks like reverted card payment. Let me help with that right away."
    check("(c) label with trailing '?' consistent after stripping",
         first_sentence_consistent_with_injected_label(cont_c, "reverted_card_payment?"))

    cont_d = "This looks like lost or stolen. Let me help with that right away."
    check("(d) truncated phrase (missing trailing word) consistent with injected label",
         first_sentence_consistent_with_injected_label(cont_d, "lost_or_stolen_card"))

    cont_e = "This looks like activate_my_card is what you want. Let me help with that right away."
    check("(e) phrase with trailing extra words consistent with injected label",
         first_sentence_consistent_with_injected_label(cont_e, "activate_my_card"))

    bad_1 = "This age limit message isn't what I'm looking for."
    check("genuinely bad: off-template sentence (no 'This looks like' shape) stays inconsistent",
         not first_sentence_consistent_with_injected_label(bad_1, "age_limit"))
    check("genuinely bad: off-template sentence flagged off_template_first_sentence",
         off_template_first_sentence(bad_1))

    bad_2 = "This doesn't seem to be the one. Let me try again."
    check("genuinely bad: non-committal drift stays inconsistent",
         not first_sentence_consistent_with_injected_label(bad_2, "card_arrival"))

    bad_3 = "This looks like the card payment result. Let me help with that right away."
    check("genuinely bad: on-template but wrong phrase stays inconsistent",
         not first_sentence_consistent_with_injected_label(bad_3, "supported_cards_and_currencies"))
    check("genuinely bad: on-template wrong phrase is NOT off_template_first_sentence",
         not off_template_first_sentence(bad_3))

    check("first_sentence_names_other_label: on-template but names a different KNOWN label",
         first_sentence_names_other_label(
             "This looks like card arrival. Let me help with that right away.",
             "age_limit", banking_names))
    check("first_sentence_names_other_label: false when consistent with the injected label",
         not first_sentence_names_other_label(cont_a, "verify_top_up", banking_names))
    check("first_sentence_names_other_label: false when off-template (no label named at all)",
         not first_sentence_names_other_label(bad_1, "age_limit", banking_names))

    check("clean_stop: false on stray control tag after the first sentence", not clean_stop(cont_a))
    check("clean_stop: true on a clean single-template continuation",
         clean_stop("This looks like card not working. Let me help with that right away."))
    check("clean_stop: false on a looping repeated sentence",
         not clean_stop("This looks like age limit. Let me help with that right away. This looks like"
                        " age limit. Let me help with that right away."))

    scores_a = score_first_sentence(cont_a, "verify_top_up", banking_names)
    check("score_first_sentence: bundles all six new metrics",
         set(scores_a) == {"first_sentence_consistent_with_injected_label", "off_template_first_sentence",
                           "clean_stop", "first_sentence_names_other_label", "stray_tags", "loop"})
    check("score_first_sentence: omits first_sentence_names_other_label when names=None",
         "first_sentence_names_other_label" not in score_first_sentence(cont_a, "verify_top_up", None))

    # continuation_clean / the run1combined3 first-sentence-metrics bug (see continuation_clean's
    # docstring for root cause): must-be-CONSISTENT-with-injected-label real rows (stored `continuation`
    # starting mid-call-text due to the char-offset slicing bug, now fixed at the source in
    # run_adapter_phase3/batched_generate -- these are regression tests against that class of bug
    # recurring in the ROBUST SCORER for rows stored by earlier, already-run evals) and
    # must-be-FLAGGED (genuinely off-template/inconsistent, not a boundary artifact) cases, exact strings
    # from run1combined3's inspect report.
    consistent_cases = [
        ("verify_top_up", "ult|> This looks like verify top-up. Let me help with that right away."
         " <|/verify_top-up|> This looks like verify top-up. Let me help with that right away."),
        ("activate_my_card", "ge_rate_for_cash_withdrawal]<|result|>activate_my_card<|/result|> This looks"
         " like activate_my_card. Let me help with that right away.\nAgent: Let me know if you need"
         " anything else."),
        ("verify_top_up", "p_up<|/result|> This looks like verify top-up. Let me help with that right"
         " away. <|/verify_top-up|> This is all there is to it."),
        ("get_disposable_virtual_card", " This looks like get_disposable_virtual_card. Let me help with"
         " that right away. <|/result|> trailing text."),
        ("verify_top_up", "|/result|> This looks like verify top-up. Let me help with that right away."
         " Which of these options would you like?"),
        ("activate_my_card", "|/result|> This looks like activate_my_card is what you want. Let me help"
         " with that right away."),
        ("reverted_card_payment?", "ult|> This looks like reverted card payment. Let me help with that"
         " right away."),
        ("automatic_top_up", "hdrawal]<|result|>automatic_top_up<|/result|> This looks like automatic"
         " top-up. Let me help with that right away."),
        ("verify_top_up", "_for_cash_withdrawal]<|result|>verify_top_up<|/result|> This looks like verify"
         " top-up. Let me help with that right away."),
        ("lost_or_stolen_card", " This looks like lost or stolen. Let me help with that right away."
         " <|/result|> trailing text."),
        ("automatic_top_up", "p_up<|/result|> This looks like automatic top-up. Let me help with that"
         " right away."),
    ]
    flagged_cases = [
        ("age_limit", "ult|>age_limit<|/result|> This age limit message isn't what I'm looking for."
         " |age_limit|> trailing text."),
        ("wrong_amount_of_cash_received", "ult|> This doesn't seem to be the one. Let me try again."
         "<|/result|> This is not what I wanted."),
        ("age_limit", "...]<|result|>age_limit<|/result|> This age limit message isn't for me. Let me"
         " know if you need anything else."),
        ("supported_cards_and_currencies", " This looks like the card payment result. Let me help with"
         " that right away. <|/card_payment_result|> trailing text."),
        ("unable_to_verify_identity", " This <|card|><|/card|> is not working. <|/card|>}|\nAgent: ..."),
    ]
    n_consistent_ok = sum(
        first_sentence_consistent_with_injected_label(cont, label) for label, cont in consistent_cases)
    n_flagged_ok = sum(
        not first_sentence_consistent_with_injected_label(cont, label) for label, cont in flagged_cases)
    check(f"continuation_clean regression: {len(consistent_cases)}/{len(consistent_cases)} real"
         f" boundary-drifted rows score CONSISTENT (run1combined3 must-pass set)",
         n_consistent_ok == len(consistent_cases))
    check(f"continuation_clean regression: {len(flagged_cases)}/{len(flagged_cases)} genuinely-bad rows"
         f" stay FLAGGED (not consistent) despite leading/trailing marker text",
         n_flagged_ok == len(flagged_cases))
    check("continuation_clean regression: expected totals on the 16-case set are 11 consistent, 5 flagged",
         n_consistent_ok + n_flagged_ok == 16 and n_consistent_ok == 11 and n_flagged_ok == 5)

    check("continuation_clean: no-op on an already-clean continuation",
         continuation_clean("This looks like card arrival. Let me help with that right away.")
         == "This looks like card arrival. Let me help with that right away.")
    check("continuation_clean: strips a leading '<|result|>label<|/result|>' fragment",
         continuation_clean("ge_rate]<|result|>x<|/result|> This looks like x. Let me help.")
         == "This looks like x. Let me help.")
    check("continuation_clean: returns text unchanged (nothing to strip) when truly off-template",
         continuation_clean("I see, let me look into your account.")
         == "I see, let me look into your account.")

    check("has_stray_tags true on cont_a's leaked '<|/verify_top-up|>' tag", has_stray_tags(cont_a))
    check("has_loop false on cont_a (the repeated text isn't an exact-sentence duplicate -- the leaked"
         " tag breaks it; cont_a's failure mode is stray_tags, not loop)", not has_loop(cont_a))
    check("has_loop true on an exact repeated sentence",
         has_loop("This looks like age limit. Let me help with that right away. This looks like age"
                  " limit. Let me help with that right away."))
    check("has_stray_tags false on a clean single-template continuation",
         not has_stray_tags("This looks like card not working. Let me help with that right away."))
    check("has_loop false on a clean single-template continuation",
         not has_loop("This looks like card not working. Let me help with that right away."))

    # write_jsonl must survive torch Tensors / numpy scalars that slip past score_decide_call's own
    # float() conversion (the TypeError this script hit mid-run on an earlier build: phase2_rows carried
    # a tensor "confidence" straight into json.dumps). Exercise the json.dumps(default=...) backstop with
    # fake tensor-ish and numpy-ish objects rather than requiring torch/numpy for --dry-parse.
    class _FakeTensor:
        def __init__(self, v): self._v = v
        def item(self): return self._v

    class _FakeArray:
        def __init__(self, v): self._v = v
        def tolist(self): return self._v

    tmp_jsonl = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".dry_parse_write_jsonl_test.jsonl")
    try:
        write_jsonl(tmp_jsonl, [{"id": 0, "resolved": {"label": "x", "confidence": _FakeTensor(0.5)}},
                               {"id": 1, "probs": _FakeArray([0.1, 0.9])}])
        rows_back = read_jsonl(tmp_jsonl)
        check("write_jsonl converts tensor-like .item() values",
             rows_back[0]["resolved"]["confidence"] == 0.5)
        check("write_jsonl converts array-like .tolist() values", rows_back[1]["probs"] == [0.1, 0.9])
        check("write_jsonl leaves tmp file behind after atomic rename",
             not os.path.exists(tmp_jsonl + ".tmp"))
    finally:
        if os.path.exists(tmp_jsonl):
            os.remove(tmp_jsonl)

    if os.environ.get("EVAL_PHASE7_TEST_TORCH"):
        import torch
        label, conf, probs_by_option = ("card_not_working", None, None)
        from kev.api import choice_confidence
        fake_probs_tensor = torch.tensor([0.2, 0.3, 0.5])
        converted = [float(x) for x in fake_probs_tensor]
        check("score_decide_call-style float() conversion drops Tensor-ness",
             all(isinstance(x, float) for x in converted))
        conf = choice_confidence(converted)
        check("choice_confidence on converted floats returns a plain float", isinstance(conf, float))
        write_jsonl(tmp_jsonl, [{"confidence": conf}])
        back = read_jsonl(tmp_jsonl)
        check("end-to-end converted confidence round-trips through write_jsonl",
             back[0]["confidence"] == conf)
        os.remove(tmp_jsonl)

    print(f"\n{len(failures)} failure(s)" if failures else "\nall dry-parse checks passed")
    return not failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-parse", action="store_true",
                    help="run offline unit tests on canned strings (no model/dataset/GPU) and exit")
    ap.add_argument("--report-only", action="store_true",
                    help="no model/GPU/dataset load: read already-generated row files (--adapter-rows"
                         " and/or --baseline-rows) and write the combined CHECK 7 comparison report to"
                         " oracle/phase7-eval-results-{tag}.md (also printed to stdout), then exit.")
    ap.add_argument("--adapter-rows", default=None,
                    help="--report-only: path to an adapter-path rows jsonl (e.g. the"
                         " oracle/phase7-eval-{tag}-adapter.jsonl a prior --phase 3/all run wrote).")
    ap.add_argument("--baseline-rows", default=None,
                    help="--report-only: path to a free-draft baseline rows jsonl (e.g. the"
                         " oracle/phase7-eval-{tag}-baseline.jsonl a prior --baseline free/both run"
                         " wrote). This baseline is NOT comparable to the adapter (see module docstring);"
                         " prefer --options-rows.")
    ap.add_argument("--options-rows", default=None,
                    help="--report-only: path to a chat-with-options baseline rows jsonl (e.g. the"
                         " oracle/phase7-eval-{tag}-baseline-options.jsonl a prior --baseline options run"
                         " wrote) -- the fair baseline; adds the decide-vs-chat paired comparison.")
    ap.add_argument("--inspect", action="store_true",
                    help="no model/GPU/dataset load: read an adapter-rows file (--adapter-rows) and"
                         " optionally a --phase1-raw file, write a qualitative markdown breakdown to"
                         " oracle/phase7-eval-{tag}-inspect.md (never overwrites; --tag required), then"
                         " exit. See build_inspect_report's docstring.")
    ap.add_argument("--phase1-raw", default=None,
                    help="--inspect: path to a phase-1 dump_raw jsonl (e.g. the default"
                         " oracle/phase7-eval-{tag}-phase1-raw.jsonl a prior --phase 1/all run wrote),"
                         " used to classify call_found=False rows as trigger-missed vs"
                         " trigger-fired-call-collapsed. Optional -- without it those rows are reported"
                         " as 'unknown'.")
    ap.add_argument("--test-prompt-boundary", action="store_true",
                    help="tokenizer-only unit test (no GPU/torch): asserts build_decide_prompt_ids's ids"
                         " are an exact token prefix of real training tokenization for 50"
                         " oracle/phase7-calls.jsonl rows, then exits")
    ap.add_argument("--prompt-mode", default="token", choices=["token", "string"],
                    help="phase-1 prompt construction: 'token' (default) cuts the prompt from the"
                         " training-format tokenization at the exact token boundary train_phase7.py saw"
                         " for '<|decide|>' (see build_decide_prompt_ids); 'string' is the old prompt-as-"
                         "plain-string path, which ends up one token off that boundary and is kept only"
                         " for comparison.")
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--revision", default="ee0ef6023621cff504d758262d4e04895a5af4a2")
    ap.add_argument("--adapter", help="PEFT LoRA adapter dir (runs/p7-it-lora/final or .../step-N)")
    ap.add_argument("--decision-run", default="runs/p4-e4b-final",
                    help="Kev decision checkpoint that owns the pointer-head readout (progress.md:"
                         " E4B validation 0.805 dev; served at runs/p4-e4b-final)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--heldout", default="oracle/phase7-eval-heldout.jsonl")
    ap.add_argument("--n-heldout", type=int, default=200)
    ap.add_argument("--seed", type=int, default=97)
    ap.add_argument("--limit", type=int, default=None, help="subsample the loaded held-out set (smoke runs)")
    ap.add_argument("--force-options", action="store_true",
                    help="forced-call mode (default OFF, existing behavior unchanged): phase 1 only"
                         " generates up to the '<|decide|>' trigger (stop_strings=['<|decide|>']), then"
                         " splices in the canonical '<|decide|>{INSTR} [names]<|result|>' call as token"
                         " ids (training-identical, see build_decide_and_canonical_ids) instead of"
                         " letting the model type the 77-name options list -- guarantees a well-formed"
                         " call on every triggered row (call validity 100%% by construction); the trigger"
                         " itself (whether the model opens the call at all) is the only part still"
                         " measured ('trigger rate'). Output files get a '-forced' tag suffix so runs"
                         " never collide with non-forced ones under the same --tag.")
    ap.add_argument("--max-new-before", type=int, default=None,
                    help="token cap generating up to <|result|> (up to the <|decide|> trigger in"
                         " --force-options mode). Default 128 normally, 64 with --force-options (the"
                         " trigger comes within the first ~40 tokens there, so 128 is unnecessarily"
                         " slow) -- pass this explicitly to override either default.")
    ap.add_argument("--max-new-after", type=int, default=80, help="token cap for the resumed continuation")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--conf-floor", type=float, default=DEFAULT_CONF_FLOOR)
    ap.add_argument("--baseline", default="both", choices=["adapter", "free", "options", "both", "none"],
                    help="'both' = adapter + free-draft (legacy, kept for continuity; the free-draft leg"
                         " is NOT comparable, see module docstring). 'options' = base E4B-it shown the"
                         " full label list, chat-templated (the fair baseline) -- run in its own --phase"
                         " all invocation, same as 'free'.")
    ap.add_argument("--options-max-new-tokens", type=int, default=24,
                    help="--baseline options: max new tokens for the chat-with-options answer (it's one"
                         " short line, not a draft).")
    ap.add_argument("--tag", default=None, help="output filename tag; default = unix timestamp."
                    " Refuses to overwrite existing files for a tag already used.")
    ap.add_argument("--phase", default="all", choices=["1", "2", "3", "all"],
                    help="run one step of the adapter path and exit (1: generate+parse decide calls"
                         " with the LoRA chat model; 2: score cached calls on the decision checkpoint;"
                         " 3: inject results and resume generation with the LoRA chat model, writing"
                         " the final report) so a RAM-limited box never loads two models in one process."
                         " Steps hand off through oracle/phase7-eval-{tag}-phase{1,2}-*.jsonl; run 1,"
                         " then 2, then 3 with the same --tag. Default 'all' runs the three steps in"
                         " one process, as before. Only valid with --baseline adapter/none (the free"
                         " baseline loads one model already and has no phase split -- run it separately).")
    ap.add_argument("--dump-raw", default=None,
                    help="phase 1: path to dump each row's raw pre-parse generation (id, raw_text,"
                         " n_new_tokens), for inspecting unparseable rows. Default"
                         " oracle/phase7-eval-{tag}-phase1-raw.jsonl.")
    a = ap.parse_args()

    if a.dry_parse:
        ok = _dry_parse_tests()
        sys.exit(0 if ok else 1)

    if a.test_prompt_boundary:
        ok = _check_prompt_boundary(model=a.model, revision=a.revision)
        sys.exit(0 if ok else 1)

    if a.inspect:
        if not a.adapter_rows:
            ap.error("--inspect requires --adapter-rows")
        if not a.tag:
            ap.error("--inspect requires --tag")
        inspect_out = f"oracle/phase7-eval-{a.tag}-inspect.md"
        if os.path.exists(inspect_out):
            raise FileExistsError(f"{inspect_out} already exists; pass a different --tag")
        adapter_rows = read_jsonl(a.adapter_rows)
        raw_by_id = {r["id"]: r for r in read_jsonl(a.phase1_raw)} if a.phase1_raw else {}
        heldout_by_id = {r["id"]: r for r in read_jsonl(a.heldout)} if os.path.exists(a.heldout) else {}
        _backfill_names(adapter_rows, a.heldout)
        report = build_inspect_report(adapter_rows, raw_by_id, heldout_by_id)
        with open(inspect_out, "w", encoding="utf-8") as f:
            f.write(report)
        print(report)
        print(f"wrote {inspect_out}", flush=True)
        return

    if a.report_only:
        if not a.adapter_rows and not a.baseline_rows and not a.options_rows:
            ap.error("--report-only needs at least one of --adapter-rows/--baseline-rows/--options-rows")
        if not a.tag:
            ap.error("--report-only requires --tag")
        results_out = f"oracle/phase7-eval-results-{a.tag}.md"
        if os.path.exists(results_out):
            raise FileExistsError(f"{results_out} already exists; pass a different --tag")
        adapter_rows = read_jsonl(a.adapter_rows) if a.adapter_rows else []
        baseline_rows = read_jsonl(a.baseline_rows) if a.baseline_rows else []
        options_rows = read_jsonl(a.options_rows) if a.options_rows else None
        _backfill_names(adapter_rows, a.heldout)
        report = build_combined_report(adapter_rows, baseline_rows, a.conf_floor, forced=a.force_options,
                                       options_rows=options_rows)
        with open(results_out, "w", encoding="utf-8") as f:
            f.write(report)
        print(report)
        print(f"wrote {results_out}", flush=True)
        return

    if not a.adapter and a.baseline in ("adapter", "both"):
        ap.error("--adapter is required unless --baseline=free")
    if a.phase != "all" and a.baseline not in ("adapter", "none"):
        ap.error("--phase 1/2/3 only runs the adapter path (one model per process); --baseline must be"
                 " 'adapter' or 'none' -- run the free baseline separately with --phase all --baseline free")

    if a.max_new_before is None:
        a.max_new_before = 64 if a.force_options else 128

    tag = a.tag or str(int(time.time()))
    file_tag = tag + ("-forced" if a.force_options else "")  # distinct names so forced/non-forced runs
                                                               # under the same --tag never overwrite
    adapter_out = f"oracle/phase7-eval-{file_tag}-adapter.jsonl"
    baseline_out = f"oracle/phase7-eval-{file_tag}-baseline.jsonl"
    results_out = f"oracle/phase7-eval-results-{file_tag}.md"
    phase1_out = f"oracle/phase7-eval-{file_tag}-phase1-calls.jsonl"
    phase2_out = f"oracle/phase7-eval-{file_tag}-phase2-scores.jsonl"
    a.dump_raw = a.dump_raw or f"oracle/phase7-eval-{file_tag}-phase1-raw.jsonl"
    a.misses_out = f"oracle/phase7-eval-{file_tag}-phase1-misses.jsonl"

    if a.phase in ("1", "all"):
        for p in (phase1_out, a.dump_raw, a.misses_out):
            if os.path.exists(p):
                raise FileExistsError(f"{p} already exists; pass a different --tag")
    if a.phase in ("2", "all") and os.path.exists(phase2_out):
        raise FileExistsError(f"{phase2_out} already exists; pass a different --tag")
    if a.phase in ("3", "all"):
        for p in (adapter_out, results_out):
            if os.path.exists(p):
                raise FileExistsError(f"{p} already exists; pass a different --tag")
    if a.phase == "all" and a.baseline in ("free", "both") and os.path.exists(baseline_out):
        raise FileExistsError(f"{baseline_out} already exists; pass a different --tag")
    options_out = f"oracle/phase7-eval-{tag}-baseline-options.jsonl"
    if a.phase == "all" and a.baseline == "options" and os.path.exists(options_out):
        raise FileExistsError(f"{options_out} already exists; pass a different --tag")

    rows = load_or_build_heldout(a.heldout, a.n_heldout, a.seed)
    if a.limit:
        rows = rows[:a.limit]
    print(f"eval set: {len(rows)} rows", flush=True)

    if a.phase == "1":
        phase1_rows = run_adapter_phase1(rows, a, a.device)
        write_jsonl(phase1_out, phase1_rows)
        print(f"wrote {phase1_out}; next: --phase 2 --tag {tag}", flush=True)
        return
    if a.phase == "2":
        if not os.path.exists(phase1_out):
            raise FileNotFoundError(f"{phase1_out} not found; run --phase 1 --tag {tag} first")
        phase2_rows = run_adapter_phase2(read_jsonl(phase1_out), a, a.device)
        write_jsonl(phase2_out, phase2_rows)
        print(f"wrote {phase2_out}; next: --phase 3 --tag {tag}", flush=True)
        return
    if a.phase == "3":
        if not (os.path.exists(phase1_out) and os.path.exists(phase2_out)):
            raise FileNotFoundError(f"{phase1_out} and {phase2_out} are both required; run --phase 1"
                                     f" then --phase 2 --tag {tag} first")
        adapter_rows = run_adapter_phase3(read_jsonl(phase1_out), read_jsonl(phase2_out), a, a.device)
        write_jsonl(adapter_out, adapter_rows)
        print(f"wrote {adapter_out}", flush=True)
        report = summarize(adapter_rows, [], a.conf_floor, forced=a.force_options, misses_path=a.misses_out)
        with open(results_out, "w", encoding="utf-8") as f:
            f.write(report)
        print(report)
        print(f"wrote {results_out}", flush=True)
        return

    print("--phase all: running phase 1/2/3 (and any --baseline) in one process; on a RAM-limited box"
          " prefer --phase 1, then 2, then 3 as separate processes instead (same --tag).", flush=True)
    adapter_rows, baseline_rows, options_rows = [], [], None
    if a.baseline in ("adapter", "both"):
        phase1_rows = run_adapter_phase1(rows, a, a.device)
        write_jsonl(phase1_out, phase1_rows)
        phase2_rows = run_adapter_phase2(phase1_rows, a, a.device)
        write_jsonl(phase2_out, phase2_rows)
        adapter_rows = run_adapter_phase3(phase1_rows, phase2_rows, a, a.device)
        write_jsonl(adapter_out, adapter_rows)
        print(f"wrote {adapter_out}", flush=True)
    if a.baseline in ("free", "both"):
        baseline_rows = run_free_baseline(rows, a, a.device)
        write_jsonl(baseline_out, baseline_rows)
        print(f"wrote {baseline_out}", flush=True)
    if a.baseline == "options":
        options_rows = run_options_baseline(rows, a, a.device)
        write_jsonl(options_out, options_rows)
        print(f"wrote {options_out}", flush=True)

    report = summarize(adapter_rows, baseline_rows, a.conf_floor, forced=a.force_options,
                       misses_path=a.misses_out, options_rows=options_rows)
    with open(results_out, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"wrote {results_out}", flush=True)


if __name__ == "__main__":
    main()
