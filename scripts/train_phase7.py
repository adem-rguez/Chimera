"""Phase 7 trainer (GPU box): masked-span SFT on the chat backbone (E4B-it) with LoRA.

Loss covers the Agent portion of each call example EXCEPT the masked label span
(the pointer head owns the label; the chat model learns call format + trigger).
Replay (1:10): plain banking drafts without calls + frozen-suite train states as
plain text, so ordinary drafting survives. Single GPU, bf16, peft LoRA.

Usage (GPU box):
  .venv/bin/python -u scripts/train_phase7.py --model google/gemma-4-E4B-it \\
      --calls oracle/phase7-calls.jsonl --out runs/p7-it-lora
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import glob
import json
import random
import shutil

import torch
from torch.utils.data import DataLoader, Dataset

from kev.model import load_tokenizer

AGENT = "Agent: "


def encode_call(tok, rec, max_len=1024):
    """Returns None (caller counts as skipped) rather than truncating into the masked
    label span: the pointer head owns that span, so silently cutting it would corrupt
    the training signal instead of just shortening context.

    Options-block span (for the loss(call_nonopt)/loss(call_opt) diagnostic): per
    build_phase7.py main(), lines ~42-47, every call's target_text is assembled as
    ...<|decide|>{INSTR} [{', '.join(names)}]<|result|>{label}<|/result|> {cont}
    i.e. everything strictly between the `<|decide|>` and `<|result|>` markers -- the
    instruction text plus the full bracketed 77-option list -- is the fixed block the
    model must reproduce verbatim every row. We locate that span directly off the two
    markers in this row's own target_text (not by re-deriving INSTR/names) so it can't
    drift from the real string. The masked label span (`s`/`e` below, between
    `<|result|>` and `<|/result|>`) is untouched: those tokens are already labels=-100."""
    text = rec["target_text"]
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True,
              truncation=True, max_length=max_len)
    ids = enc["input_ids"][0]
    offs = enc["offset_mapping"][0].tolist()
    s, e = rec["masked_span"]
    if not offs or offs[-1][1] < e:
        return None
    labels = ids.clone()
    agent_at = text.index(AGENT) + len(AGENT)
    for i, (a, b) in enumerate(offs):
        if b <= agent_at or (a < e and b > s):
            labels[i] = -100
    opt_s = text.index("<|decide|>") + len("<|decide|>")
    opt_e = text.index("<|result|>", opt_s)
    opt_mask = torch.tensor([1 if (a < opt_e and b > opt_s) else 0 for a, b in offs],
                             dtype=torch.long)
    return {"input_ids": ids, "labels": labels, "kind": "call", "opt_mask": opt_mask}


def encode_replay(tok, text, max_len=1024, kind="replay_plain", mask_until=0, add_special_tokens=True):
    """mask_until: char offset; tokens starting before it get labels=-100 (masked prompt prefix). 0 (default)
    keeps the old full-supervision behavior unchanged."""
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=max_len,
              add_special_tokens=add_special_tokens)
    ids = enc["input_ids"][0]
    labels = ids.clone()
    if mask_until:
        for i, (a, _b) in enumerate(enc["offset_mapping"][0].tolist()):
            if a < mask_until:
                labels[i] = -100
    return {"input_ids": ids, "labels": labels, "kind": kind}


def encode_think(tok, rec, max_len, variant="short", user_turn_cap=350):
    """Phase 7R T9: encode one built row (oracle/phase7r-built/*.jsonl, see build_phase7r.py's assemble_row)
    for the typed-trigger thinking format. variant picks target_text_short (call-bearing, typed triggers +
    injected spans) or target_text_long (call-free continuous prose, no injected spans). Tokenized with
    add_special_tokens=False: target_text_{short,long} already contain the literal '<bos>' / chat-template
    bytes (reports/17), so letting the tokenizer add its own BOS would double it.

    Masking: masked_spans_short (from the builder) covers [0, prompt_len) + each call's injected
    instruction/options/gold span; both variants share the same system+user prefix (render_row renders
    [system, user] identically regardless of variant), so masked_spans_short[0][1] is also the long
    variant's prompt boundary -- the long variant just has no injected spans after it. Returns None (and
    the caller counts it as skipped) if the row's token length exceeds max_len or rec['user'] alone tokenizes
    past user_turn_cap tokens, rather than truncating into a masked span (same rationale as encode_call)."""
    text = rec[f"target_text_{variant}"]
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = torch.tensor(enc["input_ids"], dtype=torch.long)
    if len(ids) > max_len:
        return None
    user_ids = tok(rec["user"], add_special_tokens=False)["input_ids"]
    if len(user_ids) > user_turn_cap:
        return None
    prompt_end = rec["masked_spans_short"][0][1]
    spans = rec["masked_spans_short"] if variant == "short" else [[0, prompt_end]]
    offs = enc["offset_mapping"]
    labels = ids.clone()
    for i, (a, b) in enumerate(offs):
        if any(a < e and b > s for s, e in spans):
            labels[i] = -100
    opt_mask = torch.zeros(len(ids), dtype=torch.long)
    return {"input_ids": ids, "labels": labels, "kind": "call", "opt_mask": opt_mask}


class Rows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def lora_target_modules(model):
    """Full dotted names of the language model's q/k/v/o_proj Linear leaves, never the vision or audio
    towers: those sit idle on text-only SFT batches (no pixel/audio input), so LoRA on them would be
    trainable but unreachable by backward, which is exactly how this silently produced a gradless loss.
    Gemma-4 wraps some projections in ClippableLinear (peft can't adapt it directly), with the real
    nn.Linear at an inner `.linear` attribute; target that leaf when present, else the module itself."""
    leaves = {"q_proj", "k_proj", "v_proj", "o_proj"}
    targets = []
    for name, mod in model.named_modules():
        parts = name.split(".")
        if not parts or parts[-1] not in leaves:
            continue
        if "language_model" not in parts:
            continue
        if "vision_tower" in parts or "audio" in name.lower():
            continue
        targets.append(f"{name}.linear" if hasattr(mod, "linear") else name)
    return targets


def dry_forward_backward(model, tok, call_rows, batch_size):
    """One forward+backward on the first batch of call rows, failing fast (seconds, not 20 minutes on a
    metered box) if the loss has no grad_fn or no LoRA param actually receives a gradient."""
    batch = collate(tok, call_rows[:max(1, batch_size)])
    batch.pop("kinds", None)
    batch.pop("opt_mask", None)
    batch = {k: v.cuda() for k, v in batch.items()}
    out = model(**batch)
    if not out.loss.requires_grad:
        print("DRY-RUN FAILED: loss.requires_grad is False (no grad_fn reaches the loss)."
              " Check lora target_modules against the actual module names.", flush=True)
        raise SystemExit(1)
    out.loss.backward()
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    hit = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in lora_params)
    model.zero_grad(set_to_none=True)
    peak = torch.cuda.max_memory_allocated() / 2**30
    if not hit:
        print("DRY-RUN FAILED: no LoRA parameter received a nonzero gradient.", flush=True)
        raise SystemExit(1)
    print(f"dry-run OK: loss {out.loss.item():.3f}, loss.requires_grad=True,"
          f" {len(lora_params)} lora params, at least one with nonzero grad", flush=True)
    print(f"peak CUDA memory during dry forward+backward: {peak:.2f} GiB", flush=True)
    del out, batch
    torch.cuda.empty_cache()


def collate(tok, batch):
    ids = [b["input_ids"] for b in batch]
    labs = [b["labels"] for b in batch]
    kinds = [b["kind"] for b in batch]
    # replay rows carry no opt_mask (no options block); default to all-zero so stacking
    # is uniform -- harmless since diagnostics below only ever read opt_mask for "call" rows.
    opts = [b.get("opt_mask", torch.zeros(len(b["input_ids"]), dtype=torch.long)) for b in batch]
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    L = max(len(x) for x in ids)
    inp = torch.stack([torch.cat([x, torch.full((L - len(x),), pad)]) for x in ids])
    lab = torch.stack([torch.cat([x, torch.full((L - len(x),), -100)]) for x in labs])
    opt = torch.stack([torch.cat([x, torch.zeros(L - len(x), dtype=torch.long)]) for x in opts])
    mask = (inp != pad).long()
    return {"input_ids": inp, "attention_mask": mask, "labels": lab, "kinds": kinds, "opt_mask": opt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--calls", default="oracle/phase7-calls.jsonl")
    ap.add_argument("--suite", default="evals/v7/decision-v7")
    ap.add_argument("--replay_ratio", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1536,
                     help="token cap for call/replay rows; rows whose masked span would be"
                          " truncated away are skipped (counted), never cut")
    ap.add_argument("--think-calls", default=None,
                     help="Phase 7R T9: glob of built jsonl files (oracle/phase7r-built/*.jsonl); when "
                          "given, these are encoded via encode_think (typed-trigger thinking format) and "
                          "used as the call rows INSTEAD of --calls/encode_call")
    ap.add_argument("--user-turn-cap", type=int, default=350,
                     help="encode_think: skip a row if rec['user'] alone tokenizes past this many tokens")
    ap.add_argument("--think-variant", default="random", choices=["short", "long", "random"],
                     help="encode_think variant selection for --think-calls rows: 'short'/'long' force"
                          " that variant for every row, 'random' (default) picks per-row as before")
    ap.add_argument("--replay-organic", default=None,
                     help="jsonl of finished==True organic thinking traces (oracle/t8-replay-think.jsonl "
                          "format: user/system/thinking/answer/finished); appended to the replay pool, "
                          "full-supervision except the (re-rendered) prompt prefix")
    ap.add_argument("--replay-per-call", type=int, default=3,
                     help="cap on --replay-organic rows: at most this many per call row")
    ap.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--no-grad-ckpt", action="store_true",
                     help="disable gradient checkpointing (default: enabled)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true",
                     help="load model, pick LoRA targets, run one forward+backward sanity check, exit")
    ap.add_argument("--save-every", type=int, default=2000,
                     help="rows between periodic adapter checkpoints, rounded up to a multiple of"
                          " --accum so checkpoints always land on an accumulation boundary; 0 disables")
    ap.add_argument("--resume", default=None,
                     help="directory with a previously saved adapter + progress.json; loads the adapter"
                          " weights (optimizer state is NOT restored, Adam moments reset) and skips the"
                          " rows already consumed from the identical --seed shuffle order")
    a = ap.parse_args()

    resume_prog = None
    if a.resume:
        with open(f"{a.resume}/progress.json", encoding="utf-8") as f:
            resume_prog = json.load(f)

    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    tok = load_tokenizer(a.model, a.revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, revision=a.revision, dtype=torch.bfloat16,
        attn_implementation=a.attn_impl).cuda().train()
    model.config.use_cache = False
    if not a.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    targets = lora_target_modules(model)
    assert targets, ("no q/k/v/o_proj modules found under 'language_model' in this checkpoint's module "
                      "tree; run with --dry-run and inspect model.named_modules() naming before training")
    assert all("language_model" in t.split(".") for t in targets)
    print(f"lora targets: {len(targets)} language-model modules -> {targets[:4]}...", flush=True)
    if a.resume:
        # adapter-only resume: reloads the LoRA weights saved at a checkpoint step; the AdamW
        # optimizer below is still freshly initialized, so its moment estimates are NOT restored.
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.resume, is_trainable=True).train()
        print(f"resumed from {a.resume} at step {resume_prog['step']}", flush=True)
    else:
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                         target_modules=targets, task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg).train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable}", flush=True)
    assert trainable > 0, "no trainable (LoRA) params after get_peft_model"

    if a.think_calls:
        think_paths = sorted(glob.glob(a.think_calls))
        think_recs = []
        for p in think_paths:
            with open(p, encoding="utf-8") as f:
                think_recs.extend(json.loads(l) for l in f if l.strip())
        think_rng = random.Random(a.seed)
        call_rows = []
        n_skipped = 0
        variant_counts = {"short": 0, "long": 0}
        for rec in think_recs:
            variant = think_rng.choice(["short", "long"]) if a.think_variant == "random" else a.think_variant
            row = encode_think(tok, rec, max_len=a.max_len, variant=variant,
                                user_turn_cap=a.user_turn_cap)
            if row is None:
                n_skipped += 1
            else:
                call_rows.append(row)
                variant_counts[variant] += 1
        print(f"think-calls: {len(think_paths)} file(s), kept {len(call_rows)}/{len(think_recs)} rows"
              f" (skipped {n_skipped}: over --max-len {a.max_len} or --user-turn-cap {a.user_turn_cap}),"
              f" variant={a.think_variant} (short={variant_counts['short']}, long={variant_counts['long']})"
              f" seed={a.seed}", flush=True)
    else:
        calls = [json.loads(l) for l in open(a.calls, encoding="utf-8") if l.strip()]
        call_rows = [encode_call(tok, r, max_len=a.max_len) for r in calls]
        n_skipped = sum(1 for r in call_rows if r is None)
        call_rows = [r for r in call_rows if r is not None]
        if n_skipped:
            print(f"skipped {n_skipped}/{len(calls)} call rows: masked span would be truncated"
                  f" past --max-len {a.max_len}", flush=True)
    dry_forward_backward(model, tok, call_rows, a.batch)
    if a.dry_run:
        return
    # replay: plain drafts of unused banking messages + suite train states as text
    from datasets import load_dataset
    rng = random.Random(a.seed)
    ds = load_dataset("legacy-datasets/banking77", split="train")
    used = {json.loads(l)["provenance"]["row"] for l in
            open(a.calls, encoding="utf-8") if l.strip()}
    pool = [i for i in range(len(ds)) if i not in used]
    # len(call_rows) (not the raw pre-filter `calls`, which doesn't exist when --think-calls replaces the
    # encode_call path): the actual number of call rows this run trains on.
    n_rep = len(call_rows) * a.replay_ratio
    rep_rows = []
    n_plain = min(n_rep // 2, len(pool))
    for i in rng.sample(pool, n_plain):
        rep_rows.append(encode_replay(tok, f"Customer: {ds[i]['text']}\nAgent: "
                                           f"Thanks for reaching out. Let me help with that.",
                                       max_len=a.max_len, kind="replay_plain"))
    # suite replay when the train partition is present (mirror-fetched, gitignored);
    # otherwise fill from remaining plain banking drafts (recorded in phase7.json).
    suite_n = 0
    try:
        with open(f"{a.suite}/train.jsonl", encoding="utf-8") as f:
            train_recs = [json.loads(l) for l in f if l.strip()]
        for r in rng.sample(train_recs, min(n_rep - len(rep_rows), len(train_recs))):
            st = r["state"]
            rep_rows.append(encode_replay(tok, st if isinstance(st, str) else json.dumps(st),
                                           max_len=a.max_len, kind="replay_suite"))
            suite_n += 1
    except FileNotFoundError:
        fill = [i for i in pool]
        rng.shuffle(fill)
        for i in fill[:max(0, n_rep - len(rep_rows))]:
            rep_rows.append(encode_replay(tok, f"Customer: {ds[i]['text']}\nAgent: "
                                               f"Thanks for reaching out. Let me help with that.",
                                           max_len=a.max_len, kind="replay_plain"))

    # organic replay (Phase 7R T9): the base model's own thinking+answer traces (oracle/t8-replay-think.jsonl
    # format), full-supervision except the re-rendered prompt prefix; capped at --replay-per-call per call row.
    n_organic = 0
    if a.replay_organic:
        with open(a.replay_organic, encoding="utf-8") as f:
            organic_recs = [r for r in (json.loads(l) for l in f if l.strip()) if r.get("finished")]
        organic_recs = [r for r in organic_recs if r.get("thinking") and r.get("answer")]
        organic_rng = random.Random(a.seed)
        organic_rng.shuffle(organic_recs)
        cap = len(call_rows) * a.replay_per_call
        organic_recs = organic_recs[:cap]
        for r in organic_recs:
            if r.get("prompt"):
                prompt = r["prompt"]
            else:
                messages = []
                if r.get("system"):
                    messages.append({"role": "system", "content": r["system"]})
                messages.append({"role": "user", "content": r["user"]})
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=True)
            text = prompt + f"<|channel>thought\n{r['thinking']}\n<channel|>{r['answer']}<turn|>\n"
            rep_rows.append(encode_replay(tok, text, max_len=a.max_len, kind="replay_organic",
                                           mask_until=len(prompt), add_special_tokens=False))
            n_organic += 1
        print(f"replay-organic: added {n_organic}/{cap} rows (cap = {len(call_rows)} call_rows x "
              f"--replay-per-call {a.replay_per_call}) from {a.replay_organic}", flush=True)

    # one-off sanity check: how many tokens actually carry a loss-contributing label for a
    # handful of rows of each kind (helps spot replay rows with near-empty supervision).
    print("labeled-token check (first 5 call / first 5 replay rows):", flush=True)
    for name, sample in (("call", call_rows[:5]), ("replay", rep_rows[:5])):
        for i, r in enumerate(sample):
            n_lab = int((r["labels"] != -100).sum())
            extra = ""
            if name == "call":
                n_opt = int(((r["opt_mask"] != 0) & (r["labels"] != -100)).sum())
                extra = f" options_tokens={n_opt} nonopt_labeled_tokens={n_lab - n_opt}"
            print(f"  {name}[{i}]: seq_len={len(r['input_ids'])} labeled_tokens={n_lab}{extra}", flush=True)

    rows = call_rows + rep_rows
    rng.shuffle(rows)
    print(f"train rows: {len(rows)} ({len(call_rows)} calls + {len(rep_rows)} replay)", flush=True)

    save_every_eff = 0
    if a.save_every > 0:
        save_every_eff = ((a.save_every + a.accum - 1) // a.accum) * a.accum
        print(f"--save-every {a.save_every} rounded up to {save_every_eff}"
              f" (multiple of --accum {a.accum})", flush=True)

    if resume_prog is not None and resume_prog.get("rows_total") != len(rows):
        print(f"WARNING: resume rows_total {resume_prog.get('rows_total')} != current {len(rows)};"
              " --calls/--suite/--replay_ratio/--seed must match the original run for the resumed"
              " shuffle order to line up", flush=True)

    def save_checkpoint(step, ep):
        ckpt_dir = f"{a.out}/step-{step}"
        model.save_pretrained(ckpt_dir)
        with open(f"{ckpt_dir}/progress.json", "w", encoding="utf-8") as f:
            json.dump({"step": step, "epoch": ep, "rows_total": len(rows), "seed": a.seed,
                       "oom_skips": n_oom}, f, indent=2)
        print(f"checkpoint saved: {ckpt_dir}", flush=True)
        # bound disk use: keep only the 2 most recent step-* checkpoints
        ckpts = sorted(glob.glob(f"{a.out}/step-*"), key=lambda p: int(p.rsplit("-", 1)[-1]))
        for old in ckpts[:-2]:
            shutil.rmtree(old, ignore_errors=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    import time
    n_oom = 0
    n_seen = 0
    start_step = resume_prog["step"] if resume_prog else 0
    step = start_step
    total = a.epochs * len(rows)
    t0 = time.time()
    # running loss means since the last log line, split by row kind (helps tell a high-loss
    # call row apart from a high-loss replay row; "mixed" only occurs if --batch > 1 mixes kinds)
    loss_stats = {"call": [0.0, 0], "replay_plain": [0.0, 0], "replay_suite": [0.0, 0],
                  "replay_organic": [0.0, 0], "mixed": [0.0, 0]}
    # call_nonopt/call_opt: token-weighted (sum, n) split of the same call-row supervised
    # tokens already summed into loss_stats["call"], broken into the options block
    # (opt_mask) vs everything else supervised (the <|decide|> emission, the text after
    # <|result|>..., the draft continuation) -- see encode_call's docstring for the exact
    # span definition. Diagnostic only; never backprop'd, never affects training.
    diag_stats = {"call_nonopt": [0.0, 0], "call_opt": [0.0, 0]}
    for ep in range(a.epochs):
        done_before = ep * len(rows)
        skip_in_epoch = max(0, start_step - done_before)
        if skip_in_epoch >= len(rows):
            continue  # this whole epoch was already completed before resume
        # rows are already shuffled deterministically above (rng seeded by --seed); the
        # DataLoader must not reshuffle again, or the resume row-skip would land on the
        # wrong rows, so shuffle=False here.
        epoch_rows = rows[skip_in_epoch:] if skip_in_epoch else rows
        dl = DataLoader(Rows(epoch_rows), batch_size=a.batch, shuffle=False,
                        collate_fn=lambda b: collate(tok, b))
        for b in dl:
            kinds = b.pop("kinds")
            opt_mask = b.pop("opt_mask")
            tag = kinds[0] if len(set(kinds)) == 1 else "mixed"
            b = {k: v.cuda() for k, v in b.items()}
            try:
                out = model(**b)
                (out.loss / a.accum).backward()
            except torch.cuda.OutOfMemoryError:
                n_oom += 1
                n_seen += 1
                tlen = b["input_ids"].shape[1]
                print(f"OOM skip: step {step} row len {tlen} tokens (total skips {n_oom})", flush=True)
                if step % a.accum == 0:
                    # first row of this accumulation window failed before anything was
                    # accumulated, so it's safe to clear now; otherwise leave the grads
                    # accumulated by earlier rows in this window alone (simplest-correct).
                    model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if n_seen >= 500 and n_oom / n_seen > 0.02:
                    raise SystemExit(f"aborting: OOM skip rate {n_oom}/{n_seen} exceeds 2% of rows seen")
                step += 1
                continue
            n_seen += 1
            loss_stats[tag][0] += out.loss.item()
            loss_stats[tag][1] += 1
            if tag == "call":
                try:
                    with torch.no_grad():
                        # Gather-before-upcast: index the bf16 logits at only the
                        # nonopt positions (few tokens) before .float(), never
                        # materialize a full [seq, vocab] float32 tensor (vocab ~262k).
                        shift_logits = out.logits.detach()[:, :-1, :]
                        shift_labels = b["labels"][:, 1:]
                        shift_opt = opt_mask.cuda()[:, 1:].bool()
                        sup = shift_labels != -100
                        nonopt_sel = sup & (~shift_opt)
                        idx = nonopt_sel.nonzero(as_tuple=False)
                        if idx.numel() > 0:
                            bi, si = idx[:, 0], idx[:, 1]
                            sel_logits = shift_logits[bi, si, :].float()
                            sel_labels = shift_labels[bi, si]
                            nonopt_sum = torch.nn.functional.cross_entropy(
                                sel_logits, sel_labels, reduction="sum").item()
                            nonopt_n = int(idx.shape[0])
                        else:
                            nonopt_sum, nonopt_n = 0.0, 0
                        total_sup_n = int(sup.sum().item())
                        # out.loss is HF's mean CE over this batch's valid (!=-100)
                        # tokens (batch is homogeneous "call" here), so
                        # out.loss * total_sup_n recovers the summed CE over ALL
                        # call-supervised tokens without a second full-vocab pass;
                        # the options-block sum is just the remainder.
                        total_sup_sum = out.loss.item() * total_sup_n
                        opt_sum = total_sup_sum - nonopt_sum
                        opt_n = total_sup_n - nonopt_n
                        diag_stats["call_nonopt"][0] += nonopt_sum
                        diag_stats["call_nonopt"][1] += nonopt_n
                        diag_stats["call_opt"][0] += opt_sum
                        diag_stats["call_opt"][1] += opt_n
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
            if (step + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            step += 1
            if step % 25 == 0:
                el = time.time() - t0
                eta = el / (step - start_step) * (total - step)
                call_sum, call_n = loss_stats["call"]
                rep_sum = (loss_stats["replay_plain"][0] + loss_stats["replay_suite"][0] +
                           loss_stats["replay_organic"][0] + loss_stats["mixed"][0])
                rep_n = (loss_stats["replay_plain"][1] + loss_stats["replay_suite"][1] +
                         loss_stats["replay_organic"][1] + loss_stats["mixed"][1])
                call_mean = call_sum / call_n if call_n else 0.0
                rep_mean = rep_sum / rep_n if rep_n else 0.0
                nonopt_sum, nonopt_n = diag_stats["call_nonopt"]
                optblk_sum, optblk_n = diag_stats["call_opt"]
                nonopt_mean = nonopt_sum / nonopt_n if nonopt_n else 0.0
                optblk_mean = optblk_sum / optblk_n if optblk_n else 0.0
                print(f"ep{ep} step {step}/{total} loss {out.loss.item():.3f} "
                      f"loss(call)={call_mean:.3f} n={call_n} loss(replay)={rep_mean:.3f} n={rep_n} "
                      f"loss(call_nonopt)={nonopt_mean:.3f} n={nonopt_n} "
                      f"loss(call_opt)={optblk_mean:.3f} n={optblk_n} "
                      f"{el/(step-start_step):.2f}s/step eta {eta/60:.1f}min oom_skips={n_oom}", flush=True)
                loss_stats = {k: [0.0, 0] for k in loss_stats}
                diag_stats = {k: [0.0, 0] for k in diag_stats}
            if save_every_eff and step % a.accum == 0 and step % save_every_eff == 0:
                save_checkpoint(step, ep)
    model.save_pretrained(a.out)
    with open(f"{a.out}/phase7.json", "w", encoding="utf-8") as f:
        json.dump({"model": a.model, "revision": a.revision, "calls": len(call_rows),
                   "replay": len(rep_rows), "replay_suite_rows": suite_n, "replay_organic_rows": n_organic,
                   "epochs": a.epochs, "lr": a.lr, "oom_skips": n_oom}, f, indent=2)
    print(f"saved {a.out} (oom_skips={n_oom})")


if __name__ == "__main__":
    main()
