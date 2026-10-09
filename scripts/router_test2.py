"""Phase 7G router test, part 2 (reports/25 addendum): two follow-ups to scripts/router_test.py.

PART 1 (gsm8k, "cheap pass costs ~0 generated tokens"): the original gsm8k signal (router_test.py's
numeric confidence, teacher-forced over the direct arm's OWN greedy output) was useless (near-degenerate
range, see oracle/router-test.md's gsm8k caveat). Here the cheap pass itself generates only 1-8 tokens:
  (a) immediate-answer: prompt forces a bare numeric continuation ("...The answer is"), greedy,
      max_new_tokens=8. Confidence candidates: mean_logprob / min_logprob (per-generated-token logprob of
      the model's OWN greedy choice, via output_scores=True) and first_token_entropy (full-vocab entropy
      of the distribution BEFORE any token is generated -- the one score here that is not conditioned on
      the model's own already-committed low-confidence tokens).
  (b) difficulty-probe: "can this be solved without steps? yes/no", teacher-forced single forward pass
      (no generation at all), confidence = P(yes) restricted to the yes/no token ids.
  (c) cheap text features (qlen, n_numbers, n_sentences), pure Python, no model.
Target label for AUROC: thinking_correct AND immediate_correct (does the cheap pass + escalation actually
recover the right answer). 200 fresh GSM8K TEST rows (scripts/fetch_s1_problems.py never touches gsm8k
test) are generated fresh for both arms; the 200 existing train-split rows (oracle/format-probe.jsonl /
oracle/s1-traces.jsonl) are reused for thinking+text-features but still need a fresh immediate-answer +
difficulty-probe pass (score-gsm8k below covers both pools uniformly).

PART 2 (bigger MC eval): 400 MMLU test (stratified across `subject`, fixed seed) + 400 ARC-Challenge test
rows, fetched fresh (test splits only -- guaranteed disjoint from oracle/s1-train-problems.jsonl's
auxiliary_train/validation/train sources). Arms:
  (i)   thinking baseline -- generate via scripts/gen_think_traces.py --thinking on (same settings as
        Part 1 / the existing s1-traces.jsonl pool: default sampling, max_new_tokens 1200).
  (ii)  CHEAP PASS = letter readout on the plain question+options prompt, ONE forward pass, NO
        generation (router_test.py's render_mc_prompt/_letter_token_id/softmax_restricted, reused
        directly): confidence = max_prob (the readout's own top letter's probability), margin = top1 -
        top2; the cheap-pass "answer" IS the readout's argmax letter (unlike router_test.py's original
        MC score, which scored a prior generated arm's chosen letter -- here there is no prior arm, the
        readout IS the cheap pass).
  (iii) no-think direct generation (scripts/gen_think_traces.py --thinking off) -- stored for comparison
        only, not fed into the router sim.

Router sim (both parts): cost = cheap-pass tokens for every record + thinking tokens for escalated
records only (same accounting as scripts/router_test.py's `accounting`). tau chosen on a dev half with
the <=1pt dev accuracy loss rule (scripts/router_test.py's choose_tau, reused directly); repeated_split_eval
below runs this over 200 fixed-seed 50/50 splits (scripts/router_test.py's split_dev_test, reused,
called with seed=0..199) and reports mean/p90 TEST loss, mean token saving, and the fraction of splits
meeting loss<=1pt & saving>=30%.

Pre-registered bars (reports/25's router-test addendum, part 2 of this task's own framing): MC pooled
must hit token saving >=30% with mean test loss <=1pt in >=80% of the 200 splits to PASS. gsm8k has no
pre-registered bar (reported only, per Part 1's own framing).

Usage:
  offline unit tests (AUROC, text features, repeated-split plumbing; no model, no GPU, no files):
    .venv/Scripts/python.exe scripts/router_test2.py --selftest

  fetch (laptop or box, needs `datasets` + network, no torch):
    .venv/Scripts/python.exe scripts/router_test2.py fetch-gsm8k \\
        --out oracle/router2-gsm8k-fresh.jsonl --n 300 --seed 0
    .venv/Scripts/python.exe scripts/router_test2.py fetch-mc \\
        --out oracle/router2-mc-fresh.jsonl --n-mmlu 400 --n-arc 400 --seed 0

  thinking/direct generation (GPU box, via gen_think_traces.py directly -- not reimplemented here):
    .venv/bin/python -u scripts/gen_think_traces.py --input oracle/router2-gsm8k-fresh.jsonl \\
        --output oracle/router2-gsm8k-thinking.jsonl --thinking on --max-new-tokens 1200 \\
        --batch-size 8 --resume
    .venv/bin/python -u scripts/gen_think_traces.py --input oracle/router2-mc-fresh.jsonl \\
        --output oracle/router2-mc-thinking.jsonl --thinking on --max-new-tokens 1200 \\
        --batch-size 16 --resume
    .venv/bin/python -u scripts/gen_think_traces.py --input oracle/router2-mc-fresh.jsonl \\
        --output oracle/router2-mc-direct.jsonl --thinking off --max-new-tokens 400 \\
        --batch-size 32 --resume

  cheap-pass scoring (GPU box):
    .venv/bin/python -u scripts/router_test2.py score-gsm8k \\
        --problems oracle/s1-train-problems.jsonl oracle/router2-gsm8k-fresh.jsonl \\
        --out oracle/router2-gsm8k-cheap.jsonl --resume
    .venv/bin/python -u scripts/router_test2.py score-mc \\
        --problems oracle/router2-mc-fresh.jsonl --out oracle/router2-mc-cheap.jsonl --resume

  simulate (CPU, laptop):
    .venv/Scripts/python.exe scripts/router_test2.py simulate-gsm8k \\
        --problems oracle/s1-train-problems.jsonl oracle/router2-gsm8k-fresh.jsonl \\
        --traces oracle/s1-traces.jsonl oracle/router2-gsm8k-thinking.jsonl \\
        --cheap oracle/router2-gsm8k-cheap.jsonl --out-report oracle/router-test2-gsm8k.md
    .venv/Scripts/python.exe scripts/router_test2.py simulate-mc \\
        --problems oracle/router2-mc-fresh.jsonl --traces oracle/router2-mc-thinking.jsonl \\
        --direct oracle/router2-mc-direct.jsonl --cheap oracle/router2-mc-cheap.jsonl \\
        --out-report oracle/router-test2-mc.md
"""
import argparse
import collections
import json
import math
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.s1_head_and_outcomes import (  # noqa: E402
    extract_last_number, extract_choice_letter, numbers_equal, judge_correctness,
)
from scripts.router_test import (  # noqa: E402
    softmax_restricted, render_mc_prompt, _letter_token_id, LETTERS,
    split_dev_test, choose_tau, accounting, policy_tau, policy_always_thinking,
    policy_always_direct, policy_oracle, candidate_taus, bootstrap_ci_accuracy_diff,
    confident_wrong_count,
)

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_SENT_RE = re.compile(r"(?<!\d)[.!?]+(?!\d)")  # doesn't split decimal points like "10.5"


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_jsonl_multi(paths):
    out = []
    for p in paths:
        out += load_jsonl(resolve(p))
    return out


def write_jsonl(path, rows, mode="w"):
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# pure helpers: AUROC, text features (no model, no GPU)
# ---------------------------------------------------------------------------

def auroc(scores, labels):
    """-> float in [0, 1]. Mann-Whitney U / rank-sum AUROC: P(score of a random positive > score of a
    random negative), ties counted as 0.5. labels: any truthy/falsy sequence same length as scores.
    Returns 0.5 (uninformative) if there are no positives or no negatives (degenerate -- caller decides
    what to do with that, never raises)."""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n = len(pairs)
    # rank-based AUROC via assigning average ranks to ties, then U = sum(ranks of positives) - n_pos*(n_pos+1)/2
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # 1-indexed average rank over the tie block [i, j)
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    n_pos = sum(1 for _, lab in pairs if lab)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum_pos = sum(r for r, (_, lab) in zip(ranks, pairs) if lab)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def best_direction_auroc(scores, labels):
    """-> (auroc, direction) where direction is +1 if higher score = more likely positive, -1 if the
    signal needs flipping (lower score = more likely positive) to get the better of the two AUROCs.
    AUROC(flip) = 1 - AUROC(raw) exactly, so this is just max(a, 1-a) with the matching sign."""
    a = auroc(scores, labels)
    if a >= 0.5:
        return a, 1
    return 1.0 - a, -1


def text_features(question):
    """-> dict(qlen, n_numbers, n_sentences), pure string features, no model."""
    qlen = len(question or "")
    n_numbers = len(_NUM_RE.findall(question or ""))
    n_sentences = len([s for s in _SENT_RE.split(question or "") if s.strip()])
    return dict(qlen=qlen, n_numbers=n_numbers, n_sentences=n_sentences)


def entropy_from_logits(logits):
    """-> float. Full-distribution Shannon entropy (nats) of softmax(logits). `logits`: list[float] (full
    vocab or any restricted set -- caller decides which); pure math, no torch, so --selftest can exercise
    it with a tiny fabricated vector."""
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    z = sum(exps)
    probs = [e / z for e in exps]
    return -sum(p * math.log(p) for p in probs if p > 0)


# ---------------------------------------------------------------------------
# repeated-split router sim (reuses router_test.py's split_dev_test/choose_tau/accounting verbatim)
# ---------------------------------------------------------------------------

def repeated_split_eval(records, n_splits=200, base_seed=0, max_accuracy_loss=0.01,
                         saving_bar=0.30):
    """-> dict(mean_loss, p90_loss, mean_saving, frac_meeting_bar, n_splits_used). For each seed in
    [base_seed, base_seed+n_splits): split_dev_test(records, seed) -> dev/test; choose_tau(dev); test
    accounting at that tau vs always-thinking on TEST. loss = always_think_acc - chosen_acc (can be
    negative, i.e. the router did BETTER than always-thinking -- not clipped). saving = 1 -
    chosen_mean_tokens/always_think_mean_tokens. A split with zero always_think mean_tokens (degenerate,
    never happens with real data but guarded) is skipped and does not count toward n_splits_used."""
    losses, savings = [], []
    for seed in range(base_seed, base_seed + n_splits):
        dev, test = split_dev_test(records, seed=seed)
        if not dev or not test:
            continue
        tau, _dev_info = choose_tau(dev, max_accuracy_loss=max_accuracy_loss)
        always_think = accounting(test, policy_always_thinking)
        chosen = accounting(test, policy_tau(tau))
        if not always_think["mean_tokens"]:
            continue
        losses.append(always_think["accuracy"] - chosen["accuracy"])
        savings.append(1 - chosen["mean_tokens"] / always_think["mean_tokens"])
    if not losses:
        return dict(mean_loss=float("nan"), p90_loss=float("nan"), mean_saving=float("nan"),
                    frac_meeting_bar=float("nan"), n_splits_used=0)
    losses_sorted = sorted(losses)
    p90_idx = min(int(0.9 * len(losses_sorted)), len(losses_sorted) - 1)
    n_meet = sum(1 for l, s in zip(losses, savings) if l <= max_accuracy_loss and s >= saving_bar)
    return dict(
        mean_loss=sum(losses) / len(losses),
        p90_loss=losses_sorted[p90_idx],
        mean_saving=sum(savings) / len(savings),
        frac_meeting_bar=n_meet / len(losses),
        n_splits_used=len(losses),
    )


# ---------------------------------------------------------------------------
# fetch: gsm8k test / mmlu+arc test (needs `datasets` + network, no torch)
# ---------------------------------------------------------------------------

GSM8K_TEST_ALTERNATIVES = [
    dict(repo="openai/gsm8k", config="main", split="test"),
    dict(repo="gsm8k", config="main", split="test"),
]


def _try_load(alternatives, label):
    from datasets import load_dataset
    errors = []
    for alt in alternatives:
        try:
            ds = load_dataset(alt["repo"], alt["config"], split=alt["split"])
            print(f"  {label}: loaded {alt['repo']!r} config={alt['config']!r} split={alt['split']!r} "
                  f"({len(ds)} rows)", flush=True)
            return ds
        except Exception as e:  # noqa: BLE001
            errors.append(f"{alt}: {e!r}")
    raise SystemExit(f"{label}: all alternatives failed.\n" + "\n".join(f"  - {e}" for e in errors))


def parse_gsm8k_gold(answer_field):
    marker = "#### "
    idx = answer_field.rfind(marker)
    if idx < 0:
        raise ValueError(f"no {marker!r} marker in GSM8K answer field: {answer_field!r}")
    return answer_field[idx + len(marker):].strip().replace(",", "")


def run_fetch_gsm8k(args):
    ds = _try_load(GSM8K_TEST_ALTERNATIVES, "gsm8k-test")
    rows = list(ds)
    rng = random.Random(args.seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    chosen = idxs[:args.n]
    out = []
    for i, ridx in enumerate(chosen):
        r = rows[ridx]
        gold = parse_gsm8k_gold(r["answer"])
        out.append(dict(id=f"gsm8kT-{i:04d}", user=r["question"].strip(), kind="problem",
                        source="gsm8k", gold_answer=gold, answer_type="numeric", split="test"))
    write_jsonl(resolve(args.out), out)
    print(f"wrote {len(out)} fresh gsm8k TEST rows -> {args.out}", flush=True)
    return 0


def render_lettered_options(option_texts):
    return "\n".join(f"{LETTERS[i] if i < len(LETTERS) else chr(ord('A') + i)}. {t}"
                      for i, t in enumerate(option_texts))


def stratified_quota(groups_sizes, total):
    """groups_sizes: {key: size}. -> {key: quota} summing to exactly `total`, proportional to size,
    largest remainders get the leftover units. Pure allocation logic, --selftest-able."""
    keys = sorted(groups_sizes)
    total_size = sum(groups_sizes.values())
    raw = {k: total * groups_sizes[k] / total_size for k in keys}
    quota = {k: int(math.floor(raw[k])) for k in keys}
    remainder = total - sum(quota.values())
    # top up by largest fractional remainder, stable order over keys for determinism
    fracs = sorted(keys, key=lambda k: (raw[k] - quota[k]), reverse=True)
    for k in fracs[:remainder]:
        quota[k] += 1
    return quota


def run_fetch_mc(args):
    from datasets import load_dataset
    mmlu = list(load_dataset("cais/mmlu", "all", split="test"))
    by_subject = collections.defaultdict(list)
    for r in mmlu:
        by_subject[r["subject"]].append(r)
    quotas = stratified_quota({k: len(v) for k, v in by_subject.items()}, args.n_mmlu)
    mmlu_rows = []
    rng = random.Random(args.seed)
    for subject in sorted(by_subject):
        pool = list(by_subject[subject])
        rng.shuffle(pool)
        mmlu_rows += pool[:quotas[subject]]
    print(f"  mmlu: {len(mmlu_rows)} rows across {len(by_subject)} subjects (target {args.n_mmlu})",
          flush=True)

    arc = list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"))
    arc = [r for r in arc if r.get("answerKey")]
    rng2 = random.Random(args.seed)
    rng2.shuffle(arc)
    arc_rows = arc[:args.n_arc]
    print(f"  arc_challenge: {len(arc_rows)} rows (target {args.n_arc})", flush=True)

    out = []
    for i, r in enumerate(mmlu_rows):
        user = f"{r['question'].strip()}\n\n{render_lettered_options(r['choices'])}"
        out.append(dict(id=f"mmluT-{i:04d}", user=user, kind="problem", source="mmlu",
                        gold_answer=LETTERS[r["answer"]], answer_type="choice", split="test"))
    for i, r in enumerate(arc_rows):
        labels = r["choices"]["label"]
        texts = r["choices"]["text"]
        user = f"{r['question'].strip()}\n\n{render_lettered_options(texts)}"
        pos = labels.index(r["answerKey"])
        out.append(dict(id=f"arcT-{i:04d}", user=user, kind="problem", source="arc_challenge",
                        gold_answer=LETTERS[pos], answer_type="choice", split="test"))
    write_jsonl(resolve(args.out), out)
    print(f"wrote {len(out)} fresh MC TEST rows -> {args.out}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# score-gsm8k: GPU, immediate-answer generation + difficulty-probe teacher-forced readout
# ---------------------------------------------------------------------------

IMMEDIATE_SUFFIX = ("\n\nRespond with only the final numeric answer and nothing else -- no words, no "
                    "explanation, no units. The answer is")
DIFFICULTY_SUFFIX = ("\n\nCan this problem be solved correctly without writing out any intermediate "
                     "steps? Answer with exactly one word, yes or no.")


def render_immediate_prompt(tok, question):
    messages = [{"role": "user", "content": question + IMMEDIATE_SUFFIX}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)


def render_difficulty_prompt(tok, question):
    messages = [{"role": "user", "content": question + DIFFICULTY_SUFFIX}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)


def _word_token_id(tok, word):
    """Same convention as router_test.py's _letter_token_id: encode "\\n<word>" and take the last token
    id -- the token actually competing at a fresh line in this chat template's generation position."""
    ids = tok("\n" + word, add_special_tokens=False)["input_ids"]
    return ids[-1]


def score_immediate_batch(tok, model, device, questions, max_new_tokens=8):
    """-> list of dicts, one per question: {generated_text, n_new_tokens, mean_logprob, min_logprob,
    first_token_entropy}. Single batched greedy generate call with output_scores=True so per-step logits
    are available without a second forward pass."""
    import torch
    prompts = [render_immediate_prompt(tok, q) for q in questions]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    prompt_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, output_scores=True,
                             return_dict_in_generate=True)
    seqs = out.sequences
    scores = out.scores  # list of (batch, vocab) tensors, one per generated step
    results = []
    for j in range(len(questions)):
        new_ids = seqs[j, prompt_len:].tolist()
        # nonpad new tokens (eos/pad both use tok.pad_token_id as generate's pad fill after a row's own
        # eos -- same convention as gen_think_traces.batched_generate's nonpad count)
        nonpad_len = 0
        for t in new_ids:
            if t == tok.pad_token_id:
                break
            nonpad_len += 1
        nonpad_len = max(nonpad_len, 1)  # guard: a row that emits pad as its very first token still
        # gets its step-0 logprob counted (degenerate but never dropped)
        generated_text = tok.decode(new_ids[:nonpad_len], skip_special_tokens=False)
        step0_logits = scores[0][j].float().tolist()
        first_token_entropy = entropy_from_logits(step0_logits)
        logprobs = []
        for step in range(nonpad_len):
            step_logits = scores[step][j].float()
            log_probs = torch.log_softmax(step_logits, dim=-1)
            chosen_id = new_ids[step]
            logprobs.append(log_probs[chosen_id].item())
        results.append(dict(generated_text=generated_text, n_new_tokens=nonpad_len,
                            mean_logprob=sum(logprobs) / len(logprobs), min_logprob=min(logprobs),
                            first_token_entropy=first_token_entropy))
    return results


def score_difficulty_batch(tok, model, device, questions, yes_id, no_id):
    """-> list of float (P(yes)), one teacher-forced forward pass over the whole batch, no generation."""
    import torch
    prompts = [render_difficulty_prompt(tok, q) for q in questions]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    # left-padded (same convention as gen_think_traces) -> last real position is always index -1
    out = []
    for j in range(len(questions)):
        last_logits = logits[j, -1, :]
        probs = softmax_restricted({"yes": last_logits[yes_id].item(), "no": last_logits[no_id].item()})
        out.append(probs["yes"])
    return out


def run_score_gsm8k(args):
    import torch
    from kev.model import load_tokenizer
    from transformers import AutoModelForCausalLM

    problems = [p for p in load_jsonl_multi(args.problems) if p.get("source") == "gsm8k"]
    by_id = {}
    for p in problems:
        by_id[p["id"]] = p  # later files win on duplicate id (shouldn't happen, train+fresh ids disjoint)
    problems = list(by_id.values())

    done = set()
    out_path = resolve(args.out)
    if args.resume and os.path.exists(out_path):
        done = {r["id"] for r in load_jsonl(out_path)}
        print(f"--resume: {len(done)} ids already scored", flush=True)
    todo = [p for p in problems if p["id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"scoring {len(todo)} gsm8k problems", flush=True)
    if not todo:
        return 0

    tok = load_tokenizer(args.model, args.revision)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, attn_implementation=args.attn_impl
    ).to(args.device).eval()
    yes_id = _word_token_id(tok, "yes")
    no_id = _word_token_id(tok, "no")

    mode = "a" if (args.resume and done) else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i:i + args.batch_size]
            questions = [p["user"] for p in batch]
            imm = score_immediate_batch(tok, model, args.device, questions, args.max_new_tokens)
            p_yes = score_difficulty_batch(tok, model, args.device, questions, yes_id, no_id)
            for p, im, py in zip(batch, imm, p_yes):
                val = extract_last_number(im["generated_text"])
                try:
                    gold_val = float(str(p["gold_answer"]).replace(",", ""))
                except ValueError:
                    gold_val = None
                correct = (val is not None and gold_val is not None and numbers_equal(val, gold_val))
                feats = text_features(p["user"])
                row = dict(id=p["id"], source="gsm8k", immediate_text=im["generated_text"],
                          immediate_tokens=im["n_new_tokens"], immediate_correct=bool(correct),
                          immediate_parsed=val, mean_logprob=im["mean_logprob"],
                          min_logprob=im["min_logprob"], first_token_entropy=im["first_token_entropy"],
                          p_yes_difficulty=py, **feats)
                f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"scored {min(i + args.batch_size, len(todo))}/{len(todo)}", flush=True)
    print(f"done -> {args.out}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# score-mc: GPU, letter-readout cheap pass (1 forward pass, no generation)
# ---------------------------------------------------------------------------

def run_score_mc(args):
    import torch
    from kev.model import load_tokenizer
    from transformers import AutoModelForCausalLM

    problems = load_jsonl_multi(args.problems)
    done = set()
    out_path = resolve(args.out)
    if args.resume and os.path.exists(out_path):
        done = {r["id"] for r in load_jsonl(out_path)}
        print(f"--resume: {len(done)} ids already scored", flush=True)
    todo = [p for p in problems if p["id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"scoring {len(todo)} mc problems", flush=True)
    if not todo:
        return 0

    tok = load_tokenizer(args.model, args.revision)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, attn_implementation=args.attn_impl
    ).to(args.device).eval()
    letter_ids = {letter: _letter_token_id(tok, letter) for letter in LETTERS}

    mode = "a" if (args.resume and done) else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i:i + args.batch_size]
            prompts = [render_mc_prompt(tok, p["user"]) for p in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(args.device)
            with torch.no_grad():
                logits = model(**enc, logits_to_keep=1).logits
            for j, p in enumerate(batch):
                last_logits = logits[j, -1, :]
                letter_logits = {letter: last_logits[tid].item() for letter, tid in letter_ids.items()}
                probs = softmax_restricted(letter_logits)
                chosen = max(probs, key=probs.get)
                ranked = sorted(probs.values(), reverse=True)
                margin = ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)
                row = dict(id=p["id"], source=p["source"], chosen_letter=chosen, max_prob=probs[chosen],
                          margin=margin, cheap_correct=bool(chosen == str(p.get("gold_answer")).upper()))
                f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"scored {min(i + args.batch_size, len(todo))}/{len(todo)}", flush=True)
    print(f"done -> {args.out}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# simulate-gsm8k: CPU
# ---------------------------------------------------------------------------

SIGNAL_SPECS_GSM8K = [
    ("mean_logprob", "immediate-answer mean token logprob"),
    ("min_logprob", "immediate-answer min token logprob"),
    ("first_token_entropy", "immediate-answer first-token entropy (lower = more confident)"),
    ("p_yes_difficulty", "difficulty-probe P(yes)"),
    ("qlen", "question length in chars (shorter = easier, by convention)"),
    ("n_numbers", "count of numbers in question"),
    ("n_sentences", "count of sentences in question"),
]


def build_gsm8k_records(problems, thinking_rows, cheap_rows):
    problems_by_id = {p["id"]: p for p in problems}
    thinking_by_id = {t["id"]: t for t in thinking_rows}
    records = []
    skipped = 0
    for c in cheap_rows:
        pid = c["id"]
        p = problems_by_id.get(pid)
        t = thinking_by_id.get(pid)
        if p is None or t is None:
            skipped += 1
            continue
        t_correct, _parsed, _unparsed = judge_correctness(t, p)
        t_tokens = (t.get("n_thinking_tokens") or 0) + (t.get("n_answer_tokens") or 0)
        records.append(dict(
            id=pid, source="gsm8k", group="gsm8k",
            direct_correct=bool(c["immediate_correct"]), direct_tokens=int(c["immediate_tokens"]),
            thinking_correct=bool(t_correct), thinking_tokens=t_tokens,
            label=bool(t_correct) and bool(c["immediate_correct"]),
            mean_logprob=c["mean_logprob"], min_logprob=c["min_logprob"],
            first_token_entropy=c["first_token_entropy"], p_yes_difficulty=c["p_yes_difficulty"],
            qlen=c["qlen"], n_numbers=c["n_numbers"], n_sentences=c["n_sentences"],
        ))
    return records, skipped


def run_simulate_gsm8k(args):
    problems = [p for p in load_jsonl_multi(args.problems) if p.get("source") == "gsm8k"]
    thinking_rows_raw = load_jsonl_multi(args.traces)
    cheap_rows = load_jsonl(resolve(args.cheap))
    records, skipped = build_gsm8k_records(problems, thinking_rows_raw, cheap_rows)
    print(f"built {len(records)} gsm8k records ({skipped} skipped: missing problem/thinking)", flush=True)

    lines = ["# Router test 2 report -- gsm8k (cheap pass = immediate numeric answer, ~1-8 tokens)", "",
             f"n={len(records)} (target 500 = 200 existing train-split + 300 fresh test-split).", ""]

    lines.append("## AUROC per signal (predicting thinking_correct AND immediate_correct)")
    lines.append("")
    lines.append("| signal | AUROC (best direction) | direction |")
    lines.append("|---|---|---|")
    labels = [r["label"] for r in records]
    best_signal, best_auroc_val, best_direction = None, 0.0, 1
    for key, desc in SIGNAL_SPECS_GSM8K:
        scores = [r[key] for r in records]
        a, direction = best_direction_auroc(scores, labels)
        lines.append(f"| {key} ({desc}) | {a:.3f} | {'higher=more confident' if direction == 1 else 'lower=more confident'} |")
        if key in ("mean_logprob", "min_logprob", "first_token_entropy", "p_yes_difficulty") and a > best_auroc_val:
            best_signal, best_auroc_val, best_direction = key, a, direction
    lines.append("")
    lines.append(f"Best model-based signal by AUROC: **{best_signal}** (AUROC={best_auroc_val:.3f}, "
                 f"direction={'raw' if best_direction == 1 else 'flipped'}). Router sim below uses this "
                 "signal as the confidence score (flipped in sign first if direction=flipped, so higher "
                 "always means 'more confident' for choose_tau's `<` threshold convention).")
    lines.append("")

    for r in records:
        raw = r[best_signal]
        r["confidence"] = raw if best_direction == 1 else -raw

    lines.append("## Router sim (confidence = best signal above)")
    lines.append("")
    dev, test = split_dev_test(records, seed=0)
    tau, dev_info = choose_tau(dev)
    always_think = accounting(test, policy_always_thinking)
    always_direct = accounting(test, policy_always_direct)
    oracle_pol = accounting(test, policy_oracle)
    chosen = accounting(test, policy_tau(tau))
    saving = (1 - chosen["mean_tokens"] / always_think["mean_tokens"]) if always_think["mean_tokens"] else 0
    lines.append(f"Single 50/50 split (seed=0): dev n={len(dev)}, test n={len(test)}, chosen tau={tau}")
    lines.append("")
    lines.append("| policy | accuracy | mean tokens | % escalated | token saving vs always-thinking |")
    lines.append("|---|---|---|---|---|")
    for name, info in [("always-thinking", always_think), ("always-direct (immediate)", always_direct),
                        ("oracle router", oracle_pol), (f"confidence router (tau={tau})", chosen)]:
        s = (1 - info["mean_tokens"] / always_think["mean_tokens"]) if always_think["mean_tokens"] else 0
        lines.append(f"| {name} | {info['accuracy']:.3f} | {info['mean_tokens']:.1f} | "
                     f"{info['pct_escalated']:.3f} | {s:.3%} |")
    lines.append("")
    lo, hi = bootstrap_ci_accuracy_diff(test, tau)
    lines.append(f"95% bootstrap CI on accuracy diff (chosen tau - always-thinking), this split: "
                 f"[{lo:.3f}, {hi:.3f}]")
    lines.append("")

    lines.append("## Repeated 50/50 splits (200 splits, seeds 0..199) -- no pre-registered bar for gsm8k")
    lines.append("")
    lines.append("| signal | mean test loss | p90 test loss | mean token saving | "
                 "frac meeting loss<=1pt & saving>=30% |")
    lines.append("|---|---|---|---|---|")
    for key, desc in SIGNAL_SPECS_GSM8K:
        raw_scores = [r[key] for r in records]
        a, direction = best_direction_auroc(raw_scores, labels)
        for r in records:
            r["confidence"] = r[key] if direction == 1 else -r[key]
        stats = repeated_split_eval(records, n_splits=200, base_seed=0)
        lines.append(f"| {key} | {stats['mean_loss']:.4f} | {stats['p90_loss']:.4f} | "
                     f"{stats['mean_saving']:.3%} | {stats['frac_meeting_bar']:.3f} |")
    lines.append("")

    report = "\n".join(lines) + "\n"
    with open(resolve(args.out_report), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"wrote {args.out_report}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# simulate-mc: CPU
# ---------------------------------------------------------------------------

def build_mc_records(problems, thinking_rows, cheap_rows):
    problems_by_id = {p["id"]: p for p in problems}
    thinking_by_id = {t["id"]: t for t in thinking_rows}
    cheap_by_id = {c["id"]: c for c in cheap_rows}
    records = []
    skipped = 0
    for pid, p in problems_by_id.items():
        t = thinking_by_id.get(pid)
        c = cheap_by_id.get(pid)
        if t is None or c is None:
            skipped += 1
            continue
        t_correct, _parsed, _unparsed = judge_correctness(t, p)
        t_tokens = (t.get("n_thinking_tokens") or 0) + (t.get("n_answer_tokens") or 0)
        records.append(dict(
            id=pid, source=p["source"], group="mc",
            direct_correct=bool(c["cheap_correct"]), direct_tokens=1,
            thinking_correct=bool(t_correct), thinking_tokens=t_tokens,
            confidence=float(c["max_prob"]), margin=float(c["margin"]),
        ))
    return records, skipped


def run_simulate_mc(args):
    problems = load_jsonl_multi(args.problems)
    thinking_rows = load_jsonl_multi(args.traces)
    cheap_rows = load_jsonl(resolve(args.cheap))
    records, skipped = build_mc_records(problems, thinking_rows, cheap_rows)
    print(f"built {len(records)} mc records ({skipped} skipped)", flush=True)

    direct_rows = load_jsonl_multi(args.direct) if args.direct else []
    direct_summary = None
    if direct_rows:
        problems_by_id = {p["id"]: p for p in problems}
        correct_n, tok_n, n = 0, 0, 0
        for d in direct_rows:
            p = problems_by_id.get(d["id"])
            if p is None:
                continue
            letter = extract_choice_letter(d.get("answer") or d.get("raw_text") or "")
            is_correct = letter is not None and letter == str(p.get("gold_answer")).upper()
            correct_n += int(is_correct)
            tok_n += (d.get("n_thinking_tokens") or 0) + (d.get("n_answer_tokens") or 0)
            n += 1
        if n:
            direct_summary = dict(n=n, accuracy=correct_n / n, mean_tokens=tok_n / n)

    lines = ["# Router test 2 report -- MC (mmlu + arc_challenge test splits, cheap pass = letter readout)",
             "", f"n={len(records)} (target 800 = 400 mmlu + 400 arc_challenge).", ""]
    if direct_summary:
        lines.append(f"(iii) no-think direct generation, for comparison only (not in the router sim): "
                     f"n={direct_summary['n']}, accuracy={direct_summary['accuracy']:.3f}, "
                     f"mean_tokens={direct_summary['mean_tokens']:.1f}")
        lines.append("")

    dev, test = split_dev_test(records, seed=0)
    tau, dev_info = choose_tau(dev)
    always_think = accounting(test, policy_always_thinking)
    always_direct = accounting(test, policy_always_direct)
    oracle_pol = accounting(test, policy_oracle)
    chosen = accounting(test, policy_tau(tau))
    saving = (1 - chosen["mean_tokens"] / always_think["mean_tokens"]) if always_think["mean_tokens"] else 0
    lines.append(f"Single 50/50 split (seed=0, full dev halves): dev n={len(dev)}, test n={len(test)}, "
                 f"chosen tau={tau}")
    lines.append("")
    lines.append("| policy | accuracy | mean tokens | % escalated | token saving vs always-thinking |")
    lines.append("|---|---|---|---|---|")
    for name, info in [("always-thinking", always_think), ("always-direct (letter readout)", always_direct),
                        ("oracle router", oracle_pol), (f"confidence router (tau={tau})", chosen)]:
        s = (1 - info["mean_tokens"] / always_think["mean_tokens"]) if always_think["mean_tokens"] else 0
        lines.append(f"| {name} | {info['accuracy']:.3f} | {info['mean_tokens']:.1f} | "
                     f"{info['pct_escalated']:.3f} | {s:.3%} |")
    lines.append("")
    acc_loss = always_think["accuracy"] - chosen["accuracy"]
    lines.append(f"token saving at chosen tau: {saving:.3%}; accuracy loss vs always-thinking: "
                 f"{acc_loss:.3f}")
    lines.append("")
    lo, hi = bootstrap_ci_accuracy_diff(test, tau)
    lines.append(f"pooled paired 95% bootstrap CI on accuracy diff (chosen tau - always-thinking), this "
                 f"split: [{lo:.3f}, {hi:.3f}]")
    lines.append("")
    n_cw = confident_wrong_count(test, tau)
    lines.append(f"confident-but-wrong direct answers on test: {n_cw}/{len(test)}")
    lines.append("")

    lines.append("## Repeated 50/50 splits (200 splits, seeds 0..199)")
    lines.append("")
    stats = repeated_split_eval(records, n_splits=200, base_seed=0)
    lines.append(f"mean test loss: {stats['mean_loss']:.4f}; p90 test loss: {stats['p90_loss']:.4f}; "
                 f"mean token saving: {stats['mean_saving']:.3%}; fraction of splits meeting "
                 f"loss<=1pt & saving>=30%: {stats['frac_meeting_bar']:.3f}")
    lines.append("")
    bar_pass = stats["frac_meeting_bar"] >= 0.80
    lines.append(f"**pre-registered bar (>=30% token saving AND <=1pt mean test loss in >=80% of "
                 f"splits): {'PASS' if bar_pass else 'FAIL'}**")
    lines.append("")

    lines.append("### tau curve (test, seed=0 split)")
    lines.append("")
    lines.append("| tau | accuracy | mean tokens | % escalated |")
    lines.append("|---|---|---|---|")
    for t in candidate_taus(test):
        info = accounting(test, policy_tau(t))
        lines.append(f"| {t:.4f} | {info['accuracy']:.3f} | {info['mean_tokens']:.1f} | "
                     f"{info['pct_escalated']:.3f} |")
    lines.append("")

    report = "\n".join(lines) + "\n"
    with open(resolve(args.out_report), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"wrote {args.out_report}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# selftest (pure logic: no model, no files)
# ---------------------------------------------------------------------------

def selftest():
    # auroc: perfect separation, chance, ties.
    a = auroc([0.1, 0.2, 0.8, 0.9], [False, False, True, True])
    assert a == 1.0, a
    print(f"selftest: auroc OK (perfect separation -> {a})")

    a2 = auroc([0.9, 0.8, 0.2, 0.1], [False, False, True, True])  # perfectly anti-correlated
    assert a2 == 0.0, a2
    print(f"selftest: auroc OK (perfect anti-correlation -> {a2})")

    a3 = auroc([1, 1, 1, 1], [True, False, True, False])  # all ties -> chance
    assert abs(a3 - 0.5) < 1e-9, a3
    print(f"selftest: auroc OK (all ties -> chance = {a3})")

    best, direction = best_direction_auroc([0.9, 0.8, 0.2, 0.1], [False, False, True, True])
    assert abs(best - 1.0) < 1e-9 and direction == -1, (best, direction)
    print(f"selftest: best_direction_auroc OK (flips anti-correlated signal -> auroc={best}, dir={direction})")

    # text_features: pure counts.
    feats = text_features("A has 3 apples. B has 10.5 apples! How many total?")
    assert feats["n_numbers"] == 2, feats
    assert feats["n_sentences"] == 3, feats
    assert feats["qlen"] == len("A has 3 apples. B has 10.5 apples! How many total?")
    print(f"selftest: text_features OK ({feats})")

    # entropy_from_logits: uniform -> log(n); one-hot -> 0.
    e_uniform = entropy_from_logits([0.0, 0.0, 0.0, 0.0])
    assert abs(e_uniform - math.log(4)) < 1e-6, e_uniform
    e_onehot = entropy_from_logits([100.0, 0.0, 0.0])
    assert e_onehot < 1e-6, e_onehot
    print(f"selftest: entropy_from_logits OK (uniform={e_uniform:.4f}=ln(4), one-hot={e_onehot:.6f})")

    # stratified_quota: sums to total, proportional-ish.
    q = stratified_quota({"a": 100, "b": 300}, 40)
    assert sum(q.values()) == 40, q
    assert q["b"] > q["a"], q  # b is 3x bigger -> gets more
    print(f"selftest: stratified_quota OK ({q}, sums to 40)")

    q2 = stratified_quota({"x": 1, "y": 1, "z": 1}, 10)
    assert sum(q2.values()) == 10, q2
    print(f"selftest: stratified_quota OK (even split, remainder handled: {q2})")

    # repeated_split_eval: fabricated records where a tau exists that strictly dominates always-thinking
    # (escalates only the wrong-direct ones) -- every split should find it and report near-zero loss,
    # positive saving.
    recs = []
    for i in range(40):
        # even i: direct correct (cheap, no need to escalate); odd i: direct wrong, thinking right.
        direct_correct = (i % 2 == 0)
        recs.append(dict(
            id=f"mmlu-{i}", source="mmlu", group="mc", direct_correct=direct_correct, direct_tokens=10,
            thinking_correct=True, thinking_tokens=100,
            confidence=0.9 if direct_correct else 0.1,
        ))
    stats = repeated_split_eval(recs, n_splits=20, base_seed=0)
    assert stats["n_splits_used"] == 20, stats
    assert stats["mean_loss"] <= 0.01, stats  # should recover full accuracy (loss <= 0, clipped at 1pt bar)
    assert stats["mean_saving"] > 0.3, stats
    print(f"selftest: repeated_split_eval OK (dominant-tau fabricated data: {stats})")

    print("selftest: all checks passed")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")

    sp = sub.add_parser("fetch-gsm8k")
    sp.add_argument("--out", default="oracle/router2-gsm8k-fresh.jsonl")
    sp.add_argument("--n", type=int, default=300)
    sp.add_argument("--seed", type=int, default=0)

    sp = sub.add_parser("fetch-mc")
    sp.add_argument("--out", default="oracle/router2-mc-fresh.jsonl")
    sp.add_argument("--n-mmlu", type=int, default=400)
    sp.add_argument("--n-arc", type=int, default=400)
    sp.add_argument("--seed", type=int, default=0)

    sp = sub.add_parser("score-gsm8k")
    sp.add_argument("--problems", nargs="+", default=["oracle/s1-train-problems.jsonl",
                                                       "oracle/router2-gsm8k-fresh.jsonl"])
    sp.add_argument("--out", default="oracle/router2-gsm8k-cheap.jsonl")
    sp.add_argument("--resume", action="store_true")
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--batch-size", type=int, default=16)
    sp.add_argument("--max-new-tokens", type=int, default=8)
    sp.add_argument("--model", default="google/gemma-4-E4B-it")
    sp.add_argument("--revision", default="ee0ef6023621cff504d758262d4e04895a5af4a2")
    sp.add_argument("--device", default="cuda")
    sp.add_argument("--attn-impl", default="sdpa")

    sp = sub.add_parser("score-mc")
    sp.add_argument("--problems", nargs="+", default=["oracle/router2-mc-fresh.jsonl"])
    sp.add_argument("--out", default="oracle/router2-mc-cheap.jsonl")
    sp.add_argument("--resume", action="store_true")
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--batch-size", type=int, default=32)
    sp.add_argument("--model", default="google/gemma-4-E4B-it")
    sp.add_argument("--revision", default="ee0ef6023621cff504d758262d4e04895a5af4a2")
    sp.add_argument("--device", default="cuda")
    sp.add_argument("--attn-impl", default="sdpa")

    sp = sub.add_parser("simulate-gsm8k")
    sp.add_argument("--problems", nargs="+", default=["oracle/s1-train-problems.jsonl",
                                                       "oracle/router2-gsm8k-fresh.jsonl"])
    sp.add_argument("--traces", nargs="+", default=["oracle/s1-traces.jsonl",
                                                     "oracle/router2-gsm8k-thinking.jsonl"])
    sp.add_argument("--cheap", default="oracle/router2-gsm8k-cheap.jsonl")
    sp.add_argument("--out-report", default="oracle/router-test2-gsm8k.md")

    sp = sub.add_parser("simulate-mc")
    sp.add_argument("--problems", nargs="+", default=["oracle/router2-mc-fresh.jsonl"])
    sp.add_argument("--traces", nargs="+", default=["oracle/router2-mc-thinking.jsonl"])
    sp.add_argument("--direct", nargs="+", default=None)
    sp.add_argument("--cheap", default="oracle/router2-mc-cheap.jsonl")
    sp.add_argument("--out-report", default="oracle/router-test2-mc.md")

    args = ap.parse_args()

    if args.selftest:
        return sys.exit(selftest())
    dispatch = {
        "fetch-gsm8k": run_fetch_gsm8k, "fetch-mc": run_fetch_mc,
        "score-gsm8k": run_score_gsm8k, "score-mc": run_score_mc,
        "simulate-gsm8k": run_simulate_gsm8k, "simulate-mc": run_simulate_mc,
    }
    if args.cmd in dispatch:
        return sys.exit(dispatch[args.cmd](args))
    ap.error("one of --selftest, fetch-gsm8k, fetch-mc, score-gsm8k, score-mc, simulate-gsm8k, "
            "simulate-mc is required")


if __name__ == "__main__":
    main()
