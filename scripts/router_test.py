"""Phase 7G router test: can a cheap pre-thinking confidence signal from a no-think direct pass decide
which problems can skip thinking, on google/gemma-4-E4B-it (BASE instruction model, no training, no
router/head -- plain teacher-forced confidence readouts)?

Inputs (all already on disk, no regeneration):
  - oracle/format-probe.jsonl: per-problem "direct" arm rows (no-think greedy generation; correct, tokens,
    generated_text, parsed_value) -- scripts/format_probe.py.
  - oracle/s1-traces.jsonl: "thinking" baseline traces (n_thinking_tokens/n_answer_tokens; correctness via
    scripts/s1_head_and_outcomes.py's judge_correctness), joined by id.
  - oracle/s1-train-problems.jsonl: gold answers / answer_type / source.
  - oracle/router-conf.jsonl (produced by this script's `score` subcommand, GPU box): one row per problem,
    {id, source, answer_type, confidence, max_prob, margin} -- see `score` docstring below for the exact
    per-type definition.

Two independent pieces, like scripts/s1_head_and_outcomes.py's two subcommands:

  score    -- GPU box. Teacher-forced forward passes (no generation) against the model to build a
              confidence score for each problem's EXISTING direct answer (oracle/format-probe.jsonl,
              arm=="direct"). No sampling anywhere in this subcommand.
                MC (mmlu/arc_challenge): prompt = plain question (the problem's "user" text, which already
                embeds "A. ...\\nB. ...\\nC. ...\\nD. ..."), rendered through the chat template exactly
                like format_probe.py's "direct" arm but with an added trailing instruction asking for the
                letter only, enable_thinking=False, one bare user turn, add_generation_prompt=True. One
                forward pass over the prompt alone (no generated tokens appended) gives next-token logits
                at the final prompt position; softmax restricted to the 4 candidate letter token ids (each
                option letter's conventional "after a newline" token id -- encode "\\nA" style per letter
                and take its last token id, since that's the token distribution actually competing at a
                fresh line in this chat template; see `_letter_token_id`). confidence = that restricted
                softmax's probability mass on the LETTER THE DIRECT ARM ACTUALLY ANSWERED (format-probe's
                parsed_value for that row), not necessarily the model's own top pick here (the direct
                arm's generation and this teacher-forced single-token readout are two different passes
                over two slightly different prompts -- the "direct" row's own chosen letter is what the
                router would be trusting, so that is what gets scored). max_prob = the restricted softmax's
                own top probability (over the 4 letters, regardless of which one). margin = top1 - top2 of
                that restricted softmax. A direct row with no parsed letter (parse_failure) still gets a
                confidence row -- confidence is then 0.0 by convention (nothing to trust), max_prob/margin
                still computed from the readout.
                GSM8K (numeric): teacher-forced pass over prompt + the direct arm's OWN generated_text
                (string concat, each tokenized separately with the SAME tokenizer calls format_probe.py's
                direct arm used -- this is an approximation of the exact generation-time tokens, see
                `locate_number_token_span`'s docstring for the boundary caveat), reading per-generated-
                token logprobs of the actually-generated tokens (teacher forcing: logits at global position
                len(prompt_ids)+i-1 predict gen_ids[i]). Locates the LAST number in generated_text (same
                _NUM_RE rule as s1_head_and_outcomes.extract_last_number) and takes the token span whose
                character offsets overlap that number (via a fresh tokenization of generated_text alone
                with return_offsets_mapping=True). confidence = mean token logprob over that span (so it is
                directly comparable to a log-probability, NOT a 0-1 softmax mass -- deliberately a
                different scale than the MC confidence; tau is chosen per-group, never pooled across types,
                so the scales never need to be compared to each other). max_prob here is unused (set to
                None); margin stores the MIN token logprob over that same span (the task's "min" stat), so
                this subcommand always fills all four output fields but MC and numeric fill them with
                different meanings -- documented, not a bug.
              Problems whose direct row has no matching number at all (parse_failure, no number anywhere
              in generated_text) get confidence = mean logprob over the WHOLE generated_text's tokens
              instead (a last-resort "how confident was this generation overall" fallback) and margin = the
              min over the same span; never dropped.
              Usage (GPU box): .venv/bin/python -u scripts/router_test.py score \\
                  --format-probe oracle/format-probe.jsonl --problems oracle/s1-train-problems.jsonl \\
                  --out oracle/router-conf.jsonl --resume

  simulate -- CPU only. Joins format-probe direct rows + s1-traces thinking rows + router-conf confidence
              scores + problems (for source/answer_type), splits 50/50 dev/test per source (fixed seed 0,
              stable sort by id), groups into "mc" (mmlu+arc_challenge pooled) and "gsm8k", and for each
              group: chooses tau on dev as the threshold (over the group's own confidence values, i.e. only
              thresholds that actually change the dev partition are tried) that gives the LARGEST token
              saving vs always-thinking subject to accuracy loss <= 1 point vs always-thinking on dev; then
              reports, on TEST ONLY: accuracy / mean tokens / %escalated for the chosen tau, token saving vs
              always-thinking, the always-thinking/always-direct/oracle-router baselines (oracle router:
              escalate a problem iff direct was wrong AND thinking was right -- direct tokens are still
              paid for every problem, same accounting as the confidence router), the full tau curve
              (accuracy, mean tokens, %escalated at every candidate tau) on TEST, a 95% bootstrap CI
              (percentile, 2000 resamples, seed 0) on (chosen-tau accuracy - always-thinking accuracy) on
              TEST, and a count of "confident-wrong" direct answers (confidence >= chosen tau but direct
              was actually wrong -- the router's actual failure mode).
              Cost accounting (every policy): total tokens = direct_tokens summed over ALL problems in the
              group (the failed direct attempt is always paid) + thinking_tokens summed over only the
              ESCALATED problems.
              Usage (CPU, laptop): .venv/Scripts/python.exe scripts/router_test.py simulate \\
                  --format-probe oracle/format-probe.jsonl --traces oracle/s1-traces.jsonl \\
                  --problems oracle/s1-train-problems.jsonl --conf oracle/router-conf.jsonl \\
                  --out-report oracle/router-test.md

Pre-registered bar (reports/25-phase7g-general-decisions-plan.md's router-test addendum): MC pooled TEST
must show >=30% fewer total tokens than always-thinking with accuracy within 1 point. gsm8k reported
separately, no pre-registered bar.

Offline unit tests (routing/accounting logic + the pure confidence-math helpers; no model, no GPU, no
files read):
  .venv/Scripts/python.exe scripts/router_test.py --selftest
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

from scripts.s1_head_and_outcomes import judge_correctness  # noqa: E402

MC_SOURCES = ("mmlu", "arc_challenge")
GSM8K_SOURCE = "gsm8k"
LETTERS = ("A", "B", "C", "D")
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_UNIT_STRIP_RE = re.compile(r"[\$%]")


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows, mode="w"):
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# score: pure confidence-math helpers (no torch -- take plain floats/logits in)
# ---------------------------------------------------------------------------

def softmax_restricted(logits_by_key):
    """logits_by_key: {key: float}. -> {key: prob} softmax over exactly these keys (no other vocab mass)."""
    keys = list(logits_by_key)
    vals = [logits_by_key[k] for k in keys]
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    z = sum(exps)
    return {k: e / z for k, e in zip(keys, exps)}


def mc_confidence_from_logits(letter_logits, chosen_letter):
    """letter_logits: {letter: float} for exactly LETTERS (or a subset actually present). chosen_letter:
    the direct arm's own parsed letter, or None (parse_failure). -> (confidence, max_prob, margin).
    confidence=0.0 by convention when chosen_letter is None or not among letter_logits' keys (nothing to
    trust). max_prob/margin are always computed from the restricted softmax regardless."""
    probs = softmax_restricted(letter_logits)
    ranked = sorted(probs.values(), reverse=True)
    max_prob = ranked[0]
    margin = ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)
    confidence = probs.get(chosen_letter, 0.0) if chosen_letter else 0.0
    return confidence, max_prob, margin


def numeric_confidence_from_logprobs(token_logprobs):
    """token_logprobs: list[float], the per-token logprobs of the answer-number span (or, as a fallback,
    of the whole generation). -> (confidence=mean, margin=min). Empty list -> (None, None) (caller decides
    what to do; score's main loop never calls this with an empty list, see its own fallback)."""
    if not token_logprobs:
        return None, None
    return sum(token_logprobs) / len(token_logprobs), min(token_logprobs)


def locate_number_span_char_offsets(text):
    """-> (start, end) char offsets of the LAST number in `text` (same _NUM_RE as
    s1_head_and_outcomes.extract_last_number), or None if no number is found."""
    cleaned_is_same_length = not _UNIT_STRIP_RE.search(text or "")
    # _NUM_RE is searched on the raw text directly (not the unit-stripped copy used for parsing the VALUE)
    # so offsets stay valid against the original string; $/% immediately adjacent to a number do not
    # affect which digits match, only extract_last_number's parsed float value does unit-stripping.
    del cleaned_is_same_length
    matches = list(_NUM_RE.finditer(text or ""))
    if not matches:
        return None
    last = matches[-1]
    return last.start(), last.end()


# ---------------------------------------------------------------------------
# score: GPU-dependent (imported lazily; selftest never touches this section)
# ---------------------------------------------------------------------------

def _letter_token_id(tok, letter):
    """-> int token id for the token actually competing at a fresh line in the chat template's generation
    position: encode "\\n<letter>" and take its LAST token id (the letter itself, however the tokenizer
    happens to merge the preceding newline) -- same style as how the model would continue right after the
    rendered prompt's trailing newline."""
    ids = tok("\n" + letter, add_special_tokens=False)["input_ids"]
    return ids[-1]


def render_mc_prompt(tok, user_text):
    messages = [{"role": "user", "content": user_text + "\n\nAnswer with the letter only."}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)


def render_direct_prompt(tok, user_text):
    """Same rendering format_probe.py's 'direct' arm used (bare user turn, enable_thinking=False) -- the
    prompt the existing oracle/format-probe.jsonl generated_text was actually produced against."""
    messages = [{"role": "user", "content": user_text}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)


def score_mc_row(tok, model, device, problem, direct_row):
    import torch
    prompt = render_mc_prompt(tok, problem["user"])
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        logits = model(**enc).logits[0, -1, :]
    letter_ids = {letter: _letter_token_id(tok, letter) for letter in LETTERS}
    letter_logits = {letter: logits[tid].item() for letter, tid in letter_ids.items()}
    chosen = direct_row["parsed_value"] if not direct_row["parse_failure"] else None
    confidence, max_prob, margin = mc_confidence_from_logits(letter_logits, chosen)
    return confidence, max_prob, margin


def score_gsm8k_row(tok, model, device, problem, direct_row):
    import torch
    prompt = render_direct_prompt(tok, problem["user"])
    generated_text = direct_row["generated_text"]
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    gen_enc = tok(generated_text, add_special_tokens=False, return_offsets_mapping=True)
    gen_ids = gen_enc["input_ids"]
    offsets = gen_enc["offset_mapping"]
    if not gen_ids:
        return None, None, None
    full_ids = prompt_ids + gen_ids
    enc = torch.tensor([full_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids=enc).logits[0]
    log_probs_all = torch.log_softmax(logits.float(), dim=-1)
    n_prompt = len(prompt_ids)
    per_token_logprob = []
    for i, tid in enumerate(gen_ids):
        pos = n_prompt + i - 1
        per_token_logprob.append(log_probs_all[pos, tid].item())

    span = locate_number_span_char_offsets(generated_text)
    if span is None:
        token_logprobs = per_token_logprob  # fallback: whole generation
    else:
        start, end = span
        token_logprobs = [lp for lp, (tok_start, tok_end) in zip(per_token_logprob, offsets)
                           if tok_end > start and tok_start < end]
        if not token_logprobs:
            token_logprobs = per_token_logprob
    confidence, margin = numeric_confidence_from_logprobs(token_logprobs)
    return confidence, None, margin


def run_score(args):
    import torch
    from kev.model import load_tokenizer

    fp_rows = [r for r in load_jsonl(resolve(args.format_probe)) if r["arm"] == "direct"]
    problems_by_id = {p["id"]: p for p in load_jsonl(resolve(args.problems))}

    done = set()
    out_path = resolve(args.out)
    if args.resume and os.path.exists(out_path):
        done = {r["id"] for r in load_jsonl(out_path)}
        print(f"--resume: {len(done)} ids already scored", flush=True)

    todo = [r for r in fp_rows if r["id"] not in done and r["id"] in problems_by_id]
    if args.limit:
        todo = todo[:args.limit]
    print(f"scoring {len(todo)} rows", flush=True)
    if not todo:
        return 0

    tok = load_tokenizer(args.model, args.revision)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, attn_implementation=args.attn_impl
    ).to(args.device).eval()

    mode = "a" if (args.resume and done) else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        for i, row in enumerate(todo):
            problem = problems_by_id[row["id"]]
            if problem["source"] in MC_SOURCES:
                confidence, max_prob, margin = score_mc_row(tok, model, args.device, problem, row)
            elif problem["source"] == GSM8K_SOURCE:
                confidence, max_prob, margin = score_gsm8k_row(tok, model, args.device, problem, row)
            else:
                continue
            out = dict(id=row["id"], source=problem["source"], answer_type=problem["answer_type"],
                        confidence=confidence, max_prob=max_prob, margin=margin)
            f.write(json.dumps(out) + "\n")
            f.flush()
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                print(f"scored {i + 1}/{len(todo)}", flush=True)
    print(f"done -> {args.out}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# simulate: joining + grouping (pure Python)
# ---------------------------------------------------------------------------

def build_records(fp_rows, thinking_rows, conf_rows, problems_by_id):
    """-> list of dicts, one per problem id present in all four sources: {id, source, group ("mc"/
    "gsm8k"), direct_correct, direct_tokens, thinking_correct, thinking_tokens, confidence}. Problems
    missing any one of direct/thinking/confidence are skipped (reported by the caller)."""
    direct_by_id = {r["id"]: r for r in fp_rows if r["arm"] == "direct"}
    thinking_by_id = {r["id"]: r for r in thinking_rows}
    conf_by_id = {r["id"]: r for r in conf_rows}
    records = []
    skipped = 0
    for pid, problem in problems_by_id.items():
        source = problem.get("source")
        if source in MC_SOURCES:
            group = "mc"
        elif source == GSM8K_SOURCE:
            group = "gsm8k"
        else:
            continue
        d = direct_by_id.get(pid)
        t = thinking_by_id.get(pid)
        c = conf_by_id.get(pid)
        if d is None or t is None or c is None or c.get("confidence") is None:
            skipped += 1
            continue
        records.append(dict(
            id=pid, source=source, group=group,
            direct_correct=bool(d["correct"]), direct_tokens=int(d["tokens"]),
            thinking_correct=bool(t["correct"]), thinking_tokens=int(t["tokens"]),
            confidence=float(c["confidence"]),
        ))
    return records, skipped


def split_dev_test(records, seed=0):
    """-> (dev, test). 50/50 per SOURCE (not group), fixed seed, deterministic given input order --
    sorted by id first so result doesn't depend on input file order, then shuffled with `seed`."""
    by_source = collections.defaultdict(list)
    for r in records:
        by_source[r["source"]].append(r)
    dev, test = [], []
    for source in sorted(by_source):
        rs = sorted(by_source[source], key=lambda r: r["id"])
        rng = random.Random(seed)
        rng.shuffle(rs)
        half = len(rs) // 2
        dev += rs[:half]
        test += rs[half:]
    return dev, test


def accounting(records, escalate_fn):
    """escalate_fn(record) -> bool. -> dict(n, accuracy, mean_tokens, pct_escalated, total_tokens)."""
    n = len(records)
    if n == 0:
        return dict(n=0, accuracy=float("nan"), mean_tokens=float("nan"), pct_escalated=float("nan"),
                    total_tokens=0)
    total_tokens = 0
    n_correct = 0
    n_escalated = 0
    for r in records:
        total_tokens += r["direct_tokens"]
        if escalate_fn(r):
            n_escalated += 1
            total_tokens += r["thinking_tokens"]
            n_correct += int(r["thinking_correct"])
        else:
            n_correct += int(r["direct_correct"])
    return dict(n=n, accuracy=n_correct / n, mean_tokens=total_tokens / n,
                pct_escalated=n_escalated / n, total_tokens=total_tokens)


def policy_always_thinking(_r):
    return True


def policy_always_direct(_r):
    return False


def policy_oracle(r):
    return (not r["direct_correct"]) and r["thinking_correct"]


def policy_tau(tau):
    return lambda r: r["confidence"] < tau


def candidate_taus(records):
    """-> sorted unique confidence values seen in `records`, each usable as a threshold that actually
    changes the partition (tau == a value means that value itself is kept, per policy_tau's `<` test)."""
    return sorted({r["confidence"] for r in records})


def choose_tau(dev_records, max_accuracy_loss=0.01):
    """-> (tau, dev_accounting) maximizing token saving vs always-thinking on `dev_records` subject to
    accuracy loss <= `max_accuracy_loss` vs always-thinking on dev. Falls back to tau=+inf (always
    escalate, i.e. always-thinking) if no candidate tau satisfies the constraint."""
    baseline = accounting(dev_records, policy_always_thinking)
    best_tau = float("inf")
    best_acc_info = baseline
    best_saving = 0.0
    for tau in candidate_taus(dev_records):
        info = accounting(dev_records, policy_tau(tau))
        if baseline["accuracy"] - info["accuracy"] > max_accuracy_loss:
            continue
        saving = baseline["total_tokens"] - info["total_tokens"]
        if saving > best_saving:
            best_saving = saving
            best_tau = tau
            best_acc_info = info
    return best_tau, best_acc_info


def bootstrap_ci_accuracy_diff(test_records, tau, n_resamples=2000, seed=0):
    """-> (lo, hi) 95% percentile bootstrap CI on (tau-policy accuracy - always-thinking accuracy) over
    `test_records`, resampling records with replacement."""
    rng = random.Random(seed)
    n = len(test_records)
    diffs = []
    for _ in range(n_resamples):
        sample = [test_records[rng.randrange(n)] for _ in range(n)]
        a_tau = accounting(sample, policy_tau(tau))["accuracy"]
        a_think = accounting(sample, policy_always_thinking)["accuracy"]
        diffs.append(a_tau - a_think)
    diffs.sort()
    lo = diffs[int(0.025 * n_resamples)]
    hi = diffs[min(int(0.975 * n_resamples), n_resamples - 1)]
    return lo, hi


def confident_wrong_count(records, tau):
    return sum(1 for r in records if r["confidence"] >= tau and not r["direct_correct"])


# ---------------------------------------------------------------------------
# simulate: report rendering
# ---------------------------------------------------------------------------

def render_group_report(group_name, dev, test, tau, dev_acc_info, skipped_note=""):
    lines = [f"## {group_name}", ""]
    if skipped_note:
        lines.append(skipped_note)
        lines.append("")
    lines.append(f"dev n={len(dev)}, test n={len(test)}. Chosen tau (dev, largest token saving with "
                 f"accuracy loss <= 1pt vs always-thinking): **{tau}**")
    lines.append(f"dev accounting at chosen tau: accuracy={dev_acc_info['accuracy']:.3f}, "
                 f"mean_tokens={dev_acc_info['mean_tokens']:.1f}, "
                 f"pct_escalated={dev_acc_info['pct_escalated']:.3f}")
    lines.append("")

    always_think = accounting(test, policy_always_thinking)
    always_direct = accounting(test, policy_always_direct)
    oracle = accounting(test, policy_oracle)
    chosen = accounting(test, policy_tau(tau))
    saving = (1 - chosen["mean_tokens"] / always_think["mean_tokens"]) if always_think["mean_tokens"] else 0

    lines.append("### TEST results")
    lines.append("")
    lines.append("| policy | accuracy | mean tokens | % escalated | token saving vs always-thinking |")
    lines.append("|---|---|---|---|---|")
    for name, info in [("always-thinking", always_think), ("always-direct", always_direct),
                        ("oracle router", oracle), (f"confidence router (tau={tau})", chosen)]:
        s = (1 - info["mean_tokens"] / always_think["mean_tokens"]) if always_think["mean_tokens"] else 0
        lines.append(f"| {name} | {info['accuracy']:.3f} | {info['mean_tokens']:.1f} | "
                     f"{info['pct_escalated']:.3f} | {s:.3%} |")
    lines.append("")

    acc_loss = always_think["accuracy"] - chosen["accuracy"]
    bar_pass = (group_name == "mc" and saving >= 0.30 and acc_loss <= 0.01)
    lines.append(f"token saving at chosen tau: {saving:.3%}; accuracy loss vs always-thinking: "
                 f"{acc_loss:.3f}")
    if group_name == "mc":
        lines.append(f"**pre-registered bar (MC: >=30% token saving, <=1pt accuracy loss): "
                     f"{'PASS' if bar_pass else 'FAIL'}**")
    lines.append("")

    lo, hi = bootstrap_ci_accuracy_diff(test, tau)
    lines.append(f"95% bootstrap CI (2000 resamples) on accuracy diff (chosen tau - always-thinking) on "
                 f"test: [{lo:.3f}, {hi:.3f}]")
    lines.append("")

    n_cw = confident_wrong_count(test, tau)
    lines.append(f"confident-but-wrong direct answers on test (confidence >= tau, direct was wrong): "
                 f"{n_cw}/{len(test)}")
    lines.append("")

    lines.append("### tau curve (test)")
    lines.append("")
    lines.append("| tau | accuracy | mean tokens | % escalated |")
    lines.append("|---|---|---|---|")
    for t in candidate_taus(test):
        info = accounting(test, policy_tau(t))
        lines.append(f"| {t:.4f} | {info['accuracy']:.3f} | {info['mean_tokens']:.1f} | "
                     f"{info['pct_escalated']:.3f} |")
    lines.append("")
    return "\n".join(lines)


def run_simulate(args):
    fp_rows = load_jsonl(resolve(args.format_probe))
    thinking_rows_raw = load_jsonl(resolve(args.traces))
    problems = load_jsonl(resolve(args.problems))
    conf_rows = load_jsonl(resolve(args.conf))
    problems_by_id = {p["id"]: p for p in problems}

    # thinking correctness via judge_correctness (same rule as format_probe.py's thinking_rows_for).
    thinking_rows = []
    for t in thinking_rows_raw:
        p = problems_by_id.get(t["id"])
        if p is None:
            continue
        correct, _parsed, _unparsed = judge_correctness(t, p)
        tokens = (t.get("n_thinking_tokens") or 0) + (t.get("n_answer_tokens") or 0)
        thinking_rows.append(dict(id=t["id"], correct=bool(correct), tokens=tokens))

    records, skipped = build_records(fp_rows, thinking_rows, conf_rows, problems_by_id)
    print(f"built {len(records)} records ({skipped} problems skipped: missing direct/thinking/conf)",
          flush=True)

    dev, test = split_dev_test(records, seed=args.seed)
    dev_mc = [r for r in dev if r["group"] == "mc"]
    test_mc = [r for r in test if r["group"] == "mc"]
    dev_gsm8k = [r for r in dev if r["group"] == "gsm8k"]
    test_gsm8k = [r for r in test if r["group"] == "gsm8k"]

    report_lines = ["# Router test report", "",
                     "Caveat: dev/test is a single 50/50 split per source, fixed seed "
                     f"{args.seed} -- tau is chosen on dev only, reported numbers are TEST only, no "
                     "cross-validation / repeated splits.", ""]
    for name, dev_group, test_group in [("mc", dev_mc, test_mc), ("gsm8k", dev_gsm8k, test_gsm8k)]:
        if not dev_group or not test_group:
            report_lines.append(f"## {name}\n\n(no records -- skipped)\n")
            continue
        tau, dev_acc_info = choose_tau(dev_group)
        report_lines.append(render_group_report(name, dev_group, test_group, tau, dev_acc_info))

    report = "\n".join(report_lines) + "\n"
    out_path = resolve(args.out_report)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"wrote {out_path}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# selftest (pure logic: confidence math + routing/accounting, no model, no files)
# ---------------------------------------------------------------------------

def selftest():
    # 1. softmax_restricted / mc_confidence_from_logits.
    probs = softmax_restricted({"A": 2.0, "B": 0.0})
    assert abs(probs["A"] + probs["B"] - 1.0) < 1e-9
    assert probs["A"] > probs["B"]
    print("selftest: softmax_restricted OK (sums to 1, ranks logits correctly)")

    conf, max_prob, margin = mc_confidence_from_logits({"A": 3.0, "B": 0.0, "C": 0.0, "D": 0.0}, "A")
    assert max_prob == conf  # chosen letter IS the top letter here
    assert margin > 0.8
    print(f"selftest: mc_confidence_from_logits OK (chosen==top: conf={conf:.4f}, margin={margin:.4f})")

    conf2, max_prob2, _m = mc_confidence_from_logits({"A": 3.0, "B": 0.0, "C": 0.0, "D": 0.0}, "B")
    assert conf2 < max_prob2  # chosen letter B is NOT the top letter (A is) -- confidence reflects B's
    # own (low) mass, not the readout's top pick
    print(f"selftest: mc_confidence_from_logits OK (chosen!=top: conf={conf2:.4f} < max_prob={max_prob2:.4f})")

    conf3, _mp, _mg = mc_confidence_from_logits({"A": 1.0, "B": 1.0}, None)
    assert conf3 == 0.0  # parse_failure (no chosen letter) -> confidence 0.0 by convention
    print("selftest: mc_confidence_from_logits OK (parse_failure -> confidence 0.0)")

    # 2. numeric_confidence_from_logprobs.
    mean_lp, min_lp = numeric_confidence_from_logprobs([-0.1, -0.2, -2.0])
    assert abs(mean_lp - (-0.1 - 0.2 - 2.0) / 3) < 1e-9
    assert min_lp == -2.0
    assert numeric_confidence_from_logprobs([]) == (None, None)
    print("selftest: numeric_confidence_from_logprobs OK (mean/min, empty case)")

    # 3. locate_number_span_char_offsets: last number wins, handles $/% adjacency.
    text = "Intermediate 12 steps. The answer is $42."
    span = locate_number_span_char_offsets(text)
    assert span is not None and text[span[0]:span[1]] == "42.", (span, text[span[0]:span[1]])
    assert locate_number_span_char_offsets("no numbers") is None
    print(f"selftest: locate_number_span_char_offsets OK (span={span!r} -> {text[span[0]:span[1]]!r})")

    # 4. accounting: always-thinking / always-direct / oracle / tau accounting on a fabricated mix.
    recs = [
        dict(id="a", source="mmlu", group="mc", direct_correct=True, direct_tokens=10,
             thinking_correct=True, thinking_tokens=100, confidence=0.9),
        dict(id="b", source="mmlu", group="mc", direct_correct=False, direct_tokens=10,
             thinking_correct=True, thinking_tokens=100, confidence=0.2),
        dict(id="c", source="mmlu", group="mc", direct_correct=False, direct_tokens=10,
             thinking_correct=False, thinking_tokens=100, confidence=0.1),
        dict(id="d", source="mmlu", group="mc", direct_correct=True, direct_tokens=10,
             thinking_correct=True, thinking_tokens=100, confidence=0.95),
    ]
    at = accounting(recs, policy_always_thinking)
    assert at["accuracy"] == 0.75 and at["total_tokens"] == 4 * 10 + 4 * 100 and at["pct_escalated"] == 1.0
    ad = accounting(recs, policy_always_direct)
    assert ad["accuracy"] == 0.5 and ad["total_tokens"] == 4 * 10 and ad["pct_escalated"] == 0.0
    orc = accounting(recs, policy_oracle)
    # oracle escalates b (wrong direct, right thinking) and NOT c (wrong direct, wrong thinking -- no
    # point escalating) and not a/d (already right).
    assert orc["accuracy"] == 0.75, orc  # a,d,b correct; c still wrong even escalated-or-not
    assert orc["total_tokens"] == 4 * 10 + 1 * 100, orc
    print(f"selftest: accounting OK (always-thinking={at}, always-direct={ad}, oracle={orc})")

    tau_policy = accounting(recs, policy_tau(0.5))  # escalates b, c (confidence < 0.5)
    assert tau_policy["pct_escalated"] == 0.5
    assert tau_policy["accuracy"] == 0.75  # a,d correct (kept direct); b correct (escalated, thinking
    # right); c wrong (escalated, thinking also wrong)
    print(f"selftest: policy_tau OK ({tau_policy})")

    # 5. candidate_taus / choose_tau: a scenario where escalating the two wrong-direct problems clears the
    # 1pt bar and saves tokens vs always-thinking.
    tau, info = choose_tau(recs, max_accuracy_loss=0.01)
    assert tau == 0.9, tau  # escalates exactly b,c (both below 0.9) -- the minimal escalation that
    # recovers accuracy; 0.95 would also clear the bound but escalates a too (wasted tokens, less saving)
    assert info["accuracy"] >= at["accuracy"] - 0.01
    print(f"selftest: choose_tau OK (tau={tau}, dev accuracy={info['accuracy']:.3f})")

    # 6. choose_tau falls back to always-thinking (tau=inf) when NO threshold clears the accuracy bar.
    strict_recs = [
        dict(id="x", source="mmlu", group="mc", direct_correct=False, direct_tokens=10,
             thinking_correct=True, thinking_tokens=100, confidence=0.99),  # high-confidence but WRONG
    ]
    tau_strict, info_strict = choose_tau(strict_recs, max_accuracy_loss=0.01)
    assert tau_strict == float("inf") and info_strict["accuracy"] == 1.0
    print("selftest: choose_tau OK (no tau clears the bar -> falls back to always-thinking)")

    # 7. confident_wrong_count.
    assert confident_wrong_count(recs, 0.5) == 0  # a,d confident(>=0.5) and correct; b,c below 0.5
    assert confident_wrong_count(recs, 0.05) == 2  # now b,c (confidence >= 0.05, direct wrong) count
    print("selftest: confident_wrong_count OK")

    # 8. split_dev_test: deterministic, disjoint, covers everything, roughly 50/50 per source.
    many = [dict(id=f"mmlu-{i}", source="mmlu", group="mc", direct_correct=True, direct_tokens=1,
                 thinking_correct=True, thinking_tokens=1, confidence=0.5) for i in range(10)]
    dev, test = split_dev_test(many, seed=0)
    assert len(dev) == 5 and len(test) == 5
    assert {r["id"] for r in dev} | {r["id"] for r in test} == {r["id"] for r in many}
    assert not ({r["id"] for r in dev} & {r["id"] for r in test})
    dev2, _test2 = split_dev_test(many, seed=0)
    assert [r["id"] for r in dev2] == [r["id"] for r in dev]  # deterministic given fixed seed
    print("selftest: split_dev_test OK (50/50, disjoint, deterministic)")

    # 9. bootstrap_ci_accuracy_diff: sane shape (lo <= hi), zero-width when tau policy == always-thinking
    # policy identically (every record escalates).
    lo, hi = bootstrap_ci_accuracy_diff(recs, 1.0)  # tau=1.0 -> confidence always < 1.0 -> always escalate
    assert lo == 0.0 and hi == 0.0, (lo, hi)
    lo2, hi2 = bootstrap_ci_accuracy_diff(recs, 0.0)  # tau=0.0 -> never escalate (confidence never < 0)
    assert lo2 <= hi2
    print(f"selftest: bootstrap_ci_accuracy_diff OK (identical-policy CI=[{lo},{hi}], distinct=[{lo2:.3f},"
          f"{hi2:.3f}])")

    print("selftest: all checks passed")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")

    sp_score = sub.add_parser("score")
    sp_score.add_argument("--format-probe", default="oracle/format-probe.jsonl")
    sp_score.add_argument("--problems", default="oracle/s1-train-problems.jsonl")
    sp_score.add_argument("--out", default="oracle/router-conf.jsonl")
    sp_score.add_argument("--resume", action="store_true")
    sp_score.add_argument("--limit", type=int, default=None)
    sp_score.add_argument("--model", default="google/gemma-4-E4B-it")
    sp_score.add_argument("--revision", default="ee0ef6023621cff504d758262d4e04895a5af4a2")
    sp_score.add_argument("--device", default="cuda")
    sp_score.add_argument("--attn-impl", default="sdpa")

    sp_sim = sub.add_parser("simulate")
    sp_sim.add_argument("--format-probe", default="oracle/format-probe.jsonl")
    sp_sim.add_argument("--traces", default="oracle/s1-traces.jsonl")
    sp_sim.add_argument("--problems", default="oracle/s1-train-problems.jsonl")
    sp_sim.add_argument("--conf", default="oracle/router-conf.jsonl")
    sp_sim.add_argument("--out-report", default="oracle/router-test.md")
    sp_sim.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    if args.selftest:
        return sys.exit(selftest())
    if args.cmd == "score":
        return sys.exit(run_score(args))
    if args.cmd == "simulate":
        return sys.exit(run_simulate(args))
    ap.error("one of --selftest, score, or simulate is required")


if __name__ == "__main__":
    main()
