"""Phase 7G format probe: few-shot "decision-diagram" format vs plain direct answering vs the existing
organic-thinking baseline, on google/gemma-4-E4B-it (BASE instruction model, no training, no head) --
does writing a compact program-like format use fewer tokens than normal thinking at similar accuracy?

Model loading / batched HF generation / chat template is modeled on scripts/gen_think_traces.py (imported
directly below -- load_model_and_tokenizer, batched_generate, _strip_pad_and_turn, EOS_IDS -- same
box/revision/sampling conventions, see that script's own docstring for the chat-template facts this
relies on). Grading reuses scripts/s1_head_and_outcomes.py's extract_last_number/extract_choice_letter/
numbers_equal/judge_correctness (same rules the existing oracle/s1-traces.jsonl thinking baseline was
graded with in that script, so the three arms are graded consistently).

Problems: oracle/s1-train-problems.jsonl rows with source in {gsm8k, mmlu, arc_challenge} (the task's
"GSM8K (200) and MMLU+ARC (350, skip 'advice')" slice -- 550 rows total). Thinking baseline: the matching
rows (by id) in oracle/s1-traces.jsonl (already generated+graded data, source=`oracle/s1_head_and_outcomes.py
outcomes`'s own judge_correctness rule; no regeneration here).

Three arms, all greedy (temperature<=0 -> do_sample=False), enable_thinking=False, batch 32,
max_new_tokens 400, no system message in the prompt (a bare one-turn `{"role": "user", ...}` message,
same as gen_think_traces.py's render_prompt with no "system" key on the input row):
  1. direct  -- plain question (problem["user"] verbatim), no few-shot. Graded like the thinking arm's
               "answer" field: extract_last_number (numeric) / extract_choice_letter (choice) on the
               WHOLE generated text (enable_thinking=False means there is no thinking channel to split
               out -- the whole continuation is the answer).
  2. diagram -- a hand-written 4-example few-shot preamble (FEWSHOT_NUMERIC / FEWSHOT_CHOICE below; NOT
               taken from oracle/s1-train-problems.jsonl) + the real question, asking for:
                 - GSM8K (numeric): a tiny calculator program, lines `vN = op(a, b)` (op in
                   add/sub/mul/div, args are literal numbers or `vK` refs to a prior line), terminated by
                   `ANSWER vK`. Graded by actually EXECUTING the program (exec_diagram_program, a safe
                   hand-rolled parser -- no eval) and numerically comparing the result to gold_answer.
                   Any parse/exec problem (no program found, no ANSWER line, undefined var ref, div by
                   zero, unparseable literal) is a graded failure, not dropped.
                 - MMLU/ARC (choice): at most 4 lines `check <letter>: yes|no (<=6 words why)` then
                   `ANSWER <letter>` (parse_diagram_choice -- last `ANSWER <letter>` match in the text,
                   case-insensitive, optional `:`/parens). No match = parse failure.
  3. thinking -- no generation; read straight from oracle/s1-traces.jsonl (joined by id). Accuracy via
               judge_correctness (same as scripts/s1_head_and_outcomes.py outcomes); tokens =
               n_thinking_tokens + n_answer_tokens.

Per-row output (oracle/format-probe.jsonl, one row per (problem, arm) for direct/diagram, one row per
problem for thinking): {id, source, arm, answer_type, gold_answer, generated_text, parsed_value, correct
(bool -- parse/exec failures and unparsed count as False, never dropped), parse_failure (bool), tokens
(int)}.

Report (oracle/format-probe.md): per-source x per-arm table (accuracy, mean generated tokens, parse/exec
failure rate, tokens ratio vs thinking, accuracy restricted to the subset where the thinking arm was
itself correct), plus 3 verbatim sample generated_text per (source, arm) including at least one failure
sample where failures exist.

Usage:
  offline unit tests (parser + calculator, no model, no GPU):
    .venv/Scripts/python.exe scripts/format_probe.py --selftest

  GPU box, full run:
    .venv/bin/python -u scripts/format_probe.py \\
        --problems oracle/s1-train-problems.jsonl --traces oracle/s1-traces.jsonl \\
        --out-rows oracle/format-probe.jsonl --out-report oracle/format-probe.md \\
        --batch-size 32 --max-new-tokens 400 --resume

  smoke (5 problems, no resume):
    .venv/bin/python -u scripts/format_probe.py \\
        --problems oracle/s1-train-problems.jsonl --traces oracle/s1-traces.jsonl \\
        --out-rows oracle/format-probe-smoke.jsonl --out-report oracle/format-probe-smoke.md \\
        --batch-size 4 --max-new-tokens 400 --limit 5
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.gen_think_traces import (  # noqa: E402
    EOS_IDS, _strip_pad_and_turn, load_model_and_tokenizer, batched_generate,
)
from scripts.s1_head_and_outcomes import (  # noqa: E402
    extract_last_number, extract_choice_letter, numbers_equal, judge_correctness,
)

SOURCES = ("gsm8k", "mmlu", "arc_challenge")
DEFAULT_MAX_NEW_TOKENS = 400
DEFAULT_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# few-shot preambles (hand-written, NOT taken from oracle/s1-train-problems.jsonl)
# ---------------------------------------------------------------------------

FEWSHOT_NUMERIC = """Solve each word problem by writing a short calculator program, then output the \
answer variable. Each line is `vN = op(a, b)` where op is one of add, sub, mul, div, and a/b are either \
literal numbers copied from the problem or an earlier vN. The last line is `ANSWER vK`. Do not write \
anything else.

Question: A store had 24 apples. They sold 9 apples and then received a delivery of 15 more apples. How \
many apples does the store have now?
v1 = sub(24, 9)
v2 = add(v1, 15)
ANSWER v2

Question: A classroom has 6 rows with 5 chairs each. If 3 chairs are broken, how many chairs are usable?
v1 = mul(6, 5)
v2 = sub(v1, 3)
ANSWER v2

Question: A baker made 48 cookies and packed them equally into 8 boxes. If each box is sold for $4, how \
much money does the baker make from selling 3 boxes?
v1 = div(48, 8)
v2 = mul(v1, 4)
v3 = mul(v2, 3)
ANSWER v3

Question: Tom had 50 dollars. He spent 12 dollars on lunch and 8 dollars on a book. How much money does \
he have left?
v1 = sub(50, 12)
v2 = sub(v1, 8)
ANSWER v2"""

FEWSHOT_CHOICE = """Answer each multiple-choice question by checking at most 4 options, one line each: \
`check <letter>: yes|no (<=6 words why)`. Stop checking once you find the right one. Then output \
`ANSWER <letter>`. Do not write anything else.

Question: Which gas do plants absorb from the air during photosynthesis?

A. Oxygen
B. Carbon dioxide
C. Nitrogen
D. Hydrogen
check A: no, plants release oxygen
check B: yes, used in photosynthesis
ANSWER B

Question: What is the capital of France?

A. Berlin
B. Madrid
C. Paris
D. Rome
check A: no, capital of Germany
check B: no, capital of Spain
check C: yes, capital of France
ANSWER C

Question: A farmer noticed his crops grew taller after switching to a new fertilizer. What can he \
reasonably conclude?

A. The fertilizer had no effect
B. The fertilizer may have helped the crops grow
C. The weather caused the growth
D. The crops were a different species
check A: no, contradicts observed growth
check B: yes, matches observed growth
ANSWER B

Question: Which planet is known as the Red Planet?

A. Venus
B. Mars
C. Jupiter
D. Saturn
check A: no, Venus is not red
check B: yes, Mars is the Red Planet
ANSWER B"""


# ---------------------------------------------------------------------------
# diagram arm: numeric program parser + calculator (pure Python, no eval)
# ---------------------------------------------------------------------------

_ASSIGN_RE = re.compile(r"^v(\d+)\s*=\s*(add|sub|mul|div)\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)\s*$",
                        re.IGNORECASE)
_ANSWER_LINE_RE = re.compile(r"^ANSWER\s+v(\d+)\s*$", re.IGNORECASE)
_VAR_REF_RE = re.compile(r"^v(\d+)$", re.IGNORECASE)
_OPS = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b, "div": lambda a, b: a / b}


def _resolve_arg(tok, vars_):
    """-> float. Raises ValueError (bad literal) or KeyError (undefined vN ref) -- caller maps both to
    the "undefined_var" error code (a bad ref and an unparseable literal are both "this program doesn't
    compute", no need to distinguish further for the probe's purposes)."""
    tok = tok.strip()
    m = _VAR_REF_RE.match(tok)
    if m:
        return vars_[int(m.group(1))]
    return float(tok.replace(",", "").replace("$", ""))


def exec_diagram_program(text):
    """-> (value: float|None, error: str|None). error is None iff a valid `ANSWER vK` line referencing a
    defined var was found and evaluated cleanly; otherwise value is None and error names the failure:
    "no_program" (no assignment line ever matched), "no_answer_line" (assignments found but no ANSWER
    line), "undefined_var" (ANSWER or an arg refers to an undefined/unparseable var/literal),
    "div_by_zero". Scans line by line, ignoring blank lines and any non-matching line (the model's
    commentary, few-shot leakage, etc. are simply skipped rather than failing the whole parse) -- lenient
    on everything except the two kinds of failure above."""
    vars_ = {}
    found_assignment = False
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m_ans = _ANSWER_LINE_RE.match(line)
        if m_ans:
            key = int(m_ans.group(1))
            if key not in vars_:
                return None, "undefined_var"
            return vars_[key], None
        m = _ASSIGN_RE.match(line)
        if m:
            idx, op, a_tok, b_tok = m.groups()
            try:
                a = _resolve_arg(a_tok, vars_)
                b = _resolve_arg(b_tok, vars_)
            except (ValueError, KeyError):
                return None, "undefined_var"
            if op.lower() == "div" and b == 0:
                return None, "div_by_zero"
            vars_[int(idx)] = _OPS[op.lower()](a, b)
            found_assignment = True
    if not found_assignment:
        return None, "no_program"
    return None, "no_answer_line"


# ---------------------------------------------------------------------------
# diagram arm: MC answer-letter parser
# ---------------------------------------------------------------------------

_MC_ANSWER_RE = re.compile(r"ANSWER\s*:?\s*\(?([A-Za-z])\)?", re.IGNORECASE)


def parse_diagram_choice(text):
    """-> str|None (uppercased letter). Last `ANSWER <letter>` match wins (in case the model repeats
    itself); None if no such line is found anywhere in the text."""
    matches = _MC_ANSWER_RE.findall(text or "")
    return matches[-1].upper() if matches else None


# ---------------------------------------------------------------------------
# grading (pure Python, shared by direct/diagram/thinking rows)
# ---------------------------------------------------------------------------

def grade_direct_or_thinking(generated_text, answer_type, gold_answer):
    """-> (correct: bool, parsed_value, parse_failure: bool). Same extraction rules
    scripts/s1_head_and_outcomes.py uses for a trace's "answer" field; a failed extraction is graded
    incorrect (never dropped) and flagged via parse_failure."""
    if answer_type == "numeric":
        val = extract_last_number(generated_text)
        if val is None:
            return False, None, True
        try:
            gold_val = float(str(gold_answer).replace(",", ""))
        except ValueError:
            return False, val, True
        return numbers_equal(val, gold_val), val, False
    if answer_type == "choice":
        letter = extract_choice_letter(generated_text)
        if letter is None:
            return False, None, True
        return (letter == str(gold_answer).upper()), letter, False
    raise ValueError(f"unexpected answer_type {answer_type!r}")


def grade_diagram(generated_text, answer_type, gold_answer):
    """-> (correct: bool, parsed_value, parse_failure: bool). Numeric: executes the program
    (exec_diagram_program); any error is a parse_failure, graded incorrect. Choice: parse_diagram_choice;
    no match is a parse_failure, graded incorrect."""
    if answer_type == "numeric":
        val, err = exec_diagram_program(generated_text)
        if err is not None:
            return False, None, True
        try:
            gold_val = float(str(gold_answer).replace(",", ""))
        except ValueError:
            return False, val, True
        return numbers_equal(val, gold_val), val, False
    if answer_type == "choice":
        letter = parse_diagram_choice(generated_text)
        if letter is None:
            return False, None, True
        return (letter == str(gold_answer).upper()), letter, False
    raise ValueError(f"unexpected answer_type {answer_type!r}")


# ---------------------------------------------------------------------------
# prompt rendering (tokenizer only)
# ---------------------------------------------------------------------------

def render_arm_prompt(tok, problem, arm):
    """-> prompt text. One bare user turn (no system message), enable_thinking=False -- same
    apply_chat_template call as gen_think_traces.py's render_prompt with no "system" key on the row."""
    if arm == "direct":
        user = problem["user"]
    elif arm == "diagram":
        fewshot = FEWSHOT_NUMERIC if problem["answer_type"] == "numeric" else FEWSHOT_CHOICE
        user = f"{fewshot}\n\nNow solve this problem in the same format:\nQuestion: {problem['user']}"
    else:
        raise ValueError(f"unexpected arm {arm!r}")
    messages = [{"role": "user", "content": user}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)


# ---------------------------------------------------------------------------
# selftest (parser + calculator unit checks, no model)
# ---------------------------------------------------------------------------

def selftest():
    # exec_diagram_program: clean multi-step program.
    val, err = exec_diagram_program("v1 = add(2, 3)\nv2 = mul(v1, 4)\nANSWER v2")
    assert err is None and val == 20.0, (val, err)
    print("selftest: exec_diagram_program OK (clean multi-step program -> 20.0)")

    # $ in literals, chained refs.
    val, err = exec_diagram_program("v1 = mul($5, 2)\nANSWER v1")
    assert err is None and val == 10.0, (val, err)
    print("selftest: exec_diagram_program OK ($ literal)")

    # div by zero.
    val, err = exec_diagram_program("v1 = div(10, 0)\nANSWER v1")
    assert err == "div_by_zero" and val is None, (val, err)
    print("selftest: exec_diagram_program OK (div_by_zero)")

    # undefined var ref (ANSWER points at a var never assigned).
    val, err = exec_diagram_program("v1 = add(1, 2)\nANSWER v2")
    assert err == "undefined_var" and val is None, (val, err)
    # undefined var ref inside an arg.
    val, err = exec_diagram_program("v2 = add(v1, 2)\nANSWER v2")
    assert err == "undefined_var" and val is None, (val, err)
    print("selftest: exec_diagram_program OK (undefined_var, both as ANSWER target and as an arg)")

    # no program at all / assignments but no ANSWER line.
    val, err = exec_diagram_program("I think the answer is 5.")
    assert err == "no_program" and val is None, (val, err)
    val, err = exec_diagram_program("v1 = add(1, 2)")
    assert err == "no_answer_line" and val is None, (val, err)
    val, err = exec_diagram_program("")
    assert err == "no_program" and val is None, (val, err)
    print("selftest: exec_diagram_program OK (no_program, no_answer_line, empty text)")

    # stray commentary lines around a valid program are ignored, not fatal.
    val, err = exec_diagram_program("Let's see.\nv1 = add(2, 2)\nThat gives us 4.\nANSWER v1\nextra junk")
    assert err is None and val == 4.0, (val, err)
    print("selftest: exec_diagram_program OK (stray commentary lines skipped)")

    # parse_diagram_choice: match, case-insensitive, parens/colon, last-match-wins, no-match.
    assert parse_diagram_choice("check A: no\ncheck B: yes\nANSWER B") == "B"
    assert parse_diagram_choice("answer: (c)") == "C"
    assert parse_diagram_choice("ANSWER D\n...\nANSWER A") == "A"  # last match wins
    assert parse_diagram_choice("I am not sure") is None
    print("selftest: parse_diagram_choice OK (match/case/parens/last-wins/no-match)")

    # grade_diagram / grade_direct_or_thinking: correct, wrong, and parse-failure cases both graded False.
    correct, val, fail = grade_diagram("v1 = add(2, 3)\nANSWER v1", "numeric", "5")
    assert correct is True and val == 5.0 and not fail
    correct, val, fail = grade_diagram("v1 = add(2, 3)\nANSWER v1", "numeric", "9")
    assert correct is False and val == 5.0 and not fail
    correct, val, fail = grade_diagram("no program here", "numeric", "5")
    assert correct is False and val is None and fail
    correct, val, fail = grade_diagram("check A: yes\nANSWER A", "choice", "A")
    assert correct is True and val == "A" and not fail
    correct, val, fail = grade_diagram("no letter anywhere", "choice", "A")
    assert correct is False and val is None and fail
    print("selftest: grade_diagram OK (correct/wrong/parse-failure, numeric+choice)")

    correct, val, fail = grade_direct_or_thinking("So the answer is 42.", "numeric", "42")
    assert correct is True and val == 42.0 and not fail
    correct, val, fail = grade_direct_or_thinking("I'm not sure.", "numeric", "42")
    assert correct is False and val is None and fail
    correct, val, fail = grade_direct_or_thinking("The answer is B.", "choice", "B")
    assert correct is True and val == "B" and not fail
    print("selftest: grade_direct_or_thinking OK (correct/parse-failure, numeric+choice)")

    print("selftest: all checks passed")
    return 0


# ---------------------------------------------------------------------------
# problem/trace loading
# ---------------------------------------------------------------------------

def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_problems(path):
    rows = load_jsonl(path)
    return [r for r in rows if r.get("source") in SOURCES]


def thinking_rows_for(problems, traces_path):
    """-> list of format-probe-schema rows for the "thinking" arm, joined from --traces by id. Problems
    with no matching trace row are skipped (reported, not fatal)."""
    traces_by_id = {t["id"]: t for t in load_jsonl(traces_path)}
    rows = []
    missing = 0
    for p in problems:
        trace = traces_by_id.get(p["id"])
        if trace is None:
            missing += 1
            continue
        correct, parsed, unparsed = judge_correctness(trace, p)
        tokens = (trace.get("n_thinking_tokens") or 0) + (trace.get("n_answer_tokens") or 0)
        rows.append(dict(id=p["id"], source=p["source"], arm="thinking", answer_type=p["answer_type"],
                          gold_answer=p["gold_answer"], generated_text=trace.get("raw_text", ""),
                          parsed_value=parsed, correct=bool(correct), parse_failure=bool(unparsed),
                          tokens=tokens))
    if missing:
        print(f"thinking arm: {missing}/{len(problems)} problems had no matching trace row (skipped)",
              flush=True)
    return rows


# ---------------------------------------------------------------------------
# generation (GPU, imported lazily via gen_think_traces' already-lazy load_model_and_tokenizer)
# ---------------------------------------------------------------------------

class _GreedyArgs:
    def __init__(self, max_new_tokens):
        self.temperature = 0.0
        self.max_new_tokens = max_new_tokens
        self.top_p = 1.0
        self.top_k = 0


def run_generation(problems, arms, args):
    """-> list of format-probe-schema rows for direct/diagram arms. Batches all (problem, arm) pairs
    together, sorted by prompt token length (padding efficiency), same pattern as gen_think_traces.py."""
    import torch
    tok, model = load_model_and_tokenizer(args.model, args.revision, args.device, args.attn_impl)
    torch.manual_seed(args.seed)

    pairs = [(p, arm) for p in problems for arm in arms]
    prompt_texts = [render_arm_prompt(tok, p, arm) for p, arm in pairs]
    prompt_lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in prompt_texts]
    order = sorted(range(len(pairs)), key=lambda i: prompt_lens[i])
    pairs_sorted = [pairs[i] for i in order]
    prompts_sorted = [prompt_texts[i] for i in order]

    gen_args = _GreedyArgs(args.max_new_tokens)
    rows = []
    n_total = len(pairs_sorted)
    for i in range(0, n_total, args.batch_size):
        batch_pairs = pairs_sorted[i:i + args.batch_size]
        batch_prompts = prompts_sorted[i:i + args.batch_size]
        results = batched_generate(tok, model, batch_prompts, gen_args, args.device)
        for (problem, arm), (raw_text, _finished, n_new) in zip(batch_pairs, results):
            generated_text = _strip_pad_and_turn(raw_text)
            if arm == "diagram":
                correct, parsed, fail = grade_diagram(generated_text, problem["answer_type"],
                                                       problem["gold_answer"])
            else:
                correct, parsed, fail = grade_direct_or_thinking(generated_text, problem["answer_type"],
                                                                   problem["gold_answer"])
            rows.append(dict(id=problem["id"], source=problem["source"], arm=arm,
                              answer_type=problem["answer_type"], gold_answer=problem["gold_answer"],
                              generated_text=generated_text, parsed_value=parsed, correct=bool(correct),
                              parse_failure=bool(fail), tokens=n_new))
        print(f"generated {min(i + args.batch_size, n_total)}/{n_total}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def summarize(rows):
    """-> {(source, arm): dict(n, accuracy, mean_tokens, fail_rate, n_fail)}."""
    import collections
    by_key = collections.defaultdict(list)
    for r in rows:
        by_key[(r["source"], r["arm"])].append(r)
    out = {}
    for key, rs in by_key.items():
        n = len(rs)
        out[key] = dict(
            n=n,
            accuracy=sum(r["correct"] for r in rs) / n if n else float("nan"),
            mean_tokens=sum(r["tokens"] for r in rs) / n if n else float("nan"),
            fail_rate=sum(r["parse_failure"] for r in rs) / n if n else float("nan"),
            n_fail=sum(r["parse_failure"] for r in rs),
        )
    return out


def subset_accuracy_where_thinking_correct(rows):
    """-> {(source, arm): accuracy|None} restricted to ids where the thinking arm was correct==True."""
    import collections
    thinking_correct_ids = {r["id"] for r in rows if r["arm"] == "thinking" and r["correct"]}
    by_key = collections.defaultdict(list)
    for r in rows:
        if r["id"] in thinking_correct_ids:
            by_key[(r["source"], r["arm"])].append(r["correct"])
    return {key: (sum(vals) / len(vals) if vals else None) for key, vals in by_key.items()}


def render_report(rows):
    summary = summarize(rows)
    subset = subset_accuracy_where_thinking_correct(rows)
    sources = sorted({r["source"] for r in rows})
    arms = ["thinking", "direct", "diagram"]

    lines = ["# Format probe report", "",
             "| source | arm | n | accuracy | mean tokens | parse/exec fail rate | "
             "tokens ratio vs thinking | accuracy where thinking correct |",
             "|---|---|---|---|---|---|---|---|"]
    for source in sources:
        thinking_tokens = summary.get((source, "thinking"), {}).get("mean_tokens")
        for arm in arms:
            s = summary.get((source, arm))
            if s is None:
                continue
            ratio = (s["mean_tokens"] / thinking_tokens) if thinking_tokens else float("nan")
            sub_acc = subset.get((source, arm))
            sub_acc_str = f"{sub_acc:.3f}" if sub_acc is not None else "n/a"
            lines.append(f"| {source} | {arm} | {s['n']} | {s['accuracy']:.3f} | "
                          f"{s['mean_tokens']:.1f} | {s['fail_rate']:.3f} ({s['n_fail']}) | "
                          f"{ratio:.3f} | {sub_acc_str} |")
    lines.append("")

    lines.append("## Sample outputs (3 per source/arm, verbatim)")
    lines.append("")
    import collections
    by_key = collections.defaultdict(list)
    for r in rows:
        by_key[(r["source"], r["arm"])].append(r)
    for source in sources:
        for arm in arms:
            rs = by_key.get((source, arm), [])
            if not rs:
                continue
            fails = [r for r in rs if r["parse_failure"]]
            sample = (fails[:1] + [r for r in rs if not r["parse_failure"]][:2]) if fails else rs[:3]
            lines.append(f"### {source} / {arm}")
            lines.append("")
            for r in sample:
                lines.append(f"- id={r['id']} gold={r['gold_answer']!r} parsed={r['parsed_value']!r} "
                              f"correct={r['correct']} parse_failure={r['parse_failure']} "
                              f"tokens={r['tokens']}")
                lines.append("```")
                lines.append(r["generated_text"])
                lines.append("```")
            lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def load_done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {(r["id"], r["arm"]) for l in f if l.strip() for r in [json.loads(l)]}


def write_jsonl(path, rows, mode="w"):
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--problems", default="oracle/s1-train-problems.jsonl")
    ap.add_argument("--traces", default="oracle/s1-traces.jsonl")
    ap.add_argument("--out-rows", default="oracle/format-probe.jsonl")
    ap.add_argument("--out-report", default="oracle/format-probe.md")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="limit to first N problems (per source order)")
    ap.add_argument("--resume", action="store_true", help="skip (id, arm) pairs already in --out-rows")
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--revision", default="ee0ef6023621cff504d758262d4e04895a5af4a2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-impl", default="sdpa")
    args = ap.parse_args()

    if args.selftest:
        return sys.exit(selftest())

    problems_path = resolve(args.problems)
    traces_path = resolve(args.traces)
    out_rows_path = resolve(args.out_rows)
    out_report_path = resolve(args.out_report)

    problems = load_problems(problems_path)
    print(f"loaded {len(problems)} problems from {args.problems} (sources: {SOURCES})", flush=True)
    if args.limit:
        problems = problems[:args.limit]

    arms_to_run = ["direct", "diagram"]
    if args.resume:
        done = load_done_keys(out_rows_path)
        gen_rows = []
        for arm in arms_to_run:
            missing_problems = [p for p in problems if (p["id"], arm) not in done]
            if missing_problems:
                gen_rows += run_generation(missing_problems, [arm], args)
        if gen_rows:
            write_jsonl(out_rows_path, gen_rows, mode="a")
        else:
            print("nothing to generate (resume complete)", flush=True)
    else:
        gen_rows = run_generation(problems, arms_to_run, args)
        write_jsonl(out_rows_path, gen_rows, mode="w")

    # thinking rows never come from generation -- (re)written fresh every run, replacing any stale copy
    # from a previous --resume pass (cheap: no generation involved, just a join).
    thinking_rows = thinking_rows_for(problems, traces_path)
    existing = load_jsonl(out_rows_path) if os.path.exists(out_rows_path) else []
    all_rows = [r for r in existing if r["arm"] != "thinking"] + thinking_rows
    write_jsonl(out_rows_path, all_rows, mode="w")
    print(f"wrote {len(all_rows)} rows -> {args.out_rows}", flush=True)

    report = render_report(all_rows)
    with open(out_report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"wrote {out_report_path}", flush=True)


if __name__ == "__main__":
    main()
