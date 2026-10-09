"""PROBE 1 -- GSM8K stepwise decisions: can Chimera's DECISION ORGAN (pointer head) drive arithmetic
reasoning step by step, choosing (operation, operand, operand) at each step out of a small option list,
rather than just reading a letter off Gemma's own chain of thought? The chooser under test must be an
organ (Kev-4B or our trained head, scored via a forward pass + argmax over probs, same convention as
scripts/s1_kev_score.py / scripts/probe_phase7r_head.py); a Gemma-E4B-it LETTER-READOUT arm is included
only as a REFERENCE (it is not an organ decision, just the base model's own answer in multiple-choice
form), never as the thing being validated.

Same 200 GSM8K problems as oracle/s1-train-problems.jsonl (source=="gsm8k"; reference numbers already
measured on these in oracle/format-probe.jsonl: thinking arm 0.870 acc @594 tok, direct-no-think arm
0.765 acc @258 tok).

Stage A (GPU, Gemma-E4B-it, enable_thinking=False, <=250 new tokens -- a first pass at 80 tokens cut off
most generations mid-array since the model pretty-prints the JSON by default; the prompt now asks for a
single compact line and the budget was raised, see STAGE_A_PROMPT_TMPL and _try_parse_truncated_array's
own tolerant-parse fallback below): per problem, emit a compact JSON
list of the QUANTITIES named in the problem text, e.g. [{"id": "q1", "value": 4.0, "label": "incorrect
answers"}, ...], resolving number words/fractions/"dozen" etc to a plain float value. Value correctness is
NOT required by the task (just parse the problem into candidate numbers) -- Stage B's calculator is what
actually has to get the arithmetic right. Parsed via extract_quantities_json (regex for the first `[...]`
array + json.loads; any problem that fails to parse is marked stage_a_parse_failure and skipped in Stage B
-- graded incorrect, not dropped, same "never silently drop a row" convention as scripts/format_probe.py).

Stage B (organ only, NO Gemma call): a pure-Python loop (run_stage_b), up to --max-steps (8) steps. Each
step is 1-2 organ decisions against a chooser callable with the s1_kev_score.py/probe_phase7r_head.py
signature (state_text, instruction, options) -> (choice_text, confidence, max_prob, probs_by_option):
  1. OPERATION: options are add/sub/mul/div/finish (fixed, described). If the organ picks an op (not
     finish): 2. FIRST OPERAND then 3. SECOND OPERAND, each chosen from every known quantity (q1..qn) and
     every already-computed intermediate result (v1..vk), described as "key: label = value". The
     calculator (calc_op, no eval()) executes op(a, b) and appends a new vK; a div-by-zero is a
     CALCULATOR-FAIL row (stops the loop immediately, final answer = None, graded incorrect, counted
     toward calculator_fail_rate -- never crashes the run).
  2. If the organ picks finish (or --max-steps is exhausted without ever picking finish -- "terminated_by"
     records which), one more organ decision -- ANSWER -- picks which known value (q or v) is the final
     answer, from the same options list. This is the only decision that ends the loop.
State text given to the organ at every decision = the problem text + the quantities list + the executed
steps so far (formatted "v1 = mul(q1, q2) = 24.0", one per line) + a one-line sub-question naming the
decision ("Choose the next operation." / "Choose the first operand." / "Choose the second operand." /
"Choose the final answer."), exactly mirroring scripts/s1_head_and_outcomes.py:build_head_input's
state-construction convention (state ends right before the decision; the sub-question IS the `instr`
argument, not folded into state). Greedy argmax (the scorer's own best-prob choice); confidence is also
logged per decision (choice_confidence's formula, same as every other S1 probe script here).

Chooser arms:
  (i)   kev    -- jaredpalmer/kev-4b, loaded/scored exactly like scripts/s1_kev_score.py.
  (ii)  head   -- runs/p4-e4b-final, loaded/scored exactly like scripts/eval_phase7.py:load_decision_model
                 (both (i)/(ii) go through the identical kev.checkpoint.Checkpoint(...).load(...) call --
                 see make_organ_scorer below -- only the `run` argument differs).
  (iii) letterreadout (REFERENCE ONLY) -- Gemma-E4B-it, enable_thinking=False, asked to pick a lettered
                 option (A/B/C/...) for each decision (same state+instruction, options relabelled A../..
                 for the prompt) and parse the letter back to the option's own key
                 (parse_letter_readout_choice); no forward-pass probs exist for this arm, confidence is
                 logged as None.

Final answer graded against problem["gold_answer"] via numbers_equal (scripts/s1_head_and_outcomes.py's
tolerance rule).

Per-arm report (oracle/stepwise-probe.md): accuracy, mean organ calls per problem, mean Gemma-generated
tokens (Stage A only -- shared across all three arms, so reported once), calculator-fail rate, average
number of steps, first-step operation distribution, and a terminated_by ("finish" vs "max_steps") count,
plus 3 verbatim traces per arm (mix of right/wrong, preferring at least 1 wrong if any exist).

Usage:
  offline selftest (calculator, quantity-JSON parsing, loop termination with a fake organ, no model/GPU):
    .venv/Scripts/python.exe scripts/stepwise_probe.py --selftest

  GPU box, Stage A (writes oracle/stepwise-quantities.jsonl):
    .venv/bin/python -u scripts/stepwise_probe.py stage-a \\
        --problems oracle/s1-train-problems.jsonl --out oracle/stepwise-quantities.jsonl \\
        --batch-size 32 --max-new-tokens 250 --resume

  GPU box, Stage B for one arm (organ-only, no Gemma; --arm in kev/head/letterreadout):
    .venv/bin/python -u scripts/stepwise_probe.py stage-b --arm kev \\
        --problems oracle/s1-train-problems.jsonl --quantities oracle/stepwise-quantities.jsonl \\
        --out oracle/stepwise-trace-kev.jsonl --resume

  Report (after all 3 arms' stage-b outputs exist):
    .venv/bin/python -u scripts/stepwise_probe.py report \\
        --problems oracle/s1-train-problems.jsonl --quantities oracle/stepwise-quantities.jsonl \\
        --kev oracle/stepwise-trace-kev.jsonl --head oracle/stepwise-trace-head.jsonl \\
        --letterreadout oracle/stepwise-trace-letterreadout.jsonl \\
        --out-report oracle/stepwise-probe.md
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MAX_STEPS = 8
ARMS = ["kev", "head", "letterreadout"]
KEV_RUN = "jaredpalmer/kev-4b"
HEAD_RUN = "runs/p4-e4b-final"


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows, mode="w"):
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_jsonl_append_flush(f, rows):
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.flush()
    os.fsync(f.fileno())


def load_gsm8k_problems(path):
    return [r for r in load_jsonl(path) if r.get("source") == "gsm8k"]


# ---------------------------------------------------------------------------
# calculator (pure Python, no eval)
# ---------------------------------------------------------------------------

_OPS = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b, "div": lambda a, b: a / b}
OPERATIONS = ["add", "sub", "mul", "div", "finish"]
OP_DESC = {"add": "add the two operands", "sub": "subtract the second from the first",
           "mul": "multiply the two operands", "div": "divide the first by the second",
           "finish": "stop and report the final answer"}


def calc_op(op, a, b):
    """-> (value: float|None, error: str|None). error == "div_by_zero" iff op == "div" and b == 0;
    error is None for every other op/operand pair (add/sub/mul never fail)."""
    if op == "div" and b == 0:
        return None, "div_by_zero"
    return _OPS[op](a, b), None


# ---------------------------------------------------------------------------
# Stage A: quantity-JSON extraction (pure Python, tokenizer/model-free -- selftestable)
# ---------------------------------------------------------------------------

STAGE_A_PROMPT_TMPL = (
    "List the quantities (numbers) in this word problem as ONE SINGLE-LINE, COMPACT JSON array -- no "
    "pretty-printing, no newlines or indentation inside the array, one object per quantity: "
    "[{{\"id\": \"q1\", \"value\": <number>, \"label\": \"<=5 words\"}}, {{\"id\": \"q2\", ...}}]. "
    "Resolve number words, fractions, and words like \"dozen\" to plain numbers. Output ONLY that one "
    "line of JSON, nothing else -- no markdown fence, no explanation.\n\nProblem: {problem}"
)


def _try_parse_truncated_array(array_text):
    """-> parsed list | None. `array_text` starts with "[" but may be missing its closing "]" (generation
    cut off mid-array by --max-new-tokens). Tries a straight json.loads first; on failure, scans
    character-by-character (string/escape-aware, so brackets inside quoted text are ignored) tracking
    overall {}/[] nesting depth to find the end of the LAST fully-closed TOP-LEVEL element -- i.e. the
    last "}" at which depth returns to 1 (just inside the outer array). This is nesting-depth-agnostic on
    purpose: scripts/stepwise_probe.py's own quantities are flat objects (nesting depth 2: outer array +
    one object), but scripts/humaneval_probe.py's design-decisions elements nest a further "options"
    array inside each element (depth 4) -- a naive "last '}' anywhere" truncation (this function's first,
    buggy version) can land mid-nested-object and still fail to parse. Truncates to that boundary, appends
    "]", and retries -- recovers every complete leading element regardless of nesting, even when the tail
    was cut off mid-write at any depth."""
    try:
        return json.loads(array_text)
    except (json.JSONDecodeError, ValueError):
        pass
    depth = 0
    in_string = False
    escape = False
    last_element_end = None
    for i, ch in enumerate(array_text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 1 and ch == "}":
                last_element_end = i
    if last_element_end is None:
        return None
    candidate = array_text[:last_element_end + 1] + "]"
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None


def extract_quantities_json(text):
    """-> (quantities: list[dict]|None, error: str|None). Finds the first "[" in `text` and parses from
    there (DOTALL, tolerant of a missing closing "]" -- see _try_parse_truncated_array, recovers whatever
    complete leading objects exist even if generation was cut off mid-array by --max-new-tokens); validates
    each element has "id"/"value"/"label" and value is numeric (coerced to float). error is None iff a
    non-empty, well-shaped list was found; otherwise quantities is None and error names the failure
    ("no_array", "bad_json", "empty", "bad_element")."""
    if not text:
        return None, "no_array"
    start = text.find("[")
    if start < 0:
        return None, "no_array"
    raw = _try_parse_truncated_array(text[start:])
    if raw is None:
        return None, "bad_json"
    if not isinstance(raw, list) or not raw:
        return None, "empty"
    out = []
    for el in raw:
        if not isinstance(el, dict) or "id" not in el or "value" not in el:
            return None, "bad_element"
        try:
            value = float(el["value"])
        except (TypeError, ValueError):
            return None, "bad_element"
        out.append({"id": str(el["id"]), "value": value, "label": str(el.get("label", ""))[:80]})
    return out, None


# ---------------------------------------------------------------------------
# Stage B: option construction + state rendering (pure Python -- selftestable with a fake organ)
# ---------------------------------------------------------------------------

def option_text(key, desc):
    return key if not desc else f"{key}: {desc}"


def option_key(opt_text):
    return opt_text.split(": ", 1)[0]


def operand_options(quantities, results):
    """-> list of described "key: label = value" option strings, quantities (q1..) then results (v1..),
    in that fixed order (no shuffle -- this probe is about whether the organ can drive arithmetic at all,
    not an order-sensitivity study)."""
    opts = []
    for q in quantities:
        label = q["label"] or q["id"]
        opts.append(option_text(q["id"], f"{label} = {q['value']:g}"))
    for i, (expr, value) in enumerate(results, 1):
        opts.append(option_text(f"v{i}", f"{expr} = {value:g}"))
    return opts


def value_of(key, quantities, results):
    """-> float for a known option key (qN from `quantities`, vN from `results`); raises KeyError if the
    key names neither (should never happen -- the organ can only choose from `operand_options`'s own
    list)."""
    for q in quantities:
        if q["id"] == key:
            return q["value"]
    m = re.match(r"^v(\d+)$", key)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(results):
            return results[idx][1]
    raise KeyError(key)


def render_state(problem_text, quantities, results):
    lines = [problem_text, "", "Quantities:"]
    for q in quantities:
        label = q["label"] or q["id"]
        lines.append(f"  {q['id']}: {label} = {q['value']:g}")
    lines.append("Steps so far:")
    if results:
        for expr, value in results:
            lines.append(f"  {expr} = {value:g}")
    else:
        lines.append("  (none yet)")
    return "\n".join(lines)


def simulate_stage_b(problem_text, quantities, chooser, max_steps=MAX_STEPS):
    """-> dict trace. `chooser` has the shared organ-scorer signature (state_text, instr, options) ->
    (choice_text, confidence, max_prob, probs_by_option). Pure Python -- no model/torch import here;
    --selftest exercises this with a fake chooser. Trace fields: steps (list of
    {op, a_key, b_key, value, confidence} -- "op"=="answer" for the terminal decision),
    terminated_by ("finish"|"max_steps"), calculator_fail (bool), final_value (float|None),
    n_organ_calls (int), decisions (list of every single organ call made, for verbatim trace dumps)."""
    results = []  # [(expr_str, value), ...]
    steps = []
    decisions = []
    terminated_by = None
    calculator_fail = False
    final_value = None

    def ask(instr, options):
        state = render_state(problem_text, quantities, results)
        choice_text, confidence, max_prob, probs = chooser(state, instr, options)
        key = option_key(choice_text)
        decisions.append(dict(instr=instr, options=list(options), choice=key, confidence=confidence))
        return key, confidence

    for step_i in range(max_steps):
        op_key, op_conf = ask("Choose the next operation.",
                              [option_text(o, OP_DESC[o]) for o in OPERATIONS])
        if op_key not in OPERATIONS:
            op_key = "finish"  # malformed organ output -- treat as "stop" rather than crash
        if op_key == "finish":
            terminated_by = "finish"
            break
        opts = operand_options(quantities, results)
        if not opts:
            # no operands available at all (Stage A produced zero quantities) -- cannot proceed.
            terminated_by = "max_steps"
            break
        a_key, _ = ask("Choose the first operand.", opts)
        b_key, _ = ask("Choose the second operand.", opts)
        try:
            a_val, b_val = value_of(a_key, quantities, results), value_of(b_key, quantities, results)
        except KeyError:
            terminated_by = "max_steps"
            break
        value, err = calc_op(op_key, a_val, b_val)
        if err == "div_by_zero":
            calculator_fail = True
            terminated_by = "calculator_fail"
            break
        expr = f"v{len(results) + 1} = {op_key}({a_key}, {b_key})"
        results.append((expr, value))
        steps.append(dict(op=op_key, a_key=a_key, b_key=b_key, value=value, confidence=op_conf))
    else:
        terminated_by = "max_steps"

    if terminated_by in ("finish", "max_steps") and not calculator_fail:
        opts = operand_options(quantities, results)
        if opts:
            ans_key, ans_conf = ask("Choose the final answer.", opts)
            try:
                final_value = value_of(ans_key, quantities, results)
            except KeyError:
                final_value = None
            steps.append(dict(op="answer", a_key=ans_key, b_key=None, value=final_value,
                              confidence=ans_conf))

    return dict(steps=steps, decisions=decisions, terminated_by=terminated_by,
                calculator_fail=calculator_fail, final_value=final_value,
                n_organ_calls=len(decisions), n_steps=len(steps))


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------

def numbers_equal(a, b, tol=1e-6):
    return a is not None and b is not None and abs(a - b) <= tol


def grade_trace(trace, gold_answer):
    try:
        gold_val = float(str(gold_answer).replace(",", ""))
    except ValueError:
        return False
    return numbers_equal(trace["final_value"], gold_val)


# ---------------------------------------------------------------------------
# organ chooser (kev / head -- same Checkpoint load path, different `run`)
# ---------------------------------------------------------------------------

def make_organ_scorer_from_model(tok, model):
    """-> callable(state_text, instr, options) -> (choice_text, confidence, max_prob, probs_by_option), for
    an ALREADY-LOADED (tok, model) pair (e.g. moved onto the GPU in place by a resident-swap caller --
    scripts/recursive_probe.py's resident-arm mode). The scoring body make_organ_scorer below also uses,
    factored out so a caller that owns the model's lifecycle (load once, swap devices across rounds) does
    not have to reload it per call."""
    import torch
    from kev.api import choice_confidence

    @torch.no_grad()
    def scorer(state_text, instr, options):
        rec = {"state": state_text, "questions": [{"instr": instr, "options": list(options), "label": 0}]}
        enc = model.encode(tok, rec)
        probs = [float(x) for x in model.probs(enc)[0]]
        best_i = max(range(len(probs)), key=lambda k: probs[k])
        return options[best_i], choice_confidence(probs), max(probs), dict(zip(options, probs))

    return scorer


def make_organ_scorer(run, device):
    """-> callable(state_text, instr, options) -> (choice_text, confidence, max_prob, probs_by_option).
    Identical load path for jaredpalmer/kev-4b and runs/p4-e4b-final -- both are
    kev.checkpoint.Checkpoint(run).load(device, LoadOptions(dtype=bf16)) (scripts/s1_kev_score.py /
    scripts/eval_phase7.py:load_decision_model both do exactly this; only `run` differs between arms)."""
    import torch
    from kev.checkpoint import Checkpoint, LoadOptions
    tok, model = Checkpoint(run).load(device, LoadOptions(dtype=torch.bfloat16))
    return make_organ_scorer_from_model(tok, model)


def make_fake_organ_scorer(policy=None):
    """-> deterministic fake scorer, no torch/GPU, for --selftest. `policy(state_text, instr, options) ->
    int index`, default: always pick the FIRST non-"finish"-looking option unless this is the "operation"
    decision and a result already exists (then pick "finish") -- good enough to drive a real 2-step
    computation end-to-end in the selftest."""
    def default_policy(state_text, instr, options):
        if instr == "Choose the next operation." and "Steps so far:\n  (none yet)" not in state_text:
            for i, o in enumerate(options):
                if option_key(o) == "finish":
                    return i
        for i, o in enumerate(options):
            if option_key(o) != "finish":
                return i
        return 0
    pol = policy or default_policy

    def scorer(state_text, instr, options):
        i = pol(state_text, instr, options)
        probs = {o: (0.9 if j == i else 0.1 / max(len(options) - 1, 1)) for j, o in enumerate(options)}
        return options[i], 0.8, 0.9, probs
    return scorer


# ---------------------------------------------------------------------------
# letter-readout chooser (REFERENCE ONLY -- Gemma-E4B-it, not an organ)
# ---------------------------------------------------------------------------

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_READOUT_ANSWER_RE = re.compile(r"ANSWER\s*:?\s*\(?([A-Za-z])\)?", re.IGNORECASE)


def render_letter_readout_prompt(state_text, instr, options):
    lines = [state_text, "", instr]
    for i, o in enumerate(options):
        lines.append(f"{_LETTERS[i]}. {o}")
    lines.append("\nReply with exactly one line: ANSWER <letter>")
    return "\n".join(lines)


def parse_letter_readout_choice(text, options):
    """-> option key chosen (str). Falls back to the first option if no letter is parseable or the parsed
    letter is out of range (never crashes the loop)."""
    matches = _READOUT_ANSWER_RE.findall(text or "")
    if matches:
        letter = matches[-1].upper()
        idx = _LETTERS.find(letter)
        if 0 <= idx < len(options):
            return option_key(options[idx])
    return option_key(options[0])


class _GreedyArgs:
    """Plain args object for scripts.gen_think_traces.batched_generate's greedy path. A constructor
    parameter (not a class-body statement assigning `max_new_tokens = max_new_tokens`) on purpose: a
    class body that both reads and assigns the SAME name resolves the read as a not-yet-defined class
    local (NameError), not the enclosing function's parameter -- this is exactly the bug this shape
    avoids (hit for real in make_letter_readout_scorer's first version)."""
    def __init__(self, max_new_tokens):
        self.temperature = 0.0
        self.max_new_tokens = max_new_tokens
        self.top_p = 1.0
        self.top_k = 0


def make_letter_readout_scorer(tok, model, device, max_new_tokens=40):
    """-> callable with the shared scorer signature; confidence/max_prob/probs are always
    None/None/{} (no forward-pass probs exist for a letter-readout -- this arm is a REFERENCE, not an
    organ decision)."""
    from scripts.gen_think_traces import batched_generate, _strip_pad_and_turn

    def scorer(state_text, instr, options):
        prompt_text = render_letter_readout_prompt(state_text, instr, options)
        messages = [{"role": "user", "content": prompt_text}]
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
        (raw_text, _fin, _n), = batched_generate(tok, model, [rendered], _GreedyArgs(max_new_tokens), device)
        generated = _strip_pad_and_turn(raw_text)
        key = parse_letter_readout_choice(generated, options)
        chosen_text = next(o for o in options if option_key(o) == key)
        return chosen_text, None, None, {}

    return scorer


# ---------------------------------------------------------------------------
# selftest (no model/GPU)
# ---------------------------------------------------------------------------

def selftest():
    # 1. calculator: all 4 ops, div-by-zero.
    assert calc_op("add", 2, 3) == (5, None)
    assert calc_op("sub", 5, 2) == (3, None)
    assert calc_op("mul", 4, 3) == (12, None)
    assert calc_op("div", 10, 2) == (5.0, None)
    assert calc_op("div", 10, 0) == (None, "div_by_zero")
    print("selftest: calc_op OK (add/sub/mul/div, div_by_zero)")

    # 2. extract_quantities_json: clean array, array embedded in commentary, bad/no array, bad element.
    qs, err = extract_quantities_json(
        'Sure! [{"id": "q1", "value": 4, "label": "incorrect answers"}, '
        '{"id": "q2", "value": "6", "label": "extra points"}]')
    assert err is None and len(qs) == 2 and qs[0]["value"] == 4.0 and qs[1]["value"] == 6.0, (qs, err)
    print("selftest: extract_quantities_json OK (clean array + string-numeric value coerced)")
    qs, err = extract_quantities_json("I don't see any numbers.")
    assert qs is None and err == "no_array", (qs, err)
    qs, err = extract_quantities_json("[1, 2, 3]")
    assert qs is None and err == "bad_element", (qs, err)
    qs, err = extract_quantities_json("[]")
    assert qs is None and err == "empty", (qs, err)
    print("selftest: extract_quantities_json OK (no_array/bad_element/empty failures)")

    # 2b. truncated array (cut off mid-element, no closing "]", pretty-printed -- the real failure mode
    # that first shipped at --max-new-tokens 80) recovers every COMPLETE leading object.
    truncated = ('Sure, here they are:\n[\n  {"id": "q1", "value": 4, "label": "incorrect answers"},\n'
                 '  {"id": "q2", "value": 6, "label": "extra po')
    qs, err = extract_quantities_json(truncated)
    assert err is None and len(qs) == 1 and qs[0]["id"] == "q1", (qs, err)
    print("selftest: extract_quantities_json OK (truncated mid-element array recovers leading objects)")

    # 3. operand_options / value_of: quantities then results, in order; lookups both kinds.
    quantities = [{"id": "q1", "value": 4.0, "label": "a"}, {"id": "q2", "value": 2.0, "label": "b"}]
    results = [("v1 = mul(q1, q2)", 8.0)]
    opts = operand_options(quantities, results)
    assert opts == ["q1: a = 4", "q2: b = 2", "v1: v1 = mul(q1, q2) = 8"], opts
    assert value_of("q1", quantities, results) == 4.0
    assert value_of("v1", quantities, results) == 8.0
    print("selftest: operand_options/value_of OK (quantities-then-results order, both lookup kinds)")

    # 4. run_stage_b with a fake organ that always finishes immediately (no quantities needed) ->
    #    terminated_by "finish", 1 organ call for the op decision + 1 for the answer, but with NO
    #    quantities, the answer decision has no options to choose from -> final_value stays None.
    def always_finish(state_text, instr, options):
        return [option_key(o) for o in options].index("finish") if instr == "Choose the next operation." \
            else 0
    trace = simulate_stage_b("A trivial problem.", [], make_fake_organ_scorer(always_finish))
    assert trace["terminated_by"] == "finish", trace
    assert trace["final_value"] is None, trace  # no quantities -> answer step skipped (no options)
    print(f"selftest: run_stage_b OK (immediate finish, no quantities -> final_value None): {trace}")

    # 5. run_stage_b with quantities and a fake organ that does ONE mul step then finishes, picking the
    #    result as the answer -- exercises the full multi-decision loop end-to-end, no model involved.
    quantities2 = [{"id": "q1", "value": 4.0, "label": "a"}, {"id": "q2", "value": 2.0, "label": "b"}]

    def one_mul_then_finish(state_text, instr, options):
        if instr == "Choose the next operation.":
            if "Steps so far:\n  (none yet)" in state_text:
                return [option_key(o) for o in options].index("mul")
            return [option_key(o) for o in options].index("finish")
        if instr == "Choose the first operand.":
            return [option_key(o) for o in options].index("q1")
        if instr == "Choose the second operand.":
            return [option_key(o) for o in options].index("q2")
        if instr == "Choose the final answer.":
            return [option_key(o) for o in options].index("v1")
        raise AssertionError(instr)

    trace2 = simulate_stage_b("4 times 2.", quantities2, make_fake_organ_scorer(one_mul_then_finish))
    assert trace2["terminated_by"] == "finish", trace2
    assert trace2["final_value"] == 8.0, trace2
    assert trace2["n_steps"] == 2, trace2  # 1 mul step + 1 answer step
    assert grade_trace(trace2, "8") is True
    assert grade_trace(trace2, "9") is False
    print(f"selftest: run_stage_b OK (full mul-then-finish loop -> final_value 8.0, graded correct): "
          f"{trace2['n_organ_calls']} organ calls")

    # 6. run_stage_b: a fake organ that NEVER picks finish hits max_steps (terminated_by == "max_steps"),
    #    loop does not run forever.
    def never_finish(state_text, instr, options):
        if instr == "Choose the next operation.":
            return [option_key(o) for o in options].index("add")
        return 0
    trace3 = simulate_stage_b("x", quantities2, make_fake_organ_scorer(never_finish), max_steps=3)
    assert trace3["terminated_by"] == "max_steps", trace3
    assert len(trace3["steps"]) <= 4, trace3  # 3 add steps + at most 1 answer step
    print(f"selftest: run_stage_b OK (never-finish organ hits max_steps, loop bounded): {trace3['steps']}")

    # 7. run_stage_b: div-by-zero -> calculator_fail, loop stops immediately, final_value None.
    def div_by_zero_policy(state_text, instr, options):
        if instr == "Choose the next operation.":
            return [option_key(o) for o in options].index("div")
        if instr == "Choose the first operand.":
            return [option_key(o) for o in options].index("q1")
        if instr == "Choose the second operand.":
            zeros = [{"id": "q1", "value": 4.0, "label": "a"}, {"id": "zero", "value": 0.0, "label": "z"}]
            return 1
        return 0
    quantities3 = [{"id": "q1", "value": 4.0, "label": "a"}, {"id": "zero", "value": 0.0, "label": "z"}]
    trace4 = simulate_stage_b("x", quantities3, make_fake_organ_scorer(div_by_zero_policy))
    assert trace4["calculator_fail"] is True and trace4["final_value"] is None, trace4
    assert trace4["terminated_by"] == "calculator_fail", trace4
    print("selftest: run_stage_b OK (div-by-zero -> calculator_fail, loop stops, final_value None)")

    # 8. parse_letter_readout_choice: match, no-match fallback.
    opts4 = ["add: x", "sub: y"]
    assert parse_letter_readout_choice("Reasoning...\nANSWER B", opts4) == "sub"
    assert parse_letter_readout_choice("I am not sure.", opts4) == "add"  # fallback = first option
    print("selftest: parse_letter_readout_choice OK (match + no-match fallback)")

    print("selftest: all checks passed")
    return 0


# ---------------------------------------------------------------------------
# Stage A: GPU generation
# ---------------------------------------------------------------------------

def run_stage_a(args):
    import torch
    from scripts.gen_think_traces import load_model_and_tokenizer, batched_generate, _strip_pad_and_turn

    problems = load_gsm8k_problems(resolve(args.problems))
    print(f"loaded {len(problems)} gsm8k problems", flush=True)
    out_path = resolve(args.out)
    done = set()
    if args.resume and os.path.exists(out_path):
        done = {r["id"] for r in load_jsonl(out_path)}
        problems = [p for p in problems if p["id"] not in done]
        print(f"--resume: {len(done)} already done, {len(problems)} remaining", flush=True)
    if args.limit:
        problems = problems[:args.limit]
    if not problems:
        print("nothing to do", flush=True)
        return 0

    tok, model = load_model_and_tokenizer(args.model, args.revision, args.device, args.attn_impl)
    torch.manual_seed(args.seed)

    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": STAGE_A_PROMPT_TMPL.format(problem=p["user"])}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in problems]
    lens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    order = sorted(range(len(problems)), key=lambda i: lens[i])
    problems_sorted = [problems[i] for i in order]
    prompts_sorted = [prompts[i] for i in order]

    mode = "a" if args.resume else "w"
    n_parse_fail = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for i in range(0, len(problems_sorted), args.batch_size):
            batch_p = problems_sorted[i:i + args.batch_size]
            batch_prompts = prompts_sorted[i:i + args.batch_size]
            results = batched_generate(tok, model, batch_prompts, _GreedyArgs(args.max_new_tokens),
                                       args.device)
            out_rows = []
            for problem, (raw_text, _fin, n_new) in zip(batch_p, results):
                generated = _strip_pad_and_turn(raw_text)
                qs, err = extract_quantities_json(generated)
                if err is not None:
                    n_parse_fail += 1
                out_rows.append(dict(id=problem["id"], generated_text=generated, quantities=qs,
                                      parse_error=err, n_tokens=n_new))
            write_jsonl_append_flush(f, out_rows)
            print(f"stage-a: {min(i + args.batch_size, len(problems_sorted))}/{len(problems_sorted)} "
                  f"done, {n_parse_fail} parse failures so far", flush=True)
    print(f"stage-a done -> {args.out}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Stage B: GPU (organ arms) / GPU (letterreadout arm) loop over problems
# ---------------------------------------------------------------------------

def run_stage_b(args):
    problems = load_gsm8k_problems(resolve(args.problems))
    quantities_by_id = {r["id"]: r for r in load_jsonl(resolve(args.quantities))}
    print(f"loaded {len(problems)} gsm8k problems, {len(quantities_by_id)} stage-a rows", flush=True)

    out_path = resolve(args.out)
    done = set()
    if args.resume and os.path.exists(out_path):
        done = {r["id"] for r in load_jsonl(out_path)}
        problems = [p for p in problems if p["id"] not in done]
        print(f"--resume: {len(done)} already done, {len(problems)} remaining", flush=True)
    if args.limit:
        problems = problems[:args.limit]
    if not problems:
        print("nothing to do", flush=True)
        return 0

    if args.arm in ("kev", "head"):
        run_name = KEV_RUN if args.arm == "kev" else HEAD_RUN
        chooser = make_organ_scorer(run_name, args.device)
    elif args.arm == "letterreadout":
        from scripts.gen_think_traces import load_model_and_tokenizer
        tok, model = load_model_and_tokenizer(args.model, args.revision, args.device, args.attn_impl)
        chooser = make_letter_readout_scorer(tok, model, args.device)
    else:
        raise ValueError(f"unknown --arm {args.arm!r}")

    mode = "a" if args.resume else "w"
    n_done = n_correct = n_stage_a_fail = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for problem in problems:
            qrow = quantities_by_id.get(problem["id"])
            if qrow is None or qrow.get("quantities") is None:
                out = dict(id=problem["id"], stage_a_fail=True, correct=False, terminated_by=None,
                           calculator_fail=False, final_value=None, n_organ_calls=0, n_steps=0,
                           steps=[], decisions=[])
                n_stage_a_fail += 1
            else:
                trace = simulate_stage_b(problem["user"], qrow["quantities"], chooser)
                correct = grade_trace(trace, problem["gold_answer"])
                out = dict(id=problem["id"], stage_a_fail=False, correct=correct, **trace)
                n_correct += int(correct)
            write_jsonl_append_flush(f, [out])
            n_done += 1
            if n_done % 10 == 0 or n_done == len(problems):
                print(f"stage-b[{args.arm}]: {n_done}/{len(problems)}; "
                      f"acc-so-far={n_correct / max(n_done - n_stage_a_fail, 1):.3f}; "
                      f"stage_a_fail={n_stage_a_fail}", flush=True)
    print(f"stage-b[{args.arm}] done -> {args.out}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# report (pure Python -- CPU only, reads the 3 arms' stage-b outputs + stage-a quantities)
# ---------------------------------------------------------------------------

def summarize_arm(rows):
    n = len(rows)
    scorable = [r for r in rows if not r.get("stage_a_fail")]
    n_correct = sum(1 for r in scorable if r["correct"])
    n_calc_fail = sum(1 for r in rows if r.get("calculator_fail"))
    n_finish = sum(1 for r in rows if r.get("terminated_by") == "finish")
    n_max_steps = sum(1 for r in rows if r.get("terminated_by") == "max_steps")
    organ_calls = [r["n_organ_calls"] for r in scorable if "n_organ_calls" in r]
    n_steps_list = [r["n_steps"] for r in scorable if "n_steps" in r]
    first_ops = [r["steps"][0]["op"] for r in scorable if r.get("steps")]
    import collections
    first_op_dist = dict(collections.Counter(first_ops))
    return dict(
        n=n, n_stage_a_fail=n - len(scorable),
        accuracy=(n_correct / len(scorable)) if scorable else float("nan"),
        mean_organ_calls=(sum(organ_calls) / len(organ_calls)) if organ_calls else float("nan"),
        mean_steps=(sum(n_steps_list) / len(n_steps_list)) if n_steps_list else float("nan"),
        calculator_fail_rate=(n_calc_fail / n) if n else float("nan"),
        terminated_finish=n_finish, terminated_max_steps=n_max_steps,
        first_op_dist=first_op_dist,
    )


def pick_sample_traces(rows, k=3):
    """-> up to `k` rows, preferring a mix of correct/incorrect (at least 1 wrong if any exist)."""
    scorable = [r for r in rows if not r.get("stage_a_fail")]
    wrong = [r for r in scorable if not r["correct"]]
    right = [r for r in scorable if r["correct"]]
    sample = []
    if wrong:
        sample.append(wrong[0])
    sample += right[: k - len(sample)]
    sample += wrong[1: k - len(sample)]
    return sample[:k]


def render_trace_verbatim(row):
    lines = [f"id={row['id']} correct={row.get('correct')} terminated_by={row.get('terminated_by')} "
             f"calculator_fail={row.get('calculator_fail')} final_value={row.get('final_value')}"]
    for d in row.get("decisions", []):
        lines.append(f"  [{d['instr']}] options={d['options']} -> choice={d['choice']!r} "
                      f"confidence={d['confidence']}")
    return "\n".join(lines)


def render_stepwise_report(mean_stage_a_tokens, n_stage_a_total, n_stage_a_fail, arm_rows):
    lines = ["# Stepwise GSM8K probe (PROBE 1)", "",
             f"Stage A: {n_stage_a_total} problems, {n_stage_a_fail} quantity-parse failures "
             f"({n_stage_a_fail / n_stage_a_total:.3f}), mean Gemma-generated tokens "
             f"{mean_stage_a_tokens:.1f} (shared across all arms below -- Stage A runs once).",
             "",
             "## Per-arm summary", "",
             "| arm | n | accuracy | mean organ calls | mean steps | calculator-fail rate | "
             "terminated=finish | terminated=max_steps | first-op distribution |",
             "|---|---|---|---|---|---|---|---|---|"]
    for arm, rows in arm_rows.items():
        s = summarize_arm(rows)
        lines.append(f"| {arm} | {s['n']} | {s['accuracy']:.3f} | {s['mean_organ_calls']:.2f} | "
                      f"{s['mean_steps']:.2f} | {s['calculator_fail_rate']:.3f} | "
                      f"{s['terminated_finish']} | {s['terminated_max_steps']} | {s['first_op_dist']} |")
    lines.append("")
    lines.append("## Verbatim traces (3 per arm)")
    lines.append("")
    for arm, rows in arm_rows.items():
        lines.append(f"### {arm}")
        lines.append("")
        for row in pick_sample_traces(rows):
            lines.append("```")
            lines.append(render_trace_verbatim(row))
            lines.append("```")
        lines.append("")
    return "\n".join(lines) + "\n"


def run_report(args):
    quantities_rows = load_jsonl(resolve(args.quantities))
    n_stage_a_fail = sum(1 for r in quantities_rows if r.get("quantities") is None)
    tok_counts = [r["n_tokens"] for r in quantities_rows if "n_tokens" in r]
    mean_tokens = (sum(tok_counts) / len(tok_counts)) if tok_counts else float("nan")

    arm_rows = {}
    for arm, path in (("kev", args.kev), ("head", args.head), ("letterreadout", args.letterreadout)):
        if path:
            arm_rows[arm] = load_jsonl(resolve(path))
    if not arm_rows:
        print("ERROR: no arm files given (--kev/--head/--letterreadout)", file=sys.stderr)
        return 1

    report = render_stepwise_report(mean_tokens, len(quantities_rows), n_stage_a_fail, arm_rows)
    out_path = resolve(args.out_report)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"wrote {out_path}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")

    sp_a = sub.add_parser("stage-a")
    sp_a.add_argument("--problems", default="oracle/s1-train-problems.jsonl")
    sp_a.add_argument("--out", required=True)
    sp_a.add_argument("--batch-size", type=int, default=32)
    sp_a.add_argument("--max-new-tokens", type=int, default=250)
    sp_a.add_argument("--seed", type=int, default=0)
    sp_a.add_argument("--limit", type=int, default=None)
    sp_a.add_argument("--resume", action="store_true")
    sp_a.add_argument("--model", default="google/gemma-4-E4B-it")
    sp_a.add_argument("--revision", default="ee0ef6023621cff504d758262d4e04895a5af4a2")
    sp_a.add_argument("--device", default="cuda")
    sp_a.add_argument("--attn-impl", default="sdpa")

    sp_b = sub.add_parser("stage-b")
    sp_b.add_argument("--arm", required=True, choices=ARMS)
    sp_b.add_argument("--problems", default="oracle/s1-train-problems.jsonl")
    sp_b.add_argument("--quantities", required=True)
    sp_b.add_argument("--out", required=True)
    sp_b.add_argument("--limit", type=int, default=None)
    sp_b.add_argument("--resume", action="store_true")
    sp_b.add_argument("--model", default="google/gemma-4-E4B-it")
    sp_b.add_argument("--revision", default="ee0ef6023621cff504d758262d4e04895a5af4a2")
    sp_b.add_argument("--device", default="cuda")
    sp_b.add_argument("--attn-impl", default="sdpa")

    sp_r = sub.add_parser("report")
    sp_r.add_argument("--quantities", required=True)
    sp_r.add_argument("--kev", default=None)
    sp_r.add_argument("--head", default=None)
    sp_r.add_argument("--letterreadout", default=None)
    sp_r.add_argument("--out-report", required=True)
    # --problems accepted for symmetry/documentation but unused (the per-arm trace rows already carry
    # everything the report needs; kept as a CLI arg since the module docstring's usage example passes it)
    sp_r.add_argument("--problems", default="oracle/s1-train-problems.jsonl")

    args = ap.parse_args()
    if args.selftest:
        return sys.exit(selftest())
    if args.cmd == "stage-a":
        return sys.exit(run_stage_a(args))
    if args.cmd == "stage-b":
        return sys.exit(run_stage_b(args))
    if args.cmd == "report":
        return sys.exit(run_report(args))
    ap.error("one of --selftest / stage-a / stage-b / report is required")


if __name__ == "__main__":
    main()
