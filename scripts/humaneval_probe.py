"""PROBE 2 -- HumanEval (164 problems, openai_humaneval): does inserting a small set of DECISION-ORGAN
picks (algorithm family / data structure / edge-case policy / return-format detail) between the planning
and the writing of a function change pass@1, versus Gemma-E4B-it just writing the code directly? Honest
expectation per the task spec: no gain -- this is a no-harm/data-point probe, not a bet that it wins.

Dataset: openai_humaneval via `datasets.load_dataset("openai/openai_humaneval")["test"]` (164 rows: task_id,
prompt, canonical_solution, test, entry_point). Fetched once by fetch_humaneval_problems, cached to
--problems-cache (oracle/humaneval-problems.jsonl) so later runs/stages never re-hit the network.

Execution grading (run_test_program): assembles `generated_code (or prompt+generated_code if the
generated code is a bare continuation, see ensure_self_contained) + "\\n" + problem["test"] +
"\\ncheck({entry_point})\\n"` (openai_humaneval's own `test` field is itself a single `check(candidate)`
function bundling every assertion for that problem -- there is no finer-grained per-assertion split in
the dataset, so "per-test subprocess timeout 10s" here means: one subprocess, running that one bundled
check() call, killed after 10s). Runs via `subprocess.run([sys.executable, "-c", program], timeout=10,
capture_output=True)` in a throwaway cwd (--sandbox-dir, "sandbox-minimal": no network/fs jail beyond a
scratch working directory + timeout -- exactly the task's own phrase). A SyntaxError caught by a
`compile()` pre-check is recorded separately as syntactic_failure (so a pass@1 miss can be told apart from
"didn't even parse"); a subprocess timeout/non-zero exit after a clean compile is just a normal fail.

Arms (all on google/gemma-4-E4B-it, same model/revision/load path as scripts/gen_think_traces.py):
  (a) direct       -- thinking OFF, max_new_tokens 512, plain "write the function" prompt (problem prompt
                      verbatim + "Return ONLY a single ```python fenced code block containing the complete
                      function.").
  (b) thinking     -- thinking ON, max_new_tokens 1800, same user content; truncation rate = share of rows
                      that never close the thinking channel + emit visible code within the budget
                      (split_thinking_answer from scripts/gen_think_traces.py; code is extracted from the
                      "answer" half only, same as format_probe.py's treatment of the thinking arm).
  (c) planned      -- DECISION-PLANNED, 3 organ sub-arms sharing Stage A/C:
        Stage A (Gemma, thinking OFF, <=300 new tokens): writes <=4 design decisions as a JSON array
        [{"question": str, "options": [{"key": str, "desc": str}, ...2-4 options...]}, ...], parsed by
        extract_decisions_json (same "first [...] array, json.loads, validate shape" pattern as
        scripts/stepwise_probe.py:extract_quantities_json). Parse failure -> stage_a_fail row (skipped in
        Stage B/C, graded incorrect, not dropped).
        Stage B (organ only, no Gemma): for every decision, in order, scores options described "key: desc"
        against state = task prompt + the decision's own question (same build-input convention as
        scripts/s1_head_and_outcomes.py:build_head_input), via one of:
          (c-kev)           jaredpalmer/kev-4b, scripts/stepwise_probe.py:make_organ_scorer.
          (c-head)          runs/p4-e4b-final, same loader, different run.
          (c-letterreadout) REFERENCE ONLY -- Gemma letter-readout over the same options
                             (scripts/stepwise_probe.py:make_letter_readout_scorer); not an organ decision.
        Stage C (Gemma, thinking OFF, max_new_tokens 512): writes the function again, with a
        "Design decisions:\\n- {question}: {chosen desc}\\n..." block inserted before the task prompt.
        Batched (--batch-size, default 24 -- the first version generated one prompt per model.generate
        call, ~1.2 problems/min on this box; test execution (run_test_program's subprocess, 10s timeout)
        is also run concurrently per batch via a --exec-workers-size thread pool, so per-row exec
        timeouts overlap instead of serializing).

Per-row output (oracle/humaneval-<arm>.jsonl): {task_id, arm, generated_text, extracted_code, passed (bool
-- never dropped, parse/exec failures count as False), syntactic_failure (bool), truncated (bool, arm
"thinking" only), tokens (int; for "planned" arms this is Stage A + Stage C combined), decisions (list,
"planned" arms only -- the chosen key/desc per decision, for inspection)}.

Report (oracle/humaneval-probe.md): pass@1, mean tokens, syntactic-failure rate per arm; truncation rate
for "thinking"; Stage A failure rate for "planned" arms.

Usage:
  offline selftest (harness + 2 hand-written trivial solutions, no model/GPU/network):
    .venv/Scripts/python.exe scripts/humaneval_probe.py --selftest

  GPU box, fetch + cache the dataset once (needs network):
    .venv/bin/python -u scripts/humaneval_probe.py fetch --out oracle/humaneval-problems.jsonl

  GPU box, arm (a)/(b) (direct generation, no organ):
    .venv/bin/python -u scripts/humaneval_probe.py gen --arm direct \\
        --problems oracle/humaneval-problems.jsonl --out oracle/humaneval-direct.jsonl \\
        --batch-size 16 --max-new-tokens 512 --resume
    .venv/bin/python -u scripts/humaneval_probe.py gen --arm thinking \\
        --problems oracle/humaneval-problems.jsonl --out oracle/humaneval-thinking.jsonl \\
        --batch-size 8 --max-new-tokens 1800 --resume

  GPU box, arm (c) stage A (writes design decisions, shared across c-kev/c-head/c-letterreadout):
    .venv/bin/python -u scripts/humaneval_probe.py planned-stage-a \\
        --problems oracle/humaneval-problems.jsonl --out oracle/humaneval-planned-decisions.jsonl \\
        --batch-size 16 --resume

  GPU box, arm (c) stage B (organ ONLY, no Gemma -- --organ in kev/head/letterreadout; letterreadout
  loads Gemma for itself here, but never alongside kev/head):
    .venv/bin/python -u scripts/humaneval_probe.py planned-stage-b --organ kev \\
        --problems oracle/humaneval-problems.jsonl \\
        --decisions oracle/humaneval-planned-decisions.jsonl \\
        --out oracle/humaneval-planned-chosen-kev.jsonl --resume

  GPU box, arm (c) stage C (Gemma ONLY, reads stage B's chosen-decisions file, never touches the organ):
    .venv/bin/python -u scripts/humaneval_probe.py planned-stage-c --organ kev \\
        --problems oracle/humaneval-problems.jsonl \\
        --chosen oracle/humaneval-planned-chosen-kev.jsonl \\
        --out oracle/humaneval-planned-kev.jsonl --batch-size 24 --exec-workers 6 --resume

  Report:
    .venv/bin/python -u scripts/humaneval_probe.py report \\
        --rows oracle/humaneval-direct.jsonl oracle/humaneval-thinking.jsonl \\
              oracle/humaneval-planned-kev.jsonl oracle/humaneval-planned-head.jsonl \\
              oracle/humaneval-planned-letterreadout.jsonl \\
        --out-report oracle/humaneval-probe.md
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.stepwise_probe import _try_parse_truncated_array  # noqa: E402 -- pure-Python, no torch/kev
                                                                 # at import time; see that module's own
                                                                 # import list.

DEFAULT_MAX_NEW_TOKENS = {"direct": 512, "thinking": 1800, "planned_stage_a": 400, "planned_stage_c": 512}
EXEC_TIMEOUT_S = 10


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl_append_flush(f, rows):
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.flush()
    os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# dataset fetch (network, GPU box)
# ---------------------------------------------------------------------------

def fetch_humaneval_problems():
    """-> list of {task_id, prompt, canonical_solution, test, entry_point}, via
    datasets.load_dataset("openai/openai_humaneval")["test"] (164 rows, the only split this dataset has).
    The bare "openai_humaneval" id (no namespace) 404s/HfUriErrors against current huggingface_hub --
    the dataset now lives under the "openai/" namespace on the Hub."""
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval")["test"]
    return [dict(task_id=r["task_id"], prompt=r["prompt"], canonical_solution=r["canonical_solution"],
                 test=r["test"], entry_point=r["entry_point"]) for r in ds]


def run_fetch(args):
    problems = fetch_humaneval_problems()
    out_path = resolve(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")
    print(f"fetched {len(problems)} problems -> {args.out}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# code extraction + execution harness (pure Python, selftestable)
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code_block(text):
    """-> extracted code (str). Prefers the content of the FIRST fenced ```python``` (or bare ```) block;
    falls back to the whole text verbatim if no fence is found (some generations just emit code with no
    markdown)."""
    if not text:
        return ""
    m = _CODE_FENCE_RE.search(text)
    return m.group(1) if m else text


def ensure_self_contained(prompt, entry_point, code_block):
    """-> a program fragment that should define `entry_point` when exec'd. If `code_block` already
    contains "def {entry_point}", it is assumed self-contained (common case: the model restates the full
    function, imports and all) and returned as-is; otherwise it is treated as a bare CONTINUATION of the
    task's own prompt (the prompt's signature/docstring + this text concatenated) -- the standard
    HumanEval completion framing when a model only continues from the given signature."""
    if re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", code_block):
        return code_block
    return prompt + code_block


def check_syntax(program_source):
    """-> True iff `program_source` compiles (SyntaxError -> False); never raises."""
    try:
        compile(program_source, "<humaneval-candidate>", "exec")
        return True
    except SyntaxError:
        return False


def run_test_program(prompt, entry_point, test_src, generated_text, sandbox_dir=None, timeout=EXEC_TIMEOUT_S):
    """-> (passed: bool, syntactic_failure: bool, extracted_code: str). Builds
    ensure_self_contained(prompt, entry_point, extract_code_block(generated_text)), appends
    `test_src + f"\\ncheck({entry_point})\\n"`, compiles (syntactic_failure=True on SyntaxError, passed
    forced False, no subprocess spawned), else runs it as `[sys.executable, "-c", program]` in
    `sandbox_dir` (a throwaway cwd; a system tempdir if None) with a `timeout`-second subprocess timeout
    -- a timeout or non-zero exit is just `passed=False`, never raised."""
    code = extract_code_block(generated_text)
    candidate = ensure_self_contained(prompt, entry_point, code)
    program = f"{candidate}\n\n{test_src}\n\ncheck({entry_point})\n"
    if not check_syntax(program):
        return False, True, code
    cwd = sandbox_dir or tempfile.gettempdir()
    try:
        result = subprocess.run([sys.executable, "-c", program], cwd=cwd, capture_output=True,
                                 timeout=timeout, text=True)
        return result.returncode == 0, False, code
    except subprocess.TimeoutExpired:
        return False, False, code


# ---------------------------------------------------------------------------
# design-decisions JSON extraction (Stage A, pure Python -- selftestable)
# ---------------------------------------------------------------------------

STAGE_A_DECISIONS_PROMPT_TMPL = (
    "Before writing code, list at most 3 short DESIGN DECISIONS for implementing this function (e.g. "
    "algorithm family, data structure, edge-case policy, return/format detail) as ONE SINGLE-LINE, "
    "COMPACT JSON array -- no pretty-printing, no newlines or indentation inside the array: "
    "[{{\"question\": \"<short question>\", \"options\": [{{\"key\": \"<short key>\", \"desc\": "
    "\"<short description>\"}}, {{\"key\": \"...\", \"desc\": \"...\"}}]}}, ...]. 2-3 options per "
    "decision. Output ONLY that one line of JSON, nothing else -- no markdown fence, no explanation.\n\n"
    "Task:\n{prompt}"
)


def extract_decisions_json(text):
    """-> (decisions: list[dict]|None, error: str|None), decisions each shaped
    {"question": str, "options": [{"key": str, "desc": str}, ...]} (>=2 options, <=3 decisions kept).
    Tolerant like scripts/stepwise_probe.py:extract_quantities_json -- finds the first "[" and parses from
    there even if the closing "]" was never generated (_try_parse_truncated_array, same mid-array
    cut-off recovery), AND salvages at the per-decision level: a malformed decision (not a dict, missing
    "question"/"options") or a decision left with <2 usable options after filtering out malformed option
    entries is DROPPED, not fatal to the whole row -- only "no_array"/"bad_json"/"empty" (zero usable
    decisions survive) fail the row. Same failure vocabulary as extract_quantities_json."""
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
    for el in raw[:3]:
        if not isinstance(el, dict) or "question" not in el or "options" not in el:
            continue
        opts = el["options"]
        if not isinstance(opts, list):
            continue
        parsed_opts = [{"key": str(o["key"]), "desc": str(o.get("desc", ""))[:120]}
                       for o in opts if isinstance(o, dict) and "key" in o]
        if len(parsed_opts) < 2:
            continue
        out.append({"question": str(el["question"])[:200], "options": parsed_opts[:3]})
    if not out:
        return None, "empty"
    return out, None


# ---------------------------------------------------------------------------
# option text helpers (shared "key: desc" convention, copied from s1_head_and_outcomes.py)
# ---------------------------------------------------------------------------

def option_text(key, desc):
    return key if not desc else f"{key}: {desc}"


def option_key(opt_text):
    return opt_text.split(": ", 1)[0]


def render_decisions_block(chosen):
    """-> "Design decisions:\\n- {question}: {desc}\\n..." text, chosen = [{"question", "key", "desc"}]."""
    lines = ["Design decisions:"]
    for c in chosen:
        lines.append(f"- {c['question']}: {c['desc']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# selftest (no model/GPU/network)
# ---------------------------------------------------------------------------

def selftest():
    # 1. extract_code_block: fenced python block, fenced bare block, no fence at all.
    assert extract_code_block("blah\n```python\ndef f():\n    return 1\n```\nmore") == \
        "def f():\n    return 1\n"
    assert extract_code_block("```\ndef g(): return 2\n```") == "def g(): return 2\n"
    assert extract_code_block("def h(): return 3") == "def h(): return 3"
    print("selftest: extract_code_block OK (python-fence/bare-fence/no-fence)")

    # 2. ensure_self_contained: self-contained code kept as-is; bare continuation gets the prompt prefixed.
    prompt = "def add(a, b):\n    \"\"\"add two numbers\"\"\"\n"
    assert ensure_self_contained(prompt, "add", "def add(a, b):\n    return a + b\n") == \
        "def add(a, b):\n    return a + b\n"
    assert ensure_self_contained(prompt, "add", "    return a + b\n") == prompt + "    return a + b\n"
    print("selftest: ensure_self_contained OK (self-contained kept, bare continuation prefixed)")

    # 3. check_syntax: valid code True, SyntaxError False.
    assert check_syntax("def f(): return 1") is True
    assert check_syntax("def f(: return 1") is False
    print("selftest: check_syntax OK (valid/SyntaxError)")

    # 4. run_test_program end-to-end on 2 hand-written trivial HumanEval-shaped problems: a correct
    #    solution (passes) and a wrong one (fails, not a crash), plus a syntactically broken one.
    prompt4 = 'def add(a, b):\n    """Return a + b."""\n'
    test4 = ("def check(candidate):\n"
             "    assert candidate(2, 3) == 5\n"
             "    assert candidate(-1, 1) == 0\n")
    correct_gen = "```python\ndef add(a, b):\n    return a + b\n```"
    passed, synfail, code = run_test_program(prompt4, "add", test4, correct_gen)
    assert passed is True and synfail is False, (passed, synfail, code)
    print("selftest: run_test_program OK (correct solution passes)")

    wrong_gen = "```python\ndef add(a, b):\n    return a - b\n```"
    passed, synfail, code = run_test_program(prompt4, "add", test4, wrong_gen)
    assert passed is False and synfail is False, (passed, synfail, code)
    print("selftest: run_test_program OK (wrong solution fails, no crash)")

    broken_gen = "```python\ndef add(a, b)\n    return a + b\n```"  # missing colon
    passed, synfail, code = run_test_program(prompt4, "add", test4, broken_gen)
    assert passed is False and synfail is True, (passed, synfail, code)
    print("selftest: run_test_program OK (syntax error -> syntactic_failure, passed False, no subprocess)")

    # 5. a bare continuation (no signature restated) still resolves via ensure_self_contained + passes.
    continuation_gen = "```python\n    return a + b\n```"
    passed, synfail, code = run_test_program(prompt4, "add", test4, continuation_gen)
    assert passed is True and synfail is False, (passed, synfail, code)
    print("selftest: run_test_program OK (bare continuation, prompt prefixed, passes)")

    # 6. a timeout (infinite loop) is graded as a clean fail, not an exception.
    hang_gen = "```python\ndef add(a, b):\n    while True:\n        pass\n```"
    passed, synfail, code = run_test_program(prompt4, "add", test4, hang_gen, timeout=1)
    assert passed is False and synfail is False, (passed, synfail, code)
    print("selftest: run_test_program OK (infinite loop times out -> passed False, no exception)")

    # 7. extract_decisions_json: clean array; no-array; a malformed decision (missing "options") and a
    # <2-option decision are each DROPPED (salvage), not fatal -- only "no usable decisions at all" fails.
    decs, err = extract_decisions_json(
        'Sure! [{"question": "Which algorithm?", "options": [{"key": "iter", "desc": "iterative"}, '
        '{"key": "rec", "desc": "recursive"}]}]')
    assert err is None and len(decs) == 1 and len(decs[0]["options"]) == 2, (decs, err)
    print("selftest: extract_decisions_json OK (clean array)")
    decs, err = extract_decisions_json("no array here")
    assert decs is None and err == "no_array", (decs, err)
    decs, err = extract_decisions_json('[{"question": "q"}]')  # missing "options" -> no usable decisions
    assert decs is None and err == "empty", (decs, err)
    decs, err = extract_decisions_json('[{"question": "q", "options": [{"key": "a"}]}]')  # only 1 option
    assert decs is None and err == "empty", (decs, err)
    # a malformed decision alongside a good one: the good one survives (salvage), row is NOT failed.
    decs, err = extract_decisions_json(
        '[{"question": "bad"}, {"question": "good", "options": [{"key": "a"}, {"key": "b"}]}]')
    assert err is None and len(decs) == 1 and decs[0]["question"] == "good", (decs, err)
    print("selftest: extract_decisions_json OK (no_array/empty failures, malformed-decision salvage)")

    # 7b. truncated array (cut off mid-SECOND-decision, no closing "]") recovers the complete first
    # decision (nested "options" array means a naive "last '}' anywhere" truncation -- this function's
    # first, buggy version -- would land mid-nested-object and still fail; this must track real depth).
    truncated = ('[{"question": "Algorithm?", "options": [{"key": "iter", "desc": "iterative"}, '
                 '{"key": "rec", "desc": "recursive"}]}, {"question": "Edge case?", "options": '
                 '[{"key": "a", "desc": "raise err')
    decs, err = extract_decisions_json(truncated)
    assert err is None and len(decs) == 1 and decs[0]["question"] == "Algorithm?", (decs, err)
    print("selftest: extract_decisions_json OK (truncated mid-SECOND-decision recovers complete first)")

    # 8. render_decisions_block: formatting.
    block = render_decisions_block([{"question": "Which algorithm?", "key": "iter", "desc": "iterative"}])
    assert block == "Design decisions:\n- Which algorithm?: iterative", block
    print("selftest: render_decisions_block OK")

    print("selftest: all checks passed")
    return 0


# ---------------------------------------------------------------------------
# generation: arms (a) direct / (b) thinking (GPU)
# ---------------------------------------------------------------------------

def render_direct_prompt(tok, problem, thinking_on, extra_block=None):
    user = problem["prompt"]
    if extra_block:
        user = f"{extra_block}\n\n{user}"
    user += "\nReturn ONLY a single ```python fenced code block containing the complete function."
    messages = [{"role": "user", "content": user}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=thinking_on)


class _GreedyArgs:
    def __init__(self, max_new_tokens):
        self.temperature = 0.0
        self.max_new_tokens = max_new_tokens
        self.top_p = 1.0
        self.top_k = 0


def run_gen(args):
    import torch
    from scripts.gen_think_traces import (
        load_model_and_tokenizer, batched_generate, split_thinking_answer, _strip_pad_and_turn,
    )

    problems = load_jsonl(resolve(args.problems))
    print(f"loaded {len(problems)} humaneval problems", flush=True)
    out_path = resolve(args.out)
    done = set()
    if args.resume and os.path.exists(out_path):
        done = {r["task_id"] for r in load_jsonl(out_path)}
        problems = [p for p in problems if p["task_id"] not in done]
        print(f"--resume: {len(done)} already done, {len(problems)} remaining", flush=True)
    if args.limit:
        problems = problems[:args.limit]
    if not problems:
        print("nothing to do", flush=True)
        return 0

    thinking_on = args.arm == "thinking"
    max_new_tokens = args.max_new_tokens or DEFAULT_MAX_NEW_TOKENS[args.arm]
    tok, model = load_model_and_tokenizer(args.model, args.revision, args.device, args.attn_impl)
    torch.manual_seed(args.seed)

    prompts = [render_direct_prompt(tok, p, thinking_on) for p in problems]
    lens = [len(tok(pr, add_special_tokens=False)["input_ids"]) for pr in prompts]
    order = sorted(range(len(problems)), key=lambda i: lens[i])
    problems_sorted = [problems[i] for i in order]
    prompts_sorted = [prompts[i] for i in order]

    mode = "a" if args.resume else "w"
    n_pass = n_syn_fail = n_trunc = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for i in range(0, len(problems_sorted), args.batch_size):
            batch_p = problems_sorted[i:i + args.batch_size]
            batch_prompts = prompts_sorted[i:i + args.batch_size]
            results = batched_generate(tok, model, batch_prompts, _GreedyArgs(max_new_tokens), args.device)
            out_rows = []
            for problem, (raw_text, finished, n_new) in zip(batch_p, results):
                generated = _strip_pad_and_turn(raw_text)
                if thinking_on:
                    _thinking, code_text = split_thinking_answer(raw_text, True)
                    truncated = not finished
                else:
                    code_text = generated
                    truncated = not finished
                passed, synfail, extracted = run_test_program(
                    problem["prompt"], problem["entry_point"], problem["test"], code_text)
                n_pass += int(passed)
                n_syn_fail += int(synfail)
                n_trunc += int(truncated)
                out_rows.append(dict(task_id=problem["task_id"], arm=args.arm, generated_text=generated,
                                      extracted_code=extracted, passed=passed, syntactic_failure=synfail,
                                      truncated=truncated, tokens=n_new))
            write_jsonl_append_flush(f, out_rows)
            n_done = min(i + args.batch_size, len(problems_sorted))
            print(f"gen[{args.arm}]: {n_done}/{len(problems_sorted)}; pass_so_far={n_pass}/{n_done}; "
                  f"syn_fail={n_syn_fail}; truncated={n_trunc}", flush=True)
    print(f"gen[{args.arm}] done -> {args.out}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# arm (c): planned, Stage A (Gemma, design decisions)
# ---------------------------------------------------------------------------

def run_planned_stage_a(args):
    import torch
    from scripts.gen_think_traces import load_model_and_tokenizer, batched_generate, _strip_pad_and_turn

    problems = load_jsonl(resolve(args.problems))
    out_path = resolve(args.out)
    done = set()
    if args.resume and os.path.exists(out_path):
        done = {r["task_id"] for r in load_jsonl(out_path)}
        problems = [p for p in problems if p["task_id"] not in done]
        print(f"--resume: {len(done)} already done, {len(problems)} remaining", flush=True)
    if args.limit:
        problems = problems[:args.limit]
    if not problems:
        print("nothing to do", flush=True)
        return 0

    tok, model = load_model_and_tokenizer(args.model, args.revision, args.device, args.attn_impl)
    torch.manual_seed(args.seed)

    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": STAGE_A_DECISIONS_PROMPT_TMPL.format(prompt=p["prompt"])}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in problems]
    lens = [len(tok(pr, add_special_tokens=False)["input_ids"]) for pr in prompts]
    order = sorted(range(len(problems)), key=lambda i: lens[i])
    problems_sorted = [problems[i] for i in order]
    prompts_sorted = [prompts[i] for i in order]
    max_new_tokens = DEFAULT_MAX_NEW_TOKENS["planned_stage_a"]

    mode = "a" if args.resume else "w"
    n_fail = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for i in range(0, len(problems_sorted), args.batch_size):
            batch_p = problems_sorted[i:i + args.batch_size]
            batch_prompts = prompts_sorted[i:i + args.batch_size]
            results = batched_generate(tok, model, batch_prompts, _GreedyArgs(max_new_tokens), args.device)
            out_rows = []
            for problem, (raw_text, _fin, n_new) in zip(batch_p, results):
                generated = _strip_pad_and_turn(raw_text)
                decisions, err = extract_decisions_json(generated)
                if err is not None:
                    n_fail += 1
                out_rows.append(dict(task_id=problem["task_id"], generated_text=generated,
                                      decisions=decisions, parse_error=err, n_tokens=n_new))
            write_jsonl_append_flush(f, out_rows)
            n_done = min(i + args.batch_size, len(problems_sorted))
            print(f"planned-stage-a: {n_done}/{len(problems_sorted)}; parse failures so far {n_fail}",
                  flush=True)
    print(f"planned-stage-a done -> {args.out}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# arm (c): planned, Stage B (organ ONLY -- no Gemma) -- picks each decision's option, one organ per run.
# Split from Stage C (below) so the two never need Gemma + the organ model resident at once: Kev-4B
# (~10GB) or runs/p4-e4b-final (~7GB) alongside Gemma-E4B-it (~16GB) would not fit the 23GB L4 (per this
# project's own box notes). "letterreadout" is the one organ choice that IS Gemma, so this stage loads
# Gemma only for that case -- still never concurrently with a true organ model.
# ---------------------------------------------------------------------------

def run_planned_stage_b(args):
    from scripts.stepwise_probe import make_organ_scorer, make_letter_readout_scorer, KEV_RUN, HEAD_RUN

    problems_by_id = {p["task_id"]: p for p in load_jsonl(resolve(args.problems))}
    decisions_rows = {r["task_id"]: r for r in load_jsonl(resolve(args.decisions))}
    task_ids = list(problems_by_id.keys())

    out_path = resolve(args.out)
    done = set()
    if args.resume and os.path.exists(out_path):
        done = {r["task_id"] for r in load_jsonl(out_path)}
        task_ids = [t for t in task_ids if t not in done]
        print(f"--resume: {len(done)} already done, {len(task_ids)} remaining", flush=True)
    if args.limit:
        task_ids = task_ids[:args.limit]
    if not task_ids:
        print("nothing to do", flush=True)
        return 0

    if args.organ == "kev":
        organ_scorer = make_organ_scorer(KEV_RUN, args.device)
    elif args.organ == "head":
        organ_scorer = make_organ_scorer(HEAD_RUN, args.device)
    elif args.organ == "letterreadout":
        from scripts.gen_think_traces import load_model_and_tokenizer
        tok, model = load_model_and_tokenizer(args.model, args.revision, args.device, args.attn_impl)
        organ_scorer = make_letter_readout_scorer(tok, model, args.device)
    else:
        raise ValueError(f"unknown --organ {args.organ!r}")

    mode = "a" if args.resume else "w"
    n_stage_a_fail = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for n_done, task_id in enumerate(task_ids, 1):
            problem = problems_by_id[task_id]
            drow = decisions_rows.get(task_id)
            if drow is None or drow.get("decisions") is None:
                n_stage_a_fail += 1
                write_jsonl_append_flush(f, [dict(task_id=task_id, stage_a_fail=True, decisions=[],
                                                   n_tokens_stage_a=drow["n_tokens"] if drow else 0)])
                continue
            chosen = []
            for d in drow["decisions"]:
                options = [option_text(o["key"], o["desc"]) for o in d["options"]]
                choice_text, _conf, _mp, _probs = organ_scorer(problem["prompt"], d["question"], options)
                key = option_key(choice_text)
                desc = next((o["desc"] for o in d["options"] if o["key"] == key), key)
                chosen.append(dict(question=d["question"], key=key, desc=desc))
            write_jsonl_append_flush(f, [dict(task_id=task_id, stage_a_fail=False, decisions=chosen,
                                               n_tokens_stage_a=drow["n_tokens"])])
            if n_done % 20 == 0 or n_done == len(task_ids):
                print(f"planned-stage-b[{args.organ}]: {n_done}/{len(task_ids)}; "
                      f"stage_a_fail={n_stage_a_fail}", flush=True)
    print(f"planned-stage-b[{args.organ}] done -> {args.out}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# arm (c): planned, Stage C (Gemma ONLY) -- writes the function conditioned on Stage B's chosen
# decisions; reads a Stage B output file, never touches the organ model.
# ---------------------------------------------------------------------------

def run_planned_stage_c(args):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from scripts.gen_think_traces import load_model_and_tokenizer, batched_generate, _strip_pad_and_turn

    problems_by_id = {p["task_id"]: p for p in load_jsonl(resolve(args.problems))}
    chosen_rows = {r["task_id"]: r for r in load_jsonl(resolve(args.chosen))}
    task_ids = list(problems_by_id.keys())

    out_path = resolve(args.out)
    done = set()
    if args.resume and os.path.exists(out_path):
        done = {r["task_id"] for r in load_jsonl(out_path)}
        task_ids = [t for t in task_ids if t not in done]
        print(f"--resume: {len(done)} already done, {len(task_ids)} remaining", flush=True)
    if args.limit:
        task_ids = task_ids[:args.limit]
    if not task_ids:
        print("nothing to do", flush=True)
        return 0

    tok, model = load_model_and_tokenizer(args.model, args.revision, args.device, args.attn_impl)
    torch.manual_seed(args.seed)

    arm_name = f"planned-{args.organ}"
    max_new_tokens = DEFAULT_MAX_NEW_TOKENS["planned_stage_c"]

    # stage_a_fail rows need no generation at all -- written immediately, not batched with the rest.
    gen_task_ids = []
    mode = "a" if args.resume else "w"
    n_pass = n_syn_fail = n_stage_a_fail = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for task_id in task_ids:
            brow = chosen_rows.get(task_id)
            if brow is None or brow.get("stage_a_fail"):
                n_stage_a_fail += 1
                write_jsonl_append_flush(f, [dict(
                    task_id=task_id, arm=arm_name, stage_a_fail=True, generated_text="",
                    extracted_code="", passed=False, syntactic_failure=False, truncated=False,
                    tokens=brow["n_tokens_stage_a"] if brow else 0, decisions=[])])
            else:
                gen_task_ids.append(task_id)

        # batch the Gemma generation step (the GPU-bound part -- was 1 prompt/call before this fix, which
        # left the GPU mostly idle at ~1.2 problems/min; sorted by prompt token length for padding
        # efficiency, same convention as run_gen/run_planned_stage_a), then grade each batch's rows
        # CONCURRENTLY (ThreadPoolExecutor, not a process pool -- run_test_program already spawns a real
        # subprocess per row via subprocess.run, so threads just let up to --exec-workers of those
        # subprocess waits overlap instead of serializing one 10s-timeout exec after another; no pickling
        # needed since nothing but builtins crosses the thread boundary).
        prompts = {}
        for task_id in gen_task_ids:
            problem = problems_by_id[task_id]
            chosen = chosen_rows[task_id]["decisions"]
            block = render_decisions_block(chosen)
            prompts[task_id] = render_direct_prompt(tok, problem, thinking_on=False, extra_block=block)
        lens = {tid: len(tok(p, add_special_tokens=False)["input_ids"]) for tid, p in prompts.items()}
        gen_task_ids.sort(key=lambda tid: lens[tid])

        n_done = 0
        with ThreadPoolExecutor(max_workers=args.exec_workers) as pool:
            for i in range(0, len(gen_task_ids), args.batch_size):
                batch_ids = gen_task_ids[i:i + args.batch_size]
                batch_prompts = [prompts[tid] for tid in batch_ids]
                results = batched_generate(tok, model, batch_prompts, _GreedyArgs(max_new_tokens),
                                           args.device)
                batch_generated = []
                for tid, (raw_text, finished, n_new) in zip(batch_ids, results):
                    batch_generated.append((tid, _strip_pad_and_turn(raw_text), finished, n_new))

                exec_futures = {
                    tid: pool.submit(run_test_program, problems_by_id[tid]["prompt"],
                                      problems_by_id[tid]["entry_point"], problems_by_id[tid]["test"],
                                      generated)
                    for tid, generated, _fin, _n in batch_generated}

                out_rows = []
                for tid, generated, finished, n_new in batch_generated:
                    passed, synfail, extracted = exec_futures[tid].result()
                    n_pass += int(passed)
                    n_syn_fail += int(synfail)
                    chosen = chosen_rows[tid]["decisions"]
                    total_tokens = n_new + chosen_rows[tid]["n_tokens_stage_a"]
                    out_rows.append(dict(
                        task_id=tid, arm=arm_name, stage_a_fail=False, generated_text=generated,
                        extracted_code=extracted, passed=passed, syntactic_failure=synfail,
                        truncated=not finished, tokens=total_tokens, decisions=chosen))
                write_jsonl_append_flush(f, out_rows)
                n_done += len(batch_ids)
                print(f"planned-stage-c[{args.organ}]: {n_done}/{len(gen_task_ids)}; "
                      f"pass_so_far={n_pass}; syn_fail={n_syn_fail}; stage_a_fail={n_stage_a_fail}",
                      flush=True)
    print(f"planned-stage-c[{args.organ}] done -> {args.out}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# report (pure Python)
# ---------------------------------------------------------------------------

def summarize_rows(rows):
    n = len(rows)
    scorable = [r for r in rows if not r.get("stage_a_fail")]
    n_pass = sum(1 for r in scorable if r["passed"])
    n_syn_fail = sum(1 for r in rows if r.get("syntactic_failure"))
    n_trunc = sum(1 for r in rows if r.get("truncated"))
    tokens = [r["tokens"] for r in rows if "tokens" in r]
    return dict(
        n=n, n_stage_a_fail=n - len(scorable),
        pass_at_1=(n_pass / len(scorable)) if scorable else float("nan"),
        syntactic_failure_rate=(n_syn_fail / n) if n else float("nan"),
        truncation_rate=(n_trunc / n) if n else float("nan"),
        mean_tokens=(sum(tokens) / len(tokens)) if tokens else float("nan"),
    )


def render_humaneval_report(arm_rows):
    lines = ["# HumanEval probe (PROBE 2)", "", "## Per-arm summary", "",
             "| arm | n | pass@1 | mean tokens | syntactic-failure rate | truncation rate | "
             "stage-A failures |",
             "|---|---|---|---|---|---|---|"]
    for arm, rows in arm_rows.items():
        s = summarize_rows(rows)
        lines.append(f"| {arm} | {s['n']} | {s['pass_at_1']:.3f} | {s['mean_tokens']:.1f} | "
                      f"{s['syntactic_failure_rate']:.3f} | {s['truncation_rate']:.3f} | "
                      f"{s['n_stage_a_fail']} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def run_report(args):
    arm_rows = {}
    for path in args.rows:
        rows = load_jsonl(resolve(path))
        if not rows:
            continue
        arm = rows[0]["arm"]
        arm_rows[arm] = rows
    report = render_humaneval_report(arm_rows)
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

    sp_f = sub.add_parser("fetch")
    sp_f.add_argument("--out", default="oracle/humaneval-problems.jsonl")

    sp_g = sub.add_parser("gen")
    sp_g.add_argument("--arm", required=True, choices=["direct", "thinking"])
    sp_g.add_argument("--problems", default="oracle/humaneval-problems.jsonl")
    sp_g.add_argument("--out", required=True)
    sp_g.add_argument("--batch-size", type=int, default=16)
    sp_g.add_argument("--max-new-tokens", type=int, default=None)
    sp_g.add_argument("--seed", type=int, default=0)
    sp_g.add_argument("--limit", type=int, default=None)
    sp_g.add_argument("--resume", action="store_true")
    sp_g.add_argument("--model", default="google/gemma-4-E4B-it")
    sp_g.add_argument("--revision", default="ee0ef6023621cff504d758262d4e04895a5af4a2")
    sp_g.add_argument("--device", default="cuda")
    sp_g.add_argument("--attn-impl", default="sdpa")

    sp_pa = sub.add_parser("planned-stage-a")
    sp_pa.add_argument("--problems", default="oracle/humaneval-problems.jsonl")
    sp_pa.add_argument("--out", required=True)
    sp_pa.add_argument("--batch-size", type=int, default=16)
    sp_pa.add_argument("--seed", type=int, default=0)
    sp_pa.add_argument("--limit", type=int, default=None)
    sp_pa.add_argument("--resume", action="store_true")
    sp_pa.add_argument("--model", default="google/gemma-4-E4B-it")
    sp_pa.add_argument("--revision", default="ee0ef6023621cff504d758262d4e04895a5af4a2")
    sp_pa.add_argument("--device", default="cuda")
    sp_pa.add_argument("--attn-impl", default="sdpa")

    sp_pb = sub.add_parser("planned-stage-b")
    sp_pb.add_argument("--organ", required=True, choices=["kev", "head", "letterreadout"])
    sp_pb.add_argument("--problems", default="oracle/humaneval-problems.jsonl")
    sp_pb.add_argument("--decisions", required=True)
    sp_pb.add_argument("--out", required=True)
    sp_pb.add_argument("--limit", type=int, default=None)
    sp_pb.add_argument("--resume", action="store_true")
    sp_pb.add_argument("--model", default="google/gemma-4-E4B-it")
    sp_pb.add_argument("--revision", default="ee0ef6023621cff504d758262d4e04895a5af4a2")
    sp_pb.add_argument("--device", default="cuda")
    sp_pb.add_argument("--attn-impl", default="sdpa")

    sp_pc = sub.add_parser("planned-stage-c")
    sp_pc.add_argument("--organ", required=True, choices=["kev", "head", "letterreadout"],
                       help="only used to name the output arm (planned-{organ}); no model is loaded "
                            "for it here -- this stage reads --chosen, it never touches the organ model")
    sp_pc.add_argument("--problems", default="oracle/humaneval-problems.jsonl")
    sp_pc.add_argument("--chosen", required=True)
    sp_pc.add_argument("--out", required=True)
    sp_pc.add_argument("--batch-size", type=int, default=24,
                       help="Gemma generation batch size (was hardcoded to 1 -- the bottleneck fixed "
                            "here); drop to 16 on OOM")
    sp_pc.add_argument("--exec-workers", type=int, default=6,
                       help="thread-pool size for concurrently running each batch's test-execution "
                            "subprocesses (run_test_program), so 10s per-row timeouts overlap")
    sp_pc.add_argument("--seed", type=int, default=0)
    sp_pc.add_argument("--limit", type=int, default=None)
    sp_pc.add_argument("--resume", action="store_true")
    sp_pc.add_argument("--model", default="google/gemma-4-E4B-it")
    sp_pc.add_argument("--revision", default="ee0ef6023621cff504d758262d4e04895a5af4a2")
    sp_pc.add_argument("--device", default="cuda")
    sp_pc.add_argument("--attn-impl", default="sdpa")

    sp_r = sub.add_parser("report")
    sp_r.add_argument("--rows", nargs="+", required=True)
    sp_r.add_argument("--out-report", required=True)

    args = ap.parse_args()
    if args.selftest:
        return sys.exit(selftest())
    if args.cmd == "fetch":
        return sys.exit(run_fetch(args))
    if args.cmd == "gen":
        return sys.exit(run_gen(args))
    if args.cmd == "planned-stage-a":
        return sys.exit(run_planned_stage_a(args))
    if args.cmd == "planned-stage-b":
        return sys.exit(run_planned_stage_b(args))
    if args.cmd == "planned-stage-c":
        return sys.exit(run_planned_stage_c(args))
    if args.cmd == "report":
        return sys.exit(run_report(args))
    ap.error("one of --selftest / fetch / gen / planned-stage-a / planned-stage-b / planned-stage-c / "
             "report is required")


if __name__ == "__main__":
    main()
