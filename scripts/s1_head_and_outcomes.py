"""Stage S1 head-scoring + outcome-judging for the general-decisions go/no-go
(reports/25-phase7g-general-decisions-plan.md's S1 gates (c) head agreement >= 0.85 and
(d) head accuracy on outcome-verified choices >= 70%, consumed by scripts/s1_report.py alongside
scripts/s1_measure.py's (a) coverage and (b) replaceable-share gates).

Two independent subcommands (see each one's own --help / docstring below for exact I/O):

  headscore  -- GPU box. For every valid extracted decision point (scripts/extract_decision_points.py's
               output schema), builds the EXACT runtime head input (scripts/run_phase7r_runtime.py's
               stage2_score_row / scripts/probe_phase7r_head.make_real_scorer convention: state = user
               text + "\\n" + thinking-so-far, instruction = the point's own question, options = the
               point's own options rendered via kev.api.option_text "key: desc") and scores it with the
               real pointer head (runs/p4-e4b-final by default). Writes one row per scored point.

  outcomes   -- CPU only. Joins traces against their problems' gold answers to judge final correctness,
               then (for traces with extracted points, if --headscores is given) reports head accuracy
               restricted to outcome-verified choices -- see that subcommand's own docstring for the
               exact "determinable" rule.

Both subcommands' pure logic (state/option construction for headscore; numeric/letter extraction and the
head-vs-outcome join for outcomes) is split out into top-level functions with no torch/model dependency,
exercised by `--selftest` (headscore, fabricated fixtures + probe_phase7r_head.make_fake_scorer -- no
GPU) and directly by ordinary Python unittest-free asserts for outcomes (CPU always, no --selftest flag
needed since outcomes never touches a model).

Usage:
  CPU, always runnable:
    .venv/Scripts/python.exe scripts/s1_head_and_outcomes.py headscore --selftest
    .venv/Scripts/python.exe scripts/s1_head_and_outcomes.py headscore --traces <t> --points <p> \\
        --out <scratch.jsonl> --dry-run --limit 2
    .venv/Scripts/python.exe scripts/s1_head_and_outcomes.py outcomes \\
        --traces oracle/s1-traces.jsonl --problems oracle/s1-train-problems.jsonl \\
        --points oracle/s1-points.jsonl --headscores oracle/s1-headscores.jsonl \\
        --out oracle/s1-outcomes.jsonl

  GPU box (real pointer head, runs/p4-e4b-final):
    .venv/bin/python -u scripts/s1_head_and_outcomes.py headscore \\
        --traces oracle/s1-traces.jsonl --points oracle/s1-points.jsonl \\
        --out oracle/s1-headscores.jsonl --decision-run runs/p4-e4b-final --resume --shuffle-check
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_DECISION_RUN = "runs/p4-e4b-final"


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_done_keys(path):
    """-> set of (id, point_idx) already present in an existing --out file (--resume)."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {(r["id"], r["point_idx"]) for l in f if l.strip() for r in [json.loads(l)]}


def write_jsonl_append(f, rows):
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.flush()
    os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# headscore: pure state/option construction (CPU-testable, no torch/kev)
# ---------------------------------------------------------------------------

def option_text(name, desc):
    """kev.api.option_text's own rule ("key" alone if desc is empty/None, else "key: desc"), copied so
    --selftest/--dry-run never import kev.api (which pulls in kev.model -> torch at module scope).
    Matches kev/api.py:option_text byte-for-byte for the str-desc case this script ever produces."""
    return name if desc is None or desc == "" else f"{name}: {desc}"


def build_head_input(trace_row, point):
    """-> (state: str, instruction: str, options: list[str]). Exactly the runtime's stage2 convention
    (scripts/run_phase7r_runtime.py:stage2_score_row / scripts/probe_phase7r_head.py's S1 state rule):
    state = user text + "\\n" + the trace text BEFORE the point's quote_start (the thinking so far at
    the moment the model starts weighing alternatives -- NOT up to quote_choice, which is the commit
    itself and would leak the answer into the state). instruction = the point's own question. options =
    the point's own options, described ("key: desc"), in the point's own order (no reordering here --
    that is --shuffle-check's job, applied on top of this function's output)."""
    user = trace_row.get("user") or ""
    thinking = trace_row.get("thinking") or ""
    start_offset = point["quote_start_offset"]
    state = f"{user}\n{thinking[:start_offset]}"
    instruction = point["question"]
    options = [option_text(o["key"], o["desc"]) for o in point["options"]]
    return state, instruction, options


def option_key(opt_text):
    """-> bare key of a "key: desc"-described option string ("key" if no ": "). Same convention as
    scripts/gate_phase7r_stageA.py:option_key, copied (not imported) to avoid pulling that module's own
    glob-based I/O helpers into a pure-logic function."""
    return opt_text.split(": ", 1)[0]


def seeded_shuffle_options(options, seed):
    """-> a shuffle of `options` (list of described strings), order-stable for a given seed, re-rolled
    (same pattern as scripts/probe_phase7r_head.py:seeded_shuffle) if it happens to reproduce the
    original order (which would tell --shuffle-check nothing about order sensitivity)."""
    import random
    rng = random.Random(seed)
    shuffled = list(options)
    rng.shuffle(shuffled)
    tries = 0
    while shuffled == options and len(options) > 1 and tries < 10:
        seed += 1
        rng = random.Random(seed)
        shuffled = list(options)
        rng.shuffle(shuffled)
        tries += 1
    return shuffled


def iter_scorable_points(traces, points_rows):
    """-> [(trace_row, point_idx, point), ...] -- every (trace, point) pair where the trace row has a
    matching --points row AND the point passes the same basic shape check extract_decision_points.py's
    own validate_points already guaranteed (options non-empty, chosen_key present) -- defensive here
    only in case an --points file was hand-edited or partially written."""
    points_by_id = {r["id"]: r.get("points", []) for r in points_rows}
    traces_by_id = {t["id"]: t for t in traces}
    out = []
    for row_id, points in points_by_id.items():
        trace_row = traces_by_id.get(row_id)
        if trace_row is None:
            continue
        for idx, p in enumerate(points):
            if not p.get("options") or not p.get("chosen_key") or "quote_start_offset" not in p:
                continue
            out.append((trace_row, idx, p))
    return out


def score_point(scorer, trace_row, point, shuffle_seed=None):
    """-> dict, one headscore output row (no "id"/"point_idx" -- caller adds those). `shuffle_seed`, if
    given, also runs a second forward pass on a shuffled option order and adds shuffle_head_choice/
    shuffle_agree (agreement with the UNSHUFFLED head_choice, per the module docstring's --shuffle-check
    spec: "agreement of head choice under shuffled option order")."""
    state, instruction, options = build_head_input(trace_row, point)
    pred_text, conf, _max_prob, probs = scorer(state, instruction, options)
    head_choice = option_key(pred_text)
    model_choice = point["chosen_key"]
    out = dict(head_choice=head_choice, confidence=conf,
                probs={option_key(k): v for k, v in probs.items()},
                model_choice=model_choice, agree=(head_choice == model_choice))
    if shuffle_seed is not None:
        shuffled = seeded_shuffle_options(options, shuffle_seed)
        s_pred_text, _s_conf, _s_max, _s_probs = scorer(state, instruction, shuffled)
        s_head_choice = option_key(s_pred_text)
        out["shuffle_head_choice"] = s_head_choice
        out["shuffle_agree"] = (s_head_choice == head_choice)
    return out


# ---------------------------------------------------------------------------
# headscore: GPU scoring loop
# ---------------------------------------------------------------------------

def make_real_scorer(decision_run, device):
    from scripts.probe_phase7r_head import make_real_scorer as _make
    return _make(decision_run, device)


def run_headscore(args):
    from scripts.probe_phase7r_head import make_fake_scorer

    traces = load_jsonl(resolve(args.traces))
    points_rows = load_jsonl(resolve(args.points))
    pairs = iter_scorable_points(traces, points_rows)
    print(f"loaded {len(traces)} traces, {len(points_rows)} points-rows -> {len(pairs)} scorable points",
          flush=True)

    if args.limit:
        pairs = pairs[:args.limit]

    if args.dry_run:
        scorer = make_fake_scorer()
        for trace_row, idx, point in pairs[:5]:
            out = score_point(scorer, trace_row, point,
                               shuffle_seed=0 if args.shuffle_check else None)
            print(f"  id={trace_row['id']!r} point_idx={idx} (fake) head_choice={out['head_choice']!r} "
                  f"model_choice={out['model_choice']!r} agree={out['agree']} conf={out['confidence']:.3f}")
        return 0

    out_path = resolve(args.out)
    done = load_done_keys(out_path) if args.resume else set()
    if done:
        before = len(pairs)
        pairs = [(t, i, p) for t, i, p in pairs if (t["id"], i) not in done]
        print(f"--resume: {before - len(pairs)}/{before} points already in {args.out}, "
              f"{len(pairs)} remaining", flush=True)
    if not pairs:
        print("nothing to do", flush=True)
        return 0

    scorer = make_real_scorer(args.decision_run, args.device)
    mode = "a" if args.resume else "w"
    n_agree = n_shuffle_agree = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for i, (trace_row, idx, point) in enumerate(pairs):
            scored = score_point(scorer, trace_row, point,
                                  shuffle_seed=i if args.shuffle_check else None)
            out = dict(id=trace_row["id"], point_idx=idx, **scored)
            write_jsonl_append(f, [out])
            n_agree += int(out["agree"])
            n_shuffle_agree += int(out.get("shuffle_agree", False))
            if (i + 1) % 50 == 0 or i + 1 == len(pairs):
                msg = f"scored {i + 1}/{len(pairs)}; agreement so far {n_agree / (i + 1):.3f}"
                if args.shuffle_check:
                    msg += f"; shuffle-agreement {n_shuffle_agree / (i + 1):.3f}"
                print(msg, flush=True)
    print(f"done: {len(pairs)} points scored -> {args.out}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# headscore: --selftest (fabricated, labelled fixtures; fake scorer; no GPU)
# ---------------------------------------------------------------------------

def selftest_headscore():
    from scripts.probe_phase7r_head import make_fake_scorer

    # fabricated trace: thinking = "...weigh highway vs back road... I will recommend the back road..."
    trace_row = dict(
        id="fab-0001",
        user="Should I take the highway or the back road today?",
        thinking=("Let me think about the highway option: faster but more traffic. "
                   "The back road is slower but predictable. "
                   "I will recommend the back road."),
    )
    quote_start = "Let me think about the highway option"
    start_offset = trace_row["thinking"].find(quote_start)
    assert start_offset >= 0
    point = dict(
        quote_start_offset=start_offset,
        question="highway or back road?",
        options=[{"key": "highway", "desc": "faster but more traffic"},
                 {"key": "back_road", "desc": "slower but predictable"}],
        chosen_key="back_road",
    )

    # 1. build_head_input: state must end right BEFORE quote_start, never include the commit text.
    state, instruction, options = build_head_input(trace_row, point)
    assert state == trace_row["user"] + "\n" + trace_row["thinking"][:start_offset], state
    assert "I will recommend" not in state, "state leaked past quote_start into the commit text"
    assert instruction == "highway or back road?"
    assert options == ["highway: faster but more traffic", "back_road: slower but predictable"], options
    print("selftest: build_head_input OK (state cut at quote_start, options described)")

    # 2. option_key round-trips option_text's format, and the bare-key case (no ": ").
    assert option_key("back_road: slower but predictable") == "back_road"
    assert option_key("bare_key") == "bare_key"
    print("selftest: option_key OK (described + bare)")

    # 3. option_text matches kev.api.option_text's rule for the str-desc / None-desc cases this script
    #    ever constructs from extract_decision_points.py's schema (desc is always a short string there).
    assert option_text("k", "d") == "k: d"
    assert option_text("k", "") == "k"
    assert option_text("k", None) == "k"
    print("selftest: option_text OK (desc / empty-desc / None-desc)")

    # 4. score_point with the fake scorer: fields present, agree flag matches head_choice==model_choice,
    #    shuffle fields only present when requested.
    scorer = make_fake_scorer()
    out = score_point(scorer, trace_row, point)
    assert set(out) == {"head_choice", "confidence", "probs", "model_choice", "agree"}, out
    assert out["model_choice"] == "back_road"
    assert out["agree"] == (out["head_choice"] == "back_road")
    assert set(out["probs"]) == {"highway", "back_road"}, out["probs"]
    print(f"selftest: score_point (no shuffle) OK (fake head_choice={out['head_choice']!r})")

    out_shuf = score_point(scorer, trace_row, point, shuffle_seed=0)
    assert "shuffle_head_choice" in out_shuf and "shuffle_agree" in out_shuf
    assert out_shuf["shuffle_agree"] == (out_shuf["shuffle_head_choice"] == out_shuf["head_choice"])
    print("selftest: score_point (with --shuffle-check) OK (shuffle fields present, agree flag correct)")

    # 5. iter_scorable_points: joins traces<->points by id, skips points missing required fields.
    traces = [trace_row, dict(id="fab-0002", user="u2", thinking="t2")]
    points_rows = [
        dict(id="fab-0001", points=[point, dict(quote_start_offset=0)]),  # 2nd point missing options/key
        dict(id="fab-9999", points=[point]),  # no matching trace -- excluded, not an error
    ]
    pairs = iter_scorable_points(traces, points_rows)
    assert len(pairs) == 1 and pairs[0][0]["id"] == "fab-0001" and pairs[0][1] == 0, pairs
    print("selftest: iter_scorable_points OK (joins by id, drops malformed point, "
          "ignores unmatched trace-id)")

    # 6. seeded_shuffle_options re-rolls rather than silently returning the identity order.
    two_opt = ["a: x", "b: y"]
    shuf = seeded_shuffle_options(two_opt, 0)
    assert sorted(shuf) == sorted(two_opt) and shuf != two_opt, shuf
    print("selftest: seeded_shuffle_options OK (2-option list actually reorders)")

    print("selftest: all headscore checks passed")
    return 0


# ---------------------------------------------------------------------------
# outcomes: final-answer correctness judging (pure Python, CPU only)
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_UNIT_STRIP_RE = re.compile(r"[\$%]")


def extract_last_number(text):
    """-> float | None. gsm8k rule: the LAST number in `text`, commas/units ($, %) stripped before
    parsing. None if no number is found."""
    cleaned = _UNIT_STRIP_RE.sub("", text or "")
    matches = _NUM_RE.findall(cleaned)
    if not matches:
        return None
    last = matches[-1].replace(",", "")
    try:
        return float(last)
    except ValueError:
        return None


_LETTER_PATTERNS = [
    re.compile(r"answer\s+is\s*:?\s*\(?([A-Za-z])\)?\b", re.IGNORECASE),
    re.compile(r"\*\*\(?([A-Za-z])\)?\*\*"),
    re.compile(r"\b([A-Za-z])\.\s"),
    re.compile(r"\b([A-Za-z])\)"),
    re.compile(r"^\s*\(?([A-Za-z])\)?\s*$"),
]


def extract_choice_letter(text):
    """-> str | None (uppercased single letter). Tries, in order: "answer is B"/"answer is (B)",
    "**B**"/"**(B)**", "B. ..." , "B)", and finally a whole-text bare letter. First pattern that matches
    ANYWHERE in the text wins (patterns are tried in this fixed priority order across the whole text,
    not just at the end, since models sometimes restate the letter early then elaborate). None if no
    pattern matches anywhere (caller counts this as "unparsed")."""
    if not text:
        return None
    for pat in _LETTER_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    return None


def numbers_equal(a, b, tol=1e-6):
    return abs(a - b) <= tol


def judge_correctness(trace_row, problem_row):
    """-> (correct: bool | None, parsed_value: object | None, unparsed: bool). None/unparsed=False for
    "advice" (answer_type == "none", no gold -- skipped, not a failure). unparsed=True means the
    extractor found nothing in the visible answer (counted toward the "unparsed rate", correct=None)."""
    answer_type = problem_row.get("answer_type")
    gold = problem_row.get("gold_answer")
    answer_text = trace_row.get("answer") or ""
    if answer_type == "none" or gold is None:
        return None, None, False
    if answer_type == "numeric":
        val = extract_last_number(answer_text)
        if val is None:
            return None, None, True
        try:
            gold_val = float(str(gold).replace(",", ""))
        except ValueError:
            return None, val, True
        return numbers_equal(val, gold_val), val, False
    if answer_type == "choice":
        letter = extract_choice_letter(answer_text)
        if letter is None:
            return None, None, True
        return (letter == str(gold).upper()), letter, False
    return None, None, False


# ---------------------------------------------------------------------------
# outcomes: head-vs-outcome analysis (pure Python)
# ---------------------------------------------------------------------------

def head_accuracy_on_outcome_verified(traces, points_rows, problems_by_id, headscores_rows):
    """-> dict(n_checkable=int, n_determinable=int, n_scored=int, n_correct=int, accuracy=float|None).
    "Determinable" (per the task spec) means: the problem is a multiple-choice problem (answer_type ==
    "choice") whose final answer was itself correctly judged AND whose point's own option KEYS are
    exactly the problem's lettered choices (so "the correct final answer's key" unambiguously identifies
    one of the point's own options) -- i.e. the point IS the final-answer decision, not some other
    intermediate one. For every determinable point with a matching --headscores row, the point counts as
    "head correct" iff head_choice == the gold letter. n_checkable = points belonging to a trace with a
    checkable (non-"none", successfully-parsed) final answer; n_determinable <= n_checkable; n_scored =
    determinable points that also have a --headscores row (the rest are excluded from `accuracy`, not
    counted as wrong)."""
    points_by_id = {r["id"]: r.get("points", []) for r in points_rows}
    headscores_by_key = {(r["id"], r["point_idx"]): r for r in headscores_rows}
    n_checkable = n_determinable = n_scored = n_correct = 0
    for trace in traces:
        row_id = trace["id"]
        problem = problems_by_id.get(row_id)
        if problem is None:
            continue
        correct, parsed, unparsed = judge_correctness(trace, problem)
        if correct is None or unparsed:
            continue  # not checkable (advice, or final answer itself unparsed)
        points = points_by_id.get(row_id, [])
        for idx, p in enumerate(points):
            n_checkable += 1
            if problem.get("answer_type") != "choice":
                continue
            option_keys = {o["key"].upper() for o in p.get("options", [])}
            gold_letter = str(problem["gold_answer"]).upper()
            if option_keys != {chr(ord("A") + i) for i in range(len(option_keys))} or \
                    gold_letter not in option_keys:
                continue
            n_determinable += 1
            hs = headscores_by_key.get((row_id, idx))
            if hs is None:
                continue
            n_scored += 1
            n_correct += int(hs["head_choice"].upper() == gold_letter)
    accuracy = (n_correct / n_scored) if n_scored else None
    return dict(n_checkable=n_checkable, n_determinable=n_determinable, n_scored=n_scored,
                n_correct=n_correct, accuracy=accuracy)


# ---------------------------------------------------------------------------
# outcomes: summary table by source
# ---------------------------------------------------------------------------

def summarize_by_source(traces, points_rows, problems_by_id):
    """-> {source: dict(n, correct_rate, mean_thinking_tokens, coverage, points_per_trace, n_unparsed)}.
    correct_rate/n_unparsed are None for sources with no checkable gold (advice)."""
    import collections
    points_by_id = {r["id"]: r.get("points", []) for r in points_rows}
    by_source = collections.defaultdict(list)
    for trace in traces:
        row_id = trace["id"]
        problem = problems_by_id.get(row_id)
        source = (problem or trace).get("source", "unknown")
        n_points = len(points_by_id.get(row_id, []))
        correct = unparsed = None
        if problem is not None:
            correct, _parsed, unparsed = judge_correctness(trace, problem)
        by_source[source].append((trace.get("n_thinking_tokens") or 0, n_points, correct, unparsed))

    out = {}
    for source, records in sorted(by_source.items()):
        n = len(records)
        checkable = [c for _t, _p, c, u in records if c is not None]
        unparsed_flags = [u for _t, _p, c, u in records if u]
        out[source] = dict(
            n=n,
            correct_rate=(sum(checkable) / len(checkable)) if checkable else None,
            mean_thinking_tokens=sum(t for t, _p, _c, _u in records) / n if n else float("nan"),
            coverage=sum(1 for _t, p, _c, _u in records if p > 0) / n if n else float("nan"),
            points_per_trace=sum(p for _t, p, _c, _u in records) / n if n else float("nan"),
            n_unparsed=sum(1 for u in unparsed_flags if u),
            n_graded=len(checkable) + len(unparsed_flags),
        )
    return out


def render_outcomes_report(by_source_summary, head_vs_outcome):
    lines = ["# S1 outcomes report", "", "## By source", "",
             "| source | n traces | correct rate | mean thinking tokens | coverage | points/trace | "
             "unparsed/graded |",
             "|---|---|---|---|---|---|---|"]
    for source, s in by_source_summary.items():
        cr = f"{s['correct_rate']:.3f}" if s["correct_rate"] is not None else "n/a (no gold)"
        lines.append(f"| {source} | {s['n']} | {cr} | {s['mean_thinking_tokens']:.1f} | "
                      f"{s['coverage']:.3f} | {s['points_per_trace']:.3f} | "
                      f"{s['n_unparsed']}/{s['n_graded']} |")
    lines.append("")
    lines.append("## Head accuracy on outcome-verified choices")
    lines.append("")
    if head_vs_outcome is None:
        lines.append("(no --headscores given -- n/a)")
    else:
        acc = head_vs_outcome["accuracy"]
        lines.append(f"- checkable points (trace has a parsed, gradeable final answer): "
                      f"{head_vs_outcome['n_checkable']}")
        lines.append(f"- determinable (MC problem, point's options == the lettered choices, gold letter "
                      f"among them): {head_vs_outcome['n_determinable']}")
        lines.append(f"- scored by head (--headscores row present): {head_vs_outcome['n_scored']}")
        lines.append(f"- head accuracy on outcome-verified choices: "
                      f"{'n/a (0 scored)' if acc is None else f'{acc:.3f}'}")
    lines.append("")
    return "\n".join(lines) + "\n"


def run_outcomes(args):
    traces = load_jsonl(resolve(args.traces))
    problems = load_jsonl(resolve(args.problems))
    points_rows = load_jsonl(resolve(args.points)) if args.points else []
    headscores_rows = load_jsonl(resolve(args.headscores)) if args.headscores else []
    problems_by_id = {p["id"]: p for p in problems}

    by_source = summarize_by_source(traces, points_rows, problems_by_id)
    head_vs_outcome = (head_accuracy_on_outcome_verified(traces, points_rows, problems_by_id,
                                                          headscores_rows)
                        if args.headscores else None)

    out_rows = []
    points_by_id = {r["id"]: r.get("points", []) for r in points_rows}
    for trace in traces:
        problem = problems_by_id.get(trace["id"])
        if problem is None:
            continue
        correct, parsed, unparsed = judge_correctness(trace, problem)
        out_rows.append(dict(id=trace["id"], source=problem.get("source"), correct=correct,
                              parsed_value=parsed, unparsed=unparsed,
                              n_points=len(points_by_id.get(trace["id"], []))))

    out_path = resolve(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        write_jsonl_append(f, out_rows)
    print(f"wrote {len(out_rows)} rows -> {args.out}", flush=True)

    report = render_outcomes_report(by_source, head_vs_outcome)
    print(report)
    if args.report_out:
        report_path = resolve(args.report_out)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"wrote {report_path}")
    return 0


# ---------------------------------------------------------------------------
# outcomes: --selftest-equivalent checks (run unconditionally -- CPU only, no flag needed, but exposed
# as a function so `main --selftest` under the `outcomes` subcommand can also call it for symmetry)
# ---------------------------------------------------------------------------

def selftest_outcomes():
    # 1. extract_last_number: strips commas/units, takes the LAST number.
    assert extract_last_number("The total is $1,234.50 after tax, so 2 items remain.") == 2.0
    assert extract_last_number("Answer: 42") == 42.0
    assert extract_last_number("no numbers here") is None
    print("selftest: extract_last_number OK (comma/unit strip, last-number rule, no-number case)")

    # 2. extract_choice_letter: all documented patterns, plus a genuinely-unparseable case.
    assert extract_choice_letter("Reasoning...\nThe answer is B.") == "B"
    assert extract_choice_letter("So the answer is (C) because...") == "C"
    assert extract_choice_letter("Final: **D**") == "D"
    assert extract_choice_letter("A. This is correct") == "A"
    assert extract_choice_letter("I'd go with B) since...") == "B"
    assert extract_choice_letter("   C   ") == "C"
    assert extract_choice_letter("I am not sure what the answer is") is None
    print("selftest: extract_choice_letter OK (6 patterns + unparseable case)")

    # 3. judge_correctness: numeric, choice, advice(no gold), unparsed.
    numeric_problem = dict(answer_type="numeric", gold_answer="24")
    correct, val, unparsed = judge_correctness(dict(answer="So Marion's score is 24."), numeric_problem)
    assert correct is True and val == 24.0 and not unparsed

    choice_problem = dict(answer_type="choice", gold_answer="B")
    correct, val, unparsed = judge_correctness(dict(answer="The answer is C."), choice_problem)
    assert correct is False and val == "C" and not unparsed

    advice_problem = dict(answer_type="none", gold_answer=None)
    correct, val, unparsed = judge_correctness(dict(answer="Here is some advice..."), advice_problem)
    assert correct is None and val is None and not unparsed

    unparsed_problem = dict(answer_type="numeric", gold_answer="5")
    correct, val, unparsed = judge_correctness(dict(answer="I am still thinking about this."),
                                                 unparsed_problem)
    assert correct is None and val is None and unparsed
    print("selftest: judge_correctness OK (numeric/choice/advice/unparsed)")

    # 4. head_accuracy_on_outcome_verified: fabricated MC trace with a correct final answer and a point
    #    whose options ARE the lettered choices (determinable) vs one whose options are NOT (excluded).
    traces = [dict(id="mc-01", answer="The answer is B.")]
    problems_by_id = {"mc-01": dict(id="mc-01", source="mmlu", answer_type="choice", gold_answer="B")}
    points_rows = [dict(id="mc-01", points=[
        dict(options=[{"key": "A", "desc": "x"}, {"key": "B", "desc": "y"}], chosen_key="B"),
        dict(options=[{"key": "fast", "desc": "x"}, {"key": "slow", "desc": "y"}], chosen_key="fast"),
    ])]
    headscores_rows = [dict(id="mc-01", point_idx=0, head_choice="B"),
                        dict(id="mc-01", point_idx=1, head_choice="fast")]
    result = head_accuracy_on_outcome_verified(traces, points_rows, problems_by_id, headscores_rows)
    assert result["n_checkable"] == 2, result
    assert result["n_determinable"] == 1, result  # only point 0's options == {A, B}
    assert result["n_scored"] == 1, result
    assert result["accuracy"] == 1.0, result
    print(f"selftest: head_accuracy_on_outcome_verified OK ({result})")

    # 5. a determinable point with a WRONG head choice -> accuracy 0.0, not 1.0 or n/a.
    headscores_wrong = [dict(id="mc-01", point_idx=0, head_choice="A")]
    result_wrong = head_accuracy_on_outcome_verified(traces, points_rows, problems_by_id,
                                                       headscores_wrong)
    assert result_wrong["accuracy"] == 0.0, result_wrong
    print("selftest: head_accuracy_on_outcome_verified OK (wrong head choice -> accuracy 0.0)")

    # 6. summarize_by_source: coverage/points-per-trace/correct-rate over a tiny fabricated mix.
    traces2 = [
        dict(id="g1", answer="Answer: 7", n_thinking_tokens=10, source="gsm8k"),
        dict(id="g2", answer="no number", n_thinking_tokens=20, source="gsm8k"),
        dict(id="a1", answer="some advice", n_thinking_tokens=30, source="advice"),
    ]
    problems2 = {"g1": dict(id="g1", source="gsm8k", answer_type="numeric", gold_answer="7"),
                 "g2": dict(id="g2", source="gsm8k", answer_type="numeric", gold_answer="9"),
                 "a1": dict(id="a1", source="advice", answer_type="none", gold_answer=None)}
    points2 = [dict(id="g1", points=[dict(options=[{"key": "x", "desc": "y"}], chosen_key="x")])]
    summary = summarize_by_source(traces2, points2, problems2)
    # g1 is correct and parsed; g2 is unparsed (correct=None, excluded from correct_rate, counted in
    # n_unparsed instead) -- correct_rate is over PARSED/checkable rows only, so 1/1 = 1.0 here, not 0.5.
    assert summary["gsm8k"]["n"] == 2 and summary["gsm8k"]["correct_rate"] == 1.0, summary["gsm8k"]
    assert summary["gsm8k"]["n_unparsed"] == 1, summary["gsm8k"]
    assert summary["gsm8k"]["coverage"] == 0.5, summary["gsm8k"]
    assert summary["advice"]["correct_rate"] is None, summary["advice"]
    print(f"selftest: summarize_by_source OK (gsm8k={summary['gsm8k']}, advice={summary['advice']})")

    print("selftest: all outcomes checks passed")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def add_headscore_args(sp):
    sp.add_argument("--traces")
    sp.add_argument("--points")
    sp.add_argument("--out")
    sp.add_argument("--decision-run", default=DEFAULT_DECISION_RUN)
    sp.add_argument("--device", default="cuda")
    sp.add_argument("--resume", action="store_true")
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--shuffle-check", action="store_true")
    sp.add_argument("--selftest", action="store_true")


def add_outcomes_args(sp):
    # none of these are argparse-`required` (even though the module docstring's usage always passes
    # --traces/--problems/--out) so that `outcomes --selftest` alone works; non-selftest invocations are
    # checked for these in main() instead, where the error message can say so plainly.
    sp.add_argument("--traces", default=None)
    sp.add_argument("--problems", default=None)
    sp.add_argument("--points", default=None)
    sp.add_argument("--headscores", default=None)
    sp.add_argument("--out", default=None)
    sp.add_argument("--report-out", default=None)
    sp.add_argument("--selftest", action="store_true")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_headscore_args(sub.add_parser("headscore"))
    add_outcomes_args(sub.add_parser("outcomes"))
    args = ap.parse_args()

    if args.cmd == "headscore":
        if args.selftest:
            return sys.exit(selftest_headscore())
        if not (args.traces and args.points and (args.out or args.dry_run)):
            ap.error("headscore requires --traces/--points, and --out (unless --dry-run)")
        return sys.exit(run_headscore(args))

    if args.cmd == "outcomes":
        if args.selftest:
            return sys.exit(selftest_outcomes())
        if not (args.traces and args.problems and args.out):
            ap.error("outcomes requires --traces, --problems, and --out")
        return sys.exit(run_outcomes(args))


if __name__ == "__main__":
    main()
