"""PROBE 2 -- recursive decisional process: Gemma-E4B-it REASONS and writes the question + candidate
options; the DECISION ORGAN (Kev-4B / our trained head) ANSWERS; the runtime injects the organ's answer
back into Gemma's context; Gemma continues. This is the correct way to exercise the organ (it answers
questions a reasoner poses, scripts/stepwise_probe.py's mistake was asking the organ to PLAN instead) --
see this module's own file-header history in the repo for that prior probe's negative result (organ asked
to choose the next OPERATION picked "finish" at step 0, acc 0.015/0.010).

Same 200 GSM8K problems as oracle/s1-train-problems.jsonl (source=="gsm8k").

Lock-step multi-round protocol, max 8 rounds, run as independent per-ARM event logs (each arm's context
diverges after round 1 because the organ's picks differ, so Gemma is called once per (problem, arm, round)
-- see module docstring's own design note below):

Round r (<=8): Gemma-E4B-it (thinking OFF, greedy, max_new_tokens 90, a fixed 3-example few-shot prompt,
see FEWSHOT below) sees the problem + every previously RESOLVED ask for that arm and emits ONE line:
  ASK: <question> | A: <candidate> ; B: <candidate> ; C: <candidate>     (2-4 lettered candidates)
  FINAL: <number>                                                        (answer known)
parse_gemma_line parses this (never crashes on malformed output -- see its own docstring). For an ASK
line, the chooser (organ) picks one candidate letter from state = problem + resolved-so-far + question,
options = the candidates verbatim (already lettered by Gemma, so no re-lettering needed); the runtime
appends "RESOLVED: <question> -> <chosen candidate desc>" (the ORGAN's choice, never Gemma's own
preference) to that arm's context for the next round.

Choosers (arms):
  kev        -- jaredpalmer/kev-4b, scored via scripts.stepwise_probe.make_organ_scorer (forward pass +
                argmax over probs, same convention as scripts/s1_kev_score.py).
  head       -- runs/p4-e4b-final, same scorer, different `run`.
  first      -- control: always picks candidate "A" (option_key index 0). No model at all.
  gemma_self -- control (REFERENCE ONLY, not an organ decision): Gemma's own letter readout over the
                candidates it just proposed (a second, short generate call on the SAME already-loaded
                Gemma -- no extra model load; see run_gemma_round). Confidence is None (no forward-pass
                probs for a readout).

Design note on lock-step / memory: Gemma (~16GB) and Kev-4B (~10GB)/our head (~7GB) cannot co-reside on a
23GB L4. Orchestration alternates, per round: ONE Gemma load (batched across every arm's active problems
at once, plus `gemma_self`'s inline second readout call, done before freeing Gemma) -> free -> one `kev`
chooser load (batched decisions for every kev-arm ask this round) -> free -> one `head` chooser load ->
free. `first` needs no model (resolved inline, in Python, right after the Gemma round). Each arm keeps its
own independent, append-only event-log JSONL (oracle/recursive-probe-<arm>.jsonl); replay_log reconstructs
full per-problem state from the log, so every stage is resumable/checkpointed per round: re-running a
round's command is a no-op for any (problem, round) pair whose event is already logged.

Final answer = the value from a FINAL line, if one is ever emitted; otherwise (8 rounds exhausted without
FINAL) the LAST resolved candidate's own number (extract_last_number on its desc text), graded via
numbers_equal (scripts/s1_head_and_outcomes.py's tolerance rule) against problem["gold_answer"].

Usage:
  offline selftest (no model/GPU): line parser, prompt/context building, lock-step bookkeeping
  (active_for_ask/active_for_choice), final-answer extraction, resume/idempotency:
    .venv/Scripts/python.exe scripts/recursive_probe.py --selftest

  GPU box, one round of Gemma generation (all 4 arms at once, batched; also resolves `first`/`gemma_self`
  inline -- writes/updates oracle/recursive-probe-<arm>.jsonl for every arm in --arms):
    .venv/bin/python -u scripts/recursive_probe.py gemma-round --round 1 \\
        --arms kev,head,first,gemma_self --problems oracle/s1-train-problems.jsonl \\
        --out-dir oracle --batch-size 16 --max-new-tokens 90

  GPU box, one round of organ decisions for ONE arm (kev or head only -- first/gemma_self never need this):
    .venv/bin/python -u scripts/recursive_probe.py chooser-round --round 1 --arm kev \\
        --problems oracle/s1-train-problems.jsonl --out-dir oracle

  Report (after all 8 rounds x 4 arms' logs exist in --out-dir):
    .venv/Scripts/python.exe scripts/recursive_probe.py report \\
        --problems oracle/s1-train-problems.jsonl --out-dir oracle --out-report oracle/recursive-probe.md
"""
import argparse
import collections
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.s1_head_and_outcomes import extract_last_number, numbers_equal  # noqa: E402

MAX_ROUNDS = 8
ARMS = ["kev", "head", "first", "gemma_self"]
ORGAN_ARMS = ("kev", "head")  # arms whose ask-lines need a chooser-round (not resolved inline)
KEV_RUN = "jaredpalmer/kev-4b"
HEAD_RUN = "runs/p4-e4b-final"
GEMMA_MODEL = "google/gemma-4-E4B-it"
GEMMA_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"

FEWSHOT = """You solve word problems by asking yourself ONE question at a time about the next concrete \
arithmetic step, proposing 2-4 candidate results, and waiting to be told which candidate is correct \
before continuing. Once the final numeric answer is certain, output it. Always output EXACTLY ONE line.

Problem: A farm has 18 cows. 5 more cows are born, then 3 cows are sold. How many cows are on the farm now?

Resolved so far:
(none yet)

ASK: How many cows after the births? | A: 18 + 5 = 23 ; B: 18 - 5 = 13 ; C: 18 + 5 = 22

Resolved so far:
RESOLVED: How many cows after the births? -> A: 18 + 5 = 23

ASK: How many cows after the sale? | A: 23 - 3 = 20 ; B: 23 + 3 = 26

Resolved so far:
RESOLVED: How many cows after the births? -> A: 18 + 5 = 23
RESOLVED: How many cows after the sale? -> A: 23 - 3 = 20

FINAL: 20

Problem: Maria read 40 pages on Monday and twice as many on Tuesday. How many pages did she read in total?

Resolved so far:
(none yet)

ASK: How many pages on Tuesday? | A: 40 * 2 = 80 ; B: 40 / 2 = 20 ; C: 40 + 2 = 42

Resolved so far:
RESOLVED: How many pages on Tuesday? -> A: 40 * 2 = 80

ASK: How many pages in total? | A: 40 + 80 = 120 ; B: 80 - 40 = 40

Resolved so far:
RESOLVED: How many pages on Tuesday? -> A: 40 * 2 = 80
RESOLVED: How many pages in total? -> A: 40 + 80 = 120

FINAL: 120

Problem: A box of 6 pens costs $9. How much does 1 pen cost?

Resolved so far:
(none yet)

ASK: What does one pen cost? | A: 9 / 6 = 1.5 ; B: 9 * 6 = 54 ; C: 9 - 6 = 3

Resolved so far:
RESOLVED: What does one pen cost? -> A: 9 / 6 = 1.5

FINAL: 1.5"""


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl_flush(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_gsm8k_problems(path):
    return [r for r in load_jsonl(resolve(path)) if r.get("source") == "gsm8k"]


def arm_log_path(out_dir, arm):
    return os.path.join(resolve(out_dir), f"recursive-probe-{arm}.jsonl")


# ---------------------------------------------------------------------------
# line parsing (pure Python)
# ---------------------------------------------------------------------------

def parse_gemma_line(text):
    """-> dict. {"type": "final", "value": float} | {"type": "ask", "question": str,
    "candidates": [{"key": "A", "desc": str}, ...]} | {"type": "malformed"}. Never raises. Whichever of
    "ASK:"/"FINAL:" occurs FIRST in the text wins (case-insensitive); a line with neither, or an ASK with
    no "|" separator, or fewer than 2 well-formed "letter: desc" candidates, or a FINAL with no parseable
    number, is "malformed"."""
    if not text:
        return {"type": "malformed"}
    m_final = re.search(r"FINAL\s*:\s*(.+)", text, re.IGNORECASE)
    m_ask = re.search(r"ASK\s*:\s*(.+)", text, re.IGNORECASE)
    cands = [(m.start(), kind, m) for kind, m in (("final", m_final), ("ask", m_ask)) if m]
    if not cands:
        return {"type": "malformed"}
    cands.sort(key=lambda t: t[0])
    kind, m = cands[0][1], cands[0][2]
    rest = m.group(1).splitlines()[0]
    if kind == "final":
        val = extract_last_number(rest)
        if val is None:
            return {"type": "malformed"}
        return {"type": "final", "value": val}
    if "|" not in rest:
        return {"type": "malformed"}
    question_part, cand_part = rest.split("|", 1)
    question = question_part.strip()
    candidates = []
    for piece in cand_part.split(";"):
        piece = piece.strip()
        cm = re.match(r"^([A-Za-z])\s*:\s*(.+)$", piece)
        if cm:
            candidates.append({"key": cm.group(1).upper(), "desc": cm.group(2).strip()})
    if not question or len(candidates) < 2:
        return {"type": "malformed"}
    return {"type": "ask", "question": question, "candidates": candidates}


# ---------------------------------------------------------------------------
# prompt / context construction (pure Python)
# ---------------------------------------------------------------------------

def render_resolved_block(resolved):
    if not resolved:
        return "(none yet)"
    return "\n".join(f"RESOLVED: {r['question']} -> {r['chosen_key']}: {r['chosen_desc']}"
                      for r in resolved)


def render_problem_prompt(problem_text, resolved):
    return (f"{FEWSHOT}\n\nNow solve this new problem the same way. Problem: {problem_text}\n\n"
            f"Resolved so far:\n{render_resolved_block(resolved)}\n\n"
            "Emit exactly ONE line: either 'ASK: <question> | A: <candidate> ; B: <candidate>' (2-4 "
            "candidates, each a concrete next-step result including its arithmetic) or 'FINAL: <number>' "
            "if the answer is already certain from the resolved steps above.")


_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_READOUT_ANSWER_RE = re.compile(r"ANSWER\s*:?\s*\(?([A-Za-z])\)?", re.IGNORECASE)


def render_letter_readout_prompt(problem_text, resolved, question, candidates):
    lines = [f"Problem: {problem_text}", "", f"Resolved so far:\n{render_resolved_block(resolved)}", "",
             question]
    for c in candidates:
        lines.append(f"{c['key']}. {c['desc']}")
    lines.append("\nReply with exactly one line: ANSWER <letter>")
    return "\n".join(lines)


def parse_letter_readout_choice(text, candidates):
    """-> candidate dict chosen. Falls back to the first candidate if no letter is parseable or the
    parsed letter is not among `candidates`' own keys (never crashes)."""
    matches = _READOUT_ANSWER_RE.findall(text or "")
    keys = {c["key"]: c for c in candidates}
    if matches:
        letter = matches[-1].upper()
        if letter in keys:
            return keys[letter]
    return candidates[0]


# ---------------------------------------------------------------------------
# event log replay (pure Python -- the only source of per-problem state)
# ---------------------------------------------------------------------------

def new_state():
    return dict(resolved=[], pending_ask=None, last_round=0, done=False, terminated_by=None,
                final_value=None, n_gemma_calls=0, n_gemma_tokens=0)


def replay_log(events):
    """-> {problem_id: state}. `events` = every row from one arm's log file, in append order (the file's
    own order IS round order within a problem, since a problem's ask for round r is only ever written
    after round r-1's resolution -- see active_for_ask). Never raises on a malformed/duplicate event."""
    states = collections.defaultdict(new_state)
    for e in events:
        s = states[e["id"]]
        if s["done"]:
            continue
        t = e["type"]
        if t in ("ask", "final", "malformed"):
            s["last_round"] = e["round"]
            s["n_gemma_calls"] += 1
            s["n_gemma_tokens"] += e.get("n_tokens", 0)
        if t == "ask":
            s["pending_ask"] = dict(round=e["round"], question=e["question"], candidates=e["candidates"])
        elif t == "final":
            s["done"] = True
            s["terminated_by"] = "final"
            s["final_value"] = e["value"]
        elif t == "malformed":
            s["done"] = True
            s["terminated_by"] = "malformed"
            if s["resolved"]:
                s["final_value"] = extract_last_number(s["resolved"][-1]["chosen_desc"])
        elif t == "resolved":
            pa = s["pending_ask"]
            if pa is None or pa["round"] != e["round"]:
                continue  # stray/duplicate resolution for a round with no matching pending ask
            s["resolved"].append(dict(question=pa["question"], chosen_key=e["chosen_key"],
                                       chosen_desc=e["chosen_desc"], confidence=e.get("confidence")))
            s["pending_ask"] = None
    return dict(states)


def finalize_if_exhausted(state, max_rounds=MAX_ROUNDS):
    """-> state, mutated in place. Call once all `max_rounds` rounds have been run (report time): a
    problem that never got a FINAL/malformed event by then is forced to terminate with the last resolved
    candidate's own number (module docstring's final-answer rule), terminated_by="max_rounds"."""
    if not state["done"] and state["last_round"] >= max_rounds:
        state["done"] = True
        state["terminated_by"] = "max_rounds"
        state["final_value"] = (extract_last_number(state["resolved"][-1]["chosen_desc"])
                                 if state["resolved"] else None)
    return state


def active_for_ask(state, round_num):
    """-> True iff this problem needs a Gemma ASK/FINAL call for `round_num` (not already done, not
    waiting on a chooser decision, and exactly caught up to round_num - 1)."""
    if state["done"] or state["pending_ask"] is not None:
        return False
    expected_prev = 0 if round_num == 1 else round_num - 1
    return state["last_round"] == expected_prev


def active_for_choice(state, round_num):
    """-> True iff this problem has an unresolved ASK logged for exactly `round_num`."""
    return (not state["done"] and state["pending_ask"] is not None
            and state["pending_ask"]["round"] == round_num)


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------

def grade(final_value, gold_answer):
    if final_value is None:
        return False
    try:
        gold_val = float(str(gold_answer).replace(",", ""))
    except ValueError:
        return False
    return numbers_equal(final_value, gold_val)


# ---------------------------------------------------------------------------
# selftest (no model/GPU)
# ---------------------------------------------------------------------------

def selftest():
    # 1. parse_gemma_line: well-formed ASK/FINAL, malformed variants.
    p = parse_gemma_line("ASK: how many? | A: 2 + 2 = 4 ; B: 2 - 2 = 0 ; C: 2 * 2 = 4")
    assert p == {"type": "ask", "question": "how many?",
                 "candidates": [{"key": "A", "desc": "2 + 2 = 4"}, {"key": "B", "desc": "2 - 2 = 0"},
                                 {"key": "C", "desc": "2 * 2 = 4"}]}, p
    assert parse_gemma_line("FINAL: 42") == {"type": "final", "value": 42.0}
    assert parse_gemma_line("blah blah\nFINAL: the answer is $1,234.5") == {"type": "final", "value": 1234.5}
    assert parse_gemma_line("ASK: no pipe here")["type"] == "malformed"
    assert parse_gemma_line("ASK: q | A: only one candidate")["type"] == "malformed"
    assert parse_gemma_line("I don't know")["type"] == "malformed"
    assert parse_gemma_line("")["type"] == "malformed"
    print("selftest: parse_gemma_line OK (ask/final/malformed variants)")

    # 2. first-occurrence-wins when text contains both markers (shouldn't normally happen, must not crash).
    both = parse_gemma_line("FINAL: 5\nASK: x | A: a ; B: b")
    assert both == {"type": "final", "value": 5.0}, both
    print("selftest: parse_gemma_line OK (FINAL before ASK -> FINAL wins)")

    # 3. render_problem_prompt / render_resolved_block: empty vs populated history.
    assert "(none yet)" in render_problem_prompt("P", [])
    resolved = [dict(question="q1", chosen_key="A", chosen_desc="2+2=4")]
    rendered = render_problem_prompt("P", resolved)
    assert "RESOLVED: q1 -> A: 2+2=4" in rendered
    print("selftest: render_problem_prompt OK (empty + populated resolved history)")

    # 4. parse_letter_readout_choice: match + out-of-range/no-match fallback to first candidate.
    cands = [{"key": "A", "desc": "x"}, {"key": "B", "desc": "y"}]
    assert parse_letter_readout_choice("reasoning...\nANSWER B", cands) == cands[1]
    assert parse_letter_readout_choice("not sure", cands) == cands[0]
    assert parse_letter_readout_choice("ANSWER Z", cands) == cands[0]  # Z not among candidates
    print("selftest: parse_letter_readout_choice OK (match + fallback cases)")

    # 5. replay_log + active_for_ask/active_for_choice + finalize_if_exhausted: a full 2-round fake run,
    #    reaching FINAL, end to end, no model involved.
    events = [
        dict(id="p1", round=1, type="ask", question="q1", n_tokens=10,
             candidates=[{"key": "A", "desc": "4"}, {"key": "B", "desc": "9"}]),
    ]
    states = replay_log(events)
    s = states["p1"]
    assert active_for_choice(s, 1) is True and active_for_ask(s, 1) is False
    assert active_for_ask(s, 2) is False  # still pending round-1 choice
    events.append(dict(id="p1", round=1, type="resolved", chosen_key="A", chosen_desc="4"))
    states = replay_log(events)
    s = states["p1"]
    assert s["pending_ask"] is None and len(s["resolved"]) == 1
    assert active_for_ask(s, 2) is True and active_for_choice(s, 2) is False
    events.append(dict(id="p1", round=2, type="final", value=4.0, n_tokens=5))
    states = replay_log(events)
    s = states["p1"]
    assert s["done"] is True and s["terminated_by"] == "final" and s["final_value"] == 4.0
    assert active_for_ask(s, 3) is False  # done -- no more rounds needed
    assert grade(s["final_value"], "4") is True
    assert grade(s["final_value"], "9") is False
    print(f"selftest: replay_log/active_for_ask/active_for_choice OK (2-round fake run -> FINAL=4.0, "
          f"n_gemma_calls={s['n_gemma_calls']}, n_gemma_tokens={s['n_gemma_tokens']})")

    # 6. a problem that never emits FINAL within MAX_ROUNDS -> finalize_if_exhausted uses the last
    #    resolved candidate's own number, terminated_by "max_rounds", never crashes with no resolved at all.
    events2 = [dict(id="p2", round=r, type="ask", question=f"q{r}", n_tokens=1,
                     candidates=[{"key": "A", "desc": f"{r}"}, {"key": "B", "desc": "0"}])
               for r in range(1, MAX_ROUNDS + 1)]
    for r in range(1, MAX_ROUNDS + 1):
        events2.append(dict(id="p2", round=r, type="resolved", chosen_key="A", chosen_desc=f"{r}"))
    states2 = replay_log(events2)
    s2 = finalize_if_exhausted(states2["p2"])
    assert s2["done"] is True and s2["terminated_by"] == "max_rounds" and s2["final_value"] == 8.0, s2
    print(f"selftest: finalize_if_exhausted OK (8 rounds, no FINAL -> last resolved value 8.0): {s2['terminated_by']}")

    # simulate round 1 asked but never resolved (e.g. chooser-round never ran) -- the pending ask alone
    # keeps last_round stuck at 1, well short of MAX_ROUNDS, so finalize_if_exhausted correctly leaves it
    # untouched (still "in progress", not exhausted) rather than wrongly declaring it done.
    events3 = [dict(id="p3", round=1, type="ask", question="q", n_tokens=1,
                     candidates=[{"key": "A", "desc": "0"}])]
    states3 = replay_log(events3)
    s3 = finalize_if_exhausted(states3["p3"])
    assert s3["done"] is False, s3  # not exhausted -- last_round (1) < MAX_ROUNDS, correctly left pending
    print("selftest: finalize_if_exhausted OK (in-progress problem under MAX_ROUNDS left untouched)")

    # a problem that IS exhausted (8 rounds of asks) but with NO resolved history at all (every chooser
    # call somehow never landed) -> final_value stays None, never crashes.
    events3b = [dict(id="p3b", round=r, type="ask", question="q", n_tokens=1,
                      candidates=[{"key": "A", "desc": "0"}]) for r in range(1, MAX_ROUNDS + 1)]
    states3b = replay_log(events3b)
    s3 = finalize_if_exhausted(states3b["p3b"])
    assert s3["final_value"] is None and s3["terminated_by"] == "max_rounds", s3
    print("selftest: finalize_if_exhausted OK (no resolved history at all -> final_value None, no crash)")

    # 7. resume/idempotency: replaying a log with a duplicate "ask" event for an already-resolved round
    #    must not corrupt state (last_round only advances off ask/final/malformed events; a stray repeat
    #    ask for round 1 after round 1 is already resolved and round 2 started is simply ignored by
    #    active_for_ask's own round-math, since last_round is already 2 by then).
    dup_events = list(events[:2])  # p1's round-1 ask + resolved
    dup_events.append(dict(id="p1", round=1, type="ask", question="q1-dup", n_tokens=1,
                            candidates=[{"key": "A", "desc": "x"}, {"key": "B", "desc": "y"}]))
    states_dup = replay_log(dup_events)
    s_dup = states_dup["p1"]
    # the duplicate round-1 ask re-sets pending_ask (last write wins) -- resume's real safeguard is at the
    # orchestration layer (never issuing a duplicate generate call once an ask for that round already
    # exists in the log, see run_gemma_round's own "already logged" skip), exercised next:
    assert s_dup["pending_ask"] is not None  # confirms replay_log itself does not dedup, by design
    done_ids_round1 = {e["id"] for e in dup_events if e["type"] == "ask" and e["round"] == 1}
    assert done_ids_round1 == {"p1"}
    print("selftest: resume dedup contract OK (orchestration skips ids with an existing ask for the round)")

    # 8. resident-swap bookkeeping (no GPU/torch): swap_gemma_device is a plain .to(device) call; swap_
    #    chooser_device must ALSO update the fake model's own `.device` attribute (the kev.model.
    #    DecisionModel bug this whole mode exists to work around -- see that function's docstring), and
    #    both must return a non-negative elapsed time and never touch CUDA for device="cpu".
    class FakeModel:
        def __init__(self):
            self.to_calls = []
            self.device = "cpu"

        def to(self, device):
            self.to_calls.append(device)
            return self

    fake_gemma = FakeModel()
    t = swap_gemma_device(fake_gemma, "cpu")
    assert fake_gemma.to_calls == ["cpu"] and t >= 0, (fake_gemma.to_calls, t)
    assert fake_gemma.device == "cpu"  # swap_gemma_device never sets .device -- Gemma has no such field
    print("selftest: swap_gemma_device OK (plain .to(), no .device bookkeeping -- Gemma needs none)")

    fake_chooser = FakeModel()
    t_on = swap_chooser_device(fake_chooser, "cpu")   # device="cpu" throughout -- no CUDA needed to test
    assert fake_chooser.to_calls == ["cpu"] and t_on >= 0
    assert fake_chooser.device == "cpu", fake_chooser.device
    fake_chooser.device = "stale"  # simulate .to() having been called without the fix (what a bare
                                    # nn.Module.to() swap would leave behind for kev.model.DecisionModel)
    t_off = swap_chooser_device(fake_chooser, "cpu")
    assert fake_chooser.device == "cpu", fake_chooser.device  # swap_chooser_device corrects it every call
    print("selftest: swap_chooser_device OK (.to() AND .device both updated every swap, "
          "even correcting a stale .device)")

    # 9. ram_check: proceeds (True) when required_gb is far below any real machine's RAM (if psutil is
    #    installed) or when psutil isn't installed (can't check -> proceed with a warning, never block);
    #    refuses (False) only when psutil IS installed and required_gb is absurdly large.
    assert ram_check(0.001, "selftest-tiny") is True
    if ram_available_gb() is not None:
        assert ram_check(10 ** 9, "selftest-huge") is False
    print("selftest: ram_check OK (tiny requirement passes; huge requirement refuses when psutil can check)")

    print("selftest: all checks passed")
    return 0


# ---------------------------------------------------------------------------
# chooser scorers (kev/head via stepwise_probe's organ scorer; first needs no model)
# ---------------------------------------------------------------------------

def make_first_scorer():
    def scorer(candidates):
        return candidates[0], None
    return scorer


def _wrap_organ_raw_scorer(raw_scorer):
    """-> callable(problem_text, resolved, question, candidates) -> (chosen_candidate, confidence), given
    a raw_scorer of scripts.stepwise_probe.make_organ_scorer's / make_organ_scorer_from_model's shape.
    Builds the same state/instr/options shape those expect (options = "key: desc" strings, already
    lettered by Gemma -- no re-lettering)."""
    from scripts.stepwise_probe import option_key

    def scorer(problem_text, resolved, question, candidates):
        state = f"{problem_text}\n\nResolved so far:\n{render_resolved_block(resolved)}"
        options = [f"{c['key']}: {c['desc']}" for c in candidates]
        choice_text, confidence, _max_prob, _probs = raw_scorer(state, question, options)
        chosen_key = option_key(choice_text)
        chosen = next((c for c in candidates if c["key"] == chosen_key), candidates[0])
        return chosen, confidence

    return scorer


def make_organ_choice_scorer(run, device):
    """-> callable(problem_text, resolved, question, candidates) -> (chosen_candidate, confidence).
    Loads the chooser from disk onto `device` (the reload-per-round path)."""
    from scripts.stepwise_probe import make_organ_scorer
    return _wrap_organ_raw_scorer(make_organ_scorer(run, device))


def make_organ_choice_scorer_from_model(tok, model):
    """-> same callable as make_organ_choice_scorer, but for an ALREADY-LOADED (tok, model) pair -- the
    resident-swap path (scripts/recursive_probe.py's resident-arm mode moves `model` onto the GPU in place
    with swap_chooser_device before calling this, and back to CPU after)."""
    from scripts.stepwise_probe import make_organ_scorer_from_model
    return _wrap_organ_raw_scorer(make_organ_scorer_from_model(tok, model))


# ---------------------------------------------------------------------------
# Gemma round (GPU): ask/final generation for every active problem of every requested arm, batched
# together; `gemma_self` additionally resolved inline (2nd short generate call); `first` resolved inline
# in pure Python (no model) right after Gemma is freed.
# ---------------------------------------------------------------------------

class _GreedyArgs:
    def __init__(self, max_new_tokens):
        self.temperature = 0.0
        self.max_new_tokens = max_new_tokens
        self.top_p = 1.0
        self.top_k = 0


def gemma_round_body(tok, model, arms, round_num, problems, problems_by_id, out_dir, batch_size,
                      max_new_tokens, device):
    """The GPU-dependent work of one Gemma round for an already-loaded (tok, model) -- shared by
    run_gemma_round (loads/frees the model itself, the reload-per-round path) and the resident-arm loop
    (loads the model once, swaps it on/off the GPU, calls this every round). Writes/appends every arm's
    event log directly; returns the number of (arm, problem) pairs processed (0 = nothing to do, no log
    writes happened)."""
    from scripts.gen_think_traces import batched_generate, _strip_pad_and_turn

    # gather every (arm, problem) pair that needs a Gemma call this round, per arm's own log/state.
    todo = []  # (arm, problem_dict, state)
    for arm in arms:
        log_path = arm_log_path(out_dir, arm)
        states = replay_log(load_jsonl(log_path))
        for p in problems:
            s = states.get(p["id"], new_state())
            if active_for_ask(s, round_num):
                todo.append((arm, p, s))
    print(f"gemma-round {round_num}: {len(todo)} (arm, problem) pairs need a Gemma call "
          f"across arms {arms}", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return 0

    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": render_problem_prompt(p["user"], s["resolved"])}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False) for _arm, p, s in todo]
    lens = [len(tok(pr, add_special_tokens=False)["input_ids"]) for pr in prompts]
    order = sorted(range(len(todo)), key=lambda i: lens[i])
    todo_sorted = [todo[i] for i in order]
    prompts_sorted = [prompts[i] for i in order]

    events_by_arm = collections.defaultdict(list)
    gen_args = _GreedyArgs(max_new_tokens)
    for i in range(0, len(todo_sorted), batch_size):
        batch = todo_sorted[i:i + batch_size]
        batch_prompts = prompts_sorted[i:i + batch_size]
        results = batched_generate(tok, model, batch_prompts, gen_args, device)
        for (arm, p, s), (raw_text, _fin, n_new) in zip(batch, results):
            generated = _strip_pad_and_turn(raw_text)
            parsed = parse_gemma_line(generated)
            base = dict(id=p["id"], round=round_num, n_tokens=n_new, raw_text=generated)
            events_by_arm[arm].append({**base, **parsed})
        print(f"gemma-round {round_num}: {min(i + batch_size, len(todo_sorted))}/{len(todo_sorted)} "
              f"generated", flush=True)

    # gemma_self: for every "ask" event just produced, do a 2nd short readout call on the SAME loaded
    # model (no extra load) and resolve inline.
    if "gemma_self" in events_by_arm:
        self_asks = [e for e in events_by_arm["gemma_self"] if e["type"] == "ask"]
        if self_asks:
            states_gs = replay_log(load_jsonl(arm_log_path(out_dir, "gemma_self")))
            readout_prompts = []
            for e in self_asks:
                p = problems_by_id[e["id"]]
                prior_resolved = states_gs.get(p["id"], new_state())["resolved"]
                readout_prompts.append(tok.apply_chat_template(
                    [{"role": "user", "content": render_letter_readout_prompt(
                        p["user"], prior_resolved, e["question"], e["candidates"])}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False))
            readout_args = _GreedyArgs(40)
            resolved_events = []
            for i in range(0, len(self_asks), batch_size):
                batch_e = self_asks[i:i + batch_size]
                batch_p = readout_prompts[i:i + batch_size]
                results = batched_generate(tok, model, batch_p, readout_args, device)
                for e, (raw_text, _fin, _n) in zip(batch_e, results):
                    generated = _strip_pad_and_turn(raw_text)
                    chosen = parse_letter_readout_choice(generated, e["candidates"])
                    resolved_events.append(dict(id=e["id"], round=round_num, type="resolved",
                                                 chosen_key=chosen["key"], chosen_desc=chosen["desc"],
                                                 confidence=None))
            events_by_arm["gemma_self"].extend(resolved_events)

    # first: resolve inline, pure Python, no model needed at all.
    if "first" in events_by_arm:
        first_resolved = []
        for e in events_by_arm["first"]:
            if e["type"] == "ask":
                c = e["candidates"][0]
                first_resolved.append(dict(id=e["id"], round=round_num, type="resolved",
                                            chosen_key=c["key"], chosen_desc=c["desc"], confidence=None))
        events_by_arm["first"].extend(first_resolved)

    for arm, events in events_by_arm.items():
        append_jsonl_flush(arm_log_path(out_dir, arm), events)
        print(f"gemma-round {round_num}[{arm}]: wrote {len(events)} events -> "
              f"{arm_log_path(out_dir, arm)}", flush=True)
    return len(todo)


def run_gemma_round(args):
    from scripts.gen_think_traces import load_model_and_tokenizer

    arms = args.arms.split(",")
    problems = load_gsm8k_problems(args.problems)
    problems_by_id = {p["id"]: p for p in problems}

    tok, model = load_model_and_tokenizer(GEMMA_MODEL, GEMMA_REVISION, args.device, args.attn_impl)
    try:
        gemma_round_body(tok, model, arms, args.round, problems, problems_by_id, args.out_dir,
                          args.batch_size, args.max_new_tokens, args.device)
    finally:
        del model
        import torch
        torch.cuda.empty_cache()
    return 0


# ---------------------------------------------------------------------------
# chooser round (GPU, kev/head only): resolve every pending ask for this round for ONE organ arm.
# ---------------------------------------------------------------------------

def chooser_round_body(scorer, arm, round_num, problems_by_id, out_dir):
    """The scoring work of one chooser round for an already-built `scorer` (make_organ_choice_scorer /
    make_organ_choice_scorer_from_model) -- shared by run_chooser_round (loads/frees the model itself) and
    the resident-arm loop (loads once, swaps on/off the GPU, calls this every round). Appends `arm`'s event
    log directly; returns the number of asks resolved (0 = nothing to do, no log write happened)."""
    log_path = arm_log_path(out_dir, arm)
    states = replay_log(load_jsonl(log_path))

    todo = [(pid, s) for pid, s in states.items() if active_for_choice(s, round_num)]
    print(f"chooser-round {round_num}[{arm}]: {len(todo)} pending asks to resolve", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return 0

    events = []
    for pid, s in todo:
        p = problems_by_id[pid]
        pa = s["pending_ask"]
        chosen, confidence = scorer(p["user"], s["resolved"], pa["question"], pa["candidates"])
        events.append(dict(id=pid, round=round_num, type="resolved", chosen_key=chosen["key"],
                            chosen_desc=chosen["desc"], confidence=confidence))
    append_jsonl_flush(log_path, events)
    print(f"chooser-round {round_num}[{arm}]: resolved {len(events)} -> {log_path}", flush=True)
    return len(events)


def run_chooser_round(args):
    if args.arm not in ORGAN_ARMS:
        print(f"ERROR: chooser-round only applies to {ORGAN_ARMS} (first/gemma_self resolve inline "
              f"during gemma-round)", file=sys.stderr)
        return 1
    problems = load_gsm8k_problems(args.problems)
    problems_by_id = {p["id"]: p for p in problems}
    states = replay_log(load_jsonl(arm_log_path(args.out_dir, args.arm)))
    if not any(active_for_choice(s, args.round) for s in states.values()):
        print(f"chooser-round {args.round}[{args.arm}]: 0 pending asks to resolve", flush=True)
        print("nothing to do", flush=True)
        return 0
    run_name = KEV_RUN if args.arm == "kev" else HEAD_RUN
    scorer = make_organ_choice_scorer(run_name, args.device)
    chooser_round_body(scorer, args.arm, args.round, problems_by_id, args.out_dir)
    return 0


# ---------------------------------------------------------------------------
# resident-swap mode: ONE process per organ arm (+ its piggybacking control arms), Gemma and the chooser
# each loaded ONCE from disk onto CPU, then moved to the GPU and back every round instead of reloaded.
# Gemma is a plain HF PreTrainedModel: nn.Module.to(device) is enough, its own .device property is
# derived from its parameters, nothing else to fix up. The kev chooser (kev.model.DecisionModel) is NOT
# move-clean: __init__ stores `self.device` as a separate string attribute (kev/model.py:384) that
# encode()/probs() use to build every fresh tensor (ids/pos/att/masks, kev/model.py:423-465) -- nn.Module.
# to() moves the parameters and buffers but does NOT touch that attribute, so a swap that only calls
# model.to(device) leaves self.device stale and the next encode() call mixes CPU-built index tensors with
# GPU weights (device-mismatch crash). swap_chooser_device below sets model.device = device right after
# .to() to fix this. (CUDA graphs / fused kernels, kev.checkpoint.LoadOptions.cuda_graphs / .fused, are
# off by default and never requested here, so those harder-to-move paths -- a CudaGraphs capture is bound
# to the device it was captured on -- do not come up.)
# ---------------------------------------------------------------------------

GEMMA_RAM_GB = 16.0     # bf16 Gemma-E4B-it resident on CPU
CHOOSER_RAM_GB = 9.0    # bf16 kev-4b / our head resident on CPU
RAM_HEADROOM_GB = 2.0


def ram_available_gb():
    """-> available RAM in GB, or None if psutil isn't installed (caller must then skip the check, not
    guess)."""
    try:
        import psutil
    except ImportError:
        return None
    return psutil.virtual_memory().available / 1e9


def ram_check(required_gb, label):
    """-> True if resident mode should proceed. Prints the available/required RAM either way. No psutil
    -> cannot check, proceeds with a warning (resident mode's CPU residency is the only thing at risk;
    worst case is an OOM kill, not silent wrong results)."""
    avail = ram_available_gb()
    if avail is None:
        print(f"[ram] psutil not installed -- cannot check available RAM for {label} "
              f"(need ~{required_gb:.1f}GB); proceeding without the check", flush=True)
        return True
    ok = avail >= required_gb
    print(f"[ram] available={avail:.1f}GB required~={required_gb:.1f}GB ({label}): "
          f"{'OK' if ok else 'INSUFFICIENT'}", flush=True)
    return ok


def gpu_mem_report(tag, device):
    if not str(device).startswith("cuda"):
        return
    import torch
    free, total = torch.cuda.mem_get_info()
    print(f"[mem] {tag}: free={free / 1e9:.2f}GB total={total / 1e9:.2f}GB "
          f"max_allocated={torch.cuda.max_memory_allocated() / 1e9:.2f}GB", flush=True)


def swap_gemma_device(model, device):
    """-> elapsed seconds. A plain HF model: .to(device) is the whole story (its .device property reads
    off its own parameters, nothing cached to fix up)."""
    import time
    t0 = time.time()
    model.to(device)
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize()
    return time.time() - t0


def swap_chooser_device(model, device):
    """-> elapsed seconds. kev.model.DecisionModel keeps `self.device` as a separate string attribute that
    encode()/probs() use to build fresh tensors (see this section's header comment) -- nn.Module.to() does
    NOT update it, so this sets it explicitly right after .to(). Must be used for every chooser swap;
    swap_gemma_device above is NOT a substitute (Gemma has no such attribute; the chooser needs this one
    extra line or it crashes on the first encode() call after the swap)."""
    import time
    t0 = time.time()
    model.to(device)
    model.device = device
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize()
    return time.time() - t0


def run_resident_arm(args):
    """One process, one organ arm (plus any piggybacking control arms in --arms that need only Gemma):
    load Gemma and (if an organ arm is present) the chooser ONCE from disk onto CPU, then for each of
    MAX_ROUNDS rounds, swap Gemma onto the GPU, run gemma_round_body for every arm in --arms, swap it back
    to CPU, swap the chooser onto the GPU (if present), run chooser_round_body, swap it back. Falls back
    to printing a message and returning exit code 2 (never silently wrong) if --no-ram-check wasn't passed
    and psutil reports insufficient RAM for Gemma + chooser resident on CPU at once; the caller (
    run_recursive.sh) is expected to fall back to the existing per-round reload commands in that case."""
    import time
    from scripts.gen_think_traces import load_model_and_tokenizer

    arms = args.arms.split(",")
    organ_arms = [a for a in arms if a in ORGAN_ARMS]
    if len(organ_arms) > 1:
        print(f"ERROR: resident-arm handles at most one organ arm per process, got {organ_arms}",
              file=sys.stderr)
        return 1
    organ_arm = organ_arms[0] if organ_arms else None

    required = GEMMA_RAM_GB + (CHOOSER_RAM_GB if organ_arm else 0.0) + RAM_HEADROOM_GB
    if not args.no_ram_check and not ram_check(required, f"resident-arm --arms {args.arms}"):
        print("INSUFFICIENT RAM for resident-swap mode; falling back to the reload-per-round path: run "
              "gemma-round / chooser-round per round instead (see scripts/run_recursive.sh's non-resident "
              "branch).", file=sys.stderr)
        return 2

    problems = load_gsm8k_problems(args.problems)
    problems_by_id = {p["id"]: p for p in problems}

    t0 = time.time()
    tok, gemma = load_model_and_tokenizer(GEMMA_MODEL, GEMMA_REVISION, "cpu", args.attn_impl)
    print(f"[resident] Gemma loaded on CPU in {time.time() - t0:.1f}s", flush=True)

    ctok = cmodel = None
    if organ_arm:
        from kev.checkpoint import Checkpoint, LoadOptions
        import torch
        run_name = KEV_RUN if organ_arm == "kev" else HEAD_RUN
        t0 = time.time()
        ctok, cmodel = Checkpoint(run_name).load("cpu", LoadOptions(dtype=torch.bfloat16))
        print(f"[resident] chooser ({organ_arm}) loaded on CPU in {time.time() - t0:.1f}s", flush=True)

    total_gemma_on = total_gemma_off = total_chooser_on = total_chooser_off = 0.0
    for r in range(1, MAX_ROUNDS + 1):
        t_round = time.time()

        t = swap_gemma_device(gemma, args.device)
        total_gemma_on += t
        gpu_mem_report(f"round {r} gemma-on-gpu", args.device)
        print(f"[resident] round {r}: gemma -> {args.device} in {t:.2f}s", flush=True)

        gemma_round_body(tok, gemma, arms, r, problems, problems_by_id, args.out_dir, args.batch_size,
                          args.max_new_tokens, args.device)

        t = swap_gemma_device(gemma, "cpu")
        total_gemma_off += t
        if str(args.device).startswith("cuda"):
            import torch
            torch.cuda.empty_cache()
        gpu_mem_report(f"round {r} gemma-off-gpu", args.device)
        print(f"[resident] round {r}: gemma -> cpu in {t:.2f}s", flush=True)

        if organ_arm:
            t = swap_chooser_device(cmodel, args.device)
            total_chooser_on += t
            gpu_mem_report(f"round {r} chooser-on-gpu", args.device)
            print(f"[resident] round {r}: chooser -> {args.device} in {t:.2f}s", flush=True)

            scorer = make_organ_choice_scorer_from_model(ctok, cmodel)
            chooser_round_body(scorer, organ_arm, r, problems_by_id, args.out_dir)

            t = swap_chooser_device(cmodel, "cpu")
            total_chooser_off += t
            if str(args.device).startswith("cuda"):
                import torch
                torch.cuda.empty_cache()
            gpu_mem_report(f"round {r} chooser-off-gpu", args.device)
            print(f"[resident] round {r}: chooser -> cpu in {t:.2f}s", flush=True)

        print(f"[resident] round {r} wall time: {time.time() - t_round:.1f}s", flush=True)

    print(f"[resident] totals: gemma-on={total_gemma_on:.1f}s gemma-off={total_gemma_off:.1f}s "
          f"chooser-on={total_chooser_on:.1f}s chooser-off={total_chooser_off:.1f}s "
          f"(vs. a 24-load reload-per-round run, this is 2 disk loads total instead of up to 24)",
          flush=True)
    return 0


# ---------------------------------------------------------------------------
# report (CPU only)
# ---------------------------------------------------------------------------

def summarize_arm(problems, states):
    n = len(problems)
    n_correct = n_final = 0
    rounds_list = []
    tokens_list = []
    pick_dist = collections.Counter()
    n_differs = 0
    divergent_ids = set()
    for p in problems:
        s = finalize_if_exhausted(dict(states.get(p["id"], new_state())))
        correct = grade(s["final_value"], p["gold_answer"])
        n_correct += int(correct)
        n_final += int(s["terminated_by"] == "final")
        rounds_list.append(s["n_gemma_calls"])
        tokens_list.append(s["n_gemma_tokens"])
        for r in s["resolved"]:
            pick_dist[r["chosen_key"]] += 1
            if r["chosen_key"] != "A":
                n_differs += 1
                divergent_ids.add(p["id"])
    n_resolutions = sum(pick_dist.values())
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")

    divergent_correct = sum(1 for p in problems if p["id"] in divergent_ids
                             and grade(finalize_if_exhausted(dict(states.get(p["id"], new_state())))
                                       ["final_value"], p["gold_answer"]))
    non_divergent_ids = [p for p in problems if p["id"] not in divergent_ids]
    non_divergent_correct = sum(1 for p in non_divergent_ids
                                 if grade(finalize_if_exhausted(dict(states.get(p["id"], new_state())))
                                          ["final_value"], p["gold_answer"]))

    return dict(
        n=n, accuracy=n_correct / n if n else float("nan"), frac_final=n_final / n if n else float("nan"),
        mean_rounds=mean(rounds_list), mean_tokens=mean(tokens_list),
        pick_dist=dict(pick_dist), frac_differs_from_a=(n_differs / n_resolutions) if n_resolutions else float("nan"),
        n_divergent=len(divergent_ids),
        acc_divergent=(divergent_correct / len(divergent_ids)) if divergent_ids else float("nan"),
        acc_non_divergent=(non_divergent_correct / len(non_divergent_ids)) if non_divergent_ids else float("nan"),
    )


def render_trace_verbatim(problem, state):
    s = finalize_if_exhausted(dict(state))
    correct = grade(s["final_value"], problem["gold_answer"])
    lines = [f"id={problem['id']} correct={correct} terminated_by={s['terminated_by']} "
             f"final_value={s['final_value']} gold={problem['gold_answer']}"]
    for r in s["resolved"]:
        lines.append(f"  RESOLVED: {r['question']} -> {r['chosen_key']}: {r['chosen_desc']} "
                      f"(confidence={r['confidence']})")
    return "\n".join(lines)


def pick_sample_traces(problems, states, k=3):
    scored = [(p, grade(finalize_if_exhausted(dict(states.get(p["id"], new_state())))["final_value"],
                         p["gold_answer"])) for p in problems]
    wrong = [p for p, c in scored if not c]
    right = [p for p, c in scored if c]
    sample = []
    if wrong:
        sample.append(wrong[0])
    sample += right[: k - len(sample)]
    sample += wrong[1: k - len(sample)]
    return sample[:k]


def render_report(problems, arm_states, ref_tokens):
    lines = ["# Recursive decisional probe (PROBE 2)", "",
              f"Reference Gemma-generated tokens (format-probe.jsonl, same problems, direct answering): "
              f"thinking {ref_tokens.get('thinking', 'n/a')}, direct {ref_tokens.get('direct', 'n/a')}.",
              "", "## Per-arm summary", "",
              "| arm | n | accuracy | frac reaching FINAL | mean rounds | mean Gemma tokens/problem | "
              "pick dist (A/B/C/D) | frac differs from A | acc if ever differs | acc if never differs |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, states in arm_states.items():
        s = summarize_arm(problems, states)
        lines.append(f"| {arm} | {s['n']} | {s['accuracy']:.3f} | {s['frac_final']:.3f} | "
                      f"{s['mean_rounds']:.2f} | {s['mean_tokens']:.1f} | {s['pick_dist']} | "
                      f"{s['frac_differs_from_a']:.3f} | {s['acc_divergent']:.3f} ({s['n_divergent']}) | "
                      f"{s['acc_non_divergent']:.3f} |")
    lines.append("")
    lines.append("## Verbatim traces (3 per arm)")
    lines.append("")
    for arm, states in arm_states.items():
        lines.append(f"### {arm}")
        lines.append("")
        for p in pick_sample_traces(problems, states):
            lines.append("```")
            lines.append(render_trace_verbatim(p, states.get(p["id"], new_state())))
            lines.append("```")
        lines.append("")
    return "\n".join(lines) + "\n"


def run_report(args):
    problems = load_gsm8k_problems(args.problems)
    arm_states = {}
    for arm in ARMS:
        arm_states[arm] = replay_log(load_jsonl(arm_log_path(args.out_dir, arm)))

    ref_tokens = {}
    fp_path = resolve("oracle/format-probe.jsonl")
    if os.path.exists(fp_path):
        fp_rows = load_jsonl(fp_path)
        gsm_rows = [r for r in fp_rows if r.get("source") == "gsm8k"]
        for arm_name in ("thinking", "direct"):
            toks = [r["tokens"] for r in gsm_rows if r.get("arm") == arm_name]
            if toks:
                ref_tokens[arm_name] = round(sum(toks) / len(toks), 1)

    report = render_report(problems, arm_states, ref_tokens)
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

    sp_g = sub.add_parser("gemma-round")
    sp_g.add_argument("--round", type=int, required=True)
    sp_g.add_argument("--arms", default=",".join(ARMS))
    sp_g.add_argument("--problems", default="oracle/s1-train-problems.jsonl")
    sp_g.add_argument("--out-dir", default="oracle")
    sp_g.add_argument("--batch-size", type=int, default=16)
    sp_g.add_argument("--max-new-tokens", type=int, default=90)
    sp_g.add_argument("--device", default="cuda")
    sp_g.add_argument("--attn-impl", default="sdpa")

    sp_c = sub.add_parser("chooser-round")
    sp_c.add_argument("--round", type=int, required=True)
    sp_c.add_argument("--arm", required=True, choices=ORGAN_ARMS)
    sp_c.add_argument("--problems", default="oracle/s1-train-problems.jsonl")
    sp_c.add_argument("--out-dir", default="oracle")
    sp_c.add_argument("--device", default="cuda")

    sp_ra = sub.add_parser("resident-arm")
    sp_ra.add_argument("--arms", required=True,
                        help="one organ arm (kev or head) plus any piggybacking control arms, e.g. "
                             "kev,first,gemma_self or head")
    sp_ra.add_argument("--problems", default="oracle/s1-train-problems.jsonl")
    sp_ra.add_argument("--out-dir", default="oracle")
    sp_ra.add_argument("--batch-size", type=int, default=16)
    sp_ra.add_argument("--max-new-tokens", type=int, default=90)
    sp_ra.add_argument("--device", default="cuda")
    sp_ra.add_argument("--attn-impl", default="sdpa")
    sp_ra.add_argument("--no-ram-check", action="store_true",
                        help="skip the psutil RAM check and proceed unconditionally")

    sp_r = sub.add_parser("report")
    sp_r.add_argument("--problems", default="oracle/s1-train-problems.jsonl")
    sp_r.add_argument("--out-dir", default="oracle")
    sp_r.add_argument("--out-report", required=True)

    args = ap.parse_args()
    if args.selftest:
        return sys.exit(selftest())
    if args.cmd == "gemma-round":
        return sys.exit(run_gemma_round(args))
    if args.cmd == "chooser-round":
        return sys.exit(run_chooser_round(args))
    if args.cmd == "resident-arm":
        return sys.exit(run_resident_arm(args))
    if args.cmd == "report":
        return sys.exit(run_report(args))
    ap.error("one of --selftest / gemma-round / chooser-round / resident-arm / report is required")


if __name__ == "__main__":
    main()
