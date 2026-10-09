"""Phase 7 CHECK 7 eval: pure-Python parsing/metrics (no torch/transformers/kev imports).

Kept separate from scripts/eval_phase7.py so every function here is importable and unit-testable
on a machine where torch is broken (the laptop). scripts/eval_phase7.py --dry-parse exercises this
module against canned strings; nothing here touches the network, a GPU, or a model.

Mirrors scripts/build_phase7.py's exact target shape:
  "...<|decide|>{instructions} [{names, comma-space joined}]<|result|>{label}<|/result|> {continuation}"
  continuation == f"This looks like {label.replace('_', ' ')}. Let me help with that right away."
"""
import random
import re

DECIDE_RE = re.compile(r"<\|decide\|>(?P<instr>.*?)\[(?P<opts>.*?)\]<\|result\|>", re.DOTALL)
RESULT_SPAN_RE = re.compile(r"<\|result\|>(?P<label>.*?)<\|/result\|>", re.DOTALL)
TEMPLATE_RE = re.compile(
    r"This looks like (?P<phrase>.+?)\.\s*Let me help with that right away\.", re.IGNORECASE)


def label_phrase(label):
    return label.replace("_", " ")


def parse_decide_call(text):
    """-> {"instructions": str, "options": [str, ...]} for the first <|decide|>...[opts]<|result|> span in
    `text`, or None if no well-formed call is present. Options are split on ', ' (build_phase7's join) and
    stripped; empty option lists are treated as malformed (None)."""
    m = DECIDE_RE.search(text)
    if not m:
        return None
    opts_raw = m.group("opts").strip()
    if not opts_raw:
        return None
    options = [o.strip() for o in opts_raw.split(",")]
    options = [o for o in options if o]
    if not options:
        return None
    return {"instructions": m.group("instr").strip(), "options": options,
            "call_end": m.end()}


def parse_injected_result(text):
    """-> label string inside the first <|result|>...<|/result|> span, or None (call not resolved/injected
    yet, e.g. generation stopped right after emitting <|result|> and before injection)."""
    m = RESULT_SPAN_RE.search(text)
    return m.group("label").strip() if m else None


def extract_named_intent(text, names):
    """-> the banking77 `names` entry whose underscore-free phrase appears in `text` (case-insensitive),
    preferring the longest (most specific) match so e.g. 'card' never shadows 'card_not_working'. None if
    no name's phrase appears. Used for the free-draft baseline, which never emits a <|result|> span."""
    low = text.lower()
    best = None
    for name in sorted(names, key=lambda n: -len(n)):
        if label_phrase(name).lower() in low:
            best = name
            break
    return best


def template_match(continuation, label):
    """True if `continuation` follows build_phase7's fixed template AND names `label` (not some other
    label). A continuation that follows the template shape but for the wrong label is NOT a template
    match -- CHECK 7's bar is template-match AND label-correct, and error-propagation needs 'named the
    wrong thing' to count as corruption even when the shape is right.

    KEPT FOR COMPARISON ONLY -- this is the OLD, overly strict definition: it requires the FULL two-
    clause template ("...X. Let me help with that right away.") to appear anywhere in the whole
    continuation (not just the first sentence), and compares the named phrase to label_phrase(label) with
    exact (not normalized/prefix-tolerant) string equality -- so e.g. the model naming the label with
    literal underscores ('activate_my_card' instead of 'activate my card'), a hyphenated rendering
    ('verify top-up'), a truncated/extended phrase ('lost or stolen' for 'lost_or_stolen_card'), or a
    label ending in '?' (banking77's 'reverted_card_payment?') all count as NOT matching even though
    they're consistent with the injected label. See first_sentence_consistent_with_injected_label for the
    fixed replacement; see is_corrupted for why 'corrupted' == 'not template_match' by construction."""
    m = TEMPLATE_RE.search(continuation)
    if not m:
        return False
    return m.group("phrase").strip().lower() == label_phrase(label).lower()


def template_phrase(continuation):
    """-> the phrase TEMPLATE_RE captures if `continuation` follows build_phase7's fixed shape at all
    ("This looks like X. Let me help with that right away."), lowercased/stripped, regardless of whether
    X matches any particular label; None if the shape isn't present at all. Used to split corrupted rows
    (template_match False) into "off-template entirely" vs "on-template but naming something else" --
    see is_corrupted's docstring for why corrupted == not template_match by construction, and why that
    split needs this instead of just re-deriving corrupted again."""
    m = TEMPLATE_RE.search(continuation)
    return m.group("phrase").strip().lower() if m else None


def is_corrupted(continuation, injected_label):
    """Cheap, code-only 'corrupted' check for the error-propagation analysis: the continuation is
    corrupted if it does NOT faithfully follow the fixed template naming the label that was actually
    injected (contradicts it, names a different label, or ignores the call and drifts into free text).
    This is deliberately template_match's negation: the call-trained adapter is supposed to always
    produce the fixed template naming whatever label it was given, so any deviation from that, given the
    injected label, is the corruption signal -- independent of whether the injected label itself was
    correct (that's label-correctness, scored separately)."""
    return not template_match(continuation, injected_label)


def bootstrap_ci(values, n_boot=2000, seed=0, lo_pct=2.5, hi_pct=97.5):
    """Mean + percentile bootstrap CI over a list of 0/1 (or float) values. -> (mean, lo, hi). mean is the
    plain sample mean (not the bootstrap mean) so point estimates don't wobble with n_boot/seed; empty
    input -> (None, None, None) rather than dividing by zero."""
    n = len(values)
    if n == 0:
        return None, None, None
    rng = random.Random(seed)
    mean = sum(values) / n
    boots = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo_i = max(0, int(n_boot * lo_pct / 100))
    hi_i = min(n_boot - 1, int(n_boot * hi_pct / 100))
    return mean, boots[lo_i], boots[hi_i]


def paired_bootstrap_diff(a_values, b_values, n_boot=2000, seed=0, lo_pct=2.5, hi_pct=97.5):
    """Paired percentile bootstrap CI for mean(a_values) - mean(b_values), where index i in both lists is
    the SAME unit (e.g. the same held-out row id scored on two different paths) -- every bootstrap draw
    resamples row INDICES once and applies that same resample to both lists, preserving the pairing
    (unlike running bootstrap_ci on each list separately, which would treat them as independent samples
    and overstate the CI width for a paired difference). -> (mean_diff, lo, hi); mean_diff is the plain
    (non-bootstrap) difference of means. Mismatched lengths or empty input -> (None, None, None)."""
    n = len(a_values)
    if n == 0 or n != len(b_values):
        return None, None, None
    rng = random.Random(seed)
    mean_diff = sum(a_values) / n - sum(b_values) / n
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        a_mean = sum(a_values[k] for k in idx) / n
        b_mean = sum(b_values[k] for k in idx) / n
        boots.append(a_mean - b_mean)
    boots.sort()
    lo_i = max(0, int(n_boot * lo_pct / 100))
    hi_i = min(n_boot - 1, int(n_boot * hi_pct / 100))
    return mean_diff, boots[lo_i], boots[hi_i]


OPTIONS_ANSWER_RE = re.compile(r"\[\s*answer\s*\]\s*(.+)", re.IGNORECASE)


def build_options_baseline_prompt(state, names, instr):
    """Chat-with-options baseline prompt: the customer message, the full label list as bullet options
    (same INSTR/names/order as scripts/build_phase7.py's canonical call), and an explicit answer-format
    instruction. Wording mirrors scripts/oracle_labels.py's build_options_prompt/format_options (the
    established "[id] instructions\\nOptions:\\n- opt" shape used for the Phase 5/6 73-75%
    chat-with-options numbers) so this baseline's accuracy is comparable to those. Returned string is the
    chat template's user-turn content (run_options_baseline wraps it via
    tokenizer.apply_chat_template)."""
    opts = "\n".join(f"- {n}" for n in names)
    return (f"Customer: {state}\n\n{instr}\nOptions:\n{opts}\n\n"
            "Answer with exactly one line in the form `[answer] <option>`, using exactly one of the"
            " options above, written exactly as it appears in the list.")


def _normalize_label_token(s):
    s = s.strip().strip("`'\"")
    s = s.rstrip(".,;:!? \t")
    s = s.strip("[]")
    return s.strip()


def parse_options_answer(text, names):
    """-> (pred_label, parsed): parses one options-baseline generation against the exact `names` list.
    Tries, in order: (1) an explicit '[answer] <label>' marker anywhere in the text; (2) each non-empty
    line, normalized (whitespace/case folded, brackets/quotes/trailing punctuation stripped), as an exact
    match against a label; (3) the whole normalized text as one label; (4) a longest-label
    substring/prefix match anywhere in the text (covers e.g. 'card_not_working because...' or the
    underscore-free phrase form). pred_label is None and parsed is False only if none of these find a
    listed label -- callers must score that as wrong (unparseable), never skip it."""
    names_norm = {n.lower(): n for n in names}

    m = OPTIONS_ANSWER_RE.search(text)
    if m:
        cand = _normalize_label_token(m.group(1).split("\n")[0])
        if cand.lower() in names_norm:
            return names_norm[cand.lower()], True

    for line in text.splitlines():
        cand = _normalize_label_token(line)
        if cand.lower() in names_norm:
            return names_norm[cand.lower()], True

    cand = _normalize_label_token(text)
    if cand.lower() in names_norm:
        return names_norm[cand.lower()], True

    low = text.lower()
    for n in sorted(names, key=lambda x: -len(x)):
        if n.lower() in low or label_phrase(n).lower() in low:
            return n, True

    return None, False


def classify_miss_row(raw_text, triggered=None):
    """-> 'trigger-missed' | 'trigger-fired-call-collapsed' | 'unknown' for one call_found=False adapter
    row, from its raw pre-parse generation text (a --phase1-raw / dump_raw row, if one was supplied).
    `triggered`, when given (force-options dump_raw rows carry this directly), is used verbatim instead of
    re-deriving it from raw_text. 'unknown' means raw_text is None -- no --phase1-raw was given and the
    adapter-rows file alone carries no raw generation text for this row."""
    if triggered is not None:
        return "trigger-fired-call-collapsed" if triggered else "trigger-missed"
    if raw_text is None:
        return "unknown"
    return "trigger-fired-call-collapsed" if "<|decide|>" in raw_text else "trigger-missed"


FIRST_SENTENCE_SHAPE_RE = re.compile(r"^This looks like (?P<phrase>.+)\.$", re.IGNORECASE)
CONTROL_TAG_RE = re.compile(r"<\|/?[a-zA-Z0-9_\-]+\|>")
THIS_LOOKS_LIKE_RE = re.compile(r"this looks like", re.IGNORECASE)


def continuation_clean(text):
    """-> `text` with any leading call-marker fragment stripped off, so first_sentence_phrase/clean_stop
    never mistake a char-offset artifact for the model's actual first sentence. Root cause (CHECK 7,
    run1combined3): run_adapter_phase3's stored `continuation` was sliced out of the resumed generation
    by CHARACTER offset (`full[len(resume_prompt):]`), but `full` is a re-decode of re-tokenized prompt
    ids, which does not always decode back to exactly `resume_prompt` character-for-character -- so the
    slice boundary drifts and `continuation` frequently starts mid-call-text, e.g.
    'ult|> This looks like verify top-up. ...' or
    'ge_rate_for_cash_withdrawal]<|result|>activate_my_card<|/result|> This looks like activate_my_card...'
    instead of cleanly at ' This looks like ...' (see run_adapter_phase3's fix: decode the newly
    generated token ids directly instead of slicing a re-decoded string, so future runs never need this).

    Strategy (robust to the drifted boundary on ALREADY-stored rows, and a no-op on clean rows): find the
    FIRST occurrence of 'this looks like' (case-insensitive) within the first 200 chars and start there --
    whatever character-offset garbage precedes it (a stray '<|result|>label<|/result|>', a bare 'ult|>'
    fragment, or nothing at all) is discarded. If no such occurrence exists in that window, `text` is
    returned unchanged: it is either genuinely off-template (no 'This looks like' at all, e.g. the model
    drifted into free text) or names the label with the control tag itself rather than a 'This looks
    like' sentence -- both must stay flagged as off-template/inconsistent, not silently swallowed by
    stripping a prefix that was never a call-marker fragment to begin with."""
    m = THIS_LOOKS_LIKE_RE.search(text[:200])
    return text[m.start():] if m else text


def first_sentence(continuation):
    """-> the text of `continuation` up to and including the first '. ' (period+space) or, if a newline
    comes first, up to (not including) that newline; falls back to the whole stripped `continuation` if
    neither delimiter is present. This is deliberately NOT "the template's two clauses" (the fixed
    template itself reads as two grammatical sentences, "This looks like X." + "Let me help with that
    right away.") -- CHECK 7 only needs the FIRST of those two to check which label got named; whatever
    follows (the second template clause, or any drift/looping/markup after it) is checked separately by
    clean_stop, and training targets carried no end-of-text marker, so rambling after this point is
    expected, not an error signal on its own."""
    s = continuation.strip()
    dot_idx = s.find(". ")
    nl_idx = s.find("\n")
    candidates = [i for i in (dot_idx, nl_idx) if i != -1]
    if not candidates:
        return s
    idx = min(candidates)
    return s[:idx + 1] if idx == dot_idx else s[:idx]


def first_sentence_phrase(continuation):
    """-> the phrase named by `continuation`'s first_sentence, if that sentence follows the shape "This
    looks like X." (case-insensitive), stripped; None if the first sentence doesn't have that shape at
    all (off-template). Unlike TEMPLATE_RE/template_phrase (which require the FULL two-clause template,
    "...X. Let me help with that right away.", anywhere in the whole continuation), this only looks at
    the first sentence and only requires the first clause -- see first_sentence's docstring for why."""
    m = FIRST_SENTENCE_SHAPE_RE.match(first_sentence(continuation_clean(continuation)))
    return m.group("phrase").strip() if m else None


def _normalize_phrase(s):
    """-> `s` lowercased, with '_'/'-' folded to whitespace (so the model naming a label as its literal
    underscored form, e.g. 'activate_my_card', or a hyphenated training-template rendering, e.g.
    'verify top-up', is the same phrase as the space-joined form label_phrase produces), trailing
    punctuation (including '?' -- banking77 has a label literally named 'reverted_card_payment?')
    stripped, and internal whitespace collapsed."""
    s = re.sub(r"[-_]", " ", s.strip().lower())
    s = s.rstrip("?.!,;: \t")
    return re.sub(r"\s+", " ", s).strip()


def _phrase_words(s):
    return _normalize_phrase(s).split()


def _phrase_consistent(actual_phrase, expected_phrase):
    """-> True if `actual_phrase` (the span the model's first sentence actually named) and
    `expected_phrase` (label_phrase of the injected label) agree up to a word-level PREFIX in either
    direction: the model truncating before finishing the phrase (e.g. 'lost or stolen' for
    'lost_or_stolen_card') and the model appending trailing words after it (e.g. 'activate_my_card is
    what you want') are both still naming the same label; anything that diverges before the shorter of
    the two phrases ends is a genuine mismatch. Word-level (not substring) so e.g. expected 'card' never
    prefix-matches an unrelated actual 'card not working'."""
    a, e = _phrase_words(actual_phrase), _phrase_words(expected_phrase)
    if not a or not e:
        return False
    n = min(len(a), len(e))
    return a[:n] == e[:n]


def first_sentence_consistent_with_injected_label(continuation, injected_label):
    """-> the FIXED template_match: True if the first sentence after <|/result|> names the label that was
    actually injected (normalized/prefix-tolerant, see _phrase_consistent), False if it's off-template
    entirely or names something inconsistent with the injected label. This is CHECK 7's corrected bar
    (see template_match's docstring for the old, stricter definition kept for comparison)."""
    phrase = first_sentence_phrase(continuation)
    if phrase is None:
        return False
    return _phrase_consistent(phrase, label_phrase(injected_label))


def off_template_first_sentence(continuation):
    """-> True if the first sentence after <|/result|> doesn't even have the "This looks like X." shape
    (no label named at all, consistent or not)."""
    return first_sentence_phrase(continuation) is None


def first_sentence_names_other_label(continuation, injected_label, names):
    """-> True if the first sentence is on-template (names SOME label-shaped phrase) but that phrase is
    consistent with a DIFFERENT entry of `names` than `injected_label` -- the genuine 'ignores/overrides
    the injected label' failure mode, as opposed to off_template_first_sentence (no label named at all)
    or a phrase that matches neither the injected label nor any other known one (off-template-content,
    still counted under off_template/inconsistent but not "names another label" specifically)."""
    phrase = first_sentence_phrase(continuation)
    if phrase is None or _phrase_consistent(phrase, label_phrase(injected_label)):
        return False
    return any(n != injected_label and _phrase_consistent(phrase, label_phrase(n)) for n in names)


def _clean_rest(continuation):
    """-> continuation_clean(continuation) with its own first sentence stripped off the front -- the
    text has_stray_tags/has_loop inspect, so a leading call-marker fragment (continuation_clean's job)
    is never mistaken for a stray tag/loop before the model's actual first sentence even starts."""
    cleaned = continuation_clean(continuation)
    return cleaned[len(first_sentence(cleaned)):].strip()


def has_stray_tags(continuation):
    """-> True if a control/markup tag (<|...|> style, e.g. a leaked '<|/verify_top-up|>') appears
    anywhere after the first sentence (post continuation_clean cleanup). Reported separately from
    has_loop so the two failure modes -- leaked markup vs. a repeated-sentence loop -- can be told apart;
    clean_stop is their combined (either-fails) rate."""
    return bool(CONTROL_TAG_RE.search(_clean_rest(continuation)))


def has_loop(continuation):
    """-> True if the same sentence (case-folded) appears >=2 times anywhere in continuation_clean's
    output -- a generation loop. Reported separately from has_stray_tags (see its docstring)."""
    cleaned = continuation_clean(continuation)
    sentences = [s.strip().lower() for s in re.split(r"\.\s+|\n", cleaned) if len(s.strip()) >= 3]
    return len(sentences) != len(set(sentences))


def clean_stop(continuation):
    """-> True if `continuation` has neither a stray control tag (has_stray_tags) nor a repeated-sentence
    loop (has_loop) after the first sentence. NOTE: scripts/build_phase7.py's training targets never
    carried an end-of-text marker after the template continuation, so plain-text rambling after the first
    sentence is EXPECTED generation behavior, not a label-propagation error -- this flags only
    loops/stray markup, not rambling itself. See has_stray_tags/has_loop to see which one dominates in a
    given run."""
    return not has_stray_tags(continuation) and not has_loop(continuation)


def score_first_sentence(continuation, injected_label, names=None):
    """-> dict of the new, fixed CHECK 7 metrics for one scored adapter row, derived ONLY from
    `continuation`/`injected_label`/`names` (no stored booleans needed -- safe to recompute from an
    already-written oracle/phase7-eval-*-adapter.jsonl row's stored continuation/resolved_label/names at
    --report-only / --inspect time). `names` is the full banking77 label list; when None (e.g. an older
    row file written before this field existed), 'first_sentence_names_other_label' is omitted from the
    result entirely rather than guessed at -- callers must exclude it from any aggregate in that case."""
    out = {
        "first_sentence_consistent_with_injected_label":
            first_sentence_consistent_with_injected_label(continuation, injected_label),
        "off_template_first_sentence": off_template_first_sentence(continuation),
        "clean_stop": clean_stop(continuation),
        "stray_tags": has_stray_tags(continuation),
        "loop": has_loop(continuation),
    }
    if names is not None:
        out["first_sentence_names_other_label"] = first_sentence_names_other_label(
            continuation, injected_label, names)
    return out


def classify_corrupted_row(continuation, label_correct):
    """-> category string for one corrupted (not template_match) adapter row, splitting WHY it's
    corrupted: 'off-template' (the fixed template shape is absent from the continuation entirely), or
    'names-different-intent, injected label RIGHT'/'WRONG' (on-template but naming something other than
    the label that was actually injected -- split by whether the injected label itself was correct
    against gold, i.e. whether the continuation is overriding a bad label (arguably fine) or ignoring a
    good one (a real propagation failure))."""
    phrase = template_phrase(continuation)
    if phrase is None:
        return "off-template"
    return ("names-different-intent, injected label RIGHT" if label_correct
            else "names-different-intent, injected label WRONG")


def conditional_rate(event, condition, **boot_kw):
    """P(event | condition) + bootstrap CI over the subset where `condition` is True. event/condition are
    same-length lists of bool/0-1. -> (mean, lo, hi, n_condition); (None, None, None, 0) if no rows meet
    the condition (e.g. zero confident-wrong rows in a small sample -- expected at n~200)."""
    sub = [e for e, c in zip(event, condition) if c]
    mean, lo, hi = bootstrap_ci([float(x) for x in sub], **boot_kw)
    return mean, lo, hi, len(sub)
