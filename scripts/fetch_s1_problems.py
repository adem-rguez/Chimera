"""Stage S1 problem-set builder for the general decision-organ go/no-go (reports/23's successor --
S1 measures whether organic thinking traces of the BASE instruction model, google/gemma-4-E4B-it @
ee0ef6023621cff504d758262d4e04895a5af4a2, contain real "weigh alternatives -> commit" decision points
during ordinary reasoning -- math, multiple choice, and non-benchmark advice/conversation prompts).

This script does the DATA side only (no model, no torch) -- it downloads TRAIN/validation-side splits
of three public benchmarks via HF `datasets`, plus 50 hand-authored non-benchmark prompts, and writes
everything into ONE file in scripts/gen_think_traces.py's input format:
  {"id": str, "user": str, "kind": "problem", "source": str,
   "gold_answer": str | int | None, "answer_type": "numeric" | "choice" | "none",
   "split": "train" | "heldout"}
("split" and "source"/"gold_answer"/"answer_type" are all extra keys beyond gen_think_traces.py's
required id/user -- that script passes every extra input key through to its output verbatim, so this
is safe to feed straight into it with `--thinking on`.)

Sources (NEVER test splits -- go/no-go criteria must never be measured against a benchmark's held-out
test set):
  - GSM8K: `gsm8k` config `main`, split `train`. Gold is the integer after the final "#### " line
    (the dataset's own canonical answer format). answer_type="numeric".
  - MMLU: `cais/mmlu` config `all`. Tries split `auxiliary_train` first (HF's own non-test MMLU-style
    training pool, sourced from other MC datasets, NOT the 57-subject MMLU test set); falls back to
    split `validation` (also non-test) if `auxiliary_train` is unavailable on this dataset revision.
    Each row's `choices` (already a list, in order) are rendered as lettered options A, B, C, ...
    in the user turn; gold_answer is the letter at `choices[answer]`. answer_type="choice".
  - ARC-Challenge: `allenai/ai2_arc` config `ARC-Challenge`, split `train`. Each row's
    `choices["text"]` are RE-LETTERED A, B, C, ... in the rendered user turn (the dataset's own labels
    are sometimes "1".."4" instead of "A".."D", so a fixed A/B/C/... relabelling keeps the rendered
    prompt format identical across rows); gold_answer is the NEW letter at the position matching
    `answerKey` in the original `choices["label"]` list. answer_type="choice".
  - Advice/conversation: 50 hand-authored prompts (ADVICE_PROMPTS below), fully deterministic (same
    list every run, no sampling) -- no benchmark, no gold_answer (None), answer_type="none". These
    exist so S1 can also look at the model's organic thinking register on non-benchmark, open-ended
    requests (career/relationship/scheduling/writing/planning advice), not just graded problems.

Counts: GSM8K 200 + MMLU 200 + ARC-Challenge 150 + advice 50 = 600 "train"-split rows into
oracle/s1-problems.jsonl, per the spec this script was written against. A further HELDOUT_PER_SOURCE
(default 20) items per benchmark source (60 total; advice has no natural "more items" pool beyond the
fixed 50 and is not sampled for heldout) are drawn from the SAME downloaded pool, DISJOINT from the 600
(simple deterministic slice: train takes pool[:n], heldout takes the next HELDOUT_PER_SOURCE after it,
both from the same seeded shuffle, so they can never overlap) and written into the SAME
oracle/s1-problems.jsonl with `"split": "heldout"` -- a go/no-go metric must never be computed against
a prompt the model/head will be trained on. oracle/s1-heldout-ids.txt is a convenience sidecar: just
the `"id"` of every heldout row, one per line, so a downstream training script can exclude them by id
alone without re-deriving this split logic.

"Seeded, balanced": `--seed` (default 0) drives one `random.Random(seed).shuffle(...)` per source
BEFORE slicing train/heldout, so both subsets are a reproducible random sample of that source's pool
rather than e.g. the file's first N rows (which can be subject/skill-skewed in these datasets).
"Balanced" here means balanced ACROSS sources by construction (fixed per-source counts: 200/200/150/50
train, 20/20/20/0 heldout) -- this script does NOT additionally rebalance a single source's own label
distribution (e.g. MMLU subject mix, ARC answerKey letter distribution); each source's shuffled sample
is assumed close enough to that source's natural distribution for a go/no-go check, not a final eval.

Dataset-name/config defensiveness: each source's loader tries a short list of (repo, config, split)
alternatives in order (see *_ALTERNATIVES below) and raises a single clear SystemExit naming every
alternative tried and the last error, if all fail -- this script never silently substitutes a
different split or guesses.

Usage:
  Dry run (no network, no `datasets` import -- just prints the plan):
    .venv/Scripts/python.exe scripts/fetch_s1_problems.py --dry-run

  Real fetch (needs `datasets`; network required the first time per dataset, then HF cache):
    .venv/Scripts/python.exe scripts/fetch_s1_problems.py \\
        --output oracle/s1-problems.jsonl --heldout-ids oracle/s1-heldout-ids.txt --seed 0
"""
import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N_GSM8K_TRAIN = 200
N_MMLU_TRAIN = 200
N_ARC_TRAIN = 150
N_ADVICE = 50
HELDOUT_PER_SOURCE = 20  # GSM8K/MMLU/ARC only -- advice has no larger pool to draw a disjoint set from

GSM8K_ALTERNATIVES = [
    dict(repo="gsm8k", config="main", split="train"),
    dict(repo="openai/gsm8k", config="main", split="train"),
]
MMLU_ALTERNATIVES = [
    dict(repo="cais/mmlu", config="all", split="auxiliary_train"),
    dict(repo="cais/mmlu", config="all", split="validation"),
]
ARC_ALTERNATIVES = [
    dict(repo="allenai/ai2_arc", config="ARC-Challenge", split="train"),
    dict(repo="ai2_arc", config="ARC-Challenge", split="train"),
]

LETTERS = [chr(ord("A") + i) for i in range(26)]


# ---------------------------------------------------------------------------
# 50 hand-authored advice/conversation prompts -- deterministic, no gold.
# ---------------------------------------------------------------------------

ADVICE_PROMPTS = [
    "I got two job offers: one pays more but has a long commute, the other pays less but I could walk "
    "to work. How should I think about which one to take?",
    "My roommate keeps leaving dishes in the sink for days. I don't want to start a fight, but I'm "
    "getting resentful. What's a good way to bring this up?",
    "I'm trying to decide between renting a bigger apartment alone or staying in my current cheaper "
    "place with a roommate. What should I weigh here?",
    "A friend asked to borrow a significant amount of money. I want to help but I'm worried about being "
    "repaid. How do I handle this without damaging the friendship?",
    "I've been offered a promotion that means more responsibility and travel, but less time with my "
    "family. How should I approach this decision?",
    "I'm learning a new language and can't decide between focusing on one language deeply or two "
    "languages at a shallower level. Which approach makes more sense?",
    "My manager keeps assigning me tasks outside my job description. I like helping but I'm worried "
    "about scope creep. How should I bring this up?",
    "I want to start exercising regularly but I keep losing motivation after a week or two. What's a "
    "realistic way to build the habit?",
    "I'm choosing between two graduate programs: one is more prestigious but farther from family, the "
    "other is close to home but less well known in my field. How should I decide?",
    "A close friend is making a decision I think is a mistake, but it's their life. Should I say "
    "something, and if so, how?",
    "I have a deadline next week and realize I underestimated the work. Should I ask for an extension, "
    "work overtime, or scope down the deliverable?",
    "My partner and I disagree about how to split household chores. What's a fair way to work this out?",
    "I was offered a freelance project that pays well but would require turning down a smaller, "
    "more interesting one. How do I think about this tradeoff?",
    "I keep procrastinating on a big writing project. Should I set smaller daily goals, find an "
    "accountability partner, or change my environment?",
    "A coworker takes credit for ideas in meetings that came from our team discussions. How should I "
    "address this without escalating unnecessarily?",
    "I'm deciding whether to go back to school part-time while working, or wait until I've saved more "
    "money. What should I consider?",
    "My elderly parent needs more help than before, but I live far away. What are reasonable options "
    "to think through?",
    "I received critical feedback on a project I worked hard on. How should I respond constructively?",
    "I'm choosing between two apartments: one is cheaper but in a noisier area, the other is quieter "
    "but a bit of a stretch on rent. How should I decide?",
    "A friend consistently cancels plans last minute. Should I keep making plans with them, bring it "
    "up directly, or scale back the friendship?",
    "I want to switch careers but I'm worried about starting over at a lower level. How should I "
    "weigh this?",
    "My team is split on which approach to take for a project: a faster but riskier plan, or a slower "
    "but safer one. How should a team resolve this kind of disagreement?",
    "I'm trying to decide whether to confront a friend about a rumor I heard they started, or let it go.",
    "I have two mentors giving me conflicting advice about my career path. How do I decide whose advice "
    "to follow, or whether to blend both?",
    "I'm overwhelmed by too many commitments this semester. How should I decide what to drop?",
    "A neighbor's noise is affecting my sleep. Should I talk to them directly, involve the landlord, "
    "or try something else first?",
    "I'm deciding between taking a guaranteed smaller raise now or negotiating for a bigger one with "
    "some risk. What should I think about?",
    "My sibling and I disagree about how to care for a family heirloom after our parents pass it down. "
    "How could we approach this fairly?",
    "I want to give a friend honest feedback about a business idea I think won't work. How do I do "
    "that without sounding discouraging?",
    "I'm choosing between two internships: one at a big company with name recognition, one at a small "
    "startup with more hands-on experience. How should I weigh these?",
    "I feel burned out at work but I'm not sure if I need a vacation, a role change, or a new job "
    "entirely. How should I figure out which it is?",
    "A friend keeps asking for relationship advice but ignores whatever I suggest. Should I keep "
    "giving advice, change how I give it, or stop?",
    "I'm deciding whether to confront a store about a billing error or just let the small amount go.",
    "My partner wants to move to a new city for their career, but I'd have to leave my job and friends. "
    "How should we think through this together?",
    "I have a chance to take on a high-visibility project with a tight deadline, or a lower-visibility "
    "one with more flexibility. Which should I take and why?",
    "I'm trying to decide whether to tell my boss I'm job-searching before I have an offer, or wait "
    "until after.",
    "A group project teammate isn't contributing. Should I cover for them, talk to them directly, or "
    "escalate to the instructor?",
    "I want to improve my public speaking but I'm not sure whether to join a class, practice alone, or "
    "seek more speaking opportunities at work.",
    "I'm deciding between investing extra savings in index funds or paying down a low-interest student "
    "loan faster. How should I weigh this?",
    "My friend wants to start a business together but I'm nervous about mixing friendship and money. "
    "How should I think about whether to join them?",
    "I keep saying yes to social invitations I don't actually want, and it's draining me. How can I "
    "start saying no without feeling guilty?",
    "I'm unsure whether to bring up a sensitive family topic at an upcoming holiday gathering or avoid "
    "it to keep the peace.",
    "A close friend moved away and we've drifted. Should I make more effort to reconnect, or accept "
    "that the friendship has naturally run its course?",
    "I'm deciding whether to renovate my kitchen now, taking on debt, or wait and save first. What "
    "should factor into that decision?",
    "My child is struggling in one subject at school. Should I get a tutor, talk to the teacher first, "
    "or try to help them myself at home?",
    "I'm torn between pursuing a stable corporate job or a riskier but more meaningful nonprofit role. "
    "How should I think about this?",
    "A friend asked me to be a reference for a job but I have reservations about their fit for the "
    "role. Should I agree, decline, or give a qualified reference?",
    "I want to downsize my living space to save money but I'm attached to the extra room I use as a "
    "hobby studio. How should I decide?",
    "I'm deciding whether to adopt a pet given my busy schedule, or wait until my life is less hectic.",
    "My partner and I disagree about how much to spend on a wedding versus saving that money. How "
    "should a couple work through a disagreement like this?",
]


# ---------------------------------------------------------------------------
# pure helpers (no `datasets`/network needed -- safe to unit-test / use in --dry-run)
# ---------------------------------------------------------------------------

def parse_gsm8k_gold(answer_field):
    """-> the integer gold answer (as a string, to keep gold_answer a plain JSON scalar) parsed off
    GSM8K's own canonical "...#### 42" answer format. Strips thousands-separator commas."""
    marker = "#### "
    idx = answer_field.rfind(marker)
    if idx < 0:
        raise ValueError(f"no {marker!r} marker in GSM8K answer field: {answer_field!r}")
    return answer_field[idx + len(marker):].strip().replace(",", "")


def render_lettered_options(option_texts):
    """-> rendered "A. text\\nB. text\\n..." block, one line per option, using fixed A/B/C/... letters
    regardless of any labels the source dataset used."""
    return "\n".join(f"{LETTERS[i]}. {t}" for i, t in enumerate(option_texts))


def render_gsm8k_row(row, idx, split):
    question = row["question"].strip()
    gold = parse_gsm8k_gold(row["answer"])
    return dict(id=f"gsm8k-{idx:04d}", user=question, kind="problem", source="gsm8k",
                gold_answer=gold, answer_type="numeric", split=split)


def render_mmlu_row(row, idx, split):
    choices = row["choices"]
    question = row["question"].strip()
    user = f"{question}\n\n{render_lettered_options(choices)}"
    gold = LETTERS[row["answer"]]
    return dict(id=f"mmlu-{idx:04d}", user=user, kind="problem", source="mmlu",
                gold_answer=gold, answer_type="choice", split=split)


def render_arc_row(row, idx, split):
    labels = row["choices"]["label"]
    texts = row["choices"]["text"]
    question = row["question"].strip()
    user = f"{question}\n\n{render_lettered_options(texts)}"
    answer_key = row["answerKey"]
    pos = labels.index(answer_key)
    gold = LETTERS[pos]
    return dict(id=f"arc-{idx:04d}", user=user, kind="problem", source="arc_challenge",
                gold_answer=gold, answer_type="choice", split=split)


def render_advice_rows():
    assert len(ADVICE_PROMPTS) == N_ADVICE, f"expected {N_ADVICE} advice prompts, got {len(ADVICE_PROMPTS)}"
    return [
        dict(id=f"advice-{i:02d}", user=text, kind="problem", source="advice",
             gold_answer=None, answer_type="none", split="train")
        for i, text in enumerate(ADVICE_PROMPTS)
    ]


def split_pool(pool, n_train, n_heldout, seed):
    """-> (train_rows, heldout_rows), disjoint by construction: shuffles a COPY of `pool` with a fresh
    `random.Random(seed)`, then slices [0:n_train] for train and [n_train:n_train+n_heldout] for
    heldout. Raises if the pool is too small for both."""
    need = n_train + n_heldout
    if len(pool) < need:
        raise SystemExit(f"pool has {len(pool)} rows, need {need} (train={n_train} + heldout={n_heldout})")
    shuffled = list(pool)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:n_train], shuffled[n_train:need]


# ---------------------------------------------------------------------------
# dataset loading (needs `datasets` + network on first call; lazy import)
# ---------------------------------------------------------------------------

def _try_load(alternatives, label):
    from datasets import load_dataset
    errors = []
    for alt in alternatives:
        try:
            ds = load_dataset(alt["repo"], alt["config"], split=alt["split"])
            print(f"  {label}: loaded {alt['repo']!r} config={alt['config']!r} split={alt['split']!r} "
                  f"({len(ds)} rows)", flush=True)
            return ds, alt
        except Exception as e:  # noqa: BLE001 -- defensively try the next alternative
            errors.append(f"{alt}: {e!r}")
    raise SystemExit(
        f"{label}: all alternatives failed.\n" + "\n".join(f"  - {e}" for e in errors)
    )


def fetch_gsm8k(n_train, n_heldout, seed):
    ds, _alt = _try_load(GSM8K_ALTERNATIVES, "gsm8k")
    rows = list(ds)
    train_pool, heldout_pool = split_pool(rows, n_train, n_heldout, seed)
    train = [render_gsm8k_row(r, i, "train") for i, r in enumerate(train_pool)]
    heldout = [render_gsm8k_row(r, n_train + i, "heldout") for i, r in enumerate(heldout_pool)]
    return train, heldout


def fetch_mmlu(n_train, n_heldout, seed):
    ds, _alt = _try_load(MMLU_ALTERNATIVES, "mmlu")
    rows = list(ds)
    train_pool, heldout_pool = split_pool(rows, n_train, n_heldout, seed)
    train = [render_mmlu_row(r, i, "train") for i, r in enumerate(train_pool)]
    heldout = [render_mmlu_row(r, n_train + i, "heldout") for i, r in enumerate(heldout_pool)]
    return train, heldout


def fetch_arc(n_train, n_heldout, seed):
    ds, _alt = _try_load(ARC_ALTERNATIVES, "arc_challenge")
    rows = [r for r in ds if r.get("answerKey")]  # a handful of ARC rows have an empty answerKey
    train_pool, heldout_pool = split_pool(rows, n_train, n_heldout, seed)
    train = [render_arc_row(r, i, "train") for i, r in enumerate(train_pool)]
    heldout = [render_arc_row(r, n_train + i, "heldout") for i, r in enumerate(heldout_pool)]
    return train, heldout


# ---------------------------------------------------------------------------
# dry run (no `datasets` import, no network)
# ---------------------------------------------------------------------------

def run_dry(args):
    print("fetch_s1_problems --dry-run: no network, no `datasets` import -- plan only\n")
    plan = [
        ("gsm8k", GSM8K_ALTERNATIVES, N_GSM8K_TRAIN, HELDOUT_PER_SOURCE, "numeric"),
        ("mmlu", MMLU_ALTERNATIVES, N_MMLU_TRAIN, HELDOUT_PER_SOURCE, "choice"),
        ("arc_challenge", ARC_ALTERNATIVES, N_ARC_TRAIN, HELDOUT_PER_SOURCE, "choice"),
    ]
    total_train = total_heldout = 0
    for source, alts, n_train, n_heldout, answer_type in plan:
        print(f"source={source!r} answer_type={answer_type!r} n_train={n_train} n_heldout={n_heldout}")
        for alt in alts:
            print(f"    try: repo={alt['repo']!r} config={alt['config']!r} split={alt['split']!r} "
                  "(never 'test')")
        total_train += n_train
        total_heldout += n_heldout
    print(f"source='advice' answer_type='none' n_train={N_ADVICE} n_heldout=0 "
          f"(hand-authored, deterministic, no network)")
    total_train += N_ADVICE
    print(f"\nTOTAL: {total_train} train rows + {total_heldout} heldout rows "
          f"-> {args.output} (train+heldout together, split field distinguishes); "
          f"heldout ids also -> {args.heldout_ids}")
    assert total_train == N_GSM8K_TRAIN + N_MMLU_TRAIN + N_ARC_TRAIN + N_ADVICE == 600
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output", default="oracle/s1-problems.jsonl")
    ap.add_argument("--heldout-ids", default="oracle/s1-heldout-ids.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout-per-source", type=int, default=HELDOUT_PER_SOURCE)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the fetch plan only, no network, no `datasets` import")
    args = ap.parse_args()

    if args.dry_run:
        return sys.exit(run_dry(args))

    out_path = args.output if os.path.isabs(args.output) else os.path.join(ROOT, args.output)
    heldout_path = (args.heldout_ids if os.path.isabs(args.heldout_ids)
                     else os.path.join(ROOT, args.heldout_ids))

    print("fetching GSM8K ...", flush=True)
    gsm8k_train, gsm8k_heldout = fetch_gsm8k(N_GSM8K_TRAIN, args.heldout_per_source, args.seed)
    print("fetching MMLU ...", flush=True)
    mmlu_train, mmlu_heldout = fetch_mmlu(N_MMLU_TRAIN, args.heldout_per_source, args.seed)
    print("fetching ARC-Challenge ...", flush=True)
    arc_train, arc_heldout = fetch_arc(N_ARC_TRAIN, args.heldout_per_source, args.seed)
    advice_train = render_advice_rows()

    all_train = gsm8k_train + mmlu_train + arc_train + advice_train
    all_heldout = gsm8k_heldout + mmlu_heldout + arc_heldout
    all_rows = all_train + all_heldout

    with open(out_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")
    with open(heldout_path, "w", encoding="utf-8") as f:
        for row in all_heldout:
            f.write(row["id"] + "\n")

    print(f"wrote {len(all_rows)} rows ({len(all_train)} train, {len(all_heldout)} heldout) "
          f"-> {out_path}")
    print(f"wrote {len(all_heldout)} heldout ids -> {heldout_path}")
    by_source = {}
    for row in all_rows:
        key = (row["source"], row["split"])
        by_source[key] = by_source.get(key, 0) + 1
    for key in sorted(by_source):
        print(f"  {key[0]:15s} {key[1]:8s} {by_source[key]}")


if __name__ == "__main__":
    main()
