"""Phase 7R arm-B (think-only) baseline eval, joined against the natural eval set.

Joins `oracle/t3-armB-think.fixed.jsonl` (arm B: `--thinking on`, no calls, organic thinking+answer --
see scripts/gen_think_traces.py) by "id" with `oracle/phase7r-natural-eval.jsonl` (the frozen natural
eval prompts/gold labels), restricted to `kind == "decision"` rows, and extracts the label B committed to
from B's own OUTPUT TEXT -- NOT from any structured field, since arm B never calls anything; it is asked
to just answer naturally and the label has to be read back out of prose.

Extraction method (heuristic, not exact -- see caveats in reports/19): for each row, search the VISIBLE
ANSWER (case-insensitively) for each candidate option's bare key plus its SYNONYMS (imported from
scripts/check_phase7r_pilot.py, the same table the Phase 7R pilot validator trusts for its own
label-leak check) plus a small set of natural-language fallbacks added here for types/options the
pilot's SYNONYMS table does not cover (kb_category's non-synonym-listed keys, answer_type's "number",
order_outcome's bare "cancelled"). If the answer is empty (channel never closed, see gen_think_traces.py
"finished"), the last 400 chars of "thinking" is used instead, on the theory that an unclosed/empty
answer still usually carries the model's eventual commitment near the end of its reasoning. A label
counts as EXTRACTED only if exactly one candidate option's surface set matches -- zero matches or two+
matches (tie) both count as "unextracted", never guessed.

Usage (laptop, no model, no GPU):
    python scripts/eval_phase7r_armB.py
    python scripts/eval_phase7r_armB.py --armb oracle/t3-armB-think.fixed.jsonl \\
        --natural oracle/phase7r-natural-eval.jsonl --out reports/19-phase7r-armB-baseline.md

--llm-extract FILE: use `scripts/extract_labels_llm.py`'s output (one row per id: "chosen",
"committed", "p_top", "margin") instead of the regex/substring heuristic above. "extraction rate"
becomes "committed rate" (chosen != "__none__"), "acc | extracted" becomes "acc | committed", and
"acc | all" is still accuracy over every row (uncommitted rows count as wrong). Requires a GPU run
of extract_labels_llm.py first; this flag itself is laptop-only (just joins/reports existing files).
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_phase7r_pilot import SYNONYMS  # noqa: E402

# head S1 reference: per-type dev accuracy of the shipped E4B decision checkpoint, argmax vs gold, clean
# variant only -- reports/16-phase7-reasoning-proposal.md lines ~27-34 ("Decision types and the
# competence cutoff" table). banking_intent (0.800) is omitted: it does not appear in the natural eval
# set's 5 types.
HEAD_S1_REFERENCE = {
    "claim_handling": 1.000,
    "order_outcome": 1.000,
    "answer_type": 0.963,
    "kb_category": 0.950,
    "news_topic": 0.812,
}

# Natural-language fallbacks beyond scripts/check_phase7r_pilot.py's SYNONYMS, for option keys that
# table does not cover (it was built for a different, smaller pilot option set) or where the task
# explicitly calls out bare phrasing the SYNONYMS table only has in a longer form.
EXTRA_SYNONYMS = {
    # order_outcome: SYNONYMS has "is cancelled"/"be cancelled" for far_over, not the bare word.
    "far_over": ["cancelled", "canceled"],
    # answer_type: natural eval's option set includes "number", absent from the pilot's trec entry.
    "number": ["a number", "a numeric answer", "a numerical answer", "a date", "a count", "a quantity"],
    # kb_category: natural eval's 14-way option set includes several keys the pilot's SYNONYMS table
    # (built for an 8-way subset) never needed.
    "educationalinstitution": ["educational institution", "a school", "a university", "a college"],
    "artist": ["an artist", "a musician", "a painter", "a performer"],
    "athlete": ["an athlete", "a sportsperson", "a sports player"],
    "building": ["a building", "a structure", "a landmark"],
    "plant": ["a plant", "a species of plant"],
    "film": ["a film", "a movie"],
}

THINKING_FALLBACK_CHARS = 400


def surface_set(key):
    """-> lowercased candidate substrings for one option key: bare forms (as-is, spaced, hyphenated)
    plus scripts/check_phase7r_pilot.py's SYNONYMS plus EXTRA_SYNONYMS above."""
    bare = {key, key.replace("_", " "), key.replace("_", "-")}
    extra = SYNONYMS.get(key, []) + EXTRA_SYNONYMS.get(key, [])
    return {s.lower() for s in (bare | set(extra)) if s}


def extract_label(text, option_keys):
    """-> (label or None). None iff zero or 2+ of option_keys' surface sets match `text`
    (case-insensitive substring search) -- ties and misses are both "unextracted", never guessed."""
    text_l = text.lower()
    matched = [k for k in option_keys if any(s in text_l for s in surface_set(k))]
    return matched[0] if len(matched) == 1 else None


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--armb", default="oracle/t3-armB-think.fixed.jsonl")
    ap.add_argument("--natural", default="oracle/phase7r-natural-eval.jsonl")
    ap.add_argument("--out", default="reports/19-phase7r-armB-baseline.md")
    ap.add_argument("--llm-extract", default=None,
                    help="path to scripts/extract_labels_llm.py output; use it instead of the "
                         "regex/substring heuristic (see module docstring)")
    a = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    armb_path = a.armb if os.path.isabs(a.armb) else os.path.join(root, a.armb)
    nat_path = a.natural if os.path.isabs(a.natural) else os.path.join(root, a.natural)
    out_path = a.out if os.path.isabs(a.out) else os.path.join(root, a.out)
    llm_path = None
    if a.llm_extract:
        llm_path = a.llm_extract if os.path.isabs(a.llm_extract) else os.path.join(root, a.llm_extract)

    armb_rows = {r["id"]: r for r in load_jsonl(armb_path)}
    nat_rows = load_jsonl(nat_path)
    decision_rows = [r for r in nat_rows if r["kind"] == "decision"]
    llm_rows = {r["id"]: r for r in load_jsonl(llm_path)} if llm_path else None

    missing = [r["id"] for r in decision_rows if r["id"] not in armb_rows]
    if missing:
        print(f"WARNING: {len(missing)} decision rows have no arm-B match: {missing[:5]}...")
    if llm_rows is not None:
        llm_missing = [r["id"] for r in decision_rows if r["id"] not in llm_rows]
        if llm_missing:
            print(f"WARNING: {len(llm_missing)} decision rows have no --llm-extract match: "
                  f"{llm_missing[:5]}...")

    by_type = collections.defaultdict(list)
    for nr in decision_rows:
        b = armb_rows.get(nr["id"])
        if b is None:
            continue
        option_keys = [k for k, _desc in nr["options_with_descriptions"]]
        gold = nr["gold_label"]
        answer = b.get("answer") or ""
        if llm_rows is not None:
            l = llm_rows.get(nr["id"])
            if l is None:
                continue
            extracted = l["chosen"] if l.get("committed") else None
            used_fallback = False
            search_text = ""
        else:
            used_fallback = False
            search_text = answer
            if not search_text.strip():
                search_text = (b.get("thinking") or "")[-THINKING_FALLBACK_CHARS:]
                used_fallback = True
            extracted = extract_label(search_text, option_keys)
        by_type[nr["type"]].append(dict(
            id=nr["id"], gold=gold, extracted=extracted, correct=(extracted == gold),
            n_thinking_tokens=b.get("n_thinking_tokens", 0), n_answer_tokens=b.get("n_answer_tokens", 0),
            finished=b.get("finished", False), used_fallback=used_fallback, answer=answer,
            thinking_tail=search_text if used_fallback else "",
        ))

    table_rows = []
    unextracted_examples = []
    for t in sorted(by_type):
        recs = by_type[t]
        n = len(recs)
        n_extracted = sum(1 for r in recs if r["extracted"] is not None)
        n_correct_among_extracted = sum(1 for r in recs if r["extracted"] is not None and r["correct"])
        n_correct_all = sum(1 for r in recs if r["correct"])
        extraction_rate = n_extracted / n if n else 0.0
        acc_among_extracted = n_correct_among_extracted / n_extracted if n_extracted else float("nan")
        acc_over_all = n_correct_all / n if n else 0.0
        mean_think = sum(r["n_thinking_tokens"] for r in recs) / n if n else 0.0
        mean_ans = sum(r["n_answer_tokens"] for r in recs) / n if n else 0.0
        unfinished_rate = sum(1 for r in recs if not r["finished"]) / n if n else 0.0
        head_ref = HEAD_S1_REFERENCE.get(t)
        meets_head = (acc_over_all >= head_ref) if head_ref is not None else None
        table_rows.append(dict(
            type=t, n=n, extraction_rate=extraction_rate, acc_among_extracted=acc_among_extracted,
            acc_over_all=acc_over_all, mean_think=mean_think, mean_ans=mean_ans,
            unfinished_rate=unfinished_rate, head_ref=head_ref, meets_head=meets_head,
        ))
        for r in recs:
            if r["extracted"] is None:
                unextracted_examples.append(dict(type=t, **r))

    # ---- console report ----
    extr_word = "commit" if llm_rows is not None else "extr"
    hdr = (f"{'type':16s} {'n':>4s} {extr_word+'%':>7s} {'acc|'+extr_word:>9s} {'acc|all':>8s} "
           f"{'think_tok':>9s} {'ans_tok':>8s} {'unfin%':>7s} {'head_ref':>8s} {'>=head':>7s}")
    print(hdr)
    print("-" * len(hdr))
    n_fail = 0
    for row in table_rows:
        meets_str = "" if row["meets_head"] is None else ("YES" if row["meets_head"] else "NO")
        if row["meets_head"] is False:
            n_fail += 1
        print(f"{row['type']:16s} {row['n']:4d} {row['extraction_rate']*100:6.1f}% "
              f"{row['acc_among_extracted']*100:8.1f}% {row['acc_over_all']*100:7.1f}% "
              f"{row['mean_think']:9.1f} {row['mean_ans']:8.1f} {row['unfinished_rate']*100:6.1f}% "
              f"{(row['head_ref']*100 if row['head_ref'] is not None else float('nan')):7.1f}% "
              f"{meets_str:>7s}")
    print(f"\ntypes below head S1 reference: {n_fail}/{len(table_rows)}")

    sample_label = "not-committed" if llm_rows is not None else "unextracted"
    print(f"\n5 {sample_label} examples:")
    sample = unextracted_examples[:5]
    for ex in sample:
        shown = ex["thinking_tail"] if ex["used_fallback"] else ex["answer"]
        label_src = "thinking[-400:]" if ex["used_fallback"] else "answer"
        print(f"  id={ex['id']} type={ex['type']} gold={ex['gold']!r} source={label_src}")
        safe = shown[:300].encode("ascii", "backslashreplace").decode("ascii")
        print(f"    {safe!r}")

    # ---- write report ----
    lines = [
        "# Report 19 -- Phase 7R arm-B (think-only) baseline eval\n",
        "Arm B = `oracle/t3-armB-think.fixed.jsonl` (`--thinking on`, no calls, organic generation; see "
        "scripts/gen_think_traces.py for the THINK_CLOSE-marker split-bug fix this file depends on). "
        "Joined by `id` with `oracle/phase7r-natural-eval.jsonl`, `kind == \"decision\"` rows only "
        f"(300 of 440 rows; {len(missing)} unmatched).\n",
        "## Extraction method and caveats\n",
    ]
    if llm_rows is not None:
        lines.append(
            f"Labels from `--llm-extract {a.llm_extract}` (`scripts/extract_labels_llm.py`): a forced-"
            "choice LLM reader (base google/gemma-4-E4B-it, no generation, single forward pass over "
            "lettered options incl. a 'none of these' choice) scores which option arm B's visible "
            "answer commits to. A row counts as committed iff the model's argmax letter is not the "
            "'none of these' choice; uncommitted rows count as wrong in \"acc | all\" (same convention "
            "as the regex method's unextracted rows).\n")
    else:
        lines.append(
            "The label B committed to is NOT structured output -- arm B never calls anything, so it has to be "
            "read back out of B's own prose. Method: search the visible ANSWER (case-insensitive substring) "
            "for each candidate option's bare key plus SYNONYMS (`scripts/check_phase7r_pilot.py`, the same "
            "table the Phase 7R pilot validator trusts) plus a small set of fallbacks added in "
            "`scripts/eval_phase7r_armB.py` for option keys that table doesn't cover (kb_category's "
            "non-pilot keys, answer_type's `number`, order_outcome's bare `cancelled`). If the answer is "
            "empty, the last 400 chars of `thinking` is used instead. A label counts as extracted only if "
            "EXACTLY ONE option's surface set matches -- ties and zero matches both count as unextracted "
            "(and as wrong, in the \"acc over all\" column). This is a substring heuristic, not a parser: it "
            "can false-positive when an option's surface text appears in a quoted excerpt of the input, or "
            "in a line rejecting that option by name, and it can false-negative when the model phrases its "
            "commitment in words not in SYNONYMS/EXTRA_SYNONYMS. Numbers below should be read as a baseline "
            "estimate, not a precise accuracy figure.\n")
    lines += [
        "## Results\n",
        f"| type | n | {extr_word} rate | acc \\| {extr_word} | acc \\| all | mean think tok | "
        "mean ans tok | unfinished rate | head S1 ref | >= head |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in table_rows:
        meets_str = "" if row["meets_head"] is None else ("YES" if row["meets_head"] else "**NO**")
        head_str = f"{row['head_ref']*100:.1f}%" if row["head_ref"] is not None else "n/a"
        lines.append(
            f"| {row['type']} | {row['n']} | {row['extraction_rate']*100:.1f}% | "
            f"{row['acc_among_extracted']*100:.1f}% | {row['acc_over_all']*100:.1f}% | "
            f"{row['mean_think']:.1f} | {row['mean_ans']:.1f} | {row['unfinished_rate']*100:.1f}% | "
            f"{head_str} | {meets_str} |"
        )
    lines.append(f"\n**{n_fail}/{len(table_rows)} types below head S1 reference** "
                 "(stop criterion 1 in reports/16 triggers at >=3 of the types tracked there; "
                 "banking_intent is not in this eval set, so at most 5 types are checkable here).\n")
    lines.append(f"## 5 {sample_label} examples (for manual inspection)\n")
    for ex in sample:
        shown = ex["thinking_tail"] if ex["used_fallback"] else ex["answer"]
        label_src = "thinking[-400:] (answer was empty)" if ex["used_fallback"] else "answer"
        lines.append(f"- `{ex['id']}` type=`{ex['type']}` gold=`{ex['gold']}` source={label_src}")
        lines.append(f"  > {shown[:300]!r}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
