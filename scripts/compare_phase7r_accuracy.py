"""Phase 7R: paired final-answer accuracy of the runtime system (arm C) vs plain thinking (arm B),
on the same prompts, per decision type, using the forced-choice LLM extractor outputs.

Inputs (all jsonl):
  --armC-extract FILE   extractor run (scripts/extract_labels_llm.py) on arm C final answers.
                         Schema: {id, type, gold_label, chosen, p_top, margin, committed}
  --armB-extract FILE   extractor run on arm B (think-only) answers. Same schema.
  --armC-s3 FILE         arm C's own rtf-*-s3.jsonl (scripts/run_phase7r_runtime.py stage3 output).
                         Relevant fields (after normalize_s1_row): id, kind, type, gold_label,
                         trigger_type, valid, chosen_label, abstain, final_answer, thinking_so_far,
                         thinking_tail, n_decoded_tokens, n_injected_tokens, gold_type.
  --natural FILE         the eval file (oracle/phase7r-natural-eval.jsonl) -- source of truth for
                         gold_label/type/kind (NOT the extractor's own copies of these fields).
  --out FILE             optional: write the markdown report here (always also printed to stdout).

Restricted throughout to kind=="decision" rows (per --natural) present in BOTH --armC-extract and
--armB-extract (missing-row counts are reported, not silently dropped). --armC-s3 is joined in
separately for the arm-C-only breakdowns (3)/(4); rows missing there are also counted and excluded
from those specific breakdowns only.

"committed" is extract_labels_llm.py's own flag (chosen != "__none__"); an uncommitted/"__none__"
row is always scored WRONG for accuracy purposes, never excluded, unless computing the
"committed-only accuracy" variant.

Usage:
  .venv/Scripts/python.exe scripts/compare_phase7r_accuracy.py --selftest
  .venv/Scripts/python.exe scripts/compare_phase7r_accuracy.py \\
      --armC-extract oracle/armC-extract.jsonl --armB-extract oracle/t3-armB-extract.jsonl \\
      --armC-s3 oracle/rtf-run4-s3.jsonl --natural oracle/phase7r-natural-eval.jsonl \\
      --out reports/phase7r-armC-vs-armB-accuracy.md
"""
import argparse
import json
import os
import random
import sys
from math import comb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.run_phase7r_runtime import normalize_s1_row  # noqa: E402 -- reused, not re-derived

NONE_KEY = "__none__"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


# ---------------------------------------------------------------------------
# core join
# ---------------------------------------------------------------------------

def build_universe(natural_rows, armc_extract_rows, armb_extract_rows, s3_rows):
    """-> dict with joined lookups + diagnostic counts. Universe = kind=="decision" ids (per
    natural) present in BOTH extractor files. s3 is a separate, best-effort join (reported, not
    required) used only for the arm-C-only breakdowns."""
    natural = {r["id"]: r for r in natural_rows}
    armc = {r["id"]: r for r in armc_extract_rows}
    armb = {r["id"]: r for r in armb_extract_rows}
    s3 = {}
    for r in s3_rows:
        nr = normalize_s1_row(dict(r))
        s3[nr["id"]] = nr

    decision_ids = {i for i, r in natural.items() if r.get("kind") == "decision"}
    missing_armc = decision_ids - armc.keys()
    missing_armb = decision_ids - armb.keys()
    ids = sorted(decision_ids & armc.keys() & armb.keys())
    missing_s3 = set(ids) - s3.keys()

    return dict(natural=natural, armc=armc, armb=armb, s3=s3, ids=ids,
                n_decision_natural=len(decision_ids),
                missing_from_armc=sorted(missing_armc), missing_from_armb=sorted(missing_armb),
                missing_from_s3=sorted(missing_s3))


def is_correct(extract_row, gold_label):
    return bool(extract_row.get("committed")) and extract_row.get("chosen") == gold_label


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

def bootstrap_mean_ci(diffs, n_resamples=10000, seed=0):
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(diffs)
    mean = sum(diffs) / n
    if n == 1:
        return mean, mean, mean
    means = []
    for _ in range(n_resamples):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[min(n_resamples - 1, int(0.975 * n_resamples))]
    return mean, lo, hi


def mcnemar_exact(b, c):
    """-> (b, c, p). Exact two-sided McNemar on the discordant pairs, via the exact binomial test
    (X ~ Binomial(n=b+c, p=0.5), two-sided p = 2 * P(X <= min(b,c)), capped at 1)."""
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b, c, min(p, 1.0)


def rate(xs):
    xs = list(xs)
    return (sum(xs) / len(xs)) if xs else float("nan")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def accuracy_block(u, ids_by_type, ids_all):
    """-> list of markdown lines: per-type + overall accuracy/committed-accuracy/commit-rate for
    both arms, and paired diff with bootstrap CI + exact McNemar."""
    lines = ["| type | n | armC acc | armC committed-acc | armC commit rate | armB acc | "
             "armB committed-acc | armB commit rate | C-B diff | 95% CI | McNemar b,c | p |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def row_for(label, ids):
        armc_correct = [is_correct(u["armc"][i], u["natural"][i]["gold_label"]) for i in ids]
        armb_correct = [is_correct(u["armb"][i], u["natural"][i]["gold_label"]) for i in ids]
        armc_committed = [bool(u["armc"][i].get("committed")) for i in ids]
        armb_committed = [bool(u["armb"][i].get("committed")) for i in ids]
        armc_committed_acc = rate(c for c, cm in zip(armc_correct, armc_committed) if cm)
        armb_committed_acc = rate(c for c, cm in zip(armb_correct, armb_committed) if cm)
        diffs = [int(a) - int(b) for a, b in zip(armc_correct, armb_correct)]
        mean, lo, hi = bootstrap_mean_ci(diffs)
        b = sum(1 for a, bb in zip(armc_correct, armb_correct) if a and not bb)
        c = sum(1 for a, bb in zip(armc_correct, armb_correct) if not a and bb)
        _, _, p = mcnemar_exact(b, c)
        return (f"| {label} | {len(ids)} | {rate(armc_correct):.3f} | {armc_committed_acc:.3f} | "
                f"{rate(armc_committed):.3f} | {rate(armb_correct):.3f} | {armb_committed_acc:.3f} | "
                f"{rate(armb_committed):.3f} | {mean:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {b},{c} | "
                f"{p:.4f} |")

    for t in sorted(ids_by_type):
        lines.append(row_for(t, ids_by_type[t]))
    lines.append(row_for("**overall**", ids_all))
    return lines


def armc_breakdowns(u, ids_all):
    """-> list of markdown lines for breakdown (3)/(4): triggered vs non-triggered, head-correct vs
    head-wrong (agreement = extractor chosen == head chosen_label), abstained rows, and the
    would-be-accuracy-if-always-follow-head reference value. Rows missing from --armC-s3 are
    excluded here (already counted in u['missing_from_s3'])."""
    lines = []
    have_s3 = [i for i in ids_all if i in u["s3"]]
    triggered = [i for i in have_s3 if u["s3"][i].get("valid")]
    nontriggered = [i for i in have_s3 if not u["s3"][i].get("valid")]

    def acc(ids):
        return rate(is_correct(u["armc"][i], u["natural"][i]["gold_label"]) for i in ids)

    lines.append("| group | n | armC accuracy |")
    lines.append("|---|---|---|")
    lines.append(f"| triggered (valid typed call) | {len(triggered)} | {acc(triggered):.3f} |")
    lines.append(f"| non-triggered | {len(nontriggered)} | {acc(nontriggered):.3f} |")
    lines.append("")

    nonabstain = [i for i in triggered if u["s3"][i].get("abstain") is False]
    abstained = [i for i in triggered if u["s3"][i].get("abstain") is True]
    head_correct = [i for i in nonabstain
                    if u["s3"][i].get("chosen_label") == u["natural"][i]["gold_label"]]
    head_wrong = [i for i in nonabstain if i not in set(head_correct)]

    def agree_rate(ids):
        return rate(u["armc"][i].get("chosen") == u["s3"][i].get("chosen_label") for i in ids)

    lines.append("| group (triggered, non-abstain) | n | final answer agrees w/ head label | "
                 "armC accuracy |")
    lines.append("|---|---|---|---|")
    lines.append(f"| head-correct (head chosen_label == gold) | {len(head_correct)} | "
                 f"{agree_rate(head_correct):.3f} | {acc(head_correct):.3f} |")
    lines.append(f"| head-wrong | {len(head_wrong)} | {agree_rate(head_wrong):.3f} | "
                 f"{acc(head_wrong):.3f} |")
    lines.append(f"| abstained | {len(abstained)} | n/a (no injected label) | {acc(abstained):.3f} |")
    lines.append("")

    would_be_nonabstain = (len(head_correct) / len(nonabstain)) if nonabstain else float("nan")
    would_be_all_triggered = (len(head_correct) / len(triggered)) if triggered else float("nan")
    lines.append(f"- would-be accuracy if the answer always followed the head (head_correct rate), "
                 f"non-abstain triggered only: {would_be_nonabstain:.3f} ({len(head_correct)}/"
                 f"{len(nonabstain)})")
    lines.append(f"- same, over ALL triggered rows (abstain counted as not-head-correct): "
                 f"{would_be_all_triggered:.3f} ({len(head_correct)}/{len(triggered)})")
    return lines


def top_disagreements(u, ids_all, n=10):
    """-> list of (id, type, gold, armC chosen/committed/p_top, armB chosen/committed/p_top) for
    the n discordant (armC correct != armB correct) ids with the highest combined p_top (i.e. the
    pair the extractor itself was most confident in, on each side)."""
    disc = []
    for i in ids_all:
        gold = u["natural"][i]["gold_label"]
        ac, bc = is_correct(u["armc"][i], gold), is_correct(u["armb"][i], gold)
        if ac != bc:
            score = u["armc"][i].get("p_top", 0.0) + u["armb"][i].get("p_top", 0.0)
            disc.append((score, i))
    disc.sort(reverse=True)
    out = []
    for _score, i in disc[:n]:
        gold = u["natural"][i]["gold_label"]
        out.append((i, u["natural"][i]["type"], gold, u["armc"][i], u["armb"][i]))
    return out, len(disc)


def build_report(u):
    ids_by_type = {}
    for i in u["ids"]:
        ids_by_type.setdefault(u["natural"][i]["type"], []).append(i)

    lines = ["# Phase 7R: arm C (runtime) vs arm B (plain thinking) paired accuracy", ""]
    lines.append(f"decision rows (per --natural): {u['n_decision_natural']}; "
                 f"missing from --armC-extract: {len(u['missing_from_armc'])}; "
                 f"missing from --armB-extract: {len(u['missing_from_armb'])}; "
                 f"joined universe: {len(u['ids'])}; "
                 f"missing from --armC-s3 (excluded from arm-C-only breakdowns): "
                 f"{len(u['missing_from_s3'])}")
    lines.append("")

    lines.append("## (1)+(2) Accuracy per type, overall, and paired C-B difference")
    lines.append("")
    lines.append("Uncommitted (\"__none__\") rows always count as WRONG for 'acc'; "
                 "'committed-acc' restricts to committed rows only; '95% CI' is a 10000-resample "
                 "bootstrap on the paired per-row (armC correct - armB correct) difference; "
                 "McNemar b,c are the discordant counts (b = C right/B wrong, c = C wrong/B right), "
                 "p is the exact two-sided McNemar p-value.")
    lines.append("")
    lines += accuracy_block(u, ids_by_type, u["ids"])
    lines.append("")

    lines.append("## (3)+(4) Arm C breakdowns: triggered/non-triggered, head-correct/head-wrong, "
                 "abstained, would-be-accuracy-if-always-follow-head")
    lines.append("")
    lines += armc_breakdowns(u, u["ids"])
    lines.append("")

    lines.append("## (5) 10 largest disagreements (by combined extractor confidence)")
    lines.append("")
    top, n_disc = top_disagreements(u, u["ids"])
    lines.append(f"{n_disc} discordant ids total (armC correct != armB correct); showing top "
                 f"{len(top)} by p_top(C)+p_top(B)")
    lines.append("")
    lines.append("| id | type | gold | armC chosen | armC committed | armC p_top | armB chosen | "
                 "armB committed | armB p_top |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for i, t, gold, ac, bc in top:
        lines.append(f"| {i} | {t} | {gold} | {ac.get('chosen')} | {ac.get('committed')} | "
                     f"{ac.get('p_top', float('nan')):.3f} | {bc.get('chosen')} | "
                     f"{bc.get('committed')} | {bc.get('p_top', float('nan')):.3f} |")
    lines.append("")

    if u["missing_from_armc"] or u["missing_from_armb"]:
        lines.append("## Missing-row ids")
        lines.append("")
        if u["missing_from_armc"]:
            lines.append(f"missing from --armC-extract ({len(u['missing_from_armc'])}): "
                         f"{', '.join(u['missing_from_armc'][:30])}"
                         f"{' ...' if len(u['missing_from_armc']) > 30 else ''}")
        if u["missing_from_armb"]:
            lines.append(f"missing from --armB-extract ({len(u['missing_from_armb'])}): "
                         f"{', '.join(u['missing_from_armb'][:30])}"
                         f"{' ...' if len(u['missing_from_armb']) > 30 else ''}")
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run(args):
    natural_rows = load_jsonl(resolve(args.natural))
    armc_rows = load_jsonl(resolve(args.armc_extract))
    armb_rows = load_jsonl(resolve(args.armb_extract))
    s3_rows = load_jsonl(resolve(args.armc_s3))
    u = build_universe(natural_rows, armc_rows, armb_rows, s3_rows)
    report = build_report(u)
    print(report)
    if args.out:
        out_path = resolve(args.out)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"wrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
# selftest -- small FABRICATED inputs, no real model/data involved
# ---------------------------------------------------------------------------

def _fab_inputs():
    """6 fabricated decision rows, 2 per real SHIPPED_TYPE ('claim_handling', 'news_topic',
    'kb_category' -- normalize_s1_row requires a real type, since it looks up canonical options
    from the real oracle/phase7r-natural-eval.jsonl), covering: armC right/armB wrong, armC wrong/
    armB right, both right, both wrong/uncommitted, an abstain row, and a row missing from
    armC-s3 (to exercise the missing-join counting)."""
    natural = [
        dict(id="d1", kind="decision", type="claim_handling", gold_label="auto_approved"),
        dict(id="d2", kind="decision", type="claim_handling", gold_label="director_signoff"),
        dict(id="d3", kind="decision", type="news_topic", gold_label="world"),
        dict(id="d4", kind="decision", type="news_topic", gold_label="sports"),
        dict(id="d5", kind="decision", type="kb_category", gold_label="company"),
        dict(id="d6", kind="decision", type="kb_category", gold_label="artist"),
        dict(id="ctrl1", kind="control", type="claim_handling", gold_label="auto_approved"),  # excluded
    ]
    armc = [
        dict(id="d1", type="claim_handling", gold_label="auto_approved", chosen="auto_approved",
             p_top=0.95, margin=0.9, committed=True),
        dict(id="d2", type="claim_handling", gold_label="director_signoff", chosen="rejected",
             p_top=0.60, margin=0.2, committed=True),
        dict(id="d3", type="news_topic", gold_label="world", chosen="world", p_top=0.80, margin=0.5,
             committed=True),
        dict(id="d4", type="news_topic", gold_label="sports", chosen=NONE_KEY, p_top=0.55, margin=0.1,
             committed=False),
        dict(id="d5", type="kb_category", gold_label="company", chosen="company", p_top=0.70,
             margin=0.3, committed=True),
        # d6 missing from armC-extract on purpose
    ]
    armb = [
        dict(id="d1", type="claim_handling", gold_label="auto_approved", chosen="director_signoff",
             p_top=0.55, margin=0.1, committed=True),
        dict(id="d2", type="claim_handling", gold_label="director_signoff", chosen="director_signoff",
             p_top=0.90, margin=0.8, committed=True),
        dict(id="d3", type="news_topic", gold_label="world", chosen="world", p_top=0.65, margin=0.3,
             committed=True),
        dict(id="d4", type="news_topic", gold_label="sports", chosen="sports", p_top=0.60, margin=0.2,
             committed=True),
        dict(id="d5", type="kb_category", gold_label="company", chosen=NONE_KEY, p_top=0.50,
             margin=0.0, committed=False),
        dict(id="d6", type="kb_category", gold_label="artist", chosen="artist", p_top=0.85,
             margin=0.7, committed=True),
    ]
    # s3: d1 triggered+head-correct, d2 triggered+head-wrong, d3 triggered+abstain, d4 missing from s3,
    # d5 non-triggered (valid=False)
    s3 = [
        dict(id="d1", kind="decision", type="claim_handling", gold_label="auto_approved",
             trigger_type="claim_handling", valid=True, chosen_label="auto_approved", abstain=False,
             final_answer="auto_approved", thinking_so_far="t", thinking_tail="",
             n_decoded_tokens=10, n_injected_tokens=5),
        dict(id="d2", kind="decision", type="claim_handling", gold_label="director_signoff",
             trigger_type="claim_handling", valid=True, chosen_label="rejected", abstain=False,
             final_answer="rejected", thinking_so_far="t", thinking_tail="",
             n_decoded_tokens=10, n_injected_tokens=5),
        dict(id="d3", kind="decision", type="news_topic", gold_label="world",
             trigger_type="news_topic", valid=True, chosen_label="sports", abstain=True,
             final_answer="world", thinking_so_far="t", thinking_tail="",
             n_decoded_tokens=10, n_injected_tokens=0),
        dict(id="d5", kind="decision", type="kb_category", gold_label="company", trigger_type=None,
             valid=False, chosen_label=None, abstain=None, final_answer="company",
             thinking_so_far="t", thinking_tail="", n_decoded_tokens=10, n_injected_tokens=0),
    ]
    return natural, armc, armb, s3


def selftest():
    natural, armc, armb, s3 = _fab_inputs()
    u = build_universe(natural, armc, armb, s3)
    assert u["n_decision_natural"] == 6, u["n_decision_natural"]
    assert u["missing_from_armc"] == ["d6"], u["missing_from_armc"]
    assert u["missing_from_armb"] == [], u["missing_from_armb"]
    assert u["ids"] == ["d1", "d2", "d3", "d4", "d5"], u["ids"]
    assert u["missing_from_s3"] == ["d4"], u["missing_from_s3"]

    # spot-check correctness
    assert is_correct(u["armc"]["d1"], "auto_approved") is True
    assert is_correct(u["armc"]["d2"], "director_signoff") is False  # chosen rejected != gold
    assert is_correct(u["armc"]["d4"], "sports") is False  # uncommitted -> wrong

    report = "SELFTEST (fabricated data, not real model output)\n\n" + build_report(u)
    print(report)
    print("selftest: build_universe/is_correct/report generation OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--armC-extract", dest="armc_extract")
    ap.add_argument("--armB-extract", dest="armb_extract")
    ap.add_argument("--armC-s3", dest="armc_s3")
    ap.add_argument("--natural", default="oracle/phase7r-natural-eval.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    missing = [name for name, val in
               (("--armC-extract", args.armc_extract), ("--armB-extract", args.armb_extract),
                ("--armC-s3", args.armc_s3))
               if not val]
    if missing:
        ap.error(f"required unless --selftest: {', '.join(missing)}")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
