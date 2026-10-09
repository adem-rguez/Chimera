"""Compares OUR trained Gemma decision head (runs/p4-e4b-final, scored the way
scripts/s1_head_and_outcomes.py / scripts/probe_phase7r_head.py:make_real_scorer do) against Kev-4B
original (jaredpalmer/kev-4b, scored the way scripts/s1_kev_score.py does) on our head's OWN
in-domain HELD-OUT TEST set: evals/v7/decision-v7/test.jsonl -- the suite's locked test partition
(1176 records / 1440 questions, never used for training or architecture search; sha256-verified
against evals/v7/decision-v7/manifest.json, kev/suite.py:load_split's own check, reimplemented here
with hashlib/json only -- importing kev.suite pulls in kev.model/kev.data -> torch/datasets at module
scope, which --build and --selftest below must NOT need). The matching out-of-domain partition the
same suite ships, evals/v4/transfer-v4/test.jsonl (764 records / 764 questions, MMLU/emotion/sciq/etc,
no training overlap -- reports/08-e4b-validation.md's "transfer-v4"), is scored and reported
separately, never pooled into the in-domain numbers.

Both locked files already exist on this laptop (`evals/v7/decision-v7/test.jsonl`,
`evals/v4/transfer-v4/test.jsonl`) -- no fetch needed. If they are ever missing on a box, the
one-line fetch is `kev.suite.load_split("evals/v7/decision-v7", "test", allow_test=True)` (and the
`evals/v4/transfer-v4` equivalent), which pulls the missing partition from the suite's pinned HF
mirror and re-checks its sha256 -- see kev/suite.py:fetch_partition.

A labelled suite record ({"state": ..., "questions": {qid: {"type", "instructions", "criteria",
"label", "src"}}, "_meta": {...}}) is turned into one scorable point per question via kev/api.py's
own to_record (the exact rule kev/data.py:materialize uses to attach an int gold index -- choice:
the criteria key's position; noul/score: int(label) directly, since to_record already builds their
options in keys order) -- reimplemented against kev.api directly (pydantic-only, no torch) so this
script never imports kev.data (pulls in `datasets`) or kev.suite (pulls in kev.model -> torch).

Three subcommands:
  build  -- CPU, no model. Loads + sha256-verifies both locked test.jsonl files, builds one point per
            (record, question), draws a fixed-seed sample (--cap each, default 600) stratified by each
            question's own `src` (round-robin across sources so a cap below the source count still
            touches every source once first), writes the points file.
  score  -- GPU box, ONE chooser at a time (--chooser head|kev). Batch size 1 throughout: the head
            scorer (make_real_scorer) and the kev scorer (make_kev_scorer) both encode one record per
            forward pass already (scripts/probe_phase7r_head.py, scripts/s1_kev_score.py); Kev-4B's
            Qwen3.5 backbone is "hybrid" (Gated DeltaNet layers, kev/model.py:is_hybrid) so batching
            would only pipeline independent passes, not help memory pressure (scripts/s1_kev_score.py's
            own docstring). Writes one row per point, optionally also scoring a seeded shuffle of the
            point's own options (--shuffle-check) for the shuffle-consistency metric.
  report -- CPU, no model. Joins a --head-scores file and a --kev-scores file against the points file
            and renders oracle/head-vs-kev-indomain.md + .jsonl: accuracy (overall, by domain, by
            source), head/kev agreement, position bias (chosen vs gold option index), shuffle
            consistency, and accuracy at 100/80/60% coverage by confidence.

--selftest exercises the key-remap (seeded_shuffle_with_remap), gold-index (build_point), sampling
(stratified_sample) and metrics (report's compute_* functions) logic on fabricated fixtures -- no
model, no real suite files, no network.

Usage (laptop, build the points file -- run this first):
  .venv/Scripts/python.exe scripts/head_vs_kev_indomain.py build \\
      --out oracle/head-vs-kev-indomain-points.jsonl

Usage (GPU box -- see module docstring's own runtime note at the bottom of this file for timing):
  nohup .venv/bin/python -u scripts/head_vs_kev_indomain.py score \\
      --points oracle/head-vs-kev-indomain-points.jsonl --chooser head \\
      --decision-run runs/p4-e4b-final --device cuda --shuffle-check --resume \\
      --out oracle/head-vs-kev-indomain-headscores.jsonl \\
      > ~/hk.log 2>&1 && \\
  .venv/bin/python -u scripts/head_vs_kev_indomain.py score \\
      --points oracle/head-vs-kev-indomain-points.jsonl --chooser kev \\
      --run jaredpalmer/kev-4b --device cuda --shuffle-check --resume \\
      --out oracle/head-vs-kev-indomain-kevscores.jsonl \\
      >> ~/hk.log 2>&1 && touch ~/hk.done &

Usage (laptop, after pulling both score files down):
  .venv/Scripts/python.exe scripts/head_vs_kev_indomain.py report \\
      --points oracle/head-vs-kev-indomain-points.jsonl \\
      --head-scores oracle/head-vs-kev-indomain-headscores.jsonl \\
      --kev-scores oracle/head-vs-kev-indomain-kevscores.jsonl \\
      --out-md oracle/head-vs-kev-indomain.md --out-jsonl oracle/head-vs-kev-indomain.jsonl

Expected runtime/memory on the box: ~600 in-domain + up to 600 OOD points per chooser (<=1200 forward
passes). The head (bf16 E4B, no LoRA loaded -- reports/08-e4b-validation.md's ~7GB peak) and Kev-4B
(bf16 LoRA over Qwen3.5-4B-Base, batch size 1) each run in well under an hour on an L4/H100/H200,
sequentially (one model loaded at a time, ~9-10GB peak each) -- run head's `score` call, then kev's,
before `report`.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_INDOMAIN_DIR = "evals/v7/decision-v7"
DEFAULT_OOD_DIR = "evals/v4/transfer-v4"
DEFAULT_CAP = 600


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_jsonl(path):
    with open(resolve(path), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    with open(resolve(path), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# build: locked-suite loading (sha256-verified, stdlib only, no torch/datasets)
# ---------------------------------------------------------------------------

def load_locked_split(directory, split="test"):
    """-> list[dict]. Reads <directory>/manifest.json + <directory>/<split>.jsonl and checks the
    jsonl's sha256 + record count against the manifest -- kev/suite.py:load_split's own check,
    reimplemented with hashlib/json only so `build`/`--selftest` never import kev.suite (pulls in
    kev.model/kev.data -> torch/datasets at module scope)."""
    directory = resolve(directory)
    with open(os.path.join(directory, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    filename = f"{split}.jsonl"
    path = os.path.join(directory, filename)
    with open(path, "rb") as f:
        raw = f.read()
    entry = manifest["files"][filename]
    sha = hashlib.sha256(raw).hexdigest()
    if sha != entry["sha256"]:
        raise ValueError(f"{path}: sha256 mismatch ({sha} != {entry['sha256']}) -- suite file changed or corrupt")
    records = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    if len(records) != entry["records"]:
        raise ValueError(f"{path}: record count mismatch ({len(records)} != {entry['records']})")
    return records


# ---------------------------------------------------------------------------
# build: labelled record+question -> scorable point (kev/api.py's to_record, no kev.data/kev.suite)
# ---------------------------------------------------------------------------

def build_point(record, qid):
    """-> dict(state, instr, options, gold_index, gold_label, qtype, source) for ONE question of a
    labelled suite record. Uses kev/api.py's to_record directly (pydantic-only, no torch) and
    kev/data.py:materialize's exact rule for the gold index: choice -> the criteria key's position in
    `options`/`keys` (same order, both built by iterating the same criteria dict); noul/score ->
    int(label) directly, since to_record already puts their options in keys order (noul: [no, yes];
    score: level 0, 1, ...)."""
    from kev.api import SystemOneRequest, to_record

    q = record["questions"][qid]
    req = {"state": record["state"],
           "questions": {qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}}}
    rec, meta = to_record(SystemOneRequest.model_validate(req))
    m = meta[0]
    rq = rec["questions"][0]
    gold_label = q["label"]
    gold_index = m["keys"].index(gold_label) if m["type"] == "choice" else int(gold_label)
    return dict(state=rec["state"], instr=rq["instr"], options=rq["options"], gold_index=gold_index,
                gold_label=gold_label, qtype=m["type"], source=q.get("src", "unknown"))


def iter_points(records):
    """-> [(record_id, qid, record), ...] -- every (record, question) pair in `records`, in file order."""
    out = []
    for r in records:
        for qid in r["questions"]:
            out.append((r["_meta"]["id"], qid, r))
    return out


def point_source(point_tuple):
    rid, qid, record = point_tuple
    return record["questions"][qid].get("src", "unknown")


def stratified_sample(points, cap, seed, group_key=point_source):
    """-> up to `cap` of `points` (fixed-seed), covering every group `group_key` names: a seeded
    shuffle within each group, then round-robin across groups in deterministic (sorted) group order,
    one pick per group per round -- so a cap smaller than the number of groups still touches every
    group once before any group gets a second pick."""
    groups = collections.defaultdict(list)
    for p in points:
        groups[group_key(p)].append(p)
    rng = random.Random(seed)
    for g in groups.values():
        rng.shuffle(g)
    order = sorted(groups)
    out = []
    i = 0
    while len(out) < cap and any(groups[g] for g in order):
        g = order[i % len(order)]
        if groups[g]:
            out.append(groups[g].pop())
        i += 1
    return out[:cap]


def run_build(args):
    indomain = iter_points(load_locked_split(args.indomain_dir, "test"))
    ood = iter_points(load_locked_split(args.ood_dir, "test"))
    sample_in = stratified_sample(indomain, args.cap, args.seed)
    sample_ood = stratified_sample(ood, args.cap, args.seed)
    rows = []
    for domain, sample in (("indomain", sample_in), ("ood", sample_ood)):
        for rid, qid, record in sample:
            point = build_point(record, qid)
            rows.append(dict(domain=domain, id=rid, qid=qid, n_options=len(point["options"]), **point))
    write_jsonl(args.out, rows)
    print(f"wrote {len(sample_in)} in-domain + {len(sample_ood)} OOD points -> {args.out}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# score: key-remap (seeded shuffle) + GPU scoring loop
# ---------------------------------------------------------------------------

def seeded_shuffle_with_remap(options, seed):
    """-> (shuffled_options, remap) where shuffled_options[i] == options[remap[i]]. Re-rolls (like
    scripts/probe_phase7r_head.py:seeded_shuffle) if it happens to reproduce the identity order, which
    would tell shuffle-consistency nothing about order sensitivity."""
    idx = list(range(len(options)))
    shuffled_idx = list(idx)
    rng = random.Random(seed)
    rng.shuffle(shuffled_idx)
    tries = 0
    while shuffled_idx == idx and len(idx) > 1 and tries < 10:
        seed += 1
        rng = random.Random(seed)
        shuffled_idx = list(idx)
        rng.shuffle(shuffled_idx)
        tries += 1
    return [options[i] for i in shuffled_idx], shuffled_idx


def make_chooser_scorer(args):
    if args.chooser == "head":
        from scripts.probe_phase7r_head import make_real_scorer
        return make_real_scorer(args.decision_run, args.device)
    from scripts.s1_kev_score import load_kev_model, make_kev_scorer
    tok, model = load_kev_model(args.run, args.device)
    return make_kev_scorer(tok, model)


def score_point(scorer, point, shuffle_seed=None):
    """-> dict(chosen_index, confidence, max_prob, correct, [shuffle_chosen_index, shuffle_agree]).
    `scorer(state, instr, options)` -> (choice_text, confidence, max_prob, probs_by_option), the shared
    convention of scripts/probe_phase7r_head.py:make_real_scorer and scripts/s1_kev_score.py's own
    scorer -- `choice_text` is always one of `options`, so its position IS the chosen index (robust
    across choice/noul/score, whose option text does not always echo the suite's own key -- e.g. noul's
    options are "no"/"yes" text while kev/api.py:question_keys reports ["false","true"])."""
    options = point["options"]
    choice_text, confidence, max_prob, _probs_by_option = scorer(point["state"], point["instr"], options)
    chosen_index = options.index(choice_text)
    out = dict(chosen_index=chosen_index, confidence=confidence, max_prob=max_prob,
               correct=(chosen_index == point["gold_index"]))
    if shuffle_seed is not None and len(options) > 1:
        shuffled, remap = seeded_shuffle_with_remap(options, shuffle_seed)
        s_choice_text, _c, _m, _p = scorer(point["state"], point["instr"], shuffled)
        s_chosen_original_index = remap[shuffled.index(s_choice_text)]
        out["shuffle_chosen_index"] = s_chosen_original_index
        out["shuffle_agree"] = (s_chosen_original_index == chosen_index)
    return out


def load_done_keys(path):
    if not os.path.exists(resolve(path)):
        return set()
    with open(resolve(path), encoding="utf-8") as f:
        return {(r["id"], r["qid"]) for l in f if l.strip() for r in [json.loads(l)]}


def run_score(args):
    import time

    points = load_jsonl(args.points)
    if args.resume:
        done = load_done_keys(args.out)
        before = len(points)
        points = [p for p in points if (p["id"], p["qid"]) not in done]
        print(f"--resume: {before - len(points)}/{before} points already in {args.out}, {len(points)} remaining",
              flush=True)
    if args.limit:
        points = points[:args.limit]
    if not points:
        print("nothing to do", flush=True)
        return 0

    scorer = make_chooser_scorer(args)
    mode = "a" if args.resume else "w"
    t0 = time.time()
    with open(resolve(args.out), mode, encoding="utf-8") as f:
        for i, point in enumerate(points):
            row_t0 = time.time()
            seed = abs(hash((point["id"], point["qid"]))) % (2 ** 31) if args.shuffle_check else None
            scored = score_point(scorer, point, shuffle_seed=seed)
            out = dict(id=point["id"], qid=point["qid"], domain=point["domain"], source=point["source"],
                       gold_index=point["gold_index"], n_options=point["n_options"], chooser=args.chooser,
                       wall_time_s=time.time() - row_t0, **scored)
            f.write(json.dumps(out) + "\n")
            f.flush()
            os.fsync(f.fileno())
            if (i + 1) % 50 == 0 or i + 1 == len(points):
                print(f"scored {i + 1}/{len(points)}; {time.time() - t0:.1f}s elapsed", flush=True)
    print(f"done: {len(points)} points scored -> {args.out}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# report: metrics (pure Python) + rendering
# ---------------------------------------------------------------------------

def mean(xs):
    return sum(xs) / len(xs) if xs else None


def compute_accuracy(rows, correct_field):
    vals = [r[correct_field] for r in rows if r.get(correct_field) is not None]
    return mean(vals) if vals else None


def compute_agreement(rows, a_field, b_field):
    pairs = [r for r in rows if r.get(a_field) is not None and r.get(b_field) is not None]
    if not pairs:
        return None
    return mean([1.0 if r[a_field] == r[b_field] else 0.0 for r in pairs])


def accuracy_by_source(rows, correct_field):
    by = collections.defaultdict(list)
    for r in rows:
        if r.get(correct_field) is not None:
            by[r["source"]].append(r[correct_field])
    return {s: dict(n=len(v), accuracy=mean(v)) for s, v in sorted(by.items())}


def position_bias(rows, chosen_field):
    """-> dict(mean_norm_chosen, mean_norm_gold, first_rate_chosen, first_rate_gold). Normalized
    position = index / max(n_options - 1, 1), so a 2-option and a 50-option question both land in
    [0, 1]; "first_rate" = how often index 0 is picked/gold, the simplest first-option-bias check."""
    scored = [r for r in rows if r.get(chosen_field) is not None]
    if not scored:
        return dict(mean_norm_chosen=None, mean_norm_gold=None, first_rate_chosen=None, first_rate_gold=None)
    chosen_norm = [r[chosen_field] / max(r["n_options"] - 1, 1) for r in scored]
    gold_norm = [r["gold_index"] / max(r["n_options"] - 1, 1) for r in scored]
    return dict(mean_norm_chosen=mean(chosen_norm), mean_norm_gold=mean(gold_norm),
                first_rate_chosen=mean([1.0 if r[chosen_field] == 0 else 0.0 for r in scored]),
                first_rate_gold=mean([1.0 if r["gold_index"] == 0 else 0.0 for r in scored]))


def coverage_accuracy(rows, confidence_field, correct_field, coverage):
    """-> accuracy on the top `coverage` fraction of rows by confidence (descending), i.e. selective
    accuracy at that coverage level. None if nothing is scored."""
    scored = [r for r in rows if r.get(confidence_field) is not None and r.get(correct_field) is not None]
    if not scored:
        return None
    ranked = sorted(scored, key=lambda r: -r[confidence_field])
    k = max(1, round(len(ranked) * coverage))
    return compute_accuracy(ranked[:k], correct_field)


def join_rows(points, head_scores, kev_scores):
    """-> one row per point, head_*/kev_* fields merged in (None where a chooser has no row for that
    point -- e.g. a partial/--limit score run)."""
    by_key = {(r["id"], r["qid"]): dict(r) for r in points}
    for prefix, scores in (("head", head_scores), ("kev", kev_scores)):
        for r in scores:
            key = (r["id"], r["qid"])
            if key not in by_key:
                continue
            for field in ("chosen_index", "confidence", "max_prob", "correct", "shuffle_chosen_index",
                          "shuffle_agree"):
                by_key[key][f"{prefix}_{field}"] = r.get(field)
    return list(by_key.values())


def render_report(rows):
    lines = ["# Head vs Kev-4B, in-domain + OOD held-out test", "",
              f"Points: {len(rows)} total "
              f"({sum(1 for r in rows if r['domain'] == 'indomain')} in-domain, "
              f"{sum(1 for r in rows if r['domain'] == 'ood')} OOD).", ""]

    lines.append("## Accuracy")
    lines.append("")
    lines.append("| domain | head n | head acc | kev n | kev acc | agreement |")
    lines.append("|---|---|---|---|---|---|")
    for domain in ("indomain", "ood"):
        sub = [r for r in rows if r["domain"] == domain]
        head_n = sum(1 for r in sub if r.get("head_correct") is not None)
        kev_n = sum(1 for r in sub if r.get("kev_correct") is not None)
        head_acc = compute_accuracy(sub, "head_correct")
        kev_acc = compute_accuracy(sub, "kev_correct")
        agree = compute_agreement(sub, "head_chosen_index", "kev_chosen_index")
        lines.append(f"| {domain} | {head_n} | {fmt(head_acc)} | {kev_n} | {fmt(kev_acc)} | {fmt(agree)} |")
    lines.append("")

    for domain in ("indomain", "ood"):
        sub = [r for r in rows if r["domain"] == domain]
        lines.append(f"## Accuracy by source ({domain})")
        lines.append("")
        lines.append("| source | head n | head acc | kev n | kev acc |")
        lines.append("|---|---|---|---|---|")
        head_by = accuracy_by_source(sub, "head_correct")
        kev_by = accuracy_by_source(sub, "kev_correct")
        for source in sorted(set(head_by) | set(kev_by)):
            h = head_by.get(source, dict(n=0, accuracy=None))
            k = kev_by.get(source, dict(n=0, accuracy=None))
            lines.append(f"| {source} | {h['n']} | {fmt(h['accuracy'])} | {k['n']} | {fmt(k['accuracy'])} |")
        lines.append("")

    lines.append("## Position bias (chosen vs gold option index, normalized)")
    lines.append("")
    lines.append("| chooser | mean chosen | mean gold | P(chosen=first) | P(gold=first) |")
    lines.append("|---|---|---|---|---|")
    for prefix in ("head", "kev"):
        pb = position_bias(rows, f"{prefix}_chosen_index")
        lines.append(f"| {prefix} | {fmt(pb['mean_norm_chosen'])} | {fmt(pb['mean_norm_gold'])} | "
                      f"{fmt(pb['first_rate_chosen'])} | {fmt(pb['first_rate_gold'])} |")
    lines.append("")

    lines.append("## Shuffle consistency (agreement with the unshuffled choice)")
    lines.append("")
    lines.append("| chooser | n | shuffle agreement |")
    lines.append("|---|---|---|")
    for prefix in ("head", "kev"):
        sub = [r for r in rows if r.get(f"{prefix}_shuffle_agree") is not None]
        rate = mean([1.0 if r[f"{prefix}_shuffle_agree"] else 0.0 for r in sub]) if sub else None
        lines.append(f"| {prefix} | {len(sub)} | {fmt(rate)} |")
    lines.append("")

    lines.append("## Accuracy at coverage (top-confidence subset)")
    lines.append("")
    lines.append("| chooser | 100% | 80% | 60% |")
    lines.append("|---|---|---|---|")
    for prefix in ("head", "kev"):
        c100 = coverage_accuracy(rows, f"{prefix}_confidence", f"{prefix}_correct", 1.0)
        c80 = coverage_accuracy(rows, f"{prefix}_confidence", f"{prefix}_correct", 0.8)
        c60 = coverage_accuracy(rows, f"{prefix}_confidence", f"{prefix}_correct", 0.6)
        lines.append(f"| {prefix} | {fmt(c100)} | {fmt(c80)} | {fmt(c60)} |")
    lines.append("")
    return "\n".join(lines)


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def run_report(args):
    points = load_jsonl(args.points)
    head_scores = load_jsonl(args.head_scores) if args.head_scores else []
    kev_scores = load_jsonl(args.kev_scores) if args.kev_scores else []
    rows = join_rows(points, head_scores, kev_scores)
    write_jsonl(args.out_jsonl, rows)
    report = render_report(rows)
    with open(resolve(args.out_md), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"wrote {args.out_jsonl}, {args.out_md}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# --selftest: key-remap + metric logic, fabricated fixtures, no model/suite files
# ---------------------------------------------------------------------------

def selftest():
    # 1. seeded_shuffle_with_remap: shuffled_options[i] == options[remap[i]], and it actually reorders.
    options = ["a: x", "b: y", "c: z"]
    shuffled, remap = seeded_shuffle_with_remap(options, seed=0)
    assert sorted(shuffled) == sorted(options) and shuffled != options, (shuffled, options)
    assert shuffled == [options[i] for i in remap], (shuffled, remap)
    # a 2-option list must still re-roll away from the identity order (re-roll branch).
    two = ["x", "y"]
    s2, r2 = seeded_shuffle_with_remap(two, seed=0)
    assert s2 != two and sorted(s2) == sorted(two), (s2, r2)
    print("selftest: seeded_shuffle_with_remap OK (remap round-trips, re-rolls off identity)")

    # 2. build_point: choice (gold index = criteria key's position), noul (gold index = int(label)),
    #    score (gold index = int(label)) -- kev/data.py:materialize's exact rule, via kev.api directly.
    choice_record = dict(_meta=dict(id="r1"),
                          state={"text": "customer message"},
                          questions={"intent": dict(type="choice", instructions="pick one",
                                                     criteria={"billing": "about billing", "other": None},
                                                     label="other", src="banking77")})
    p = build_point(choice_record, "intent")
    assert p["options"] == ["billing: about billing", "other"], p["options"]
    assert p["gold_index"] == 1 and p["gold_label"] == "other" and p["qtype"] == "choice", p

    noul_record = dict(_meta=dict(id="r2"), state="is it raining?",
                        questions={"wet": dict(type="noul", instructions="yes or no", criteria=None,
                                                label=True, src="boolq")})
    p2 = build_point(noul_record, "wet")
    assert p2["options"] == ["no", "yes"], p2["options"]
    assert p2["gold_index"] == 1 and p2["qtype"] == "noul", p2

    score_record = dict(_meta=dict(id="r3"), state="rate this",
                         questions={"rating": dict(type="score", instructions="how good",
                                                    criteria=["bad", "ok", "good"], label=2, src="amazon")})
    p3 = build_point(score_record, "rating")
    assert p3["options"] == ["bad", "ok", "good"], p3["options"]
    assert p3["gold_index"] == 2 and p3["qtype"] == "score", p3
    print(f"selftest: build_point OK (choice={p['gold_index']}, noul={p2['gold_index']}, "
          f"score={p3['gold_index']})")

    # 3. stratified_sample: covers every group before any group repeats, respects a fixed seed.
    pts = []
    for src, n in (("a", 1), ("b", 5), ("c", 2)):
        for i in range(n):
            rec = dict(_meta=dict(id=f"{src}-{i}"), questions={"q": dict(src=src)})
            pts.append((rec["_meta"]["id"], "q", rec))
    sample = stratified_sample(pts, cap=3, seed=0)
    assert {point_source(p) for p in sample} == {"a", "b", "c"}, sample  # cap == #groups -> all touched
    sample_full = stratified_sample(pts, cap=100, seed=0)
    assert len(sample_full) == len(pts), sample_full  # cap above total -> everything returned
    sample_a = stratified_sample(pts, cap=3, seed=1)
    sample_b = stratified_sample(pts, cap=3, seed=1)
    assert [p[0] for p in sample_a] == [p[0] for p in sample_b], "same seed must reproduce the same sample"
    print("selftest: stratified_sample OK (covers every group first, deterministic per seed)")

    # 4. metrics: accuracy / agreement / by-source / position bias / coverage, on fabricated score rows.
    rows = [
        dict(source="s1", n_options=2, gold_index=0, head_chosen_index=0, head_correct=True, head_confidence=0.9,
             kev_chosen_index=1, kev_correct=False, kev_confidence=0.6),
        dict(source="s1", n_options=2, gold_index=1, head_chosen_index=1, head_correct=True, head_confidence=0.6,
             kev_chosen_index=1, kev_correct=True, kev_confidence=0.9),
        dict(source="s2", n_options=4, gold_index=3, head_chosen_index=0, head_correct=False, head_confidence=0.4,
             kev_chosen_index=3, kev_correct=True, kev_confidence=0.3),
        dict(source="s2", n_options=4, gold_index=0, head_chosen_index=0, head_correct=True, head_confidence=0.95,
             kev_chosen_index=0, kev_correct=True, kev_confidence=0.95),
    ]
    assert compute_accuracy(rows, "head_correct") == 0.75
    assert compute_accuracy(rows, "kev_correct") == 0.75
    assert compute_agreement(rows, "head_chosen_index", "kev_chosen_index") == 0.5  # rows 2 and 4 agree
    by_src = accuracy_by_source(rows, "head_correct")
    assert by_src["s1"] == dict(n=2, accuracy=1.0) and by_src["s2"] == dict(n=2, accuracy=0.5), by_src
    pb = position_bias(rows, "head_chosen_index")
    # chosen norm: [0, 1, 0, 0] -> mean 0.25; gold norm: [0, 1, 1, 0] -> mean 0.5
    assert abs(pb["mean_norm_chosen"] - 0.25) < 1e-9 and abs(pb["mean_norm_gold"] - 0.5) < 1e-9, pb
    assert pb["first_rate_chosen"] == 0.75 and pb["first_rate_gold"] == 0.5, pb
    # coverage: top-50% by head_confidence = rows 4 (0.95) and 1 (0.9), both head_correct=True -> 1.0;
    # top-100% = all 4 rows, 3/4 correct -> 0.75.
    assert coverage_accuracy(rows, "head_confidence", "head_correct", 0.5) == 1.0
    assert coverage_accuracy(rows, "head_confidence", "head_correct", 1.0) == 0.75
    print("selftest: accuracy/agreement/by-source/position-bias/coverage metrics OK")

    # 5. join_rows: points without a matching score row keep None fields, not a KeyError/crash.
    points = [dict(id="p1", qid="q", domain="indomain", source="s1", n_options=2, gold_index=0)]
    head_scores = [dict(id="p1", qid="q", chosen_index=0, confidence=0.8, max_prob=0.8, correct=True)]
    joined = join_rows(points, head_scores, kev_scores=[])
    assert joined[0]["head_correct"] is True and joined[0].get("kev_correct") is None, joined
    print("selftest: join_rows OK (missing chooser scores stay None, no crash)")

    print("selftest: all checks passed")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selftest", action="store_true", help="run the key-remap/metric selftest and exit")
    sub = ap.add_subparsers(dest="cmd")

    sp = sub.add_parser("build")
    sp.add_argument("--indomain-dir", default=DEFAULT_INDOMAIN_DIR)
    sp.add_argument("--ood-dir", default=DEFAULT_OOD_DIR)
    sp.add_argument("--cap", type=int, default=DEFAULT_CAP)
    sp.add_argument("--seed", type=int, default=20261009)
    sp.add_argument("--out", required=True)

    sp = sub.add_parser("score")
    sp.add_argument("--points", required=True)
    sp.add_argument("--chooser", choices=["head", "kev"], required=True)
    sp.add_argument("--decision-run", default="runs/p4-e4b-final")
    sp.add_argument("--run", default="jaredpalmer/kev-4b")
    sp.add_argument("--device", default="cuda")
    sp.add_argument("--out", required=True)
    sp.add_argument("--resume", action="store_true")
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--shuffle-check", action="store_true")

    sp = sub.add_parser("report")
    sp.add_argument("--points", required=True)
    sp.add_argument("--head-scores", default=None)
    sp.add_argument("--kev-scores", default=None)
    sp.add_argument("--out-md", required=True)
    sp.add_argument("--out-jsonl", required=True)

    args = ap.parse_args()
    if args.selftest:
        return sys.exit(selftest())
    if args.cmd == "build":
        return sys.exit(run_build(args))
    if args.cmd == "score":
        return sys.exit(run_score(args))
    if args.cmd == "report":
        return sys.exit(run_report(args))
    ap.error("one of --selftest, build, score, report is required")


if __name__ == "__main__":
    main()
