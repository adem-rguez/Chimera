"""Diagnostic: is the trained pointer head the bottleneck on the Phase 7G S1 decision points, or would
a plain forced-choice LETTER/KEY readout off the base model (no head, no adapter) do just as well?

Three choosers scored on IDENTICAL inputs, for every valid S1 decision point
(scripts/extract_decision_points.py's output, oracle/s1-points.jsonl, 277 points):

  (a) the trained pointer head -- already scored, oracle/s1-headscores.jsonl (produced by
      scripts/s1_head_and_outcomes.py headscore). Not re-run here.
  (b) base google/gemma-4-E4B-it (no adapter), forced-choice next-token-logit readout, same pattern as
      scripts/extract_labels_llm.py's letter reader (chat template, enable_thinking=False,
      add_generation_prompt=True, single forward pass, logits restricted to the candidate option keys'
      first token, resolved per-row the same prefix-diff way extract_labels_llm.py does -- generalised
      from single letters to each option's own (possibly multi-char) `key`, see resolve_key_token_ids).
      Two variants:
        b1 = state (user turn + thinking-so-far, built EXACTLY like
             scripts/s1_head_and_outcomes.py:build_head_input -- same state the head sees) + question +
             options.
        b2 = question + options only, no state at all.
  (c) Kev's original (Qwen-based) pretrained checkpoint -- jaredpalmer/kev-4b, a LoRA adapter over
      Qwen3.5-4B-Base (docs/kev-upstream-README.md). Scored by scripts/s1_kev_score.py (separate script:
      it calls DecisionModel.encode/.probs directly, the same pattern as the trained head's own scorer,
      not this script's forced-choice LM readout), writing oracle/s1-kev-readout.jsonl; joined in here
      as the "kev" chooser.

Usage:
  Offline (laptop, no torch/model, pure logic):
    .venv/Scripts/python.exe scripts/s1_readout_diagnostic.py --selftest

  GPU box, score b1+b2 (one forward pass per point per variant; writes oracle/s1-readout-lm.jsonl):
    .venv/bin/python -u scripts/s1_readout_diagnostic.py readout \\
        --traces oracle/s1-traces.jsonl --points oracle/s1-points.jsonl \\
        --out oracle/s1-readout-lm.jsonl --resume --shuffle-check --batch-size 8

  CPU, join + report (writes oracle/s1-readout-diagnostic.md + .jsonl):
    .venv/Scripts/python.exe scripts/s1_readout_diagnostic.py report \\
        --traces oracle/s1-traces.jsonl --points oracle/s1-points.jsonl \\
        --problems oracle/s1-train-problems.jsonl --headscores oracle/s1-headscores.jsonl \\
        --lm-readout oracle/s1-readout-lm.jsonl --kev-readout oracle/s1-kev-readout.jsonl \\
        --out-md oracle/s1-readout-diagnostic.md --out-jsonl oracle/s1-readout-diagnostic.jsonl
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL = "google/gemma-4-E4B-it"
REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
COVERAGE_LEVELS = (1.0, 0.8, 0.6)


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl_append(f, rows):
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.flush()
    os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# prompt construction (pure string logic -- safe for --selftest, no torch/kev)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You will be shown a question and a list of options. Pick exactly one option. "
    "Respond with the option's key only, and nothing else."
)


def build_option_lines(options):
    """`options`: list of (key, desc) pairs -> rendered '{key}: {desc}' lines (one per line, in the
    given order -- the caller is responsible for any shuffling)."""
    return "\n".join(f"{k}: {d}" if d else k for k, d in options)


def build_user_content(state, instruction, options):
    """-> user message text. `state` may be None (variant b2: question+options only)."""
    parts = []
    if state is not None:
        parts.append(state)
    parts.append(instruction)
    parts.append("OPTIONS:\n" + build_option_lines(options))
    parts.append("Answer with the option key only.")
    return "\n\n".join(parts)


def render_prompt(tok, user_content):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def resolve_key_token_ids(tok, prompt_text, keys):
    """-> (dict key -> first token id of that key as it would continue `prompt_text`, n_fallback int).
    Generalises scripts/extract_labels_llm.py:resolve_letter_token_ids from single letters to arbitrary
    (possibly multi-char) option keys: tokenizes `prompt_text` and `prompt_text + key` and diffs the
    ids, taking the first new token; falls back to the standalone tokenization of `key` on a BPE-merge
    mismatch (counted in n_fallback, same convention as that script)."""
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    out = {}
    n_fallback = 0
    for key in keys:
        combined_ids = tok(prompt_text + key, add_special_tokens=False)["input_ids"]
        if combined_ids[:len(prompt_ids)] == prompt_ids and len(combined_ids) > len(prompt_ids):
            out[key] = combined_ids[len(prompt_ids)]
        else:
            n_fallback += 1
            standalone = tok(key, add_special_tokens=False)["input_ids"]
            out[key] = standalone[0]
    return out, n_fallback


def renorm_probs(cand_values):
    """-> dict key -> probability, renormalized so the given raw (non-negative) values sum to 1. Pure
    Python; used both for the real softmax-over-candidates values and exercised directly by
    --selftest with fabricated numbers."""
    total = sum(cand_values.values()) or 1e-12
    return {k: v / total for k, v in cand_values.items()}


def top1_margin(renormed):
    """-> (top_key, top_p, margin) from a renormalized {key: prob} dict, margin = top1 - top2 (0.0 if
    only one candidate)."""
    ranked = sorted(renormed.items(), key=lambda kv: kv[1], reverse=True)
    top_key, top_p = ranked[0]
    margin = top_p - (ranked[1][1] if len(ranked) > 1 else 0.0)
    return top_key, top_p, margin


def detect_key_collision(key_to_id):
    """-> True iff two or more keys resolved to the SAME first token id (readout cannot distinguish
    them; caller should exclude the point from this variant rather than silently guessing)."""
    ids = list(key_to_id.values())
    return len(set(ids)) != len(ids)


# ---------------------------------------------------------------------------
# selftest (CPU, no torch/kev/model)
# ---------------------------------------------------------------------------

class _FakeTok:
    """Word-level fake tokenizer: each whitespace-split token is its own id (stable across calls via a
    shared vocab dict), just enough to exercise resolve_key_token_ids's prefix-diff logic without
    pulling in a real BPE tokenizer."""
    def __init__(self):
        self.vocab = {}

    def _ids(self, text):
        words = text.split()
        out = []
        for w in words:
            out.append(self.vocab.setdefault(w, len(self.vocab)))
        return out

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self._ids(text)}


def selftest():
    # 1. build_user_content: b1 (state present) vs b2 (state=None).
    opts = [("A", "cell membrane"), ("B", "cell wall")]
    u1 = build_user_content("some state text", "Which structure?", opts)
    assert u1.startswith("some state text\n\n"), u1
    assert "A: cell membrane" in u1 and "B: cell wall" in u1
    u2 = build_user_content(None, "Which structure?", opts)
    assert "some state text" not in u2
    assert u2.startswith("Which structure?"), u2
    print("selftest: build_user_content OK (b1 has state, b2 omits it)")

    # 2. build_option_lines: bare key (no desc) renders without a colon.
    assert build_option_lines([("x", "")]) == "x"
    assert build_option_lines([("x", None)]) == "x"
    assert build_option_lines([("x", "y")]) == "x: y"
    print("selftest: build_option_lines OK (desc / empty-desc / None-desc)")

    # 3. resolve_key_token_ids: single-token keys resolve cleanly via the fake tokenizer; a key whose
    #    first word collides with another key's first word is still handled (same first id -- that's
    #    exactly the collision --detect_key_collision below exists to catch).
    tok = _FakeTok()
    prompt = "pick one: "
    ids, n_fallback = resolve_key_token_ids(tok, prompt, ["highway", "back_road"])
    assert ids["highway"] != ids["back_road"], ids
    assert n_fallback == 0, n_fallback
    print(f"selftest: resolve_key_token_ids OK (distinct keys -> distinct ids, {ids})")

    collide_ids = {"a": 5, "b": 5}
    assert detect_key_collision(collide_ids) is True
    assert detect_key_collision({"a": 5, "b": 6}) is False
    print("selftest: detect_key_collision OK")

    # 4. renorm_probs / top1_margin: fabricated raw values, not necessarily summing to 1.
    renormed = renorm_probs({"A": 0.2, "B": 0.6, "C": 0.2})
    assert abs(sum(renormed.values()) - 1.0) < 1e-9, renormed
    top_key, top_p, margin = top1_margin(renormed)
    assert top_key == "B", renormed
    assert abs(top_p - 0.6) < 1e-9 and abs(margin - 0.4) < 1e-9, (top_p, margin)
    print(f"selftest: renorm_probs/top1_margin OK ({renormed} -> top={top_key} p={top_p} margin={margin})")

    # 5. single-candidate edge case (1-option points: 47 of them in oracle/s1-points.jsonl): with no
    #    second-place candidate to subtract, margin == top_p == 1.0 (not an error).
    one = renorm_probs({"only": 3.7})
    top_key, top_p, margin = top1_margin(one)
    assert top_key == "only" and top_p == 1.0 and margin == 1.0
    print("selftest: single-candidate case OK (margin == p == 1.0)")

    print("selftest: all s1_readout_diagnostic checks passed")
    return 0


# ---------------------------------------------------------------------------
# readout: GPU scoring loop (lazy torch/transformers imports)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name, revision, device, attn_impl="sdpa"):
    import torch
    from transformers import AutoModelForCausalLM
    from kev.model import load_tokenizer
    tok = load_tokenizer(model_name, revision)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, dtype=torch.bfloat16, attn_implementation=attn_impl
    ).to(device).eval()
    return tok, model


def score_variant(tok, model, device, prompts, per_row_keys, per_row_key_to_id):
    """-> list of {choice, confidence, probs, collision} for one batch, one variant (b1 or b2)."""
    import torch
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    real_lens = enc["attention_mask"].sum(dim=1)
    out = []
    for i in range(len(prompts)):
        pos = int(real_lens[i].item()) - 1
        row_logits = logits[i, pos, :].to(torch.float32)
        probs_full = torch.softmax(row_logits, dim=-1)
        key_to_id = per_row_key_to_id[i]
        cand = {k: probs_full[tid].item() for k, tid in key_to_id.items()}
        renormed = renorm_probs(cand)
        top_key, top_p, margin = top1_margin(renormed)
        out.append(dict(choice=top_key, confidence=top_p, margin=margin, probs=renormed,
                         collision=detect_key_collision(key_to_id)))
    return out


def run_readout(args):
    from scripts.s1_head_and_outcomes import build_head_input, iter_scorable_points

    traces = load_jsonl(resolve(args.traces))
    points_rows = load_jsonl(resolve(args.points))
    pairs = iter_scorable_points(traces, points_rows)
    print(f"loaded {len(traces)} traces, {len(points_rows)} points-rows -> {len(pairs)} scorable points",
          flush=True)
    if args.limit:
        pairs = pairs[:args.limit]

    out_path = resolve(args.out)
    done = set()
    if args.resume and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {(r["id"], r["point_idx"]) for l in f if l.strip() for r in [json.loads(l)]}
        before = len(pairs)
        pairs = [(t, i, p) for t, i, p in pairs if (t["id"], i) not in done]
        print(f"--resume: {before - len(pairs)}/{before} points already in {args.out}, "
              f"{len(pairs)} remaining", flush=True)
    if not pairs:
        print("nothing to do", flush=True)
        return 0

    tok, model = load_model_and_tokenizer(args.model, args.revision, args.device, args.attn_impl)

    # pre-build everything CPU-side so batching + sort-by-length is cheap.
    rows = []
    for trace_row, idx, point in pairs:
        state, instruction, described_options = build_head_input(trace_row, point)
        keys = [o["key"] for o in point["options"]]
        descs = [o["desc"] for o in point["options"]]
        options = list(zip(keys, descs))
        u1 = build_user_content(state, instruction, options)
        u2 = build_user_content(None, instruction, options)
        p1 = render_prompt(tok, u1)
        p2 = render_prompt(tok, u2)
        seed = abs(hash((trace_row["id"], idx))) % (2 ** 31)
        # shuffle the (key, desc) PAIRS directly (not s1_head_and_outcomes.seeded_shuffle_options, which
        # shuffles rendered "key: desc" strings) so descs travel with their key under reordering.
        if args.shuffle_check:
            shuffled_pairs = list(options)
            import random
            rng = random.Random(seed)
            rng.shuffle(shuffled_pairs)
            tries = 0
            while shuffled_pairs == options and len(options) > 1 and tries < 10:
                seed += 1
                rng = random.Random(seed)
                shuffled_pairs = list(options)
                rng.shuffle(shuffled_pairs)
                tries += 1
            su1 = build_user_content(state, instruction, shuffled_pairs)
            su2 = build_user_content(None, instruction, shuffled_pairs)
            sp1 = render_prompt(tok, su1)
            sp2 = render_prompt(tok, su2)
        else:
            sp1 = sp2 = None
        rows.append(dict(id=trace_row["id"], point_idx=idx, keys=keys, p1=p1, p2=p2,
                          sp1=sp1, sp2=sp2, chosen_key=point["chosen_key"]))

    rows.sort(key=lambda r: len(r["p1"]))  # reduce padding waste, same reasoning as extract_labels_llm.py

    print(f"scoring {len(rows)} points x 2 variants"
          f"{' x 2 orders (shuffle-check)' if args.shuffle_check else ''}, "
          f"batch_size={args.batch_size}", flush=True)

    mode = "a" if args.resume else "w"
    t0 = time.time()
    n_fallback_total = n_collision_total = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i + args.batch_size]
            batch_t0 = time.time()

            def key_ids_for(field):
                prompts, key_to_id_list = [], []
                nf = 0
                for r in batch:
                    key_to_id, n_fb = resolve_key_token_ids(tok, r[field], r["keys"])
                    nf += n_fb
                    prompts.append(r[field])
                    key_to_id_list.append(key_to_id)
                return prompts, key_to_id_list, nf

            out_rows = []
            variant_results = {}
            for field, name in (("p1", "b1"), ("p2", "b2")):
                prompts, key_to_id_list, nf = key_ids_for(field)
                n_fallback_total += nf
                variant_results[name] = score_variant(tok, model, args.device, prompts,
                                                        [r["keys"] for r in batch], key_to_id_list)
            if args.shuffle_check:
                for field, name in (("sp1", "sb1"), ("sp2", "sb2")):
                    prompts, key_to_id_list, nf = key_ids_for(field)
                    n_fallback_total += nf
                    variant_results[name] = score_variant(tok, model, args.device, prompts,
                                                            [r["keys"] for r in batch], key_to_id_list)
            batch_elapsed = time.time() - batch_t0

            for j, r in enumerate(batch):
                out = dict(id=r["id"], point_idx=r["point_idx"], model_choice=r["chosen_key"])
                for name in ("b1", "b2"):
                    v = variant_results[name][j]
                    out[f"{name}_choice"] = v["choice"]
                    out[f"{name}_confidence"] = v["confidence"]
                    out[f"{name}_margin"] = v["margin"]
                    out[f"{name}_probs"] = v["probs"]
                    out[f"{name}_collision"] = v["collision"]
                    out[f"{name}_agree"] = (v["choice"] == r["chosen_key"])
                    n_collision_total += int(v["collision"])
                if args.shuffle_check:
                    for name, sname in (("b1", "sb1"), ("b2", "sb2")):
                        sv = variant_results[sname][j]
                        out[f"{name}_shuffle_choice"] = sv["choice"]
                        out[f"{name}_shuffle_agree"] = (sv["choice"] == out[f"{name}_choice"])
                out["wall_time_s"] = batch_elapsed / len(batch)
                out_rows.append(out)
            write_jsonl_append(f, out_rows)
            done_n = i + len(batch)
            print(f"scored {done_n}/{len(rows)}; batch {batch_elapsed:.2f}s "
                  f"({batch_elapsed / len(batch):.3f}s/point); fallbacks={n_fallback_total} "
                  f"collisions={n_collision_total}", flush=True)

    print(f"done: {len(rows)} points scored -> {args.out}, {time.time() - t0:.1f}s total, "
          f"{n_fallback_total} key-token fallbacks, {n_collision_total} key collisions "
          f"(variant excluded from scoring on those rows -- see report)", flush=True)
    return 0


# ---------------------------------------------------------------------------
# report: join (a)/(b1)/(b2), compute all metrics (pure Python, CPU only)
# ---------------------------------------------------------------------------

def determinable_points(traces, points_rows, problems_by_id):
    """-> list of dicts {id, point_idx, gold}, the 116-point "determinable" set: trace's final answer is
    itself a correctly-judged MC choice AND the point's own option keys ARE the lettered choices (so the
    point IS the final-answer decision). Mirrors
    scripts/s1_head_and_outcomes.py:head_accuracy_on_outcome_verified's rule exactly, but returns the
    per-point list instead of an aggregate count (needed here to join against 3 choosers, not 1)."""
    from scripts.s1_head_and_outcomes import judge_correctness
    points_by_id = {r["id"]: r.get("points", []) for r in points_rows}
    out = []
    for trace in traces:
        row_id = trace["id"]
        problem = problems_by_id.get(row_id)
        if problem is None or problem.get("answer_type") != "choice":
            continue
        correct, _parsed, unparsed = judge_correctness(trace, problem)
        if correct is None or unparsed:
            continue
        gold_letter = str(problem["gold_answer"]).upper()
        for idx, p in enumerate(points_by_id.get(row_id, [])):
            option_keys = {o["key"].upper() for o in p.get("options", [])}
            if option_keys != {chr(ord("A") + i) for i in range(len(option_keys))} or \
                    gold_letter not in option_keys:
                continue
            out.append(dict(id=row_id, point_idx=idx, gold=gold_letter))
    return out


def selective_accuracy(items, coverage_levels=COVERAGE_LEVELS):
    """`items`: list of (confidence, correct_bool), already restricted to the set accuracy is defined
    over (e.g. the 116 determinable points, chooser has a score for all of them). -> {coverage:
    accuracy} by abstaining on the lowest-confidence (1 - coverage) fraction."""
    items = sorted(items, key=lambda t: t[0], reverse=True)
    n = len(items)
    out = {}
    for cov in coverage_levels:
        k = max(1, round(n * cov)) if n else 0
        top = items[:k]
        out[cov] = (sum(1 for _c, correct in top if correct) / len(top)) if top else float("nan")
    return out


def render_report(n_total, n_determinable, metrics, disagreements, caveats):
    lines = ["# S1 readout diagnostic: is the decision head the bottleneck?", ""]
    lines.append(f"{n_total} total valid S1 points; {n_determinable} determinable (MC, gold-verified) "
                 "points used for accuracy/selective-accuracy.")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| chooser | agree w/ trace (n=277) | acc vs gold (n=" + str(n_determinable) + ") | "
                 "shuffle consistency | acc@100% cov | acc@80% cov | acc@60% cov | mean wall time/decision |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, m in metrics.items():
        def fmt(x):
            return "n/a" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
        sel = m["selective"]
        lines.append(f"| {name} | {fmt(m['agree_trace'])} | {fmt(m['acc_gold'])} | "
                     f"{fmt(m['shuffle_consistency'])} | {fmt(sel.get(1.0))} | {fmt(sel.get(0.8))} | "
                     f"{fmt(sel.get(0.6))} | {fmt(m['wall_time_s'])} |")
    lines.append("")
    lines.append("## Disagreement with gold, on the determinable set")
    lines.append("")
    lines.append("| comparison | head right, LM wrong | LM right, head wrong |")
    lines.append("|---|---|---|")
    for name, d in disagreements.items():
        lines.append(f"| head vs {name} | {d['head_right_lm_wrong']} | {d['lm_right_head_wrong']} |")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    for c in caveats:
        lines.append(f"- {c}")
    lines.append("")
    return "\n".join(lines) + "\n"


def run_report(args):
    from scripts.s1_head_and_outcomes import iter_scorable_points

    traces = load_jsonl(resolve(args.traces))
    points_rows = load_jsonl(resolve(args.points))
    problems = load_jsonl(resolve(args.problems))
    problems_by_id = {p["id"]: p for p in problems}
    headscores = {(r["id"], r["point_idx"]): r for r in load_jsonl(resolve(args.headscores))}
    lm_rows = {(r["id"], r["point_idx"]): r for r in load_jsonl(resolve(args.lm_readout))}
    kev_rows = {(r["id"], r["point_idx"]): r for r in load_jsonl(resolve(args.kev_readout))} if args.kev_readout else {}

    all_pairs = iter_scorable_points(traces, points_rows)
    n_total = len(all_pairs)
    det = determinable_points(traces, points_rows, problems_by_id)
    det_by_key = {(d["id"], d["point_idx"]): d["gold"] for d in det}
    n_determinable = len(det)

    caveats = [
        "mean wall time/decision for the pointer head (a) was not re-measured here (headscores were "
        "produced by an earlier run, scripts/s1_head_and_outcomes.py, with no per-row timing); only "
        "b1/b2 wall time was measured in this run.",
        "b1/b2 wall time is batch-amortized (GPU batched forward pass / batch size), not a true "
        "per-call latency -- a single-item-at-a-time run would be slower.",
    ]
    if not kev_rows:
        caveats.append(
            "(c) Kev's original (Qwen-based) checkpoint was SKIPPED (no --kev-readout given / file "
            "not found): see scripts/s1_kev_score.py to produce oracle/s1-kev-readout.jsonl.")

    chooser_names = ["head", "b1", "b2"] + (["kev"] if kev_rows else [])
    agree_trace = {n: [] for n in chooser_names}
    acc_gold = {n: [] for n in chooser_names}
    shuffle_cons = {n: [] for n in chooser_names}
    sel_items = {n: [] for n in chooser_names}
    wall_times = {n: [] for n in chooser_names}
    n_excluded_collision = {"b1": 0, "b2": 0}
    joined_rows = []

    for trace_row, idx, point in all_pairs:
        key = (trace_row["id"], idx)
        hs = headscores.get(key)
        lm = lm_rows.get(key)
        kv = kev_rows.get(key)
        row = dict(id=key[0], point_idx=key[1], trace_choice=point["chosen_key"])
        gold = det_by_key.get(key)
        row["gold"] = gold

        if hs is not None:
            agree_trace["head"].append(hs["agree"])
            if "shuffle_agree" in hs:
                shuffle_cons["head"].append(hs["shuffle_agree"])
            row["head_choice"] = hs["head_choice"]
            row["head_agree"] = hs["agree"]
            if gold is not None:
                head_correct = hs["head_choice"].upper() == gold
                acc_gold["head"].append(head_correct)
                sel_items["head"].append((hs["confidence"], head_correct))
                row["head_correct"] = head_correct

        if lm is not None:
            for name in ("b1", "b2"):
                choice = lm.get(f"{name}_choice")
                collided = lm.get(f"{name}_collision")
                agree_trace[name].append(lm[f"{name}_agree"])
                row[f"{name}_choice"] = choice
                row[f"{name}_agree"] = lm[f"{name}_agree"]
                if f"{name}_shuffle_agree" in lm:
                    shuffle_cons[name].append(lm[f"{name}_shuffle_agree"])
                if "wall_time_s" in lm:
                    wall_times[name].append(lm["wall_time_s"])
                if gold is not None:
                    if collided:
                        n_excluded_collision[name] += 1
                        continue
                    correct = (choice or "").upper() == gold
                    acc_gold[name].append(correct)
                    sel_items[name].append((lm.get(f"{name}_confidence", 0.0), correct))
                    row[f"{name}_correct"] = correct

        if kv is not None:
            agree_trace["kev"].append(kv["kev_agree"])
            if "kev_shuffle_agree" in kv:
                shuffle_cons["kev"].append(kv["kev_shuffle_agree"])
            row["kev_choice"] = kv["kev_choice"]
            row["kev_agree"] = kv["kev_agree"]
            if "wall_time_s" in kv:
                wall_times["kev"].append(kv["wall_time_s"])
            if gold is not None:
                kev_correct = (kv["kev_choice"] or "").upper() == gold
                acc_gold["kev"].append(kev_correct)
                sel_items["kev"].append((kv.get("kev_confidence", 0.0), kev_correct))
                row["kev_correct"] = kev_correct
        joined_rows.append(row)

    def mean(xs):
        xs = list(xs)
        return (sum(xs) / len(xs)) if xs else None

    metrics = {}
    for name in chooser_names:
        metrics[name] = dict(
            agree_trace=mean(agree_trace[name]),
            acc_gold=mean(acc_gold[name]),
            shuffle_consistency=mean(shuffle_cons[name]) if shuffle_cons[name] else None,
            selective=selective_accuracy(sel_items[name]) if sel_items[name] else {c: None for c in COVERAGE_LEVELS},
            wall_time_s=mean(wall_times[name]),
        )
    for name in ("b1", "b2"):
        if n_excluded_collision[name]:
            caveats.append(f"{name}: {n_excluded_collision[name]}/{n_determinable} determinable points "
                           "excluded from acc/selective-accuracy (two option keys resolved to the same "
                           "first token -- readout genuinely cannot distinguish them).")

    disagreements = {}
    for name in ("b1", "b2") + (("kev",) if kev_rows else ()):
        hr_lw = lw_rh = 0
        for row in joined_rows:
            if "head_correct" not in row or f"{name}_correct" not in row:
                continue
            hc, lc = row["head_correct"], row[f"{name}_correct"]
            if hc and not lc:
                hr_lw += 1
            elif lc and not hc:
                lw_rh += 1
        disagreements[name] = dict(head_right_lm_wrong=hr_lw, lm_right_head_wrong=lw_rh)

    report = render_report(n_total, n_determinable, metrics, disagreements, caveats)
    print(report)

    out_md = resolve(args.out_md)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"wrote {out_md}")

    out_jsonl = resolve(args.out_jsonl)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        write_jsonl_append(f, joined_rows)
    print(f"wrote {out_jsonl} ({len(joined_rows)} rows)")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")

    sp_r = sub.add_parser("readout")
    sp_r.add_argument("--traces")
    sp_r.add_argument("--points")
    sp_r.add_argument("--out")
    sp_r.add_argument("--model", default=MODEL)
    sp_r.add_argument("--revision", default=REVISION)
    sp_r.add_argument("--device", default="cuda")
    sp_r.add_argument("--attn-impl", default="sdpa")
    sp_r.add_argument("--batch-size", type=int, default=8)
    sp_r.add_argument("--resume", action="store_true")
    sp_r.add_argument("--limit", type=int, default=None)
    sp_r.add_argument("--shuffle-check", action="store_true")

    sp_p = sub.add_parser("report")
    sp_p.add_argument("--traces")
    sp_p.add_argument("--points")
    sp_p.add_argument("--problems")
    sp_p.add_argument("--headscores")
    sp_p.add_argument("--lm-readout")
    sp_p.add_argument("--kev-readout", default=None)
    sp_p.add_argument("--out-md")
    sp_p.add_argument("--out-jsonl")

    args = ap.parse_args()
    if args.selftest:
        return sys.exit(selftest())
    if args.cmd == "readout":
        if not (args.traces and args.points and args.out):
            ap.error("readout requires --traces/--points/--out")
        return sys.exit(run_readout(args))
    if args.cmd == "report":
        if not (args.traces and args.points and args.problems and args.headscores and args.lm_readout
                and args.out_md and args.out_jsonl):
            ap.error("report requires --traces/--points/--problems/--headscores/--lm-readout/"
                      "--out-md/--out-jsonl")
        return sys.exit(run_report(args))
    ap.error("one of --selftest, readout, or report is required")


if __name__ == "__main__":
    main()
