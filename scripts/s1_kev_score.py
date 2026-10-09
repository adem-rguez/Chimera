"""(c) Kev's original pretrained decision model -- jaredpalmer/kev-4b, a LoRA adapter over
Qwen3.5-4B-Base (docs/kev-upstream-README.md) -- scored on the same 277 S1 decision points as the
head/readout choosers in scripts/s1_readout_diagnostic.py, on the SAME state (user turn +
reasoning-so-far + question, options described "key: desc") built the same way
(scripts/s1_head_and_outcomes.py:build_head_input).

Scores by calling the model directly (DecisionModel.encode/.probs), the same pattern
scripts/probe_phase7r_head.py:make_real_scorer uses for the trained pointer head -- NOT
kev.predictors.LocalPredictor (that goes through kev.api's pydantic request validation and
kev.data.materialize, built for the /v1/systemone request shape; calling the model directly needs no
request/response mapping and matches the head-scoring convention this diagnostic already uses).

One point = one forward pass (batch size 1): earlier readout runs OOM'd at batch 8 on long states, and
kev's Qwen3.5 backbone is "hybrid" (Gated DeltaNet layers, kev/model.py:is_hybrid) -- each question in a
batch runs its own causal row, so batching only pipelines independent passes, it does not help memory
pressure the way packed attention-only batching would.

Usage:
  GPU box:
    .venv/bin/python -u scripts/s1_kev_score.py score \\
        --traces oracle/s1-traces.jsonl --points oracle/s1-points.jsonl \\
        --out oracle/s1-kev-readout.jsonl --run jaredpalmer/kev-4b --device cuda \\
        --resume --shuffle-check
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_RUN = "jaredpalmer/kev-4b"


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


def load_kev_model(run, device):
    import torch
    from kev.checkpoint import Checkpoint, LoadOptions
    tok, model = Checkpoint(run).load(device, LoadOptions(dtype=torch.bfloat16))
    return tok, model


def make_kev_scorer(tok, model):
    """-> callable(state_text, instr, options) -> (choice_text, confidence, max_prob, probs_dict).
    Mirrors scripts/probe_phase7r_head.py:make_real_scorer exactly (same rec shape, same call
    sequence); `options` is a list of described "key: desc" strings, same convention as
    scripts/s1_head_and_outcomes.py:build_head_input."""
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


def option_key(opt_text):
    return opt_text.split(": ", 1)[0]


def top1_margin_from_probs(probs_by_option):
    ranked = sorted(probs_by_option.values(), reverse=True)
    top_p = ranked[0]
    margin = top_p - (ranked[1] if len(ranked) > 1 else 0.0)
    return top_p, margin


def run_score(args):
    from scripts.s1_head_and_outcomes import build_head_input, iter_scorable_points, seeded_shuffle_options

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

    tok, model = load_kev_model(args.run, args.device)
    scorer = make_kev_scorer(tok, model)

    mode = "a" if args.resume else "w"
    t0 = time.time()
    n_done = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for trace_row, idx, point in pairs:
            state, instruction, options = build_head_input(trace_row, point)
            row_t0 = time.time()
            choice_text, confidence, max_prob, probs_by_option = scorer(state, instruction, options)
            choice_key = option_key(choice_text)
            top_p, margin = top1_margin_from_probs(probs_by_option)
            out = dict(id=trace_row["id"], point_idx=idx, chosen_key=point["chosen_key"],
                        kev_choice=choice_key, kev_confidence=confidence, kev_max_prob=max_prob,
                        kev_margin=margin, kev_agree=(choice_key == point["chosen_key"]),
                        wall_time_s=time.time() - row_t0)
            if args.shuffle_check and len(options) > 1:
                seed = abs(hash((trace_row["id"], idx))) % (2 ** 31)
                shuffled = seeded_shuffle_options(options, seed)
                s_choice_text, _sc, _smp, _sp = scorer(state, instruction, shuffled)
                out["kev_shuffle_choice"] = option_key(s_choice_text)
                out["kev_shuffle_agree"] = (out["kev_shuffle_choice"] == choice_key)
            write_jsonl_append(f, [out])
            n_done += 1
            if n_done % 10 == 0 or n_done == len(pairs):
                elapsed = time.time() - t0
                print(f"scored {n_done}/{len(pairs)}; {elapsed:.1f}s elapsed, "
                      f"{elapsed / n_done:.3f}s/point avg", flush=True)

    print(f"done: {n_done} points scored -> {args.out}, {time.time() - t0:.1f}s total", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd")

    sp = sub.add_parser("score")
    sp.add_argument("--traces", required=True)
    sp.add_argument("--points", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--run", default=DEFAULT_RUN)
    sp.add_argument("--device", default="cuda")
    sp.add_argument("--resume", action="store_true")
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--shuffle-check", action="store_true")

    args = ap.parse_args()
    if args.cmd == "score":
        return sys.exit(run_score(args))
    ap.error("'score' is the only subcommand")


if __name__ == "__main__":
    main()
