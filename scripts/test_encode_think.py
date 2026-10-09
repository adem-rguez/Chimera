"""CPU-only (tokenizer only) smoke test for scripts/train_phase7.py's encode_think (Phase 7R T9).

Covers oracle/phase7r-built/stageA-000.jsonl .. stageA-010.jsonl (built rows: target_text_short/long,
masked_spans_short, user, calls, id -- see build_phase7r.py's assemble_row). For a handful of rows
(including at least one multi-call row) this:
  - encodes both variants (short/long) and asserts no row is silently dropped (unless it genuinely
    exceeds --max-len or --user-turn-cap, which is reported, not asserted away),
  - decodes the masked vs supervised token spans and checks: every char position inside
    masked_spans_short (short) / [0, prompt_end] (long) is masked end to end; the typed trigger
    (`<|decide:TYPE|>`) and the final `<turn|>` are NOT masked,
  - prints a handful of masked/supervised decoded samples for human inspection,
  - reports the max sequence length seen,
  - runs collate() on a small batch of encoded rows.

Usage (laptop, CPU, tokenizer only):
    python scripts/test_encode_think.py
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.train_phase7 import collate, encode_think  # noqa: E402

MODEL = "google/gemma-4-E4B-it"
REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"


def load_rows():
    paths = sorted(glob.glob(os.path.join(ROOT, "oracle/phase7r-built/stageA-0[0-1][0-9].jsonl")))
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rows.extend(json.loads(l) for l in f if l.strip())
    return paths, rows


def get_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"SKIP: transformers not importable ({e})")
        return None
    try:
        return AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    except Exception as e:  # noqa: BLE001 -- report and bail, don't crash the whole test file
        print(f"SKIP: could not load tokenizer {MODEL}@{REVISION} (local cache only, no network "
              f"assumed): {e}")
        return None


def char_is_masked(spans, pos):
    return any(s <= pos < e for s, e in spans)


def check_row(tok, rec, variant, max_len, user_turn_cap):
    row = encode_think(tok, rec, max_len=max_len, variant=variant, user_turn_cap=user_turn_cap)
    text = rec[f"target_text_{variant}"]
    if row is None:
        # independently verify this is a genuine over-cap row, not a silent drop
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        user_ids = tok(rec["user"], add_special_tokens=False)["input_ids"]
        over_len = len(enc["input_ids"]) > max_len
        over_user = len(user_ids) > user_turn_cap
        assert over_len or over_user, (
            f"{rec['id']} ({variant}): encode_think returned None but row is within caps "
            f"(len={len(enc['input_ids'])}, user_len={len(user_ids)}) -- SILENT DROP")
        return None, f"skipped (over_len={over_len} over_user_turn_cap={over_user})"

    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc["offset_mapping"]
    assert len(row["input_ids"]) == len(offs) == len(row["labels"])

    prompt_end = rec["masked_spans_short"][0][1]
    spans = rec["masked_spans_short"] if variant == "short" else [[0, prompt_end]]

    # every token whose char span overlaps a masked span must be labels=-100, and vice versa
    for i, (a, b) in enumerate(offs):
        should_mask = any(a < e and b > s for s, e in spans)
        is_masked = row["labels"][i].item() == -100
        assert should_mask == is_masked, (
            f"{rec['id']} ({variant}) token {i} offs=({a},{b}) should_mask={should_mask} "
            f"got masked={is_masked}")

    # trigger tokens (for short, call-bearing rows) must be supervised
    if variant == "short" and rec["calls"]:
        for c in rec["calls"]:
            trig = f"<|decide:{c['type']}|>"
            idx = text.find(trig)
            assert idx >= 0, f"{rec['id']}: trigger {trig!r} not found in target_text_short"
            # find token(s) covering the trigger's char range and assert at least one is supervised
            trig_end = idx + len(trig)
            trig_tok_labels = [row["labels"][i].item() for i, (a, b) in enumerate(offs)
                                if a < trig_end and b > idx]
            assert trig_tok_labels, f"{rec['id']}: no tokens found covering trigger {trig!r}"
            assert any(l != -100 for l in trig_tok_labels), (
                f"{rec['id']}: trigger {trig!r} tokens are all masked (-100): {trig_tok_labels}")

    # final <turn|> must be supervised
    last_turn_idx = text.rfind("<turn|>")
    assert last_turn_idx >= 0, f"{rec['id']} ({variant}): no <turn|> found at all"
    turn_end = last_turn_idx + len("<turn|>")
    turn_tok_labels = [row["labels"][i].item() for i, (a, b) in enumerate(offs)
                        if a < turn_end and b > last_turn_idx]
    assert turn_tok_labels, f"{rec['id']} ({variant}): no tokens found covering final <turn|>"
    assert all(l != -100 for l in turn_tok_labels), (
        f"{rec['id']} ({variant}): final <turn|> token(s) masked: {turn_tok_labels}")

    n_masked = int((row["labels"] == -100).sum())
    n_sup = len(row["labels"]) - n_masked
    return row, f"seq_len={len(row['input_ids'])} masked={n_masked} supervised={n_sup}"


def decode_sample(tok, rec, variant, max_len=1536, user_turn_cap=350):
    row = encode_think(tok, rec, max_len=max_len, variant=variant, user_turn_cap=user_turn_cap)
    if row is None:
        return
    text = rec[f"target_text_{variant}"]
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc["offset_mapping"]
    masked_chars, sup_chars = [], []
    for i, (a, b) in enumerate(offs):
        (masked_chars if row["labels"][i].item() == -100 else sup_chars).append(text[a:b])
    print(f"    masked sample:     {''.join(masked_chars)[:160]!r}")
    print(f"    supervised sample: {''.join(sup_chars)[:160]!r}")


def main():
    tok = get_tokenizer()
    if tok is None:
        print("RESULT: SKIPPED (tokenizer unavailable)")
        return 0

    paths, rows = load_rows()
    print(f"loaded {len(rows)} rows from {len(paths)} files: {[os.path.basename(p) for p in paths]}")

    multi_call_rows = [r for r in rows if len(r["calls"]) > 1]
    single_call_rows = [r for r in rows if len(r["calls"]) == 1]
    nocall_rows = [r for r in rows if len(r["calls"]) == 0]
    print(f"calls-per-row: multi={len(multi_call_rows)} single={len(single_call_rows)} "
          f"none={len(nocall_rows)}")
    assert multi_call_rows, "expected at least one multi-call row in stageA-000..010, found none"

    sample = (multi_call_rows[:3] + single_call_rows[:3] + nocall_rows[:2])
    max_len, user_turn_cap = 1536, 350

    max_seq_len = 0
    n_checked = n_skipped = 0
    kept_rows = []
    for rec in sample:
        for variant in ("short", "long"):
            row, msg = check_row(tok, rec, variant, max_len, user_turn_cap)
            n_checked += 1
            if row is None:
                n_skipped += 1
                print(f"  {rec['id']} ({variant}) calls={len(rec['calls'])}: {msg}")
                continue
            kept_rows.append(row)
            max_seq_len = max(max_seq_len, len(row["input_ids"]))
            print(f"  {rec['id']} ({variant}) calls={len(rec['calls'])}: {msg}")

    print(f"\nmax sequence length observed: {max_seq_len} tokens (cap --max-len {max_len})")
    print(f"checked {n_checked} (row, variant) pairs, {n_skipped} skipped (over caps)")

    print("\ndecoded samples (one multi-call row, short variant):")
    decode_sample(tok, multi_call_rows[0], "short")
    if nocall_rows:
        print("decoded samples (one control_nocall row, short variant):")
        decode_sample(tok, nocall_rows[0], "short")

    assert kept_rows, "no rows survived encoding -- cannot test collate()"
    batch = collate(tok, kept_rows[: min(4, len(kept_rows))])
    print(f"\ncollate() OK: input_ids {tuple(batch['input_ids'].shape)}, "
          f"labels {tuple(batch['labels'].shape)}, attention_mask {tuple(batch['attention_mask'].shape)}, "
          f"opt_mask {tuple(batch['opt_mask'].shape)}, kinds={set(batch['kinds'])}")

    print("\nRESULT: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
