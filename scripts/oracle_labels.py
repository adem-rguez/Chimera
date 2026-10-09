"""Phase 5 oracle-label run: every pool example through both endpoints.

Pool = decision-v7 dev (decide-candidates, gold labels in suite) + oracle/open_ended.jsonl
(generate-candidates, no labels). Per example records correctness, decision confidence,
cost (ms both sides, chat tokens), and the oracle label: `decide` only if decision mode
is correct on every question, confident (min confidence >= --min-conf), and cheaper;
else `generate`. Generation correctness is NOT judged here (gen_correct null, raw text
stored) -- that is the judge pass, CHECK 4A method (stronger model + spot checks).

Env: DECIDE_URL (kev.serve, e.g. http://...:8009), CHAT_URL (vLLM OpenAI-compatible).
Out: oracle/labels-v1.jsonl. Stdlib only.
"""
import argparse
import glob
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# stdlib only: read the frozen partition directly, no kev imports (kev.suite drags torch).

DECIDE_URL = os.environ.get("DECIDE_URL", "http://127.0.0.1:8009").rstrip("/")
CHAT_URL = os.environ.get("CHAT_URL", "http://127.0.0.1:8010").rstrip("/")


def post(url, payload, timeout=300):
    t0 = time.perf_counter()
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    return body, (time.perf_counter() - t0) * 1000


def strip_labels(record):
    return {"state": record["state"], "model": "kev-latest", "questions": {
        qid: {k: v for k, v in q.items() if k not in ("label", "src")}
        for qid, q in record["questions"].items()}}


def decide_correct(ans, q):
    y = q["label"]
    if q["type"] == "choice":
        return ans["choice"] == y
    if q["type"] == "noul":
        return (ans["noul"] >= 0.5) == bool(y)
    return int(round(ans["score"])) == int(y)


def run_decide(record):
    body, ms = post(DECIDE_URL + "/v1/systemone", strip_labels(record))
    answers = body.get("answers", body)
    per_q, confs = {}, []
    for qid, q in record["questions"].items():
        a = answers[qid]
        ok = decide_correct(a, q)
        c = a.get("confidence")
        if c is None:  # noul answers carry P(yes); confidence is in the predicted answer
            c = max(a["noul"], 1 - a["noul"])
        per_q[qid] = {"correct": ok, "confidence": c}
        confs.append(c or 0)
    in_tok = (body.get("usage") or {}).get("input_tokens")
    return per_q, min(confs), ms, in_tok


def format_options(q):
    """Render one question's instructions + criteria as text options (spec v2:
    chat-with-options must see the same question/options decide saw, no gold label)."""
    crit = q.get("criteria")
    if crit is None and q["type"] == "noul":  # noul: true/false, no criteria key
        opts = "- true\n- false"
        return f"{q['instructions']}\nOptions:\n{opts}"
    if isinstance(crit, dict):  # choice (intent-style) or noul (true/false)
        opts = "\n".join(f"- {k}" + (f": {v}" if v else "") for k, v in crit.items())
    else:  # score: ordered list of labels, pick by name
        opts = "\n".join(f"- {v}" for v in crit)
    return f"{q['instructions']}\nOptions:\n{opts}"


def build_options_prompt(record):
    state = record["state"] if isinstance(record["state"], str) else json.dumps(record["state"])
    parts = [state, ""]
    first_qid = next(iter(record["questions"]))
    for qid, q in record["questions"].items():
        parts.append(f"[{qid}] {format_options(q)}")
    parts.append("\nAnswer each question above. Answer each question on its own line as "
                  f"`[<id>] <option>`, for example `[{first_qid}] some_option`, using the real "
                  "question id in place of <id> and picking exactly one option per question.")
    return "\n".join(parts)


def run_chat(text, max_tokens=256):
    body, ms = post(CHAT_URL + "/v1/chat/completions", {
        "model": os.environ.get("CHAT_MODEL", "google/gemma-4-E4B-it"),
        "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens})
    try:
        out = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        out = json.dumps(body)[:2000]
    return out, (body.get("usage") or {}).get("total_tokens"), ms


PROTECTED_OUT = {os.path.abspath(p) for p in
                  ("oracle/labels-v1.jsonl", "oracle/labels-smoke.jsonl")}


def run_chat_options_mode(a):
    """Spec v2 chat-with-options pass: re-run CHAT_URL on already-labeled dev rows,
    this time showing it the question + options (v1 never did). Judge pass (scoring
    the picks against gold) is separate and not done here -- raw text only."""
    dev = glob.glob("evals/v7/decision-v7/development.jsonl")[0]
    with open(dev, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    with open(a.labels_in, encoding="utf-8") as f:
        labeled = [json.loads(line) for line in f if line.strip()]
    labeled = [r for r in labeled if r["kind"] == "decision"]
    if a.limit or a.offset:
        labeled = labeled[a.offset:(a.offset + a.limit) if a.limit else None]
    print(f"chat-options pool: {len(labeled)} labeled dev rows from {a.labels_in}")

    def label_row(row):
        i = int(row["id"].split("-")[1])
        text = build_options_prompt(recs[i])
        gen_text, chat_tok, c_ms = run_chat(text, a.max_tokens)
        return {"id": row["id"], "kind": "decision-chat-options",
                "oracle_v1": row["oracle"], "chat_ms": round(c_ms, 1),
                "chat_tokens": chat_tok, "gen_correct": None, "gen_text": gen_text[:2000]}

    with ThreadPoolExecutor(max_workers=a.workers) as ex, \
            open(a.out, "a" if a.append else "w", encoding="utf-8") as f:
        for out_row in ex.map(label_row, labeled):
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n"); f.flush()
    print(f"wrote {a.out}: {len(labeled)} chat-with-options rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["label", "chat-options"], default="label",
                     help="label: original decide+chat pool run. chat-options: spec v2 "
                          "pass -- re-run chat with options shown on already-labeled rows")
    ap.add_argument("--out", default=None)
    ap.add_argument("--labels-in", default="oracle/labels-v1.jsonl",
                     help="chat-options mode: source labeled dev rows (kind=decision)")
    ap.add_argument("--min-conf", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--offset", type=int, default=0, help="skip first N decision records")
    ap.add_argument("--oes-only", action="store_true", help="label only the open-ended set")
    ap.add_argument("--append", action="store_true")
    a = ap.parse_args()
    if a.out is None:
        a.out = "oracle/labels-v1.jsonl" if a.mode == "label" else "oracle/labels-chat-options.jsonl"
    if a.mode == "chat-options":
        if os.path.abspath(a.out) in PROTECTED_OUT:
            raise SystemExit(f"refusing to overwrite protected file: {a.out}")
        run_chat_options_mode(a)
        return

    dev = glob.glob("evals/v7/decision-v7/development.jsonl")[0]
    with open(dev, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    with open("oracle/open_ended.jsonl", encoding="utf-8") as f:
        oes = [json.loads(line) for line in f if line.strip()]
    if a.oes_only:
        recs = []
    elif a.limit or a.offset:
        recs = recs[a.offset:(a.offset + a.limit) if a.limit else None]
        oes = []
    print(f"pool: {len(recs)} decide-candidates + {len(oes)} open-ended")

    n_dec = 0
    gen_all = "generate"

    def label_dev(args):
        i, r = args
        i += a.offset
        per_q, conf, d_ms, in_tok = run_decide(r)
        all_ok = all(v["correct"] for v in per_q.values())
        chat_text, chat_tok, c_ms = run_chat(
            r["state"] if isinstance(r["state"], str) else json.dumps(r["state"]), a.max_tokens)
        oracle = "decide" if (all_ok and conf >= a.min_conf and d_ms < c_ms) else gen_all
        return {"id": f"dev-{i}", "kind": "decision", "per_q": per_q,
                "decide_conf": round(conf, 4), "decide_ms": round(d_ms, 1),
                "chat_ms": round(c_ms, 1), "chat_tokens": chat_tok,
                "gen_correct": None, "gen_text": chat_text[:2000], "oracle": oracle}

    def label_oe(o):
        chat_text, chat_tok, c_ms = run_chat(o["text"], a.max_tokens)
        return {"id": o["id"], "kind": "open-ended", "gen_correct": None,
                "gen_text": chat_text[:2000], "chat_ms": round(c_ms, 1),
                "chat_tokens": chat_tok, "oracle": gen_all}

    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(a.out, "a" if a.append else "w", encoding="utf-8") as f:
        for row in ex.map(label_dev, enumerate(recs)):
            n_dec += row["oracle"] == "decide"
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
        for row in ex.map(label_oe, oes):
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
    print(f"wrote {a.out}: decide={n_dec}/{len(recs)} open-ended all generate")


if __name__ == "__main__":
    main()
