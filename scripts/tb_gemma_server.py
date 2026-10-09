"""
Minimal OpenAI-compatible chat completions server for Terminal-Bench smoke
testing with base Gemma-4-E4B-it.

Endpoints:
  POST /v1/chat/completions
  GET  /v1/models

Batches concurrent requests (up to --batch-size within --batch-window-ms) into a
single left-padded generate() call. Honours messages -> chat template with
enable_thinking configurable via --thinking on|off (default on). The 'content'
field returned to the client is ONLY the final answer (thinking text is
stripped out and logged separately). temperature/top_p come from the request
or CLI defaults; temperature 0 (or unset with default 0) means greedy decoding.
max_tokens caps total generated tokens (thinking + answer), default 2048.

--fake mode returns a canned reply without loading any model, for pipeline
testing on CPU / without a GPU.

Every request is logged as one JSON line to --log-file with timings, token
counts, and whether truncation occurred.

Usage:
  python tb_gemma_server.py --fake --port 8000 --log-file /tmp/server.jsonl
  python tb_gemma_server.py --model google/gemma-... --thinking on --port 8000 \
      --log-file ~/Chimera/oracle/tbench-server.jsonl
"""

import argparse
import json
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG_LOCK = threading.Lock()
LOG_FILE = None


def log_jsonl(record: dict) -> None:
    if LOG_FILE is None:
        return
    with LOG_LOCK:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Model wrapper (real or fake)
# ---------------------------------------------------------------------------

class FakeModel:
    """Returns a canned reply; no heavy deps required. Runs on CPU."""

    def __init__(self):
        self.tokenizer = None

    def generate_batch(self, requests):
        """requests: list of dict(messages, max_tokens, temperature, top_p, thinking)
        Returns list of dict(answer, thinking, prompt_tokens, completion_tokens,
        thinking_tokens, truncated)."""
        results = []
        for req in requests:
            last_user = ""
            for m in reversed(req["messages"]):
                if m.get("role") == "user":
                    last_user = m.get("content", "")
                    break
            thinking_text = ""
            if req["thinking"]:
                thinking_text = f"[fake-thinking] considering: {last_user[:60]!r}"
            answer = (
                "FAKE_RESPONSE: echo(" + last_user[:120].replace("\n", " ") + ")"
            )
            results.append(
                {
                    "answer": answer,
                    "thinking": thinking_text,
                    "prompt_tokens": max(1, len(last_user.split())),
                    "completion_tokens": len(answer.split()),
                    "thinking_tokens": len(thinking_text.split()),
                    "truncated": False,
                }
            )
        return results


class GemmaModel:
    """Real HF transformers Gemma-4-E4B-it model, loaded once, bf16 + sdpa."""

    def __init__(self, model_name: str, default_thinking: bool):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.default_thinking = default_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda",
        )
        self.model.eval()

    # Verified against scripts/gen_think_traces.py (reports/17): these are plain
    # decoded-text markers on this tokenizer, not special tokens, and
    # THINK_CLOSE has no required preceding "\n".
    THINK_OPEN = "<|channel>thought\n"
    THINK_CLOSE = "<channel|>"

    def _split_thinking(self, text: str):
        """Split model output into (thinking, answer). Channel never opens
        (thinking ignored by model) -> ("", text). Opens but never closes
        (ran out of tokens) -> (everything after open, "")."""
        start = text.find(self.THINK_OPEN)
        if start < 0:
            return "", text.strip()
        body_start = start + len(self.THINK_OPEN)
        end = text.find(self.THINK_CLOSE, body_start)
        if end < 0:
            return text[body_start:].strip(), ""
        return text[body_start:end].strip(), text[end + len(self.THINK_CLOSE):].strip()

    def generate_batch(self, requests):
        torch = self.torch
        tok = self.tokenizer
        prompts = []
        for req in requests:
            enable_thinking = req["thinking"]
            try:
                prompt = tok.apply_chat_template(
                    req["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                # tokenizer/template doesn't support enable_thinking kwarg
                prompt = tok.apply_chat_template(
                    req["messages"], tokenize=False, add_generation_prompt=True
                )
            prompts.append(prompt)

        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()

        max_tokens = max(req["max_tokens"] for req in requests)
        # Use greedy decoding if ALL requests in the batch want temperature 0,
        # otherwise sample with the max temperature/top_p in the batch (batches
        # are small smoke-test batches so this coarse approach is acceptable).
        temps = [req["temperature"] for req in requests]
        do_sample = any(t > 0 for t in temps)
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            pad_token_id=tok.pad_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = max(temps) or 1.0
            gen_kwargs["top_p"] = max(req["top_p"] for req in requests)

        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)

        results = []
        input_len = enc["input_ids"].shape[1]
        for i, req in enumerate(requests):
            gen_ids = out[i][input_len:]
            full_text = tok.decode(gen_ids, skip_special_tokens=True)
            thinking_text, answer = (
                self._split_thinking(full_text) if req["thinking"] else ("", full_text)
            )
            completion_tokens_total = int((gen_ids != tok.pad_token_id).sum().item())
            thinking_tokens = (
                len(tok(thinking_text, add_special_tokens=False)["input_ids"])
                if thinking_text
                else 0
            )
            answer_tokens = max(0, completion_tokens_total - thinking_tokens)
            truncated = completion_tokens_total >= req["max_tokens"]
            results.append(
                {
                    "answer": answer,
                    "thinking": thinking_text,
                    "prompt_tokens": int(prompt_lens[i]),
                    "completion_tokens": answer_tokens,
                    "thinking_tokens": thinking_tokens,
                    "truncated": truncated,
                }
            )
        return results


# ---------------------------------------------------------------------------
# Batching dispatcher
# ---------------------------------------------------------------------------

class BatchDispatcher:
    def __init__(self, model, batch_size: int, batch_window_ms: int):
        self.model = model
        self.batch_size = batch_size
        self.batch_window_s = batch_window_ms / 1000.0
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, req: dict) -> dict:
        result_box = {}
        event = threading.Event()
        self._q.put((req, result_box, event))
        event.wait()
        return result_box["result"]

    def _loop(self):
        while True:
            item = self._q.get()  # blocks for first item
            batch = [item]
            deadline = time.time() + self.batch_window_s
            while len(batch) < self.batch_size:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._q.get(timeout=remaining))
                except queue.Empty:
                    break

            reqs = [b[0] for b in batch]
            try:
                results = self.model.generate_batch(reqs)
            except Exception as e:
                for _, box, event in batch:
                    box["result"] = {"error": str(e)}
                    event.set()
                continue

            for (req, box, event), result in zip(batch, results):
                box["result"] = result
                event.set()


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

DISPATCHER = None
DEFAULT_THINKING = True
DEFAULT_MAX_TOKENS = 2048
MODEL_ID = "gemma-4-e4b-it"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet; we log structured jsonl ourselves

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self._send_json(
                {
                    "object": "list",
                    "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}],
                }
            )
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send_json({"error": "not found"}, status=404)
            return

        t0 = time.time()
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid json"}, status=400)
            return

        messages = payload.get("messages", [])
        temperature = payload.get("temperature", 0.0)
        if temperature is None:
            temperature = 0.0
        top_p = payload.get("top_p", 1.0)
        if top_p is None:
            top_p = 1.0
        max_tokens = payload.get("max_tokens") or DEFAULT_MAX_TOKENS
        thinking = DEFAULT_THINKING

        req = {
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "thinking": thinking,
        }

        result = DISPATCHER.submit(req)
        t1 = time.time()

        if "error" in result:
            log_jsonl(
                {
                    "ts": t0,
                    "error": result["error"],
                    "latency_s": t1 - t0,
                    "messages": messages,
                }
            )
            self._send_json({"error": result["error"]}, status=500)
            return

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        response = {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(t1),
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["answer"]},
                    "finish_reason": "length" if result["truncated"] else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "completion_thinking_tokens": result["thinking_tokens"],
                "total_tokens": result["prompt_tokens"]
                + result["completion_tokens"]
                + result["thinking_tokens"],
            },
        }

        log_jsonl(
            {
                "ts": t0,
                "id": completion_id,
                "latency_s": round(t1 - t0, 3),
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "thinking_tokens": result["thinking_tokens"],
                "truncated": result["truncated"],
                "thinking_text": result["thinking"],
                "n_messages": len(messages),
            }
        )

        self._send_json(response)


def main():
    global DISPATCHER, DEFAULT_THINKING, LOG_FILE, DEFAULT_MAX_TOKENS

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--thinking", choices=["on", "off"], default="on")
    ap.add_argument("--fake", action="store_true", help="skip model load, CPU-safe canned replies")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--batch-window-ms", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--log-file", required=True)
    args = ap.parse_args()

    DEFAULT_THINKING = args.thinking == "on"
    DEFAULT_MAX_TOKENS = args.max_tokens
    LOG_FILE = args.log_file

    if args.fake:
        model = FakeModel()
    else:
        model = GemmaModel(args.model, DEFAULT_THINKING)

    DISPATCHER = BatchDispatcher(model, args.batch_size, args.batch_window_ms)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving on {args.host}:{args.port} (fake={args.fake}, thinking={args.thinking})")
    server.serve_forever()


if __name__ == "__main__":
    main()
