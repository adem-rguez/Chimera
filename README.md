# Chimera-LD: Research Release — Hybrid LLM + Option-Decision Head

Chimera-LD combines Gemma-4-E4B-it (instruction-tuned) with a pointer-head decision model for typed option selection (yes/no, multiple choice, rating). The decision head was ported from the open-source Kev project and adapted to run on Gemma. Three modes are supported: decision-only (fast classification without generation), LLM-only (standard chat), and auto (experimental routing based on confidence gate).

## Headline Result

**This release does NOT outperform base Gemma-4-E4B-it on standard benchmarks and does not reduce tokens at equal quality.** It is a research artifact documenting negative results, a working option-decision classifier, and the source code for our experiments.

## What Works

- **Option-decision head on Gemma-E4B**: in-domain test accuracy 0.825 (on 600 held-out points); out-of-domain 0.628. For reference, the original Kev-4B head achieves 0.865 in-domain and 0.835 out-of-domain, showing our Gemma port is usable in-domain but clearly weaker in transfer.
- **Inline typed decisions in reasoning channel**: five decision types (claim_handling, order_outcome, answer_type, kb_category, news_topic). Trigger rate 93.7% on 440 natural prompts (93.3% correct type), 90.0% on 250 out-of-distribution prompts (82.0% correct type). Token savings of approximately 198 tokens (natural) and 257 tokens (OOD) per example versus plain thinking.
- **Quality penalty measured**: versus a fair baseline (Gemma-E4B-it with thinking + options listed in prompt), accuracy declined 4.0 percentage points on natural data (89.7% vs 93.7%, p=0.012) and 5.6 pp on OOD (80.4% vs 86.0%, p=0.06). No fallback for incorrect triggers was implemented.

## What Did Not Work

| Attempt | Observation | Accuracy vs Baseline |
|---------|-------------|---------------------|
| Replacing reasoning text with decisions | 6% of Gemma-E4B thinking tokens (~390 median) was replaceable decision text; coverage 35% vs 40% needed; head agreement 0.72 vs 0.85 needed | N/A |
| Compact decision-diagram format, few-shot | 7-9% token cost of thinking | GSM8K 0.535 vs 0.870; MMLU 0.780 vs 0.860; ARC 0.893 vs 0.953 |
| Decision model as step planner (GSM8K) | Model chooses 'finish' at step 0 | Kev-4B 1.5%, our head 1.0% |
| Recursive ask-organ (GSM8K, 200 problems) | No token saving, model's picker adds no value | Our head 0.540 vs Kev-4B 0.630 vs direct answer 0.765 vs thinking 0.870 |
| Router skipping thinking (400 MMLU + 400 ARC) | Thinking adds almost nothing on multiple choice for this model | Letter readout 0.799 vs thinking 0.804; GSM8K cannot be routed (19% correct without reasoning) |
| Decision-planned code (HumanEval, 164 problems) | Model writes design, head picks, model writes code | Kev-4B 0.780, our head 0.720, letter-readout 0.726 vs thinking 0.793 |

## Why

Gemma-4-E4B thinks briefly (median ~390 tokens). On multiple choice and code, answering without reasoning already matches reasoning-augmented output. Math tasks require numerical computation that a classifier-style decision head cannot perform. Gemma tends to write its most plausible candidate first, leaving little margin for a picker to improve.

## Repo Map

- **kev/**: decision head model, training and serving code (adapted from Kev, ported from Qwen to Gemma)
  - `model.py`: Gemma-4-E4B loading, pointer head, attention mask handling
  - `train.py`: training loop, LoRA setup, think-calls and replay flags for inline-decision training
  - `serve.py`: inference server, gate rule (confidence threshold T=0.6112)
- **scripts/**: experiment, probe, and evaluation scripts (many read data files under oracle/ or evals/ that are NOT included; tests/ may fail without external eval suites)
- **tests/**: unit tests
- **modal_app.py**: Modal runner for kev studies (from upstream Kev)
- **pyproject.toml**, **uv.lock**: Python dependencies

## Reproducing

Most scripts read data files (oracle/, evals/) and adapter checkpoints (runs/) not included in this release. To retrain:

1. Install dependencies: `uv sync --extra serve`
2. Fetch dataset splits from upstream (e.g., GSM8K, MMLU) or use your own
3. Adapt `scripts/build_phase7r_hybrid.py` (templated data generation) and `kev/train.py` with your paths
4. Model weights for Gemma-4-E4B-it must be fetched from Hugging Face (access required)

## Attribution & Licenses

- Decision-head code in `kev/` descends from Kev (https://github.com/jaredpalmer/kev, Apache-2.0; see LICENSE). It was modified to run on Gemma instead of Qwen.
- Gemma-4 model weights are not redistributed; their use is subject to Google's terms for that model.
- Evaluation datasets (GSM8K, MMLU, ARC, HumanEval) belong to their respective owners.
- All modifications and new experimental code (hybrid design, auto gate, inline-decision training, Phase 7R pipeline) are provided under the same Apache-2.0 license.

## Limitations & Future Work

- A larger-model variant (e.g., Gemma-27B) is an untested hypothesis only; we make no claims of benefit.
- Robustness against control-token override attacks: approximately 1 in 7 attempts (13.8% targeted) succeeded against the inline-decision design.
- The decision head is weaker out-of-domain; retraining on mixed OOD data or per-source calibration may help, but was not tested.
