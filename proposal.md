# Proposal: Fine-tune + Serve a Contract-Clause Extraction Model Behind a REST API

## 1. The Difficulty

The task requires building a complete train-to-deploy ML pipeline: parameter-efficient
fine-tuning (LoRA) of a small causal language model to extract structured fields from
contract clauses, followed by standing up an independent, production-style FastAPI
service that loads the *trained artifact* (not the base model) and serves it correctly
under validation, batching, and error conditions.

The reasoning difficulty is not "can a model be trained" — it is keeping training and
serving consistent across a process boundary. An agent has to get right, simultaneously:

- Prompt/format consistency. Whatever prompt template and target-JSON format the
  model was trained on must be reproduced *exactly* at inference time, including
  tokenizer special tokens, truncation behavior, and generation stop conditions. A
  mismatch here doesn't crash the service — it silently degrades output quality, which
  is a much harder failure mode to catch than a stack trace.
- **Artifact separation.** The adapter (LoRA weights) must be saved separately from the
  frozen base model and re-composed at serve time via `PeftModel.from_pretrained`,
  proving the agent understands PEFT's save/load contract rather than just calling
  `.save_pretrained()` on a merged model and hoping.
- **Robust generation parsing.** The model emits free-form text; the API must parse it
  into a strict JSON contract, handle truncated/malformed generations gracefully, and
  return a deterministic typed fallback when the generation cannot be parsed.
- **API correctness under load shape variation.** Single vs. batched inference share a
  model but not necessarily a code path — batching has to actually batch (one forward
  pass class, not a Python loop dressed up as batching) or the agent has to justify why
  not, and either way every batch item must map to exactly one result in order.
- **Operational hygiene.** Structured logging of latency/status without leaking raw
  contract text (a privacy-adjacent constraint that's easy to overlook when you're
  focused on making the model work).

This is exactly the job of an **ML platform / MLOps engineer** taking a data scientist's
fine-tuning notebook and turning it into a service other teams can call — a very common
and highly transferable real-world responsibility distinct from either pure ML research
or pure backend work.

**Data.** The dataset is synthetic contract clauses generated from five clause-type
template families (termination, payment_terms, confidentiality, renewal,
leave_benefits), each with randomized party names, section numbers, boilerplate filler
sentences, and a randomized numeric value embedded in a fixed unit phrase (e.g. "45
days"). Synthetic data is the right choice here because (a) real contracts are
confidential and licensing-encumbered, and (b) the task is evaluating the *engineering
pipeline*, not novel clause-understanding capability — the template family gives a
learnable, well-defined ground truth needed for deterministic grading. It is realistically
challenging in the sense that matters for this task: the surface text varies substantially
(different parties, section numbers, filler sentences, phrasing order) around the signal,
so the model must actually learn the association between keywords and clause_type/value
rather than pattern-match a fixed string, and the held-out evaluation set uses a
disjoint value range and a different random seed than the training set, so memorizing
training examples verbatim does not transfer.

## 2. The Intended Approach

**Key insight:** decouple "does the agent understand LoRA fine-tuning" from "does the
agent understand serving a trained artifact" by making both individually inspectable —
train.py produces artifacts on disk with a well-defined contract (adapter dir +
tokenizer dir + a small `training_config.json` recording the prompt template used),
and serve.py's *only* job is to correctly consume that contract. This is also why the
prompt/target format has to be written to disk alongside the adapter rather than
hardcoded identically in both files — it forces the agent to treat format-consistency
as a real interface, not an accident of copy-pasting the same string twice.

High-level pipeline:

```
environment/data/train_clauses.jsonl (baked into image, 300 records)
        │
        ▼
train.py
  - load JSONL, build instruction-style prompts:
      "### Contract Clause:\n{text}\n### Extraction:\n{json_target}\n"
  - tokenize with base model tokenizer (pad token = eos)
  - LoraConfig(r=8, alpha=16, target_modules=["c_attn"], task_type=CAUSAL_LM)
    over distilgpt2, a small causal LM that is tractable on CPU and can also use
    CUDA when the scripts are run directly in a Colab GPU runtime
  - HF Trainer, a handful of epochs over the small dataset
  - save PEFT adapter + tokenizer to /app/model/adapter/
  - save /app/model/adapter/training_config.json recording the exact prompt
    template and generation stop string, so serve.py never has to guess it
        │
        ▼
serve.py (FastAPI)
  - on startup: load base model once, wrap with
    PeftModel.from_pretrained(base, "/app/model/adapter")
  - read training_config.json to reconstruct the identical prompt template
  - select CUDA automatically when available, otherwise use CPU
  - POST /extract: validate via schemas.py, build prompt, generate, and parse JSON
    out of the completion with a deterministic unknown-value fallback
  - POST /extract/batch: tokenizer(..., padding=True) across the whole list in
    one model.generate() call, then split decoded outputs back per-item
  - GET /health: reports "ready" only once the adapter is actually loaded
  - structured logging: one JSON line per request with timestamp, endpoint,
    status, latency_ms, and item count — never the raw clause text itself
```

**Best-case expert time estimate:** ~4 focused hours for a senior ML platform
engineer already fluent in HF Transformers/PEFT and FastAPI — roughly 1.5h for the
dataset + train.py, 2h for serve.py/schemas.py including the batching and error-path
edge cases, 30min for solution.sh/instructions.md and end-to-end smoke testing.

## 3. How It Will Be Verified

Verification is a mix of artifact/schema checks (fast, always deterministic) and a
functional accuracy check against a held-out set (has a tolerance band, explained
below).

The agent must produce, under `/app`:

- `/app/train.py`, `/app/serve.py`, `/app/schemas.py`, `/app/solution.sh`,
  `/app/instructions.md`
- `/app/model/adapter/` containing the saved LoRA adapter + tokenizer +
  `training_config.json`
- A running (or independently re-startable via `serve.py`) FastAPI service on a
  fixed port, and `/app/logs/api.log` containing structured request logs

The verifier (`tests/test_outputs.py`) independently starts the service by importing
`serve.py`'s app (it does **not** trust or execute the agent's `solution.sh`) and checks:

1. All five required files exist under `/app`.
2. `/app/model/adapter/` exists and is loadable as a PEFT adapter (schema/file check,
   not a trust-the-agent check).
3. `GET /health` returns `200` with a `status` field indicating the adapter is loaded.
4. `POST /extract` on a valid clause returns `200` with exactly the keys
   `clause_type`, `extracted_value`, `confidence` (types checked: str, str, float in
   [0,1]).
5. `POST /extract/batch` with N inputs returns exactly N results, in input order
   (checked via a per-item marker embedded in each synthetic input).
6. `POST /extract` with empty text, and a batch exceeding the documented size limit,
   both return `422`, not `200` or `500`.
7. **Accuracy band:** running all 50 held-out records from `tests/eval_clauses.jsonl`
   (generated with a disjoint seed and a disjoint numeric-value range from the training
   set — see Section 1) through `/extract`, `clause_type` exact-match accuracy must be
   **≥ 0.70** and `extracted_value` exact-match accuracy must be **≥ 0.50**.
   *Why a band, not an exact number:* LoRA training on a tiny CPU-friendly model has
   run-to-run variance even with a fixed seed (nondeterministic kernels, floating-point
   reduction order), so demanding a specific accuracy value would make the task flaky
   through no fault of the agent's engineering. The band is calibrated so that (a) the
   reference solution clears it comfortably and repeatably (typically ≥0.9 clause_type
   accuracy), (b) a no-op agent, an agent that hardcodes a single label, or an agent
   that serves the *untuned* base model (i.e., skipped fine-tuning entirely) scores at
   or near random-chance/near-zero and fails, and (c) the band rejects "wrong method"
   (no real fine-tuning happened) while tolerating "right method, unlucky training run."
8. `api.log` contains one JSON object per request with `status` and `latency_ms` keys,
   and does **not** contain the raw clause text of any held-out example verbatim
   (privacy-logging check).

`tests/test.sh` runs the standard pytest suite and writes `1`/`0` to
`/logs/verifier/reward.txt` based on the exit code; it installs nothing (all
dependencies — `torch`, `transformers`, `peft`, `fastapi`, `httpx`, `pytest` — are baked
into `environment/Dockerfile`, shared by agent and verifier).

The Docker image uses the CPU-only PyTorch wheel. The Python scripts automatically use
CUDA when available, so Colab GPU execution requires a CUDA-enabled PyTorch install and
running the scripts directly rather than using this CPU-only image.

## 4. Category & Sub-category Justification

This task belongs to **Fine-tuning + API Integration** (not "pure fine-tuning" and not
"pure backend API") because neither half is gradable in isolation and success requires
correctly bridging them: a perfectly-trained adapter that `serve.py` fails to load
correctly scores the same as no training at all (fails the accuracy band, since /extract
would be answering from an untuned or broken model), and a beautifully-built FastAPI
service that generates from the base model instead of the adapter equally fails. The
verifier is deliberately structured so the accuracy check is a *joint* function of
training quality and serving correctness — you cannot pass by faking either stage. That
joint dependency, plus the presence of both a genuine PEFT training loop and a genuine
multi-endpoint validated REST API with batching/logging/error-handling, is what
distinguishes this from a single-category ML or backend task.
