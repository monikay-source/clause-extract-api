# Contract Clause Extraction — Setup & Usage

## What this is

A LoRA-fine-tuned language model, served behind a FastAPI REST API, that extracts a
`clause_type` (one of `termination`, `payment_terms`, `confidentiality`, `renewal`,
`leave_benefits`) and a short `extracted_value` phrase from contract clause text.

## One-command reproduction

```bash
bash /app/solution.sh
```

This trains the model, starts the server, waits for it to become healthy, and runs
sample requests against every endpoint.

## Training

```bash
python3 /app/train.py
```

Reads `/app/data/train_clauses.jsonl`, fine-tunes the base model at `/opt/base_model`
with LoRA, and saves the adapter + tokenizer + `training_config.json` to
`/app/model/adapter/`. Takes a few minutes on CPU. Safe to re-run from scratch at any
time — it always overwrites `/app/model/adapter/`.

## Serving

```bash
python3 /app/serve.py
```

Starts the API on `http://0.0.0.0:8000`. Requires `/app/model/adapter/` to already
exist (run training first). Every request is logged as a JSON line to
`/app/logs/api.log`.

## API usage

**Health check**

```bash
curl http://127.0.0.1:8000/health
```

**Single extraction**

```bash
curl -X POST http://127.0.0.1:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Section 4. Termination. This Agreement may be terminated by either party upon 30 days written notice."}'
```

**Batch extraction (1-32 items)**

```bash
curl -X POST http://127.0.0.1:8000/extract/batch \
  -H "Content-Type: application/json" \
  -d '{"items": [{"text": "..."}, {"text": "..."}]}'
```

## Retraining from scratch

Delete the adapter directory and re-run training, then restart the server:

```bash
rm -rf /app/model/adapter
python3 /app/train.py
# stop any running server process, then:
python3 /app/serve.py
```

## Troubleshooting

- **`/health` never reports ready`**: check that `/app/model/adapter/` exists and
  contains `adapter_config.json` — this means training did not complete or was never
  run.
- **Server fails to start with an import error**: confirm `schemas.py` is in the same
  directory as `serve.py` (`/app/`) so the `from schemas import ...` import resolves.
- **`/extract` returns `clause_type: "unknown"` often**: the model's generation could
  not be parsed as valid JSON; re-check that `serve.py` is using the same prompt
  template recorded in `/app/model/adapter/training_config.json` that `train.py` used.
- **422 on a request that looks valid**: check text length (max 2000 characters) and,
  for batch requests, that `items` has between 1 and 32 entries.
