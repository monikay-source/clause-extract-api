# Proposal: Fine-tune and Serve Contract-Clause Extraction

## 1. The Difficulty

This project builds a complete machine-learning delivery pipeline for contract clauses.
It fine-tunes a small causal language model with LoRA, then serves the trained adapter
through a validated FastAPI REST service.

The service extracts two fields from each clause:

- `clause_type`: termination, payment_terms, confidentiality, renewal, or leave_benefits
- `extracted_value`: the short numeric phrase copied from the clause

The API provides single extraction, genuine batched extraction, health reporting,
input validation, structured logging, and privacy protection.

The key engineering challenge is preserving one contract between training and serving.
The prompt template, target JSON format, tokenizer settings, truncation behavior, and
generation rules must remain consistent across separate processes.

The adapter must also remain separate from the frozen base model. Training saves LoRA
weights independently, and serving composes them with the base model using PEFT.
This proves that the service uses the trained artifact rather than the untouched model.

Generation is free-form, so the server extracts JSON safely from model output. Invalid
or truncated output receives a deterministic typed fallback rather than malformed JSON.

Batching is implemented through one tokenizer call and one `model.generate()` call.
The results are decoded in input order, with exactly one result per valid input item.

Every request is logged with timestamp, endpoint, status, latency, and item count.
Raw contract text is never written to the log, protecting potentially sensitive data.

## 2. Intended Approach

The synthetic dataset contains 300 labeled records, with 60 examples for each class.
Each record includes `text`, `clause_type`, and `extracted_value`.

The generator varies parties, section numbers, filler sentences, and numeric values.
The five template families are termination, payment terms, confidentiality, renewal,
and leave benefits.

The held-out evaluation set contains 50 records created with another seed and a value
offset of 1000. This prevents simple memorization of the training examples and tests
generalization to unseen values, parties, and sections.

The base model is `distilgpt2`, loaded from `/opt/base_model`. LoRA uses rank 8,
alpha 16, dropout 0.05, the GPT-2 `c_attn` target module, and causal-language-model
training. The target loss is masked so the model learns the extraction response.

`train.py` loads the JSONL data and builds prompts in this form:

```text
### Contract Clause:
{clause text}
### Extraction:
{target JSON}
```

It trains for six epochs and saves the adapter, tokenizer, and `training_config.json`
to `/app/model/adapter/`. The configuration records the prompt and stop string used
by both stages, preventing silent prompt drift.

`serve.py` loads the base model once at startup, applies the adapter with PEFT, and
marks `/health` ready only after loading finishes. `schemas.py` owns all Pydantic
request and response validation.

The server accepts text from 1 to 2000 characters. Batch requests accept 1 to 32
items. Invalid input returns HTTP 422. Successful responses contain exactly the
documented fields and a confidence value from 0 to 1.

The scripts select CUDA automatically when a CUDA-enabled PyTorch installation exists;
otherwise they use CPU. The Docker image intentionally installs CPU-only PyTorch.
For Colab GPU execution, install CUDA-enabled PyTorch and run the scripts directly.

The Dockerfile installs the pinned Python dependencies, downloads `distilgpt2` into
`/opt/base_model`, copies the training data, and creates the application directories.

The complete reproduction script is `/app/solution.sh`. It trains the model, starts
the API in the background, waits for `/health`, and sends sample single and batch
requests.

The normal workflow is:

1. Build the image from `environment/`.
2. Run the solution pipeline or copy the source files into `/app`.
3. Run `python3 /app/train.py`.
4. Start `python3 /app/serve.py` on port 8000.
5. Open `/docs` or call the REST endpoints with curl.

For local browser access, publish the container port with `-p 8000:8000`.
The interactive documentation is then available at `http://localhost:8000/docs`.

## 3. How It Will Be Verified

`tests/test_outputs.py` verifies required files, adapter structure, API readiness,
response schemas, validation errors, batch count, batch ordering, accuracy, and logs.

The required accuracy thresholds are 0.70 for `clause_type` and 0.50 for
`extracted_value` across all 50 held-out records.

The logging test requires valid JSON lines containing status and latency fields and
checks that held-out clause text never appears verbatim in `api.log`.

The tested implementation passed all 14 verifier tests. Training completed in about
13 minutes in the CPU container, and the API returned ready status with the adapter
loaded. A sample termination request returned `termination` and `30 days`.

## 4. Category and Value Justification

The repository provides `train.py`, `serve.py`, `schemas.py`, `solution.sh`,
`instructions.md`, the Docker environment, deterministic datasets, and tests.

This is an ML platform and API integration project rather than only a model exercise.
It demonstrates parameter-efficient fine-tuning, artifact management, reproducible
training, production-style inference, batching, validation, observability, privacy,
and independent functional verification.

The design is suitable for CPU execution and can be accelerated in Colab or another
CUDA environment without changing the model-training or API contract.
