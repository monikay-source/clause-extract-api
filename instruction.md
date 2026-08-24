You must fine-tune a small language model to extract structured fields from contract
clauses, then serve that fine-tuned model through a REST API. Both the fine-tuning and
the serving stages will be graded, and the API must actually be backed by the model you
fine-tuned, not the untouched base model.

A pretrained base model is already available on disk at /opt/base_model (a causal
language model, saved in standard Hugging Face format: config, tokenizer, and weights).
Do not download a different base model and do not require internet access at inference
time. A labeled training dataset is available at /app/data/train_clauses.jsonl, one JSON
object per line with the keys "text", "clause_type", and "extracted_value".
"clause_type" is always one of: termination, payment_terms, confidentiality, renewal,
leave_benefits. "extracted_value" is always a short phrase copied from within "text"
(for example "45 days" or "3 years").

Fine-tune the base model using a parameter-efficient method (e.g. LoRA) so that, given a
new, unseen contract clause of the same general style, it predicts the correct
clause_type and extracted_value. Write your training code to /app/train.py. Running
`python3 /app/train.py` with no arguments must, from a clean state, train the model
using /app/data/train_clauses.jsonl and save the resulting adapter plus tokenizer to the
directory /app/model/adapter/ (create it if missing). Do not merge the adapter into the
base model weights on disk — the adapter must be saved separately and re-loadable on top
of /opt/base_model.

Write a FastAPI application to /app/serve.py that, when run, loads the base model from
/opt/base_model and applies your saved adapter from /app/model/adapter/, then serves on
0.0.0.0:8000. It must expose:

- GET /health — returns HTTP 200 with a JSON body containing a "status" field. Return a
  value indicating the model is ready only after the adapter has actually finished
  loading.
- POST /extract — body is a JSON object with a single field "text" (string). Returns
  HTTP 200 with a JSON body containing exactly the fields: "clause_type" (string),
  "extracted_value" (string), and "confidence" (a number between 0 and 1). If "text" is
  missing, empty, or exceeds 2000 characters, return HTTP 422 instead of a 200.
- POST /extract/batch — body is a JSON object with a field "items", a list of 1 to 32
  objects each shaped like the /extract request body. Returns HTTP 200 with a JSON body
  containing a field "results", a list of objects shaped like the /extract response,
  one per input item, in the same order as the input. If "items" is empty or has more
  than 32 entries, return HTTP 422.

Define all request/response models in /app/schemas.py using Pydantic, and import them
from serve.py (do not redefine the same validation logic twice).

Every request to /extract or /extract/batch must append one JSON line to
/app/logs/api.log containing at least the fields "timestamp", "endpoint", "status", and
"latency_ms" (a number). Do not write the raw contract clause text into this log file in
any field.

Write /app/solution.sh: a single bash script that, run from a clean container with
nothing else done yet, performs the full pipeline end to end — trains the model, starts
the API server in the background, waits until GET /health reports ready, sends at least
one sample request to /extract and one to /extract/batch, and prints their responses.
It must exit 0 on success.

Write /app/instructions.md: a short human-readable markdown document explaining how to
run training, how to start the server, example curl commands for each endpoint, how to
retrain from scratch, and what to check if the server fails to start.

Your model does not need to be perfect, but it must generalize beyond the exact examples
in the training file — it will be evaluated against contract clauses it has not seen
during training, of the same five clause types and general phrasing style but with
different specific numbers, party names, and section numbers. Serving the untouched base
model, or hardcoding outputs for fixed inputs, will not generalize and will fail
grading.

You have 5400 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
