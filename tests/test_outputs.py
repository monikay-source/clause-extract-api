"""
Verifier for the clause-extraction fine-tune + serve task.

Does NOT execute or trust /app/solution.sh. Imports the agent's serve.py
directly and drives it through FastAPI's TestClient (which runs the real
`lifespan` startup, i.e. actually loads the base model + the agent's saved
adapter) so every check below is against the agent's real, running service.
"""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

APP_DIR = Path("/app")
EVAL_PATH = Path(__file__).parent / "eval_clauses.jsonl"

REQUIRED_FILES = [
    "train.py",
    "serve.py",
    "schemas.py",
    "solution.sh",
    "instructions.md",
]

CLAUSE_ACCURACY_MIN = 0.70
VALUE_ACCURACY_MIN = 0.50


def load_eval_records():
    records = []
    with open(EVAL_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Static artifact checks
# ---------------------------------------------------------------------------

def test_required_files_exist():
    """All five required deliverable files exist under /app."""
    missing = [name for name in REQUIRED_FILES if not (APP_DIR / name).is_file()]
    assert not missing, f"Missing required files under /app: {missing}"


def test_adapter_directory_exists_and_is_valid():
    """/app/model/adapter is present and looks like a real, loadable PEFT adapter."""
    adapter_dir = APP_DIR / "model" / "adapter"
    assert adapter_dir.is_dir(), "/app/model/adapter does not exist"
    assert (adapter_dir / "adapter_config.json").is_file(), (
        "adapter_config.json missing -- this does not look like a saved PEFT adapter "
        "(merging into the base model or skipping fine-tuning would fail this check)"
    )
    tokenizer_files = list(adapter_dir.glob("tokenizer*")) + list(adapter_dir.glob("vocab*"))
    assert tokenizer_files, "no tokenizer files found alongside the adapter"


def test_solution_script_is_executable_bash():
    """solution.sh is a non-empty bash script."""
    content = (APP_DIR / "solution.sh").read_text()
    assert content.strip(), "solution.sh is empty"
    assert content.splitlines()[0].startswith("#!"), "solution.sh is missing a shebang line"


# ---------------------------------------------------------------------------
# Live service checks (module-scoped: load the model once, reuse across tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    sys.path.insert(0, str(APP_DIR))
    import importlib

    serve = importlib.import_module("serve")
    with TestClient(serve.app) as c:
        yield c


def test_health(client):
    """GET /health reports the service and adapter are ready."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert body["status"] == "ready"


def test_extract_valid_schema(client):
    """POST /extract on a valid clause returns exactly the documented schema."""
    resp = client.post(
        "/extract",
        json={"text": "Section 4. Termination. This Agreement may be terminated "
                       "by either party upon 30 days written notice."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"clause_type", "extracted_value", "confidence"}
    assert isinstance(body["clause_type"], str)
    assert isinstance(body["extracted_value"], str)
    assert isinstance(body["confidence"], (int, float))
    assert 0.0 <= body["confidence"] <= 1.0


def test_extract_empty_text_rejected(client):
    """Empty text is a validation error, not a 200 or a 500."""
    resp = client.post("/extract", json={"text": ""})
    assert resp.status_code == 422


def test_extract_missing_field_rejected(client):
    """A malformed request body (missing 'text') is a validation error."""
    resp = client.post("/extract", json={})
    assert resp.status_code == 422


def test_extract_oversized_text_rejected(client):
    """Text over 2000 characters is rejected."""
    resp = client.post("/extract", json={"text": "x" * 2001})
    assert resp.status_code == 422


def test_batch_count_matches_input(client):
    """POST /extract/batch returns exactly one result per input item."""
    items = [
        {"text": "Section 2. Payment Terms. Payment shall be made within 45 days "
                  "of invoice receipt."},
        {"text": "Section 9. Renewal. This Agreement shall automatically renew for "
                  "successive periods of 12 months."},
        {"text": "Section 1. Confidentiality. The Receiving Party shall keep all "
                  "information confidential for 5 years."},
    ]
    resp = client.post("/extract/batch", json={"items": items})
    assert resp.status_code == 200
    body = resp.json()
    assert "results" in body
    assert len(body["results"]) == len(items)


def test_batch_order_matches_single_extract(client):
    """Batch results are in input order (checked against independent single calls)."""
    texts = [
        "Section 3. Leave Benefits. Each employee shall be entitled to 15 days of "
        "paid leave per calendar year.",
        "Section 4. Termination. This Agreement may be terminated by either party "
        "upon 20 days written notice.",
    ]
    single_results = [
        client.post("/extract", json={"text": t}).json() for t in texts
    ]
    batch_result = client.post(
        "/extract/batch", json={"items": [{"text": t} for t in texts]}
    ).json()["results"]

    for single, batched in zip(single_results, batch_result):
        assert batched["clause_type"] == single["clause_type"]
        assert batched["extracted_value"] == single["extracted_value"]


def test_batch_empty_items_rejected(client):
    """An empty items list is a validation error."""
    resp = client.post("/extract/batch", json={"items": []})
    assert resp.status_code == 422


def test_batch_oversized_items_rejected(client):
    """More than 32 items in one batch request is a validation error."""
    items = [{"text": "Section 1. Termination. 10 days notice required."}] * 33
    resp = client.post("/extract/batch", json={"items": items})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Held-out functional accuracy (tolerance band -- see proposal.md Section 3)
# ---------------------------------------------------------------------------

def test_generalization_accuracy(client):
    """
    clause_type and extracted_value accuracy on a held-out set (disjoint random
    seed AND disjoint numeric-value range from the training data) must clear a
    tolerance band. See proposal.md for why a band, not an exact number, is used.
    """
    records = load_eval_records()
    assert len(records) == 50

    predictions = []
    chunk = 25
    for i in range(0, len(records), chunk):
        batch = records[i : i + chunk]
        resp = client.post(
            "/extract/batch", json={"items": [{"text": r["text"]} for r in batch]}
        )
        assert resp.status_code == 200
        predictions.extend(resp.json()["results"])

    assert len(predictions) == len(records)

    clause_correct = sum(
        1 for r, p in zip(records, predictions) if p["clause_type"] == r["clause_type"]
    )
    value_correct = sum(
        1 for r, p in zip(records, predictions)
        if p["extracted_value"].strip() == r["extracted_value"].strip()
    )
    clause_acc = clause_correct / len(records)
    value_acc = value_correct / len(records)

    assert clause_acc >= CLAUSE_ACCURACY_MIN, (
        f"clause_type accuracy {clause_acc:.2f} below required {CLAUSE_ACCURACY_MIN} "
        "-- the served model does not appear to be genuinely fine-tuned"
    )
    assert value_acc >= VALUE_ACCURACY_MIN, (
        f"extracted_value accuracy {value_acc:.2f} below required {VALUE_ACCURACY_MIN}"
    )


# ---------------------------------------------------------------------------
# Logging checks
# ---------------------------------------------------------------------------

def test_log_file_structured_and_private(client):
    """
    api.log has one JSON line per request with status + latency_ms, and never
    contains the raw text of held-out clause examples verbatim.
    """
    # Generate at least one fresh logged request.
    client.post("/extract", json={"text": "Section 1. Termination. 10 days notice."})

    log_path = APP_DIR / "logs" / "api.log"
    assert log_path.is_file(), "/app/logs/api.log does not exist"

    lines = [l for l in log_path.read_text().splitlines() if l.strip()]
    assert lines, "api.log is empty"

    parsed = 0
    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        assert "status" in obj, f"log line missing 'status': {line}"
        assert "latency_ms" in obj, f"log line missing 'latency_ms': {line}"
        parsed += 1
    assert parsed > 0, "no valid JSON lines found in api.log"

    log_text = log_path.read_text()
    records = load_eval_records()
    for r in records[:10]:
        assert r["text"] not in log_text, (
            "raw clause text found verbatim in api.log -- request logging must not "
            "store raw contract text"
        )
