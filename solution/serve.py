#!/usr/bin/env python3
"""
FastAPI service that loads /opt/base_model + the LoRA adapter saved at
/app/model/adapter/ and serves clause extraction.

Run with:  python3 /app/serve.py
(or:       uvicorn serve:app --host 0.0.0.0 --port 8000)
"""
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from schemas import (
    ExtractRequest,
    ExtractResponse,
    BatchExtractRequest,
    BatchExtractResponse,
    HealthResponse,
    MAX_BATCH_ITEMS,
)

BASE_MODEL_PATH = "/opt/base_model"
ADAPTER_DIR = "/app/model/adapter"
LOG_PATH = "/app/logs/api.log"
VALID_CLAUSE_TYPES = {
    "termination",
    "payment_terms",
    "confidentiality",
    "renewal",
    "leave_benefits",
}

state = {"model": None, "tokenizer": None, "cfg": None, "ready": False}

# --- structured request logging (never logs raw clause text) ---
Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
api_logger = logging.getLogger("clause_api")
api_logger.setLevel(logging.INFO)
_handler = logging.FileHandler(LOG_PATH)
_handler.setFormatter(logging.Formatter("%(message)s"))
api_logger.addHandler(_handler)


def log_request(endpoint: str, status: int, latency_ms: float, n_items: int = 1) -> None:
    api_logger.info(
        json.dumps(
            {
                "timestamp": time.time(),
                "endpoint": endpoint,
                "status": status,
                "latency_ms": round(latency_ms, 2),
                "n_items": n_items,
            }
        )
    )


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batched generation below

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()

    cfg_path = Path(ADAPTER_DIR) / "training_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {
        "prompt_template": "### Contract Clause:\n{text}\n### Extraction:\n",
        "stop_string": "\n### Contract Clause:",
    }
    return model, tokenizer, cfg


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, tokenizer, cfg = load_model()
    state["model"] = model
    state["tokenizer"] = tokenizer
    state["cfg"] = cfg
    state["ready"] = True
    yield


app = FastAPI(lifespan=lifespan)

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_completion(raw: str, stop_string: str):
    """Best-effort parse of the model's completion into (clause_type, extracted_value, confidence)."""
    cut = raw.split(stop_string)[0] if stop_string in raw else raw
    match = _JSON_RE.search(cut)
    if match:
        try:
            obj = json.loads(match.group(0))
            clause_type = str(obj.get("clause_type", "")).strip()
            extracted_value = str(obj.get("extracted_value", "")).strip()
            if clause_type in VALID_CLAUSE_TYPES and extracted_value:
                return clause_type, extracted_value, 0.9
            if clause_type in VALID_CLAUSE_TYPES:
                return clause_type, extracted_value, 0.5
        except (json.JSONDecodeError, AttributeError):
            pass
    # Fallback: never return malformed/empty output on a 200.
    return "unknown", "", 0.1


@torch.no_grad()
def run_batch_inference(texts):
    model, tokenizer, cfg = state["model"], state["tokenizer"], state["cfg"]
    prompts = [cfg["prompt_template"].format(text=t) for t in texts]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    gen = model.generate(
        **enc,
        max_new_tokens=40,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    # padding_side="left" -> every prompt ends at the same column, so new tokens
    # for row i are exactly gen[i, enc["input_ids"].shape[1]:]
    prompt_len = enc["input_ids"].shape[1]
    new_tokens = gen[:, prompt_len:]
    completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [parse_completion(c, cfg.get("stop_string", "")) for c in completions]


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ready" if state["ready"] else "loading",
        adapter_loaded=state["ready"],
    )


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    start = time.time()
    try:
        clause_type, extracted_value, confidence = run_batch_inference([req.text])[0]
    except Exception:
        log_request("/extract", 500, (time.time() - start) * 1000)
        raise HTTPException(status_code=500, detail="inference failed")
    log_request("/extract", 200, (time.time() - start) * 1000)
    return ExtractResponse(
        clause_type=clause_type, extracted_value=extracted_value, confidence=confidence
    )


@app.post("/extract/batch", response_model=BatchExtractResponse)
def extract_batch(req: BatchExtractRequest):
    start = time.time()
    if not req.items or len(req.items) > MAX_BATCH_ITEMS:
        log_request("/extract/batch", 422, (time.time() - start) * 1000, len(req.items))
        raise HTTPException(status_code=422, detail="items must contain 1-32 entries")
    try:
        outputs = run_batch_inference([it.text for it in req.items])
    except Exception:
        log_request("/extract/batch", 500, (time.time() - start) * 1000, len(req.items))
        raise HTTPException(status_code=500, detail="inference failed")
    log_request("/extract/batch", 200, (time.time() - start) * 1000, len(req.items))
    return BatchExtractResponse(
        results=[
            ExtractResponse(clause_type=ct, extracted_value=ev, confidence=cf)
            for ct, ev, cf in outputs
        ]
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    log_request(str(request.url.path), 500, 0.0)
    return JSONResponse(status_code=500, content={"detail": "internal error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
