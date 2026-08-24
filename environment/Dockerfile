FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# --- Python dependencies (shared by agent runtime and verifier) ---
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        transformers==4.44.2 \
        peft==0.12.0 \
        accelerate==0.33.0 \
        datasets==2.21.0 \
        fastapi==0.112.2 \
        uvicorn[standard]==0.30.6 \
        pydantic==2.8.2 \
        httpx==0.27.2 \
        pytest==8.3.2 \
        pytest-ctrf==0.3.0

# --- Bake the base model into the image at build time (author-side network use;
#     the agent needs zero internet access at inference time) ---
RUN python3 -c "\
from transformers import AutoModelForCausalLM, AutoTokenizer; \
tok = AutoTokenizer.from_pretrained('distilgpt2'); \
tok.pad_token = tok.eos_token; \
tok.save_pretrained('/opt/base_model'); \
model = AutoModelForCausalLM.from_pretrained('distilgpt2'); \
model.save_pretrained('/opt/base_model')"

WORKDIR /app
RUN mkdir -p /app/data /app/model /app/logs

# --- Training data available to the agent (NOT the held-out eval set, which
#     lives only in tests/ and is added at verify time) ---
COPY data/train_clauses.jsonl /app/data/train_clauses.jsonl

CMD ["/bin/bash"]
