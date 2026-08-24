#!/usr/bin/env bash
# End-to-end reproduction: train -> start API -> wait for /health -> sample requests.
set -euo pipefail

APP_DIR="/app"
PORT=8000
HEALTH_URL="http://127.0.0.1:${PORT}/health"

echo "[1/4] Training model (this writes /app/model/adapter/)..."
python3 "${APP_DIR}/train.py"

echo "[2/4] Starting API server on port ${PORT}..."
mkdir -p "${APP_DIR}/logs"
nohup python3 "${APP_DIR}/serve.py" > "${APP_DIR}/logs/server.out" 2>&1 &
SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

echo "[3/4] Waiting for /health to report ready..."
for i in $(seq 1 60); do
  if curl -s -f "${HEALTH_URL}" | grep -q '"status":"ready"'; then
    echo "Server is ready."
    break
  fi
  if [ "$i" -eq 60 ]; then
    echo "Server did not become ready in time." >&2
    exit 1
  fi
  sleep 2
done

echo "[4/4] Sending sample requests..."
echo "--- GET /health ---"
curl -s "${HEALTH_URL}"
echo
echo "--- POST /extract ---"
curl -s -X POST "http://127.0.0.1:${PORT}/extract" \
  -H "Content-Type: application/json" \
  -d '{"text": "Section 4. Termination. This Agreement may be terminated by either party upon 30 days written notice."}'
echo
echo "--- POST /extract/batch ---"
curl -s -X POST "http://127.0.0.1:${PORT}/extract/batch" \
  -H "Content-Type: application/json" \
  -d '{"items": [
        {"text": "Section 2. Payment Terms. Payment shall be made within 45 days of invoice receipt."},
        {"text": "Section 9. Renewal. This Agreement shall automatically renew for successive periods of 12 months."}
      ]}'
echo
echo "Done."
