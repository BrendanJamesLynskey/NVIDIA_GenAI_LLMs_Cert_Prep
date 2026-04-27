#!/usr/bin/env bash
# NOTE: written but not hardware-verified — smoke-test before use
#
# smoke_test.sh
# -------------
# End-to-end automated test for the Triton serving demo.
#
# 1. Checks the ONNX file exists (prompts to run prepare_model.py if not).
# 2. Starts Triton with docker compose.
# 3. Polls the readiness endpoint until the server is up (or times out).
# 4. Runs client.py.
# 5. Tears down.
#
# Exit codes:
#   0 — all steps succeeded
#   1 — any step failed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_PATH="${SCRIPT_DIR}/model_repository/resnet18_onnx/1/model.onnx"
TRITON_HTTP="http://localhost:8000/v2/health/ready"
POLL_INTERVAL=5   # seconds between readiness polls
MAX_WAIT=120      # seconds before giving up

# --------------------------------------------------------------------------- #
echo "=== Step 1: check ONNX model ==="
if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "ERROR: ${ONNX_PATH} not found."
  echo "Run: python prepare_model.py"
  exit 1
fi
echo "ONNX model found: ${ONNX_PATH}"

# --------------------------------------------------------------------------- #
echo ""
echo "=== Step 2: start Triton ==="
cd "${SCRIPT_DIR}"
docker compose up -d
echo "Triton container started."

# --------------------------------------------------------------------------- #
echo ""
echo "=== Step 3: wait for readiness ==="
elapsed=0
until curl -sf "${TRITON_HTTP}" > /dev/null 2>&1; do
  if (( elapsed >= MAX_WAIT )); then
    echo "ERROR: Triton did not become ready within ${MAX_WAIT}s."
    echo "Container logs:"
    docker compose logs --tail=40 triton
    docker compose down
    exit 1
  fi
  echo "  Waiting … (${elapsed}s elapsed)"
  sleep "${POLL_INTERVAL}"
  elapsed=$(( elapsed + POLL_INTERVAL ))
done
echo "Triton is ready."

# --------------------------------------------------------------------------- #
echo ""
echo "=== Step 4: run client ==="
python "${SCRIPT_DIR}/client.py"

# --------------------------------------------------------------------------- #
echo ""
echo "=== Step 5: tear down ==="
docker compose down
echo ""
echo "Smoke test PASSED."
