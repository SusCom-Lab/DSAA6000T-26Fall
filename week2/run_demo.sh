#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ ! -f .env ]]; then
  echo "Missing week2/.env. Copy env.example to .env and edit it first." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source ./.env
set +a

# Only commands that start the GPU service need hardware discovery.
export GPU_DEVICE=nvidia.com/gpu=unresolved
case "${1:-up}" in
  up|restore)
    GPU_DEVICE=$(python3 router.py)
    export GPU_DEVICE ;;
esac
case "${1:-up}" in
  build) docker compose build ;;
  up) docker compose up -d ;;
  cpu-only) docker compose up -d cpu-backend router ;;
  status) docker compose ps ;;
  logs) docker compose logs --tail 50 -f ;;
  down) docker compose down ;;
  fallback) docker compose stop gpu-backend ;;
  restore) docker compose up -d gpu-backend ;;
  route)
    curl --fail-with-body -sS "http://${ROUTER_BIND_HOST:-127.0.0.1}:${ROUTER_PORT:-8080}/generate" \
      -H 'Content-Type: application/json' \
      -d '{"prompt":"Explain why GPU memory matters.","max_new_tokens":16}'
    echo ;;
  *) echo "Usage: $0 {build|up|cpu-only|status|logs|route|fallback|restore|down}" >&2; exit 2 ;;
esac
