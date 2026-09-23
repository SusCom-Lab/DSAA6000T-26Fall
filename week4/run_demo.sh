#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

case "${1:-help}" in
  check|capabilities) ;;
  help|-h|--help)
    echo 'Usage: ./run_demo.sh {check|capabilities} [demo.py options]'
    echo 'Optional: copy env.example to .env to select Python and one GPU.'
    exit 0 ;;
  *) echo "Unknown command: $1" >&2; exit 2 ;;
esac

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
exec "${PYTHON:-python3}" demo.py "$@"
