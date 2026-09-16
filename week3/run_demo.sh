#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

action="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$action" in
  benchmark|check) ;;
  help|-h|--help)
    echo 'Usage: ./run_demo.sh {benchmark|check} [demo.py options]'
    echo 'Create .env from env.example first. Run check before the class demo.'
    exit 0 ;;
  *) echo "Unknown command: $action" >&2; exit 2 ;;
esac

if [[ ! -f .env ]]; then
  echo 'Missing week3/.env. Copy env.example to .env and select your GPU.' >&2
  exit 1
fi
set -a
source .env
set +a

exec "${PYTHON:-python3}" demo.py "$action" "$@"
