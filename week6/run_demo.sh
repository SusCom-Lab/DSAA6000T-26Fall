#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

case "${1:-help}" in
  build|check) ;;
  help|-h|--help)
    echo 'Usage: ./run_demo.sh {build|check}'
    echo 'Optional: copy env.example to .env to select GPUs and library paths.'
    exit 0 ;;
  *) echo "Unknown command: $1" >&2; exit 2 ;;
esac
if (( $# != 1 )); then
  echo 'Usage: ./run_demo.sh {build|check}' >&2
  exit 2
fi

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,1}"
cuda_dir="${CUDA_HOME:-/usr/local/cuda}"
library_flags=()
if [[ -n "${NCCL_HOME:-}" ]]; then
  library_flags+=("-I${NCCL_HOME}/include" "-L${NCCL_HOME}/lib"
    "-L${NCCL_HOME}/lib64" "-Wl,-rpath,${NCCL_HOME}/lib"
    "-Wl,-rpath,${NCCL_HOME}/lib64")
fi
mkdir -p build
"${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra demo.cpp -o build/demo \
  "-I${cuda_dir}/include" "-L${cuda_dir}/lib64" \
  "-Wl,-rpath,${cuda_dir}/lib64" "${library_flags[@]}" -lnccl -lcudart
if [[ "$1" == check ]]; then
  exec ./build/demo check
fi
