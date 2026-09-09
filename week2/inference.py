#!/usr/bin/env python3
"""Probe or run the complete Llama model on one explicit backend."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any
from functools import lru_cache

import torch


RESULT_PREFIX = "DEMO_JSON="


def emit(payload: dict[str, Any]) -> None:
    print(RESULT_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def memory_info() -> tuple[int | None, int | None]:
    try:
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw_value = line.split(":", maxsplit=1)
            values[key] = int(raw_value.strip().split()[0]) * 1024
        return values.get("MemTotal"), values.get("MemAvailable")
    except (OSError, ValueError, IndexError):
        return None, None


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", maxsplit=1)[1].strip()
    except (OSError, IndexError):
        pass
    return platform.processor() or platform.machine()


def probe_bfloat16(device: torch.device) -> tuple[bool, str]:
    try:
        value = torch.ones((16, 16), dtype=torch.bfloat16, device=device)
        _ = value @ value
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return True, "BF16 matrix multiplication succeeded"
    except (RuntimeError, NotImplementedError) as exc:
        return False, f"BF16 probe failed: {type(exc).__name__}: {exc}"


def probe_backend(backend: str, cpu_threads: int) -> dict[str, Any]:
    if backend == "cpu":
        torch.set_num_threads(cpu_threads)
        total_bytes, free_bytes = memory_info()
        supported, detail = probe_bfloat16(torch.device("cpu"))
        return {
            "status": "ok",
            "backend": "cpu",
            "available": True,
            "device": "cpu",
            "hardware_name": cpu_name(),
            "total_memory_bytes": total_bytes,
            "free_memory_bytes": free_bytes,
            "bfloat16_supported": supported,
            "cpu_threads": torch.get_num_threads(),
            "detail": detail,
        }

    if not torch.cuda.is_available():
        return {
            "status": "ok",
            "backend": "gpu",
            "available": False,
            "device": "cuda:0",
            "hardware_name": "not detected",
            "total_memory_bytes": None,
            "free_memory_bytes": None,
            "bfloat16_supported": False,
            "detail": "torch.cuda.is_available() returned False",
        }

    try:
        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        supported, detail = probe_bfloat16(device)
        return {
            "status": "ok",
            "backend": "gpu",
            "available": True,
            "device": "cuda:0",
            "hardware_name": properties.name,
            "total_memory_bytes": total_bytes,
            "free_memory_bytes": free_bytes,
            "bfloat16_supported": supported,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "detail": detail,
        }
    except (RuntimeError, AssertionError) as exc:
        return {
            "status": "ok",
            "backend": "gpu",
            "available": False,
            "device": "cuda:0",
            "hardware_name": "probe failed",
            "total_memory_bytes": None,
            "free_memory_bytes": None,
            "bfloat16_supported": False,
            "detail": f"{type(exc).__name__}: {exc}",
        }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def eos_token_ids(value: int | list[int] | None) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return set(value)


def greedy_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, float, float, float]:
    """Generate greedily while measuring prefill/first-token and decode time."""

    generated: list[torch.Tensor] = []
    stop_ids = eos_token_ids(model.generation_config.eos_token_id)

    synchronize(device)
    generation_start = time.perf_counter()
    first_start = generation_start
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        synchronize(device)
        ttft_seconds = time.perf_counter() - first_start
        generated.append(next_token)
        past_key_values = outputs.past_key_values

        decode_start = time.perf_counter()
        for _ in range(1, max_new_tokens):
            if int(next_token.item()) in stop_ids:
                break
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=device,
                    ),
                ),
                dim=1,
            )
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = outputs.past_key_values
            generated.append(next_token)
        synchronize(device)
        decode_seconds = time.perf_counter() - decode_start

    total_seconds = time.perf_counter() - generation_start
    return torch.cat(generated, dim=1), ttft_seconds, decode_seconds, total_seconds


@lru_cache(maxsize=1)
def load_model(backend: str, model_directory: str, cpu_threads: int):
    if backend == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU backend requested, but CUDA is unavailable")
        device = torch.device("cuda:0")
    else:
        torch.set_num_threads(cpu_threads)
        device = torch.device("cpu")

    supported, detail = probe_bfloat16(device)
    if not supported:
        raise RuntimeError(detail)

    model_path = Path(model_directory)
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"missing model configuration: {model_path / 'config.json'}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    tokenizer_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=True,
    )
    tokenizer_load_seconds = time.perf_counter() - tokenizer_start

    print(f"Loading the complete model on {device}...", file=sys.stderr, flush=True)
    model_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    synchronize(device)
    model_load_seconds = time.perf_counter() - model_start

    return model, tokenizer, device, tokenizer_load_seconds, model_load_seconds


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    backend = args.backend
    if args.simulate_failure:
        raise RuntimeError(f"simulated {backend} execution failure")
    model, tokenizer, device, tokenizer_load_seconds, model_load_seconds = load_model(
        backend, args.model_path, args.cpu_threads
    )

    messages = [
        {"role": "system", "content": "Answer concisely."},
        {"role": "user", "content": args.prompt},
    ]
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = tokenized["input_ids"].to(device)
    if input_ids.shape[1] > 512:
        raise ValueError("Prompt exceeds the demo limit of 512 input tokens")
    attention_mask = tokenized.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)
    else:
        attention_mask = attention_mask.to(device)

    if args.warmup:
        warmup_ids = input_ids[:, -min(8, input_ids.shape[1]) :]
        with torch.inference_mode():
            model(input_ids=warmup_ids, use_cache=False)
        synchronize(device)

    generated_ids, ttft_seconds, decode_seconds, total_seconds = greedy_generate(
        model,
        input_ids,
        attention_mask,
        args.max_new_tokens,
        device,
    )
    generated_count = generated_ids.shape[1]
    decoded = tokenizer.decode(generated_ids[0].cpu(), skip_special_tokens=True)

    if device.type == "cuda":
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        hardware_name = torch.cuda.get_device_name(device)
    else:
        # ru_maxrss is KiB on Linux.
        peak_memory_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        total_bytes, free_bytes = memory_info()
        hardware_name = cpu_name()

    decode_token_count = max(0, generated_count - 1)
    decode_tokens_per_second = (
        decode_token_count / decode_seconds
        if decode_token_count > 0 and decode_seconds > 0
        else None
    )
    return {
        "status": "ok",
        "backend": backend,
        "device": str(device),
        "hardware_name": hardware_name,
        "dtype": "bfloat16",
        "input_tokens": input_ids.shape[1],
        "generated_tokens": generated_count,
        "tokenizer_load_seconds": tokenizer_load_seconds,
        "model_load_seconds": model_load_seconds,
        "time_to_first_token_ms": ttft_seconds * 1000,
        "decode_tokens_per_second": decode_tokens_per_second,
        "total_generation_seconds": total_seconds,
        "peak_memory_bytes": peak_memory_bytes,
        "free_memory_after_bytes": free_bytes,
        "total_memory_bytes": total_bytes,
        "text": decoded,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("probe", "run"), required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--model-path", default="/model")
    parser.add_argument(
        "--prompt",
        default="Why is GPU memory important for large language model inference?",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=32)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--simulate-failure", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        payload = (
            probe_backend(args.backend, args.cpu_threads)
            if args.mode == "probe"
            else run_inference(args)
        )
        emit(payload)
        return 0
    except Exception as exc:  # The router needs a structured failure for fallback.
        emit(
            {
                "status": "error",
                "backend": args.backend,
                "error_type": type(exc).__name__,
                "detail": str(exc),
            }
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
