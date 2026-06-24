#!/usr/bin/env python3
"""Benchmark pre-tokenized Python batcher modes for SamatNext-Speed-8L-56M."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from build_samatnext_speed8 import SamatNextSpeed8LM, Speed8Config, parameter_count
from train_python_pretrain import (
    BATCHER_MODES,
    compute_mtp_loss,
    create_batcher,
    dtype_from_name,
    grad_has_nan_or_inf,
    load_metadata,
    lr_for_step,
    make_optimizer,
    parse_bool,
    safe_ppl,
    set_lr,
    sync,
)

ACTIVE_MODEL_NAME = "SamatNext-Speed-8L-56M"


def parse_batch_sizes(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def blank_metrics() -> dict[str, Any]:
    return {
        "average_tokens_per_sec": None,
        "median_tokens_per_sec": None,
        "best_tokens_per_sec": None,
        "step_time_sec_average": None,
        "compute_time_sec_average": None,
        "data_time_sec_average": None,
        "estimated_data_wait_percent": None,
        "peak_vram_mib": None,
        "initial_loss": None,
        "final_loss": None,
        "initial_ppl": None,
        "final_ppl": None,
        "has_nan_or_inf": None,
        "loss_decreased": None,
        "estimated_tokens_10h": None,
    }


def should_skip_gpu_resident(data_dir: Path, metadata: dict[str, Any], max_vram_fraction: float) -> tuple[bool, str | None]:
    if not torch.cuda.is_available():
        return True, "CUDA unavailable"
    train_tokens = int(metadata["train_token_count"])
    required_bytes = train_tokens * 8
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if required_bytes > free_bytes * max_vram_fraction:
        return True, (
            f"gpu token array would need {required_bytes / (1024**2):.1f} MiB, "
            f"above {max_vram_fraction:.2f} of free VRAM"
        )
    if not (data_dir / "train.bin").is_file():
        return True, "missing train.bin"
    return False, None


def run_once(args: argparse.Namespace, metadata: dict[str, Any], mode: str, batch_size: int) -> dict[str, Any]:
    if mode == "gpu_resident_tokens":
        skip, reason = should_skip_gpu_resident(Path(args.data_dir), metadata, args.gpu_resident_max_vram_fraction)
        if skip:
            return {
                "batcher_mode": mode,
                "batch_size": batch_size,
                "status": "skipped",
                "skip_reason": reason,
                **blank_metrics(),
            }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    config = Speed8Config.from_json(Path(args.config))
    config.validate()
    if config.model_name != ACTIVE_MODEL_NAME:
        raise ValueError(f"unexpected active model: {config.model_name}")
    result: dict[str, Any] = {
        "batcher_mode": mode,
        "batch_size": batch_size,
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "mtp_depth": args.mtp_depth,
        "status": "started",
        "error": None,
        **blank_metrics(),
    }
    batcher = None
    has_nan_or_inf = False
    try:
        model = SamatNextSpeed8LM(config).to(device=device, dtype=dtype)
        result["parameter_count"] = parameter_count(model)
        compile_worked = False
        compile_error = None
        if args.compile:
            try:
                model = torch.compile(model)
                compile_worked = True
            except Exception as exc:
                compile_error = repr(exc)
        result["torch_compile_requested"] = args.compile
        result["torch_compile_worked"] = compile_worked
        result["torch_compile_error"] = compile_error
        optimizer, optimizer_fused, optimizer_fallback_error = make_optimizer(model, args.optimizer, args.lr, args.weight_decay)
        result["optimizer_fused"] = optimizer_fused
        result["optimizer_fallback_error"] = optimizer_fallback_error
        batcher = create_batcher(
            mode,
            Path(args.data_dir),
            metadata,
            batch_size,
            args.seq_len,
            args.mtp_depth,
            device,
            args.seed,
            args.num_workers,
            args.prefetch_factor,
        )
        model.train()
        first_loss = None
        final_loss = None
        first_ppl = None
        final_ppl = None
        for step in range(1, args.warmup_steps + 1):
            lr = lr_for_step(step, args.warmup_steps + args.timed_steps, args.lr, args.warmup_steps, "cosine")
            set_lr(optimizer, lr)
            window, _ = batcher.next()
            optimizer.zero_grad(set_to_none=True)
            loss, _ = compute_mtp_loss(model, window, args.seq_len, args.mtp_depth, config.vocab_size)
            if not torch.isfinite(loss):
                has_nan_or_inf = True
                raise FloatingPointError(f"non-finite warmup loss at step {step}")
            loss.backward()
            if grad_has_nan_or_inf(model):
                has_nan_or_inf = True
                raise FloatingPointError(f"non-finite warmup gradients at step {step}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        sync()
        step_times: list[float] = []
        compute_times: list[float] = []
        data_times: list[float] = []
        tokens_per_sec: list[float] = []
        for timed_idx in range(1, args.timed_steps + 1):
            step_num = args.warmup_steps + timed_idx
            lr = lr_for_step(step_num, args.warmup_steps + args.timed_steps, args.lr, args.warmup_steps, "cosine")
            set_lr(optimizer, lr)
            data_start = time.perf_counter()
            window, _ = batcher.next()
            sync()
            data_time = time.perf_counter() - data_start
            compute_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss, loss_parts = compute_mtp_loss(model, window, args.seq_len, args.mtp_depth, config.vocab_size)
            loss_finite = bool(torch.isfinite(loss).item())
            if not loss_finite:
                has_nan_or_inf = True
                raise FloatingPointError(f"non-finite timed loss at step {timed_idx}")
            loss.backward()
            has_nan_or_inf = grad_has_nan_or_inf(model)
            if has_nan_or_inf:
                raise FloatingPointError(f"non-finite timed gradients at step {timed_idx}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            sync()
            compute_time = time.perf_counter() - compute_start
            elapsed = data_time + compute_time
            loss_value = loss_parts["total_loss"]
            if first_loss is None:
                first_loss = loss_value
                first_ppl = safe_ppl(loss_value)
            final_loss = loss_value
            final_ppl = safe_ppl(loss_value)
            step_times.append(elapsed)
            compute_times.append(compute_time)
            data_times.append(data_time)
            tokens_per_sec.append((batch_size * args.seq_len) / elapsed if elapsed > 0 else 0.0)
        avg_tps = statistics.mean(tokens_per_sec)
        avg_step = statistics.mean(step_times)
        avg_compute = statistics.mean(compute_times)
        avg_data = statistics.mean(data_times)
        result.update(
            {
                "status": "completed",
                "average_tokens_per_sec": avg_tps,
                "median_tokens_per_sec": statistics.median(tokens_per_sec),
                "best_tokens_per_sec": max(tokens_per_sec),
                "step_time_sec_average": avg_step,
                "compute_time_sec_average": avg_compute,
                "data_time_sec_average": avg_data,
                "estimated_data_wait_percent": (100.0 * avg_data / avg_step) if avg_step > 0 else None,
                "peak_vram_mib": torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0,
                "initial_loss": first_loss,
                "final_loss": final_loss,
                "initial_ppl": first_ppl,
                "final_ppl": final_ppl,
                "has_nan_or_inf": has_nan_or_inf,
                "loss_decreased": final_loss is not None and first_loss is not None and final_loss < first_loss,
                "estimated_tokens_10h": avg_tps * 10.0 * 3600.0,
            }
        )
    except torch.cuda.OutOfMemoryError as exc:
        result.update({"status": "failed", "error": f"CUDA OOM: {exc}", "has_nan_or_inf": has_nan_or_inf})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        result.update({"status": "failed", "error": repr(exc), "has_nan_or_inf": has_nan_or_inf})
    finally:
        if batcher is not None:
            batcher.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def best_valid(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [item for item in results if item.get("status") == "completed" and not item.get("has_nan_or_inf")]
    return max(valid, key=lambda item: item["average_tokens_per_sec"]) if valid else None


def bottleneck_analysis(best: dict[str, Any] | None, results: list[dict[str, Any]]) -> str:
    if best is None:
        return "No valid batcher mode completed; blocker is pipeline/runtime failure."
    if best["average_tokens_per_sec"] >= 100000.0:
        return "Target reached; no below-target bottleneck."
    wait = best.get("estimated_data_wait_percent") or 0.0
    if wait > 15.0:
        return "Data pipeline or CPU sampling/transfer is the primary bottleneck."
    if any(item.get("torch_compile_requested") and not item.get("torch_compile_worked") for item in results if item.get("status") == "completed"):
        return "torch.compile compatibility reduced available optimized modes."
    completed = [item for item in results if item.get("status") == "completed"]
    max_completed_batch = max((item.get("batch_size", 0) for item in completed), default=0)
    if best.get("batch_size") == max_completed_batch:
        return "Model compute/optimizer appears dominant at the largest tested batch size."
    return "Model compute/optimizer is dominant; larger batches lose throughput near the VRAM limit while data wait is low."


def write_reports(args: argparse.Namespace, metadata: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = best_valid(results)
    reached_100k = bool(best and best["average_tokens_per_sec"] >= 100000.0)
    recommended_overnight_steps = None
    if best:
        tokens_per_step = int(best["batch_size"]) * int(args.seq_len)
        recommended_overnight_steps = max(1, int(best["estimated_tokens_10h"] // max(tokens_per_step, 1)))
    summary = {
        "model": "SamatNext-Speed-8L-56M",
        "config": args.config,
        "data_dir": args.data_dir,
        "dataset_mix": metadata.get("dataset_mix"),
        "tokenizer_path": metadata.get("tokenizer_path"),
        "train_token_count": metadata.get("train_token_count"),
        "val_token_count": metadata.get("val_token_count"),
        "batcher_modes_benchmarked": list(BATCHER_MODES),
        "batch_sizes": parse_batch_sizes(args.batch_sizes),
        "best_result": best,
        "best_real_data_pretokenized_tokens_per_sec": best.get("average_tokens_per_sec") if best else None,
        "reached_100k_tokens_per_sec": reached_100k,
        "bottleneck_analysis": bottleneck_analysis(best, results),
        "recommended_overnight_steps": recommended_overnight_steps,
        "results": results,
    }
    (out_dir / "pipeline_benchmark.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    def fmt_float(value: Any, digits: int = 2) -> str:
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) and math.isfinite(float(value)) else ""

    def fmt_int(value: Any) -> str:
        return str(int(value)) if isinstance(value, (int, float)) and math.isfinite(float(value)) else ""

    def fmt_bool(value: Any) -> str:
        if value is None:
            return ""
        return str(bool(value))

    lines = [
        "# Python Pretraining Pipeline Benchmark",
        "",
        f"- Dataset mix: {summary['dataset_mix']}",
        f"- Tokenizer path: {summary['tokenizer_path']}",
        f"- Train tokens: {summary['train_token_count']}",
        f"- Val tokens: {summary['val_token_count']}",
        f"- Reached 100K tok/s: {summary['reached_100k_tokens_per_sec']}",
        f"- Bottleneck analysis: {summary['bottleneck_analysis']}",
        "",
        "## Results",
        "",
        "| mode | batch | status | avg tok/s | median tok/s | best tok/s | step avg s | data avg s | data wait % | peak VRAM MiB | loss | ppl | NaN/Inf | loss down | 10h tokens | reason |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | --- |",
    ]
    for item in results:
        lines.append(
            "| {mode} | {batch} | {status} | {avg} | {median} | {best_tps} | {step} | {data} | {wait} | {vram} | {loss} | {ppl} | {nan} | {down} | {tokens10h} | {reason} |".format(
                mode=item.get("batcher_mode"),
                batch=item.get("batch_size"),
                status=item.get("status"),
                avg=fmt_float(item.get("average_tokens_per_sec")),
                median=fmt_float(item.get("median_tokens_per_sec")),
                best_tps=fmt_float(item.get("best_tokens_per_sec")),
                step=fmt_float(item.get("step_time_sec_average"), 4),
                data=fmt_float(item.get("data_time_sec_average"), 4),
                wait=fmt_float(item.get("estimated_data_wait_percent")),
                vram=fmt_float(item.get("peak_vram_mib"), 1),
                loss=(
                    f"{item.get('initial_loss'):.4f}->{item.get('final_loss'):.4f}"
                    if item.get("initial_loss") is not None and item.get("final_loss") is not None
                    else ""
                ),
                ppl=(
                    f"{item.get('initial_ppl'):.2f}->{item.get('final_ppl'):.2f}"
                    if item.get("initial_ppl") is not None and item.get("final_ppl") is not None
                    else ""
                ),
                nan=fmt_bool(item.get("has_nan_or_inf")),
                down=fmt_bool(item.get("loss_decreased")),
                tokens10h=fmt_int(item.get("estimated_tokens_10h")),
                reason=item.get("error") or item.get("skip_reason") or "",
            )
        )
    if best:
        lines.extend(
            [
                "",
                "## Selected Mode",
                "",
                f"- Best batcher mode: {best['batcher_mode']}",
                f"- Best batch size: {best['batch_size']}",
                f"- Average tokens/sec: {best['average_tokens_per_sec']:.2f}",
                f"- Peak VRAM MiB: {best['peak_vram_mib']:.1f}",
                f"- Loss decreased over 100 steps: {best['loss_decreased']}",
                "",
                "## Recommended Overnight Command",
                "",
                "```bash",
                "python scripts/train_python_pretrain.py \\",
                "  --config configs/samatnext_speed8_640.json \\",
                "  --data-dir data_prepared/python_syntax_512 \\",
                "  --out-dir results/samatnext_speed8_python_pretrain \\",
                f"  --batcher-mode {best['batcher_mode']} \\",
                "  --seq-len 512 \\",
                f"  --batch-size {best['batch_size']} \\",
                f"  --max-steps {recommended_overnight_steps} \\",
                "  --dtype bf16 \\",
                "  --compile true \\",
                "  --optimizer fused_adamw \\",
                "  --lr 3e-4 \\",
                "  --weight-decay 0.1 \\",
                "  --warmup-steps 200 \\",
                "  --scheduler cosine \\",
                "  --grad-clip 1.0 \\",
                "  --eval-every 500 \\",
                "  --checkpoint-dir checkpoints/samatnext_speed8_python_pretrain \\",
                "  --save-every 25000 \\",
                "  --mtp-depth 1",
                "```",
            ]
        )
    (out_dir / "pipeline_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/samatnext_speed8_640.json")
    parser.add_argument("--data-dir", default="data_prepared/python_syntax_512")
    parser.add_argument("--out-dir", default="results/samatnext_speed8_python_pretrain")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-sizes", default="16,24,32,40")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--compile", type=parse_bool, default=True)
    parser.add_argument("--optimizer", choices=("fused_adamw", "adamw"), default="fused_adamw")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--timed-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--mtp-depth", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--gpu-resident-max-vram-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_metadata(Path(args.data_dir))
    results = []
    for mode in BATCHER_MODES:
        for batch_size in parse_batch_sizes(args.batch_sizes):
            print(f"Benchmarking mode={mode} batch={batch_size}", flush=True)
            result = run_once(args, metadata, mode, batch_size)
            print(json.dumps(result, indent=2), flush=True)
            results.append(result)
    summary = write_reports(args, metadata, results)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
