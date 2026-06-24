#!/usr/bin/env python3
"""Synthetic training-speed benchmark for SamatNext-Speed-8L-56M."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from build_samatnext_speed8 import DEFAULT_CONFIG, SamatNextSpeed8LM, Speed8Config, parameter_count


DEFAULT_BATCH_SIZES = [8, 16, 24, 32, 48, 64]
LOSS_IMPLS = ("standard_ce", "fused_linear_ce", "chunked_linear_ce", "liger_fused_linear_ce")
HISTORICAL_BEST_STANDARD_CE_TOK_S = 101486.4


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def reset_cuda_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_optimizer(model: torch.nn.Module, optimizer_kind: str = "fused_adamw") -> tuple[torch.optim.Optimizer, bool, str | None]:
    if optimizer_kind == "sgd":
        return torch.optim.SGD(model.parameters(), lr=1e-3), False, None
    if optimizer_kind == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8), False, None
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8, fused=True)
        return optimizer, True, None
    except Exception as exc:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8)
        return optimizer, False, str(exc)


def make_batch(config: Speed8Config, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    return input_ids, labels


def find_liger_linear_ce():
    candidates = [
        ("liger_kernel.transformers.fused_linear_cross_entropy", "LigerFusedLinearCrossEntropyLoss"),
        ("liger_kernel.transformers", "LigerFusedLinearCrossEntropyLoss"),
        ("liger_kernel.ops.fused_linear_cross_entropy", "LigerFusedLinearCrossEntropyFunction"),
    ]
    errors = []
    for module_name, attr_name in candidates:
        try:
            module = __import__(module_name, fromlist=[attr_name])
            obj = getattr(module, attr_name)
            return obj, None
        except Exception as exc:
            errors.append(f"{module_name}.{attr_name}: {exc!r}")
    return None, "; ".join(errors)


def embedding_param_count(model: torch.nn.Module) -> int:
    base = getattr(model, "_orig_mod", model)
    return int(base.embed_tokens.weight.numel())


def lm_head_param_count(model: torch.nn.Module) -> int:
    base = getattr(model, "_orig_mod", model)
    return int(base.lm_head.weight.numel())


def compute_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_impl: str,
    ce_chunk_size: int,
    liger_linear_ce,
) -> tuple[torch.Tensor, bool]:
    if loss_impl == "standard_ce":
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        if loss is None:
            raise RuntimeError("model did not return loss")
        return loss, True

    base = getattr(model, "_orig_mod", model)
    hidden = model(input_ids, return_hidden_states=True, return_logits=False)["hidden_states"]
    if hidden is None:
        raise RuntimeError("model did not return hidden states")
    flat_hidden = hidden.reshape(-1, base.config.hidden_size)
    flat_labels = labels.reshape(-1)
    weight = base.lm_head.weight

    if loss_impl in {"fused_linear_ce", "liger_fused_linear_ce"}:
        if liger_linear_ce is None:
            raise RuntimeError("Liger fused linear cross entropy is unavailable")
        return liger_linear_ce(weight, flat_hidden, flat_labels), False

    if loss_impl == "chunked_linear_ce":
        total_loss = flat_hidden.new_zeros((), dtype=torch.float32)
        token_count = flat_labels.numel()
        for start in range(0, token_count, ce_chunk_size):
            end = min(start + ce_chunk_size, token_count)
            logits = flat_hidden[start:end] @ weight.T
            total_loss = total_loss + F.cross_entropy(logits.float(), flat_labels[start:end], reduction="sum")
        return total_loss / token_count, False

    raise ValueError(f"unsupported loss_impl: {loss_impl}")


def fb_step(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_impl: str,
    ce_chunk_size: int,
    liger_linear_ce,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    loss, _ = compute_loss(model, input_ids, labels, loss_impl, ce_chunk_size, liger_linear_ce)
    loss.backward()
    return loss.detach()


def fbo_step(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_impl: str,
    ce_chunk_size: int,
    liger_linear_ce,
) -> torch.Tensor:
    loss = fb_step(model, input_ids, labels, optimizer, loss_impl, ce_chunk_size, liger_linear_ce)
    optimizer.step()
    return loss


def benchmark_loop(
    step_fn,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    timed_steps: int,
    loss_impl: str,
    ce_chunk_size: int,
    liger_linear_ce,
) -> tuple[float, float]:
    last_loss = None
    for _ in range(warmup_steps):
        last_loss = step_fn(model, input_ids, labels, optimizer, loss_impl, ce_chunk_size, liger_linear_ce)
    sync()
    start = time.perf_counter()
    for _ in range(timed_steps):
        last_loss = step_fn(model, input_ids, labels, optimizer, loss_impl, ce_chunk_size, liger_linear_ce)
    sync()
    elapsed = time.perf_counter() - start
    if last_loss is None:
        return elapsed / timed_steps, float("nan")
    return elapsed / timed_steps, float(last_loss.float().detach().cpu())


def grad_diagnostics(model: torch.nn.Module) -> dict[str, Any]:
    total_sq = 0.0
    nan_params: list[str] = []
    inf_params: list[str] = []
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        if torch.isnan(grad).any():
            nan_params.append(name)
        if torch.isinf(grad).any():
            inf_params.append(name)
        if torch.isfinite(grad).all():
            total_sq += float(grad.detach().float().norm().cpu()) ** 2
    return {
        "grad_norm": total_sq**0.5,
        "grad_norm_finite": bool(total_sq < float("inf")),
        "nan_gradient_parameters": nan_params[:50],
        "inf_gradient_parameters": inf_params[:50],
        "nan_gradient_parameter_count": len(nan_params),
        "inf_gradient_parameter_count": len(inf_params),
    }


def diagnostic_breakdown(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_impl: str,
    ce_chunk_size: int,
    liger_loss_fn,
    include_optimizer: bool,
) -> dict[str, Any]:
    base = getattr(model, "_orig_mod", model)
    timings: dict[str, float] = {}

    # Warm this exact diagnostic path so torch.compile branch compilation is
    # not included in component timings.
    optimizer.zero_grad(set_to_none=True)
    warm_hidden = model(input_ids, return_hidden_states=True, return_logits=False)["hidden_states"]
    if loss_impl == "standard_ce":
        warm_logits = base.lm_head(warm_hidden)
        warm_loss = F.cross_entropy(warm_logits.view(-1, base.config.vocab_size).float(), labels.reshape(-1))
    else:
        warm_flat_hidden = warm_hidden.reshape(-1, base.config.hidden_size)
        warm_flat_labels = labels.reshape(-1)
        if loss_impl in {"fused_linear_ce", "liger_fused_linear_ce"}:
            warm_loss = liger_loss_fn(base.lm_head.weight, warm_flat_hidden, warm_flat_labels)
        else:
            warm_loss = warm_flat_hidden.new_zeros((), dtype=torch.float32)
            token_count = warm_flat_labels.numel()
            for start in range(0, token_count, ce_chunk_size):
                end = min(start + ce_chunk_size, token_count)
                warm_logits_chunk = warm_flat_hidden[start:end] @ base.lm_head.weight.T
                warm_loss = warm_loss + F.cross_entropy(
                    warm_logits_chunk.float(), warm_flat_labels[start:end], reduction="sum"
                )
            warm_loss = warm_loss / token_count
    warm_loss.backward()
    if include_optimizer:
        optimizer.step()
    sync()

    def timed(name: str, fn):
        sync()
        start = time.perf_counter()
        value = fn()
        sync()
        timings[name] = time.perf_counter() - start
        return value

    timed("zero_grad_sec", lambda: optimizer.zero_grad(set_to_none=True))
    if loss_impl == "standard_ce":
        hidden = timed("forward_transformer_blocks_sec", lambda: model(input_ids, return_hidden_states=True, return_logits=False)["hidden_states"])
        logits = timed("lm_head_logits_sec", lambda: base.lm_head(hidden))
        loss = timed("loss_compute_sec", lambda: F.cross_entropy(logits.view(-1, base.config.vocab_size).float(), labels.reshape(-1)))
    else:
        hidden = timed("forward_transformer_blocks_sec", lambda: model(input_ids, return_hidden_states=True, return_logits=False)["hidden_states"])
        flat_hidden = hidden.reshape(-1, base.config.hidden_size)
        flat_labels = labels.reshape(-1)
        if loss_impl in {"fused_linear_ce", "liger_fused_linear_ce"}:
            logits = None
            loss = timed("loss_compute_sec", lambda: liger_loss_fn(base.lm_head.weight, flat_hidden, flat_labels))
            timings["lm_head_logits_sec"] = 0.0
        else:
            def chunked_loss():
                total = flat_hidden.new_zeros((), dtype=torch.float32)
                token_count = flat_labels.numel()
                for start in range(0, token_count, ce_chunk_size):
                    end = min(start + ce_chunk_size, token_count)
                    logits_chunk = flat_hidden[start:end] @ base.lm_head.weight.T
                    total = total + F.cross_entropy(logits_chunk.float(), flat_labels[start:end], reduction="sum")
                return total / token_count

            loss = timed("loss_compute_sec", chunked_loss)
            timings["lm_head_logits_sec"] = 0.0
    loss_before_backward = float(loss.detach().float().cpu())
    timed("backward_sec", lambda: loss.backward())
    optimizer_step_sec = 0.0
    if include_optimizer:
        timed("optimizer_step_sec", optimizer.step)
        optimizer_step_sec = timings["optimizer_step_sec"]
    else:
        timings["optimizer_step_sec"] = 0.0
    timings["total_fwd_bwd_sec"] = (
        timings["zero_grad_sec"]
        + timings["forward_transformer_blocks_sec"]
        + timings["lm_head_logits_sec"]
        + timings["loss_compute_sec"]
        + timings["backward_sec"]
    )
    timings["total_fwd_bwd_optimizer_sec"] = timings["total_fwd_bwd_sec"] + optimizer_step_sec
    timings["loss_before_backward"] = loss_before_backward
    timings["loss_before_backward_finite"] = bool(torch.isfinite(torch.tensor(loss_before_backward)))
    timings.update(grad_diagnostics(model))
    return timings


def run_one(
    config: Speed8Config,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    compile_enabled: bool,
    warmup_steps: int,
    timed_steps: int,
    device: torch.device,
    loss_impl: str,
    ce_chunk_size: int,
    liger_linear_ce,
    liger_available: bool,
    liger_error: str | None,
    include_optimizer: bool = True,
    optimizer_kind: str = "fused_adamw",
    collect_diagnostic_breakdown: bool = False,
) -> dict[str, Any]:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    reset_cuda_peak()
    model = SamatNextSpeed8LM(config).to(device=device, dtype=dtype)
    model.train()
    if compile_enabled:
        model = torch.compile(model)
    liger_loss_fn = liger_linear_ce() if loss_impl in {"fused_linear_ce", "liger_fused_linear_ce"} and liger_linear_ce else None
    optimizer, fused, optimizer_fallback_error = make_optimizer(model, optimizer_kind)
    input_ids, labels = make_batch(config, batch_size, seq_len, device)
    tokens = batch_size * seq_len

    result: dict[str, Any] = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "dtype": str(dtype),
        "torch_compile": compile_enabled,
        "optimizer": "AdamW",
        "optimizer_kind": optimizer_kind,
        "optimizer_fused": fused,
        "optimizer_fallback_error": optimizer_fallback_error,
        "parameter_count": parameter_count(model),
        "vocab_size": config.vocab_size,
        "effective_tokens_per_step": tokens,
        "loss_impl": loss_impl,
        "ce_chunk_size": ce_chunk_size if loss_impl == "chunked_linear_ce" else None,
        "liger_available": liger_available,
        "liger_error": liger_error,
        "full_logits_materialized": loss_impl == "standard_ce",
        "logits_shape_if_standard": [batch_size, seq_len, config.vocab_size] if loss_impl == "standard_ce" else None,
        "lm_head_params": lm_head_param_count(model),
        "embedding_params": embedding_param_count(model),
    }
    try:
        fb_step_time, fb_loss = benchmark_loop(
            fb_step,
            model,
            input_ids,
            labels,
            optimizer,
            warmup_steps,
            timed_steps,
            loss_impl,
            ce_chunk_size,
            liger_loss_fn,
        )
        if include_optimizer:
            fbo_step_time, fbo_loss = benchmark_loop(
                fbo_step,
                model,
                input_ids,
                labels,
                optimizer,
                warmup_steps,
                timed_steps,
                loss_impl,
                ce_chunk_size,
                liger_loss_fn,
            )
        else:
            fbo_step_time, fbo_loss = fb_step_time, fb_loss
        loss_finite = bool(torch.isfinite(torch.tensor(fbo_loss)))
        result.update(
            {
                "forward_backward_step_time_sec": fb_step_time,
                "forward_backward_tokens_per_sec": tokens / fb_step_time,
                "forward_backward_optimizer_step_time_sec": fbo_step_time,
                "forward_backward_optimizer_tokens_per_sec": tokens / fbo_step_time,
                "step_time_sec": fbo_step_time,
                "tokens_per_sec": tokens / fbo_step_time,
                "loss": fbo_loss,
                "forward_backward_loss": fb_loss,
                "loss_finite": loss_finite,
                "has_nan": bool(torch.isnan(torch.tensor(fbo_loss))),
                "peak_vram_mib": torch.cuda.max_memory_allocated() / (1024**2)
                if torch.cuda.is_available()
                else 0.0,
                "error": None,
            }
        )
        if collect_diagnostic_breakdown:
            result["diagnostic_breakdown_mode"] = True
            result["diagnostic_breakdown_note"] = "Synchronizes around components; timings are diagnostic and can distort official speed."
            result["diagnostic_breakdown"] = diagnostic_breakdown(
                model,
                input_ids,
                labels,
                optimizer,
                loss_impl,
                ce_chunk_size,
                liger_loss_fn,
                include_optimizer,
            )
    except torch.cuda.OutOfMemoryError as exc:
        result["error"] = f"CUDA OOM: {exc}"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        result["error"] = repr(exc)
    del model, optimizer, input_ids, labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_profiler_diagnostic(
    config: Speed8Config,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    compile_enabled: bool,
    device: torch.device,
    loss_impl: str,
    ce_chunk_size: int,
    liger_linear_ce,
    optimizer_kind: str,
    include_optimizer: bool,
) -> list[dict[str, Any]]:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    model = SamatNextSpeed8LM(config).to(device=device, dtype=dtype)
    model.train()
    if compile_enabled:
        model = torch.compile(model)
    liger_loss_fn = liger_linear_ce() if loss_impl in {"fused_linear_ce", "liger_fused_linear_ce"} and liger_linear_ce else None
    optimizer, _, _ = make_optimizer(model, optimizer_kind)
    input_ids, labels = make_batch(config, batch_size, seq_len, device)
    step_fn = fbo_step if include_optimizer else fb_step
    for _ in range(5):
        step_fn(model, input_ids, labels, optimizer, loss_impl, ce_chunk_size, liger_loss_fn)
    sync()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        profile_memory=True,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        step_fn(model, input_ids, labels, optimizer, loss_impl, ce_chunk_size, liger_loss_fn)
    sync()
    rows = []
    for event in prof.key_averages().table(sort_by="cuda_time_total", row_limit=10).splitlines():
        rows.append({"table_row": event})
    del model, optimizer, input_ids, labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out-dir", default="results/samatnext_speed8")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-sizes", default=",".join(str(size) for size in DEFAULT_BATCH_SIZES))
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--compile", type=parse_bool, default=False)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=10)
    parser.add_argument("--loss-impl", choices=LOSS_IMPLS, default="standard_ce")
    parser.add_argument("--ce-chunk-size", type=int, default=2048)
    parser.add_argument("--include-optimizer", type=parse_bool, default=True)
    parser.add_argument("--optimizer-kind", choices=("fused_adamw", "adamw", "sgd"), default="fused_adamw")
    parser.add_argument("--diagnostic-breakdown", type=parse_bool, default=False)
    parser.add_argument("--profile", type=parse_bool, default=False)
    parser.add_argument(
        "--max-vram-mib",
        type=float,
        default=12000.0,
        help="Stop launching larger batch sizes after a completed batch reaches this peak VRAM.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = Speed8Config.from_json(config_path)
    config.validate()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item.strip()]
    liger_linear_ce, liger_error = find_liger_linear_ce()
    liger_available = liger_linear_ce is not None

    results = []
    profiler_results: list[dict[str, Any]] = []
    stop_reason = None
    for index, batch_size in enumerate(batch_sizes):
        print(f"Benchmarking batch_size={batch_size}, compile={args.compile}, dtype={args.dtype}", flush=True)
        result = run_one(
            config=config,
            batch_size=batch_size,
            seq_len=args.seq_len,
            dtype=dtype,
            compile_enabled=args.compile,
            warmup_steps=args.warmup_steps,
            timed_steps=args.timed_steps,
            device=device,
            loss_impl=args.loss_impl,
            ce_chunk_size=args.ce_chunk_size,
            liger_linear_ce=liger_linear_ce,
            liger_available=liger_available,
            liger_error=liger_error,
            include_optimizer=args.include_optimizer,
            optimizer_kind=args.optimizer_kind,
            collect_diagnostic_breakdown=args.diagnostic_breakdown,
        )
        print(json.dumps(result, indent=2), flush=True)
        results.append(result)
        if args.profile and index == 0 and result.get("error") is None:
            try:
                profiler_results = run_profiler_diagnostic(
                    config=config,
                    batch_size=batch_size,
                    seq_len=args.seq_len,
                    dtype=dtype,
                    compile_enabled=args.compile,
                    device=device,
                    loss_impl=args.loss_impl,
                    ce_chunk_size=args.ce_chunk_size,
                    liger_linear_ce=liger_linear_ce,
                    optimizer_kind=args.optimizer_kind,
                    include_optimizer=args.include_optimizer,
                )
            except Exception as exc:
                profiler_results = [{"error": repr(exc)}]
        peak = result.get("peak_vram_mib") or 0.0
        if result.get("error") is None and peak >= args.max_vram_mib:
            stop_reason = f"stopped after batch_size={batch_size}: peak_vram_mib {peak:.2f} >= {args.max_vram_mib:.2f}"
            for skipped_batch_size in batch_sizes[index + 1 :]:
                results.append(
                    {
                        "batch_size": skipped_batch_size,
                        "seq_len": args.seq_len,
                        "dtype": str(dtype),
                        "torch_compile": args.compile,
                        "optimizer": "AdamW",
                        "optimizer_fused": None,
                        "parameter_count": parameter_count(SamatNextSpeed8LM(config)),
                        "vocab_size": config.vocab_size,
                        "effective_tokens_per_step": skipped_batch_size * args.seq_len,
                        "loss_impl": args.loss_impl,
                        "ce_chunk_size": args.ce_chunk_size if args.loss_impl == "chunked_linear_ce" else None,
                        "liger_available": liger_available,
                        "full_logits_materialized": args.loss_impl == "standard_ce",
                        "logits_shape_if_standard": [skipped_batch_size, args.seq_len, config.vocab_size]
                        if args.loss_impl == "standard_ce"
                        else None,
                        "lm_head_params": config.hidden_size * config.vocab_size,
                        "embedding_params": config.hidden_size * config.vocab_size,
                        "error": f"skipped_after_vram_limit: {stop_reason}",
                    }
                )
            break

    successful = [
        item
        for item in results
        if item.get("error") is None and item.get("loss_finite", True) and not item.get("has_nan", False)
    ]
    best = max(successful, key=lambda item: item["tokens_per_sec"]) if successful else None
    mean_tokens_per_sec = (
        sum(float(item["tokens_per_sec"]) for item in successful) / len(successful) if successful else None
    )
    mean_forward_backward_tokens_per_sec = (
        sum(float(item["forward_backward_tokens_per_sec"]) for item in successful) / len(successful)
        if successful
        else None
    )
    median_tokens_per_sec = (
        statistics.median(float(item["tokens_per_sec"]) for item in successful) if successful else None
    )
    report = {
        "model_name": config.model_name,
        "historical_best_standard_ce_tok_s": HISTORICAL_BEST_STANDARD_CE_TOK_S,
        "config": asdict(config),
        "config_path": str(config_path),
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "torch_compile": args.compile,
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.timed_steps,
        "loss_impl": args.loss_impl,
        "ce_chunk_size": args.ce_chunk_size if args.loss_impl == "chunked_linear_ce" else None,
        "include_optimizer": args.include_optimizer,
        "optimizer_kind": args.optimizer_kind,
        "diagnostic_breakdown_mode": args.diagnostic_breakdown,
        "profiler_diagnostic_only": args.profile,
        "official_speed_note": "Official speed synchronizes only around full timed step loops.",
        "liger_available": liger_available,
        "liger_error": liger_error,
        "max_vram_mib": args.max_vram_mib,
        "stop_reason": stop_reason,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "batch_size_tested": batch_sizes,
        "results": results,
        "profiler_top_ops": profiler_results,
        "best_result": best,
        "best_valid_tokens_per_sec": best["tokens_per_sec"] if best else None,
        "mean_valid_tokens_per_sec": mean_tokens_per_sec,
        "median_valid_tokens_per_sec": median_tokens_per_sec,
        "mean_tokens_per_sec": mean_tokens_per_sec,
        "mean_forward_backward_tokens_per_sec": mean_forward_backward_tokens_per_sec,
    }
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"samatnext_speed8_benchmark_{timestamp}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved benchmark report: {out_path.resolve()}")


if __name__ == "__main__":
    main()
