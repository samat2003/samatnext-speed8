#!/usr/bin/env python3
"""Run a 1000-step real-data stability check for SamatNext-Speed-8L-56M."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from build_samatnext_speed8 import SamatNextSpeed8LM, Speed8Config, parameter_count


DEFAULT_CONFIG = Path("configs/samatnext_speed8_640.json")
DEFAULT_OUT_DIR = Path("results/samatnext_speed8_realdata_1000step")
HF_DATASET = "Salesforce/wikitext"
HF_CONFIG = "wikitext-2-raw-v1"
LOG_FIELDS = [
    "step",
    "train_loss",
    "train_ppl",
    "lr",
    "grad_norm",
    "grad_max",
    "param_norm",
    "tokens_per_sec",
    "step_time_sec",
    "peak_vram_mib",
    "loss_finite",
    "grad_finite",
    "has_nan_loss",
    "has_nan_grad",
    "skipped_step",
    "dtype",
    "batch_size",
    "seq_len",
    "val_loss",
    "val_ppl",
    "validation_tokens",
    "validation_time_sec",
]


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def safe_ppl(loss: float) -> float:
    if not math.isfinite(loss):
        return float("inf")
    return math.exp(min(loss, 20.0))


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_cuda_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def discover_local_text_files(paths: list[str]) -> list[Path]:
    explicit = [Path(path) for path in paths]
    if explicit:
        files: list[Path] = []
        for path in explicit:
            if path.is_file():
                files.append(path)
            elif path.is_dir():
                files.extend(sorted(path.rglob("*.txt")))
                files.extend(sorted(path.rglob("*.md")))
                files.extend(sorted(path.rglob("*.py")))
                files.extend(sorted(path.rglob("*.jsonl")))
            else:
                raise FileNotFoundError(f"local text path not found: {path}")
        return files
    for dirname in ("data", "datasets"):
        root = Path(dirname)
        if root.exists():
            files = []
            files.extend(sorted(root.rglob("*.txt")))
            files.extend(sorted(root.rglob("*.md")))
            files.extend(sorted(root.rglob("*.py")))
            files.extend(sorted(root.rglob("*.jsonl")))
            if files:
                return files
    return []


def bytes_to_ids(text: str, vocab_size: int) -> list[int]:
    # Reserve no special tokens here; this is a deterministic real-text byte stream.
    return [int(byte) % vocab_size for byte in text.encode("utf-8", errors="ignore")]


def load_real_text(args: argparse.Namespace, vocab_size: int) -> tuple[list[int], list[int] | None, dict[str, Any]]:
    local_files = discover_local_text_files(args.local_text)
    metadata: dict[str, Any] = {
        "tokenization": "deterministic UTF-8 byte-level IDs into active vocab",
        "tokenization_note": (
            "This is a stability/token-flow check, not a tokenizer-quality or coding-quality claim. "
            "Perplexity is for this byte-derived stream only."
        ),
    }
    if local_files:
        train_text_parts = []
        for path in local_files:
            train_text_parts.append(path.read_text(encoding="utf-8", errors="ignore"))
        train_ids = bytes_to_ids("\n".join(train_text_parts), vocab_size)
        metadata.update({"dataset_source": "local_text_files", "local_files": [str(path) for path in local_files]})
        return train_ids, None, metadata

    try:
        from datasets import load_dataset

        dataset = load_dataset(HF_DATASET, HF_CONFIG)
        train_text = "\n".join(item["text"] for item in dataset["train"] if item.get("text"))
        val_split = dataset.get("validation")
        val_text = "\n".join(item["text"] for item in val_split if item.get("text")) if val_split is not None else ""
        train_ids = bytes_to_ids(train_text, vocab_size)
        val_ids = bytes_to_ids(val_text, vocab_size) if val_text else None
        metadata.update(
            {
                "dataset_source": "huggingface",
                "dataset": HF_DATASET,
                "dataset_config": HF_CONFIG,
                "train_text_chars": len(train_text),
                "validation_text_chars": len(val_text),
                "train_token_count": len(train_ids),
                "validation_token_count": len(val_ids) if val_ids is not None else 0,
            }
        )
        return train_ids, val_ids, metadata
    except Exception as exc:
        metadata.update(
            {
                "dataset_source": None,
                "dataset_error": repr(exc),
                "dataset_missing": True,
                "dataset_attempted": {"dataset": HF_DATASET, "config": HF_CONFIG},
            }
        )
        return [], None, metadata


def make_batch(token_ids: list[int], batch_size: int, seq_len: int, step: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if len(token_ids) < seq_len + 2:
        raise ValueError(f"not enough real-data tokens for seq_len={seq_len}: {len(token_ids)}")
    span = len(token_ids) - seq_len - 1
    starts = [((step * batch_size + item) * seq_len) % span for item in range(batch_size)]
    rows = [token_ids[start : start + seq_len + 1] for start in starts]
    tensor = torch.tensor(rows, dtype=torch.long, device=device)
    return tensor[:, :-1].contiguous(), tensor[:, 1:].contiguous()


def make_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> tuple[torch.optim.Optimizer, bool, str | None]:
    if args.optimizer != "fused_adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            eps=1e-8,
        ), False, None
    try:
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            eps=1e-8,
            fused=True,
        ), True, None
    except Exception as exc:
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            eps=1e-8,
        ), False, repr(exc)


def resolve_checkpoint_file(path: str) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.is_file():
        return checkpoint_path
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"checkpoint path not found: {checkpoint_path}")
    preferred = [
        checkpoint_path / "model.safetensors",
        checkpoint_path / "pytorch_model.bin",
        checkpoint_path / "checkpoint.pt",
        checkpoint_path / "model.pt",
    ]
    for candidate in preferred:
        if candidate.is_file():
            return candidate
    for pattern in ("*.safetensors", "*.pt", "*.pth", "*.bin"):
        matches = sorted(checkpoint_path.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"no supported checkpoint file found in {checkpoint_path}")


def unwrap_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError(f"unsupported checkpoint payload type: {type(payload)!r}")
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict) and value:
            return value
    if payload and all(isinstance(key, str) for key in payload):
        return payload
    raise TypeError("checkpoint payload does not contain a recognized state dict")


def load_checkpoint_into_model(model: SamatNextSpeed8LM, checkpoint: str) -> dict[str, Any]:
    checkpoint_file = resolve_checkpoint_file(checkpoint)
    suffix = checkpoint_file.suffix.lower()
    if suffix == ".safetensors":
        state = load_file(str(checkpoint_file), device="cpu")
    elif suffix in {".pt", ".pth", ".bin"}:
        try:
            payload = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(checkpoint_file, map_location="cpu")
        state = unwrap_state_dict(payload)
    else:
        raise ValueError(f"unsupported checkpoint suffix: {checkpoint_file.suffix}")

    normalized = {
        key.removeprefix("_orig_mod."): value
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    incompatible = model.load_state_dict(normalized, strict=False)
    allowed_missing = {"lm_head.weight"} if model.config.tie_word_embeddings else set()
    unexpected = list(incompatible.unexpected_keys)
    missing = list(incompatible.missing_keys)
    disallowed_missing = sorted(set(missing) - allowed_missing)
    if unexpected or disallowed_missing:
        raise RuntimeError(
            "checkpoint did not match active model; "
            f"missing={missing}, unexpected={unexpected}, loaded={checkpoint_file}"
        )
    model.tie_weights()
    return {
        "checkpoint_requested": checkpoint,
        "checkpoint_loaded_path": str(checkpoint_file),
        "checkpoint_missing_keys": missing,
        "checkpoint_unexpected_keys": unexpected,
    }


def lr_for_step(step: int, args: argparse.Namespace) -> float:
    if step <= args.warmup_steps:
        return args.lr * step / max(args.warmup_steps, 1)
    progress = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    if args.scheduler == "linear":
        return args.lr * (1.0 - progress)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def compute_norms(model: torch.nn.Module) -> tuple[float, float, bool, float]:
    grad_sq = 0.0
    grad_max = 0.0
    param_sq = 0.0
    grad_finite = True
    for parameter in model.parameters():
        param_sq += float(parameter.detach().float().norm().cpu()) ** 2
        grad = parameter.grad
        if grad is None:
            continue
        grad_float = grad.detach().float()
        if not torch.isfinite(grad_float).all():
            grad_finite = False
        if torch.isfinite(grad_float).any():
            grad_sq += float(grad_float.norm().cpu()) ** 2
            grad_max = max(grad_max, float(grad_float.abs().max().cpu()))
    return grad_sq**0.5, grad_max, grad_finite, param_sq**0.5


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_ids: list[int],
    batch_size: int,
    seq_len: int,
    device: torch.device,
    max_batches: int,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    losses = []
    tokens = 0
    start = time.perf_counter()
    for index in range(max_batches):
        input_ids, labels = make_batch(val_ids, batch_size, seq_len, index, device)
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        if loss is None:
            continue
        losses.append(float(loss.detach().float().cpu()))
        tokens += batch_size * seq_len
    sync()
    elapsed = time.perf_counter() - start
    if was_training:
        model.train()
    if not losses:
        return {"val_loss": None, "val_ppl": None, "validation_tokens": 0, "validation_time_sec": elapsed}
    loss = sum(losses) / len(losses)
    return {"val_loss": loss, "val_ppl": safe_ppl(loss), "validation_tokens": tokens, "validation_time_sec": elapsed}


def write_summary(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    validation_first = summary.get("validation_first") or {}
    validation_last = summary.get("validation_last") or {}
    lines = [
        "# SamatNext-Speed-8L-56M Real-Data 1000-Step Stability Check",
        "",
        f"- Status: {summary['status']}",
        f"- Dataset: {summary.get('dataset', summary.get('dataset_source'))}",
        f"- Dataset config: {summary.get('dataset_config')}",
        f"- Steps completed: {summary.get('steps_completed', 0)}",
        f"- Scratch initialized: {summary.get('scratch_initialized')}",
        "",
        "## Required Answers",
        "",
        f"1. Completed 1000 real-data steps without NaNs: {summary.get('status') == 'completed'}",
        f"2. Initial/final train loss: {summary.get('initial_train_loss')} -> {summary.get('final_train_loss')}",
        f"3. Initial/final train perplexity: {summary.get('initial_train_ppl')} -> {summary.get('final_train_ppl')}",
        (
            "4. Validation loss improved: "
            f"{summary.get('validation_improved')} "
            f"({validation_first.get('val_loss')} -> {validation_last.get('val_loss')})"
        ),
        f"5. Average real-data training tokens/sec: {summary.get('average_tokens_per_sec')}",
        f"6. Peak VRAM MiB: {summary.get('peak_vram_mib')}",
        f"7. Max grad_norm / max grad_max: {summary.get('max_grad_norm')} / {summary.get('max_grad_max')}",
        f"8. bf16 stable: {summary.get('bf16_stable')}",
        f"9. torch.compile stable: {summary.get('torch_compile_stable')}",
        f"10. Ready for longer real-data run: {summary.get('ready_for_longer_realdata_run')}",
        "",
        "This is a real-data stability/token-flow check, not a coding-quality claim.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = Speed8Config.from_json(Path(args.config))
    config.validate()
    train_ids, val_ids, dataset_meta = load_real_text(args, config.vocab_size)
    common_summary: dict[str, Any] = {
        "model": config.model_name,
        "config_path": args.config,
        "parameter_count": None,
        "benchmark_type": "real-data 1000-step training stability check",
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "max_steps": args.max_steps,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "betas": [args.beta1, args.beta2],
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "scheduler": args.scheduler,
        "grad_clip": args.grad_clip,
        "torch_compile_requested": args.compile,
        "standard_ce": True,
        "scratch_initialized": args.checkpoint is None,
        **dataset_meta,
    }
    if not train_ids:
        summary = {
            **common_summary,
            "status": "skipped_dataset_missing_or_failed",
            "steps_completed": 0,
            "ready_for_longer_realdata_run": False,
        }
        write_summary(out_dir, summary)
        return summary

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    reset_cuda_peak()
    model = SamatNextSpeed8LM(config).to(device=device, dtype=dtype)
    checkpoint_meta: dict[str, Any] = {
        "checkpoint_requested": args.checkpoint,
        "checkpoint_loaded_path": None,
        "checkpoint_missing_keys": [],
        "checkpoint_unexpected_keys": [],
    }
    if args.checkpoint:
        checkpoint_meta = load_checkpoint_into_model(model, args.checkpoint)
    model.train()
    compile_worked = False
    compile_error = None
    if args.compile:
        try:
            model = torch.compile(model)
            compile_worked = True
        except Exception as exc:
            compile_error = repr(exc)
    optimizer, optimizer_fused, optimizer_fallback_error = make_optimizer(model, args)
    jsonl_path = out_dir / "train_log.jsonl"
    csv_path = out_dir / "train_log.csv"
    first_loss = None
    first_ppl = None
    final_loss = None
    final_ppl = None
    max_grad_norm = 0.0
    max_grad_max = 0.0
    total_tokens = 0
    total_time = 0.0
    consecutive_bad_loss = 0
    validation_first = None
    validation_last = None
    status = "completed"
    failure_reason = None
    with jsonl_path.open("w", encoding="utf-8") as jsonl_file, csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=LOG_FIELDS)
        writer.writeheader()
        for step in range(1, args.max_steps + 1):
            lr = lr_for_step(step, args)
            set_lr(optimizer, lr)
            skipped_step = False
            val_metrics = {"val_loss": None, "val_ppl": None, "validation_tokens": 0, "validation_time_sec": 0.0}
            try:
                input_ids, labels = make_batch(train_ids, args.batch_size, args.seq_len, step, device)
                sync()
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"]
                if loss is None:
                    raise RuntimeError("model returned no loss")
                has_nan_loss = bool(torch.isnan(loss).item())
                loss_finite = bool(torch.isfinite(loss).item())
                if not loss_finite:
                    raise FloatingPointError(f"non-finite loss at step {step}: {float(loss.detach().cpu())}")
                loss.backward()
                grad_norm_before_clip, grad_max, grad_finite, param_norm = compute_norms(model)
                has_nan_grad = not grad_finite
                if not grad_finite:
                    raise FloatingPointError(f"non-finite gradients at step {step}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                sync()
                step_time = time.perf_counter() - start
                loss_value = float(loss.detach().float().cpu())
                ppl_value = safe_ppl(loss_value)
                if loss_value > args.bad_loss_threshold:
                    consecutive_bad_loss += 1
                else:
                    consecutive_bad_loss = 0
                if consecutive_bad_loss >= args.bad_loss_patience:
                    raise FloatingPointError(
                        f"loss exceeded {args.bad_loss_threshold} for {consecutive_bad_loss} consecutive steps"
                    )
                if val_ids and step % args.val_every == 0:
                    val_metrics = validate(model, val_ids, args.val_batch_size, args.seq_len, device, args.val_batches)
                    if validation_first is None and val_metrics["val_loss"] is not None:
                        validation_first = val_metrics
                    if val_metrics["val_loss"] is not None:
                        validation_last = val_metrics
            except torch.cuda.OutOfMemoryError as exc:
                status = "failed"
                failure_reason = f"CUDA OOM at step {step}: {exc}"
                skipped_step = True
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break
            except Exception as exc:
                status = "failed"
                failure_reason = repr(exc)
                skipped_step = True
                break

            tokens = args.batch_size * args.seq_len
            total_tokens += tokens
            total_time += step_time
            max_grad_norm = max(max_grad_norm, grad_norm_before_clip)
            max_grad_max = max(max_grad_max, grad_max)
            final_loss = loss_value
            final_ppl = ppl_value
            if first_loss is None:
                first_loss = loss_value
                first_ppl = ppl_value
            row = {
                "step": step,
                "train_loss": loss_value,
                "train_ppl": ppl_value,
                "lr": lr,
                "grad_norm": grad_norm_before_clip,
                "grad_max": grad_max,
                "param_norm": param_norm,
                "tokens_per_sec": tokens / step_time if step_time > 0 else 0.0,
                "step_time_sec": step_time,
                "peak_vram_mib": torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0,
                "loss_finite": loss_finite,
                "grad_finite": grad_finite,
                "has_nan_loss": has_nan_loss,
                "has_nan_grad": has_nan_grad,
                "skipped_step": skipped_step,
                "dtype": args.dtype,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                **val_metrics,
            }
            jsonl_file.write(json.dumps(row) + "\n")
            writer.writerow(row)
            if step % args.log_every == 0 or step == 1:
                print(json.dumps(row), flush=True)

    steps_completed = int(final_loss is not None and total_tokens // (args.batch_size * args.seq_len))
    validation_improved = (
        validation_first is not None
        and validation_last is not None
        and validation_first.get("val_loss") is not None
        and validation_last.get("val_loss") is not None
        and validation_last["val_loss"] <= validation_first["val_loss"]
    )
    summary = {
        **common_summary,
        "status": status,
        "failure_reason": failure_reason,
        **checkpoint_meta,
        "parameter_count": parameter_count(getattr(model, "_orig_mod", model)),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "optimizer_fused": optimizer_fused,
        "optimizer_fallback_error": optimizer_fallback_error,
        "torch_compile_worked": compile_worked,
        "torch_compile_error": compile_error,
        "steps_completed": steps_completed,
        "initial_train_loss": first_loss,
        "final_train_loss": final_loss,
        "initial_train_ppl": first_ppl,
        "final_train_ppl": final_ppl,
        "validation_first": validation_first,
        "validation_last": validation_last,
        "validation_improved": validation_improved,
        "average_tokens_per_sec": total_tokens / total_time if total_time > 0 else None,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0,
        "max_grad_norm": max_grad_norm,
        "max_grad_max": max_grad_max,
        "bf16_stable": args.dtype == "bf16" and status == "completed",
        "torch_compile_stable": bool(args.compile and compile_worked and status == "completed"),
        "ready_for_longer_realdata_run": status == "completed" and final_loss is not None and math.isfinite(final_loss),
        "train_log_jsonl": str(jsonl_path),
        "train_log_csv": str(csv_path),
    }
    write_summary(out_dir, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--local-text", action="append", default=[], help="Optional local text file or directory.")
    parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint file or directory.")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--compile", type=parse_bool, default=True)
    parser.add_argument("--optimizer", choices=("fused_adamw", "adamw"), default="fused_adamw")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--scheduler", choices=("cosine", "linear"), default="cosine")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--bad-loss-threshold", type=float, default=20.0)
    parser.add_argument("--bad-loss-patience", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
