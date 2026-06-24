#!/usr/bin/env python3
"""Train SamatNext-Speed-8L-56M on pre-tokenized Python data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from build_samatnext_speed8 import SamatNextSpeed8LM, Speed8Config, parameter_count, save_checkpoint


BATCHER_MODES = ("dataloader_workers", "custom_pinned_prefetch", "gpu_resident_tokens")


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


def load_metadata(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing metadata.json in {data_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def memmap_tokens(path: Path, dtype_name: str) -> np.memmap:
    dtype = np.dtype(dtype_name)
    return np.memmap(path, mode="r", dtype=dtype)


class TokenWindowDataset(Dataset[torch.Tensor]):
    def __init__(self, token_path: Path, dtype_name: str, seq_len: int, mtp_depth: int, seed: int):
        self.tokens = memmap_tokens(token_path, dtype_name)
        self.seq_len = seq_len
        self.window = seq_len + mtp_depth
        self.seed = seed
        self.span = len(self.tokens) - self.window
        if self.span <= 0:
            raise ValueError(f"not enough tokens in {token_path}: {len(self.tokens)}")

    def __len__(self) -> int:
        return max(self.span, 1_000_000)

    def __getitem__(self, index: int) -> torch.Tensor:
        start = ((index + self.seed) * 104729) % self.span
        array = np.asarray(self.tokens[start : start + self.window], dtype=np.int64)
        return torch.from_numpy(array)


class DataLoaderBatcher:
    def __init__(
        self,
        token_path: Path,
        dtype_name: str,
        batch_size: int,
        seq_len: int,
        mtp_depth: int,
        device: torch.device,
        seed: int,
        num_workers: int,
        prefetch_factor: int,
    ):
        dataset = TokenWindowDataset(token_path, dtype_name, seq_len, mtp_depth, seed)
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )
        self.iterator = iter(self.loader)
        self.device = device

    def next(self) -> tuple[torch.Tensor, float]:
        start = time.perf_counter()
        batch = next(self.iterator)
        batch = batch.to(self.device, non_blocking=True)
        return batch, time.perf_counter() - start

    def close(self) -> None:
        return None


def sample_numpy_window(tokens: np.memmap, batch_size: int, window: int, rng: np.random.Generator) -> torch.Tensor:
    span = len(tokens) - window
    starts = rng.integers(0, span, size=batch_size, dtype=np.int64)
    offsets = np.arange(window, dtype=np.int64)
    values = np.asarray(tokens[starts[:, None] + offsets[None, :]], dtype=np.int64)
    tensor = torch.from_numpy(values)
    if torch.cuda.is_available():
        tensor = tensor.pin_memory()
    return tensor


class CustomPinnedPrefetchBatcher:
    def __init__(
        self,
        token_path: Path,
        dtype_name: str,
        batch_size: int,
        seq_len: int,
        mtp_depth: int,
        device: torch.device,
        seed: int,
    ):
        self.tokens = memmap_tokens(token_path, dtype_name)
        self.batch_size = batch_size
        self.window = seq_len + mtp_depth
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future: Future[torch.Tensor] = self.executor.submit(self._prepare)

    def _prepare(self) -> torch.Tensor:
        return sample_numpy_window(self.tokens, self.batch_size, self.window, self.rng)

    def next(self) -> tuple[torch.Tensor, float]:
        start = time.perf_counter()
        batch = self.future.result()
        self.future = self.executor.submit(self._prepare)
        batch = batch.to(self.device, non_blocking=True)
        return batch, time.perf_counter() - start

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)


class GpuResidentBatcher:
    def __init__(self, token_path: Path, dtype_name: str, batch_size: int, seq_len: int, mtp_depth: int, device: torch.device):
        if device.type != "cuda":
            raise RuntimeError("gpu_resident_tokens requires CUDA")
        cpu_tokens = np.asarray(memmap_tokens(token_path, dtype_name), dtype=np.int64)
        self.tokens = torch.from_numpy(cpu_tokens).to(device=device, dtype=torch.long)
        self.batch_size = batch_size
        self.window = seq_len + mtp_depth
        self.device = device
        self.offsets = torch.arange(self.window, device=device, dtype=torch.long)
        self.span = self.tokens.numel() - self.window
        if self.span <= 0:
            raise ValueError(f"not enough tokens in {token_path}: {self.tokens.numel()}")

    def next(self) -> tuple[torch.Tensor, float]:
        start = time.perf_counter()
        starts = torch.randint(0, self.span, (self.batch_size,), device=self.device)
        batch = self.tokens[starts[:, None] + self.offsets[None, :]]
        return batch, time.perf_counter() - start

    def close(self) -> None:
        return None


def create_batcher(
    mode: str,
    data_dir: Path,
    metadata: dict[str, Any],
    batch_size: int,
    seq_len: int,
    mtp_depth: int,
    device: torch.device,
    seed: int,
    num_workers: int = 4,
    prefetch_factor: int = 4,
) -> Any:
    train_path = data_dir / "train.bin"
    dtype_name = metadata["token_dtype"]
    if mode == "dataloader_workers":
        return DataLoaderBatcher(train_path, dtype_name, batch_size, seq_len, mtp_depth, device, seed, num_workers, prefetch_factor)
    if mode == "custom_pinned_prefetch":
        return CustomPinnedPrefetchBatcher(train_path, dtype_name, batch_size, seq_len, mtp_depth, device, seed)
    if mode == "gpu_resident_tokens":
        return GpuResidentBatcher(train_path, dtype_name, batch_size, seq_len, mtp_depth, device)
    raise ValueError(f"unknown batcher mode: {mode}")


def make_optimizer(model: torch.nn.Module, optimizer_kind: str, lr: float, weight_decay: float) -> tuple[torch.optim.Optimizer, bool, str | None]:
    if optimizer_kind == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay, eps=1e-8), False, None
    try:
        return (
            torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay, eps=1e-8, fused=True),
            True,
            None,
        )
    except Exception as exc:
        return (
            torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay, eps=1e-8),
            False,
            repr(exc),
        )


def lr_for_step(step: int, max_steps: int, base_lr: float, warmup_steps: int, scheduler: str) -> float:
    if step <= warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    if scheduler == "linear":
        return base_lr * (1.0 - progress)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def compute_mtp_loss(
    model: torch.nn.Module,
    window: torch.Tensor,
    seq_len: int,
    mtp_depth: int,
    vocab_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    input_ids = window[:, :seq_len].contiguous()
    outputs = model(input_ids, return_logits=True)
    logits = outputs["logits"]
    if logits is None:
        raise RuntimeError("model returned no logits")
    losses = []
    for offset in range(1, mtp_depth + 1):
        labels = window[:, offset : offset + seq_len].contiguous()
        losses.append(F.cross_entropy(logits.reshape(-1, vocab_size).float(), labels.reshape(-1)))
    main_loss = losses[0]
    mtp_loss = torch.stack(losses[1:]).mean() if len(losses) > 1 else torch.zeros_like(main_loss)
    total_loss = torch.stack(losses).mean()
    return total_loss, {
        "main_loss": float(main_loss.detach().float().cpu()),
        "mtp_loss": float(mtp_loss.detach().float().cpu()),
        "total_loss": float(total_loss.detach().float().cpu()),
    }


def grad_has_nan_or_inf(model: torch.nn.Module) -> bool:
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad.detach()).all():
            return True
    return False


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    data_dir: Path,
    metadata: dict[str, Any],
    seq_len: int,
    mtp_depth: int,
    batch_size: int,
    device: torch.device,
    max_batches: int,
) -> dict[str, Any]:
    val_path = data_dir / "val.bin"
    if not val_path.is_file() or metadata.get("val_token_count", 0) < seq_len + mtp_depth + 1:
        return {"val_loss": None, "val_ppl": None, "validation_tokens": 0, "validation_time_sec": 0.0}
    was_training = model.training
    model.eval()
    tokens = memmap_tokens(val_path, metadata["token_dtype"])
    losses = []
    start_time = time.perf_counter()
    for index in range(max_batches):
        start = (index * batch_size * seq_len) % (len(tokens) - seq_len - mtp_depth)
        rows = []
        for row in range(batch_size):
            row_start = (start + row * seq_len) % (len(tokens) - seq_len - mtp_depth)
            rows.append(np.asarray(tokens[row_start : row_start + seq_len + mtp_depth], dtype=np.int64))
        window = torch.from_numpy(np.stack(rows)).to(device=device, non_blocking=True)
        loss, _ = compute_mtp_loss(model, window, seq_len, mtp_depth, int(metadata["vocab_size"]))
        losses.append(float(loss.detach().float().cpu()))
    sync()
    elapsed = time.perf_counter() - start_time
    if was_training:
        model.train()
    loss_value = sum(losses) / len(losses) if losses else None
    return {
        "val_loss": loss_value,
        "val_ppl": safe_ppl(loss_value) if loss_value is not None else None,
        "validation_tokens": max_batches * batch_size * seq_len,
        "validation_time_sec": elapsed,
    }


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(data_dir)
    config = Speed8Config.from_json(Path(args.config))
    config.validate()
    if config.model_name != "SamatNext-Speed-8L-56M":
        raise ValueError(f"unexpected active model: {config.model_name}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = SamatNextSpeed8LM(config).to(device=device, dtype=dtype)
    compile_worked = False
    compile_error = None
    if args.compile:
        try:
            model = torch.compile(model)
            compile_worked = True
        except Exception as exc:
            compile_error = repr(exc)
    optimizer, optimizer_fused, optimizer_fallback_error = make_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    batcher = create_batcher(
        args.batcher_mode,
        data_dir,
        metadata,
        args.batch_size,
        args.seq_len,
        args.mtp_depth,
        device,
        args.seed,
        args.num_workers,
        args.prefetch_factor,
    )
    train_jsonl = out_dir / "train_log.jsonl"
    train_csv = out_dir / "train_log.csv"
    eval_jsonl = out_dir / "eval_log.jsonl"
    fields = [
        "step",
        "lr",
        "main_loss",
        "mtp_loss",
        "total_loss",
        "train_ppl",
        "tokens_per_sec",
        "step_time_sec",
        "compute_time_sec",
        "data_time_sec",
        "peak_vram_mib",
        "has_nan_or_inf",
        "mtp_depth",
        "batch_size",
        "seq_len",
    ]
    first_loss = None
    final_loss = None
    total_tokens = 0
    total_time = 0.0
    status = "completed"
    failure_reason = None
    try:
        with train_jsonl.open("w", encoding="utf-8") as jsonl, train_csv.open("w", newline="", encoding="utf-8") as csv_file, eval_jsonl.open(
            "w", encoding="utf-8"
        ) as eval_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            for step in range(1, args.max_steps + 1):
                lr = lr_for_step(step, args.max_steps, args.lr, args.warmup_steps, args.scheduler)
                set_lr(optimizer, lr)
                data_start = time.perf_counter()
                window, _ = batcher.next()
                sync()
                data_time = time.perf_counter() - data_start
                compute_start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss, loss_parts = compute_mtp_loss(model, window, args.seq_len, args.mtp_depth, config.vocab_size)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {step}: {float(loss.detach().cpu())}")
                loss.backward()
                has_nan_or_inf = grad_has_nan_or_inf(model)
                if has_nan_or_inf:
                    raise FloatingPointError(f"non-finite gradients at step {step}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                sync()
                compute_time = time.perf_counter() - compute_start
                step_time = data_time + compute_time
                total_tokens += args.batch_size * args.seq_len
                total_time += step_time
                final_loss = loss_parts["total_loss"]
                if first_loss is None:
                    first_loss = final_loss
                row = {
                    "step": step,
                    "lr": lr,
                    **loss_parts,
                    "train_ppl": safe_ppl(loss_parts["total_loss"]),
                    "tokens_per_sec": (args.batch_size * args.seq_len) / step_time if step_time > 0 else 0.0,
                    "step_time_sec": step_time,
                    "compute_time_sec": compute_time,
                    "data_time_sec": data_time,
                    "peak_vram_mib": torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0,
                    "has_nan_or_inf": has_nan_or_inf,
                    "mtp_depth": args.mtp_depth,
                    "batch_size": args.batch_size,
                    "seq_len": args.seq_len,
                }
                jsonl.write(json.dumps(row) + "\n")
                writer.writerow(row)
                if step % args.log_every == 0 or step == 1:
                    print(json.dumps(row), flush=True)
                if args.eval_every > 0 and step % args.eval_every == 0:
                    val = evaluate_loss(model, data_dir, metadata, args.seq_len, args.mtp_depth, args.eval_batch_size, device, args.eval_batches)
                    val["step"] = step
                    eval_file.write(json.dumps(val) + "\n")
                if args.checkpoint_dir and args.save_every > 0 and step % args.save_every == 0:
                    save_checkpoint(getattr(model, "_orig_mod", model), Path(args.checkpoint_dir) / f"step{step}")
    except Exception as exc:
        status = "failed"
        failure_reason = repr(exc)
    finally:
        batcher.close()
    summary = {
        "status": status,
        "failure_reason": failure_reason,
        "model": config.model_name,
        "parameter_count": parameter_count(getattr(model, "_orig_mod", model)),
        "config": args.config,
        "data_dir": args.data_dir,
        "batcher_mode": args.batcher_mode,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "mtp_depth": args.mtp_depth,
        "steps_completed": total_tokens // max(args.batch_size * args.seq_len, 1),
        "initial_loss": first_loss,
        "final_loss": final_loss,
        "initial_ppl": safe_ppl(first_loss) if first_loss is not None else None,
        "final_ppl": safe_ppl(final_loss) if final_loss is not None else None,
        "loss_decreased": final_loss is not None and first_loss is not None and final_loss < first_loss,
        "average_tokens_per_sec": total_tokens / total_time if total_time > 0 else None,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0,
        "torch_compile_requested": args.compile,
        "torch_compile_worked": compile_worked,
        "torch_compile_error": compile_error,
        "optimizer_fused": optimizer_fused,
        "optimizer_fallback_error": optimizer_fallback_error,
        "train_log_jsonl": str(train_jsonl),
        "train_log_csv": str(train_csv),
        "eval_log_jsonl": str(eval_jsonl),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Python Pretraining Summary",
                "",
                f"- Status: {summary['status']}",
                f"- Batcher mode: {summary['batcher_mode']}",
                f"- Average tokens/sec: {summary['average_tokens_per_sec']}",
                f"- Initial/final loss: {summary['initial_loss']} -> {summary['final_loss']}",
                f"- Loss decreased: {summary['loss_decreased']}",
                f"- Peak VRAM MiB: {summary['peak_vram_mib']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/samatnext_speed8_640.json")
    parser.add_argument("--data-dir", default="data_prepared/python_syntax_512")
    parser.add_argument("--out-dir", default="results/samatnext_speed8_python_pretrain")
    parser.add_argument("--batcher-mode", choices=BATCHER_MODES, default="gpu_resident_tokens")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--compile", type=parse_bool, default=True)
    parser.add_argument("--optimizer", choices=("fused_adamw", "adamw"), default="fused_adamw")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--scheduler", choices=("cosine", "linear"), default="cosine")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=2500)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--mtp-depth", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
