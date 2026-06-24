#!/usr/bin/env python3
"""Prepare pre-tokenized Python syntax data for SamatNext-Speed-8L-56M."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


TEXT_FIELDS = (
    "content",
    "code",
    "text",
    "func_code_string",
    "whole_func_string",
    "function",
    "source",
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    role: str
    split: str = "train"
    optional: bool = False


DATASET_MIXES: dict[str, list[DatasetSpec]] = {
    "smoke": [
        DatasetSpec("codeparrot/codeparrot-clean-valid", "clean Python source"),
    ],
    "python-small": [
        DatasetSpec("codeparrot/codeparrot-clean-valid", "clean Python source"),
        DatasetSpec("Nan-Do/code-search-net-python", "Python functions with docstrings/comments", optional=True),
    ],
    "python-overnight": [
        DatasetSpec("codeparrot/codeparrot-clean", "larger sampled Python source"),
        DatasetSpec("codeparrot/codeparrot-clean-valid", "clean Python source"),
        DatasetSpec("Nan-Do/code-search-net-python", "Python functions with docstrings/comments", optional=True),
    ],
}


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_text(row: dict[str, Any]) -> str | None:
    parts: list[str] = []
    for field in TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    docstring = row.get("func_documentation_string")
    if isinstance(docstring, str) and docstring.strip() and parts:
        parts.append('"""\n' + docstring + '\n"""')
    if not parts:
        return None
    return "\n\n".join(parts).strip()


def load_docs_for_spec(
    spec: DatasetSpec,
    *,
    max_docs: int | None,
    max_bytes: int | None,
    dedupe_exact: bool,
    seen_hashes: set[str],
) -> tuple[list[str], dict[str, Any]]:
    docs: list[str] = []
    bytes_used = 0
    skipped_duplicates = 0
    skipped_empty = 0
    status = "loaded"
    error = None
    try:
        dataset = load_dataset(spec.name, split=spec.split, streaming=True)
        for row in dataset:
            text = extract_text(row)
            if not text:
                skipped_empty += 1
                continue
            raw = text.encode("utf-8", errors="ignore")
            digest = hashlib.sha256(raw).hexdigest()
            if dedupe_exact and digest in seen_hashes:
                skipped_duplicates += 1
                continue
            if max_docs is not None and len(docs) >= max_docs:
                break
            if max_bytes is not None and docs and bytes_used + len(raw) > max_bytes:
                break
            seen_hashes.add(digest)
            docs.append(text)
            bytes_used += len(raw)
            if max_bytes is not None and bytes_used >= max_bytes:
                break
    except Exception as exc:
        if spec.optional:
            status = "skipped_unavailable"
            error = repr(exc)
        else:
            raise
    metadata = {
        "name": spec.name,
        "role": spec.role,
        "split": spec.split,
        "optional": spec.optional,
        "status": status,
        "error": error,
        "documents_used": len(docs),
        "bytes_used": bytes_used,
        "skipped_duplicates": skipped_duplicates,
        "skipped_empty": skipped_empty,
    }
    return docs, metadata


def train_or_load_tokenizer(docs: list[str], args: argparse.Namespace, tokenizer_path: Path) -> tuple[Tokenizer, str]:
    if tokenizer_path.is_file() and args.reuse_tokenizer:
        return Tokenizer.from_file(str(tokenizer_path)), "loaded_existing"
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=["<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(iter(docs), trainer=trainer, length=len(docs))
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    return tokenizer, "trained"


def encode_docs(
    docs_by_dataset: list[tuple[DatasetSpec, list[str], dict[str, Any]]],
    tokenizer: Tokenizer,
    max_total_tokens: int | None,
) -> tuple[list[int], list[dict[str, Any]]]:
    all_ids: list[int] = []
    output_meta: list[dict[str, Any]] = []
    for spec, docs, metadata in docs_by_dataset:
        produced = 0
        docs_tokenized = 0
        for text in docs:
            if max_total_tokens is not None and len(all_ids) >= max_total_tokens:
                break
            ids = tokenizer.encode(text).ids
            if not ids:
                continue
            remaining = None if max_total_tokens is None else max_total_tokens - len(all_ids)
            if remaining is not None and len(ids) > remaining:
                ids = ids[:remaining]
            all_ids.extend(ids)
            produced += len(ids)
            docs_tokenized += 1
        item = dict(metadata)
        item.update({"documents_tokenized": docs_tokenized, "tokens_produced": produced})
        output_meta.append(item)
        if max_total_tokens is not None and len(all_ids) >= max_total_tokens:
            break
    return all_ids, output_meta


def write_bin(path: Path, ids: list[int], dtype: np.dtype[Any]) -> dict[str, Any]:
    array = np.asarray(ids, dtype=dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "dtype": str(array.dtype)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-mix", choices=sorted(DATASET_MIXES), default="smoke")
    parser.add_argument("--out-dir", default="data_prepared/python_syntax_512")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--max-total-tokens", type=int, default=None)
    parser.add_argument("--max-docs-per-dataset", type=int, default=None)
    parser.add_argument("--max-bytes-per-dataset", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=1234)
    parser.add_argument("--dedupe-exact", type=parse_bool, default=True)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--reuse-tokenizer", type=parse_bool, default=True)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--planned-steps", type=int, default=10000)
    parser.add_argument("--planned-batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(args.tokenizer_path) if args.tokenizer_path else out_dir / "tokenizer.json"
    rng = random.Random(args.shuffle_seed)
    seen_hashes: set[str] = set()
    docs_by_dataset: list[tuple[DatasetSpec, list[str], dict[str, Any]]] = []
    for spec in DATASET_MIXES[args.dataset_mix]:
        docs, metadata = load_docs_for_spec(
            spec,
            max_docs=args.max_docs_per_dataset,
            max_bytes=args.max_bytes_per_dataset,
            dedupe_exact=args.dedupe_exact,
            seen_hashes=seen_hashes,
        )
        rng.shuffle(docs)
        docs_by_dataset.append((spec, docs, metadata))

    all_docs = [doc for _, docs, _ in docs_by_dataset for doc in docs]
    if not all_docs:
        raise SystemExit("no documents loaded; cannot prepare Python pretraining data")

    tokenizer, tokenizer_status = train_or_load_tokenizer(all_docs, args, tokenizer_path)
    token_ids, dataset_meta = encode_docs(docs_by_dataset, tokenizer, args.max_total_tokens)
    if len(token_ids) < args.seq_len + 2:
        raise SystemExit(f"not enough tokens produced for seq_len={args.seq_len}: {len(token_ids)}")
    max_id = max(token_ids)
    if max_id >= args.vocab_size:
        raise SystemExit(f"token id {max_id} exceeds active vocab_size={args.vocab_size}")
    dtype = np.dtype("uint16") if max_id < 65536 else np.dtype("uint32")
    split = int(len(token_ids) * (1.0 - args.val_fraction))
    split = max(args.seq_len + 1, min(split, len(token_ids) - args.seq_len - 1))
    train_ids = token_ids[:split]
    val_ids = token_ids[split:]
    train_file = write_bin(out_dir / "train.bin", train_ids, dtype)
    val_file = write_bin(out_dir / "val.bin", val_ids, dtype)
    total_tokens = len(token_ids)
    for item in dataset_meta:
        item["mixture_percent"] = (100.0 * item.get("tokens_produced", 0) / total_tokens) if total_tokens else 0.0
    estimated_sequences = len(train_ids) // args.seq_len
    planned_tokens = args.planned_steps * args.planned_batch_size * args.seq_len
    metadata = {
        "dataset_mix": args.dataset_mix,
        "datasets_used": dataset_meta,
        "seq_len": args.seq_len,
        "vocab_size": args.vocab_size,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_status": tokenizer_status,
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "token_dtype": str(dtype),
        "max_token_id": max_id,
        "total_tokens": total_tokens,
        "train_token_count": len(train_ids),
        "val_token_count": len(val_ids),
        "estimated_512_token_sequences": estimated_sequences,
        "planned_steps": args.planned_steps,
        "planned_batch_size": args.planned_batch_size,
        "planned_tokens": planned_tokens,
        "estimated_epochs_for_planned_run": planned_tokens / max(len(train_ids), 1),
        "max_total_tokens": args.max_total_tokens,
        "max_docs_per_dataset": args.max_docs_per_dataset,
        "max_bytes_per_dataset": args.max_bytes_per_dataset,
        "shuffle_seed": args.shuffle_seed,
        "dedupe_exact": args.dedupe_exact,
        "train_bin": train_file,
        "val_bin": val_file,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
