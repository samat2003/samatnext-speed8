# Qwen2.5-Coder Local Baseline

> Archived context: this is not the active SamatNext model path. The active
> model is the all-softmax/GQA SamatNext-Speed-8L-56M documented in
> [ACTIVE_MODEL.md](ACTIVE_MODEL.md). See
> [EXPERIMENT_ARCHIVE.md](EXPERIMENT_ARCHIVE.md) for archived Qwen/GDN/hybrid
> experiments.

This is a frozen, unmodified local baseline for:

`Qwen/Qwen2.5-Coder-1.5B-Instruct`

The baseline is intended for later architecture or weight experiments. It does not modify the Qwen architecture, train, fine-tune, or quantize the model.

## Files

The Hugging Face snapshot is stored locally at:

`models/qwen2.5-coder-1.5b-instruct/`

Model weights must not be committed. The `.gitignore` excludes `models/` and common large weight formats such as `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, and `*.gguf`.

## Commands

Activate the existing virtual environment:

```bash
source .venv/bin/activate
```

Download and verify the snapshot:

```bash
python scripts/download_qwen.py
```

Run the local healthcheck:

```bash
python scripts/qwen_healthcheck.py
```

Run the baseline test:

```bash
python -m unittest tests/test_qwen_baseline.py
```

If CUDA memory is insufficient, run the healthcheck explicitly on CPU:

```bash
python scripts/qwen_healthcheck.py --cpu
```

The healthcheck uses `local_files_only=True`, so it tests the downloaded local baseline rather than fetching from the internet cache.
