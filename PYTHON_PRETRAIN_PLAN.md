# Python Syntax Pretraining Plan

The active model remains **SamatNext-Speed-8L-56M** using
`configs/samatnext_speed8_640.json`.

This plan does not use Qwen transfer, GDN, or hybrid layers. The current goal is
to build a real Python pretraining pipeline that uses pre-tokenized batches and
then optimize throughput before any overnight run.

## Current State

- Active architecture: all-softmax/GQA only
- Parameters: 56,371,840
- Synthetic verified speed: 101,486.4 tok/s
- WikiText real-data 1000-step stability: passed
- First Python pipeline smoke dataset: `codeparrot/codeparrot-clean-valid`

## Dataset Mixes

`smoke`

- `codeparrot/codeparrot-clean-valid`
- Role: clean Python source
- Use: first 100-step pipeline smoke test

`python-small`

- `codeparrot/codeparrot-clean-valid`
- `Nan-Do/code-search-net-python` if available
- Use: small mixed Python syntax run

`python-overnight`

- sampled `codeparrot/codeparrot-clean`
- `codeparrot/codeparrot-clean-valid`
- `Nan-Do/code-search-net-python` if available
- Use: overnight candidate after throughput smoke tests pass

The scripts intentionally avoid full `codeparrot/github-code-clean` and full
`bigcode/the-stack` for this stage.

## Data Preparation

The preparation stage trains or loads a 32K BPE tokenizer compatible with the
active `vocab_size=32768`, pre-tokenizes the selected corpus, and writes binary
token arrays before training.

Smoke command:

```bash
python scripts/prepare_python_pretrain_data.py \
  --dataset-mix smoke \
  --out-dir data_prepared/python_syntax_512 \
  --seq-len 512 \
  --vocab-size 32768 \
  --max-total-tokens 25000000 \
  --dedupe-exact true
```

Expected outputs:

- `data_prepared/python_syntax_512/tokenizer.json`
- `data_prepared/python_syntax_512/train.bin`
- `data_prepared/python_syntax_512/val.bin`
- `data_prepared/python_syntax_512/metadata.json`

The prepared data directory is gitignored.

## Throughput Optimization Phase

Before an overnight run, benchmark all valid batcher modes:

- `dataloader_workers`
- `custom_pinned_prefetch`
- `gpu_resident_tokens`, only when the real token file fits safely in VRAM

Smoke benchmark command:

```bash
python scripts/benchmark_python_pretrain_pipeline.py \
  --config configs/samatnext_speed8_640.json \
  --data-dir data_prepared/python_syntax_512 \
  --out-dir results/samatnext_speed8_python_pretrain \
  --seq-len 512 \
  --batch-sizes 16,24,32,40 \
  --dtype bf16 \
  --compile true \
  --optimizer fused_adamw \
  --warmup-steps 20 \
  --timed-steps 100 \
  --mtp-depth 1
```

The benchmark selects the fastest valid mode automatically and reports every
mode, including failed or skipped modes.

## MTP

MTP support is present but disabled for the first speed baseline.

- `--mtp-depth 1`: normal next-token prediction
- `--mtp-depth 2`: predict t+1 and t+2
- `--mtp-depth 4`: predict t+1 through t+4

Do not run MTP overnight until the NTP baseline is fast and stable.

## Overnight Gate

Do not start an overnight Python syntax pretraining run until:

- pre-tokenized data prep succeeds,
- all valid batcher modes are benchmarked,
- the selected mode is stable for the 100-step smoke,
- loss is finite and decreases,
- NaN/Inf checks pass,
- bottlenecks are documented if real-data throughput is below 100K tok/s.
