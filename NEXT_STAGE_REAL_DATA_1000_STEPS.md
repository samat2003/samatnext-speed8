# Next Stage: Real-Data 1000-Step Stability Check

The next stage checks whether SamatNext-Speed-8L-56M can train stably on real
text for 1000 steps.

This is not a coding-quality claim. It only verifies that:

- the real-data training loop works,
- loss stays finite,
- gradients stay finite,
- perplexity moves in a plausible direction for the byte-derived stream,
- throughput remains usable on real data.

## Dataset

Default dataset:

- Hugging Face dataset: `Salesforce/wikitext`
- Config: `wikitext-2-raw-v1`

Dataset priority:

1. Explicit CLI/local text files.
2. Existing local prepared text/code files if found.
3. Hugging Face `Salesforce/wikitext`, `wikitext-2-raw-v1`.
4. If loading fails, write a failure summary and stop.

The script must not fall back to synthetic random tokens.

## Latest Completed Check

The first local check completed 1000 steps on `Salesforce/wikitext` with
`wikitext-2-raw-v1`. The summary is saved at
`results/samatnext_speed8_realdata_1000step/summary.md`.

## Tokenization

The first stability check uses deterministic UTF-8 byte-level IDs mapped into
the active 32768-token vocabulary.

This is a stability/token-flow check, not a tokenizer-quality or coding-quality
claim. Perplexity is valid for this byte-derived training stream but should not
be compared to normal tokenizer-based LM perplexity.

## Default Command

```bash
python scripts/run_realdata_1000step_check.py \
  --config configs/samatnext_speed8_640.json \
  --out-dir results/samatnext_speed8_realdata_1000step \
  --seq-len 512 \
  --batch-size 16 \
  --max-steps 1000 \
  --dtype bf16 \
  --compile true \
  --optimizer fused_adamw \
  --lr 3e-4 \
  --weight-decay 0.1 \
  --warmup-steps 50 \
  --scheduler cosine \
  --grad-clip 1.0
```

## Outputs

- `results/samatnext_speed8_realdata_1000step/train_log.jsonl`
- `results/samatnext_speed8_realdata_1000step/train_log.csv`
- `results/samatnext_speed8_realdata_1000step/summary.json`
- `results/samatnext_speed8_realdata_1000step/summary.md`

Only the small summary files are intended for Git.

## Next Python Syntax Stage

The next stage after the WikiText stability check is a Python syntax
pretraining pipeline with a dedicated real-data throughput optimization phase.
See [PYTHON_PRETRAIN_PLAN.md](PYTHON_PRETRAIN_PLAN.md).

Do not start the overnight Python run until the pre-tokenized smoke benchmark
selects the fastest valid batcher mode and confirms stable loss/gradient
behavior.
