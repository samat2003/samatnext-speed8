# Active Model

The active model is **SamatNext-Speed-8L-56M**.

It is the all-softmax/GQA speed model defined by
`configs/samatnext_speed8_640.json`. It is not a Qwen-transfer model, not a GDN
model, and not the 6-GDN + 2-softmax hybrid experiment.

## Current Verified Speed

- Verified synthetic training speed: 101,486.4 tok/s
- Previous best: 102,670.9 tok/s
- Official metric: forward + backward + optimizer tokens/sec
- Hardware: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- dtype: bf16
- Batch size: 32
- Sequence length: 512
- torch.compile: true
- Optimizer: fused AdamW
- Loss: standard CE
- Parameter count: 56,371,840

See [SPEED_CLAIM.md](SPEED_CLAIM.md) for the frozen speed claim and benchmark
report path.

## Next Stage

The next stage is not another speed claim. It is a 1000-step real-data training
stability, loss, and perplexity check using the active all-softmax 56M model.

See [NEXT_STAGE_REAL_DATA_1000_STEPS.md](NEXT_STAGE_REAL_DATA_1000_STEPS.md).
