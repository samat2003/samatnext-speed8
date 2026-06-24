# SamatNext-Speed-8L-56M Speed Claim

See also [ACTIVE_MODEL.md](ACTIVE_MODEL.md),
[NEXT_STAGE_REAL_DATA_1000_STEPS.md](NEXT_STAGE_REAL_DATA_1000_STEPS.md), and
[EXPERIMENT_ARCHIVE.md](EXPERIMENT_ARCHIVE.md).

## Verified Throughput

SamatNext-Speed-8L-56M reaches **101,486.4 synthetic training tokens/sec**
on an **NVIDIA GeForce RTX 5070 Ti Laptop GPU** in the 100-step verification run.

This is a synthetic throughput benchmark, not a quality benchmark and not a
real-data training result.

## Benchmark Details

- Model: SamatNext-Speed-8L-56M
- Parameter count: 56,371,840
- Benchmark type: synthetic training-speed benchmark
- Official metric: forward + backward + optimizer tokens/sec
- Hardware: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- dtype: bf16
- Batch size: 32
- Sequence length: 512
- torch.compile: true
- Optimizer: fused AdamW
- Loss: standard CE
- Warmup steps: 20
- Timed steps: 100
- Verified tokens/sec: 101,486.4
- Previous best: 102,670.9 tok/s from the hybrid comparison rerun
- Verification report: `results/samatnext_speed8_baseline_102k/samatnext_speed8_benchmark_20260623-214744.json`

## Notes

The prior 30-step comparison run reached 102,670.9 tok/s with the same model,
dtype, batch size, sequence length, compiler setting, optimizer family, and loss
implementation. The longer 100-step verification measured 101,486.4 tok/s, so
the frozen verified claim is 101.5K synthetic training tokens/sec.
