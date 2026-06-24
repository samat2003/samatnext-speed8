# Experiment Archive

This file records older experiment branches so they do not get confused with
the active model.

## Active Path

- Active model: SamatNext-Speed-8L-56M
- Architecture: all-softmax/GQA
- Config: `configs/samatnext_speed8_640.json`
- Status: active throughput model and next real-data stability target

## Archived / Not Active

- Qwen-transfer GDN experiment: archived. It was useful for transfer and
  architecture exploration, but it is not the active model.
- 6-GDN + 2-softmax hybrid: archived. The optimized FLA benchmark used
  `fla.layers.gated_deltanet.GatedDeltaNet`, but the hybrid lost to the
  all-softmax 56M baseline in the measured speed comparison.
- Reference PyTorch GDN diagnostic: archived. It was diagnostic-only and is not
  valid for speed claims.
- Liger CE experiment: archived. It reduced or changed memory behavior in some
  settings but did not replace standard CE for the current active speed claim.

Historical result files remain in `results/` locally but are not the active
project claim.
