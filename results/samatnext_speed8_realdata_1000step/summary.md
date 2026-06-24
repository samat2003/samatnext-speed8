# SamatNext-Speed-8L-56M Real-Data 1000-Step Stability Check

- Status: completed
- Dataset: Salesforce/wikitext
- Dataset config: wikitext-2-raw-v1
- Steps completed: 1000
- Scratch initialized: True

## Required Answers

1. Completed 1000 real-data steps without NaNs: True
2. Initial/final train loss: 10.527115821838379 -> 1.6363881826400757
3. Initial/final train perplexity: 37313.69967971967 -> 5.1365835662230825
4. Validation loss improved: True (2.4374727606773376 -> 1.6211478412151337)
5. Average real-data training tokens/sec: 35723.71337474396
6. Peak VRAM MiB: 3252.18359375
7. Max grad_norm / max grad_max: 23.410364326708223 / 0.85546875
8. bf16 stable: True
9. torch.compile stable: True
10. Ready for longer real-data run: True

This is a real-data stability/token-flow check, not a coding-quality claim.
