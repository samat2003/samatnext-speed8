# Python Pretraining Pipeline Benchmark

- Dataset mix: smoke
- Tokenizer path: data_prepared/python_syntax_512/tokenizer.json
- Train tokens: 24500000
- Val tokens: 500000
- Reached 100K tok/s: False
- Bottleneck analysis: Model compute/optimizer is dominant; larger batches lose throughput near the VRAM limit while data wait is low.

## Results

| mode | batch | status | avg tok/s | median tok/s | best tok/s | step avg s | data avg s | data wait % | peak VRAM MiB | loss | ppl | NaN/Inf | loss down | 10h tokens | reason |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | --- |
| dataloader_workers | 16 | completed | 71599.74 | 71665.18 | 75668.06 | 0.1145 | 0.0005 | 0.43 | 4746.8 | 8.0702->6.9085 | 3197.66->1000.73 | False | True | 2577590516 |  |
| dataloader_workers | 24 | completed | 73319.08 | 73261.21 | 76867.46 | 0.1677 | 0.0005 | 0.27 | 6981.1 | 7.9851->6.5705 | 2936.83->713.70 | False | True | 2639486891 |  |
| dataloader_workers | 32 | completed | 75264.75 | 75252.02 | 78470.27 | 0.2178 | 0.0005 | 0.23 | 9156.6 | 7.9402->6.3417 | 2807.98->567.76 | False | True | 2709530892 |  |
| dataloader_workers | 40 | completed | 56285.74 | 59088.08 | 63570.76 | 0.3714 | 0.0005 | 0.14 | 11398.8 | 7.9845->6.2281 | 2935.20->506.78 | False | True | 2026286722 |  |
| custom_pinned_prefetch | 16 | completed | 63796.63 | 65710.34 | 72020.02 | 0.1292 | 0.0005 | 0.39 | 4747.1 | 8.0704->6.6560 | 3198.45->777.43 | False | True | 2296678660 |  |
| custom_pinned_prefetch | 24 | completed | 69408.13 | 71537.12 | 77036.13 | 0.1784 | 0.0005 | 0.28 | 6981.1 | 8.0825->6.5642 | 3237.27->709.22 | False | True | 2498692768 |  |
| custom_pinned_prefetch | 32 | completed | 67395.03 | 70463.36 | 75656.84 | 0.2452 | 0.0005 | 0.19 | 9156.6 | 7.8784->6.4127 | 2639.55->609.54 | False | True | 2426220948 |  |
| custom_pinned_prefetch | 40 | completed | 50938.51 | 47709.21 | 61041.40 | 0.4068 | 0.0005 | 0.12 | 11398.8 | 7.8957->6.3685 | 2685.76->583.19 | False | True | 1833786480 |  |
| gpu_resident_tokens | 16 | completed | 57260.38 | 56798.66 | 66272.39 | 0.1434 | 0.0004 | 0.25 | 4934.0 | 8.1890->6.7662 | 3601.13->868.05 | False | True | 2061373722 |  |
| gpu_resident_tokens | 24 | completed | 59616.62 | 58831.51 | 69629.18 | 0.2067 | 0.0004 | 0.18 | 7175.7 | 8.0428->6.6242 | 3111.29->753.07 | False | True | 2146198416 |  |
| gpu_resident_tokens | 32 | completed | 54062.95 | 57229.89 | 62173.72 | 0.3078 | 0.0003 | 0.10 | 9343.8 | 8.0130->6.5868 | 3019.93->725.46 | False | True | 1946266055 |  |
| gpu_resident_tokens | 40 | completed | 58339.69 | 58679.42 | 61747.12 | 0.3514 | 0.0003 | 0.09 | 11590.3 | 7.9669->6.5180 | 2883.91->677.23 | False | True | 2100228827 |  |

## Selected Mode

- Best batcher mode: dataloader_workers
- Best batch size: 32
- Average tokens/sec: 75264.75
- Peak VRAM MiB: 9156.6
- Loss decreased over 100 steps: True

## Recommended Overnight Command

```bash
python scripts/train_python_pretrain.py \
  --config configs/samatnext_speed8_640.json \
  --data-dir data_prepared/python_syntax_512 \
  --out-dir results/samatnext_speed8_python_pretrain \
  --batcher-mode dataloader_workers \
  --seq-len 512 \
  --batch-size 32 \
  --max-steps 165376 \
  --dtype bf16 \
  --compile true \
  --optimizer fused_adamw \
  --lr 3e-4 \
  --weight-decay 0.1 \
  --warmup-steps 200 \
  --scheduler cosine \
  --grad-clip 1.0 \
  --eval-every 500 \
  --checkpoint-dir checkpoints/samatnext_speed8_python_pretrain \
  --save-every 25000 \
  --mtp-depth 1
```
