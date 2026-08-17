# Benchmark Results Summary

## Deployment Results

| Metric | `Kimi-K2.6` | `FW-Kimi-K2.6` |
| --- | ---: | ---: |
| Successful / failed runs | 35 / 0 | 35 / 0 |
| Valid response rate | 28.6% | 100.0% |
| Median latency | 3.405s | 2.326s |
| P95 latency | 15.356s | 2.631s |
| Median output throughput | 110.41 tokens/sec | 209.11 tokens/sec |
| Median prompt tokens | 291 | 291 |
| Median completion tokens | 512 | 512 |
| Median total tokens | 803 | 803 |

## Fireworks Versus Baseline

| Metric | Change |
| --- | ---: |
| Median latency | +31.7% |
| Output throughput | +89.4% |
| Prompt tokens | +0 |
| Completion tokens | +0 |
| Total tokens | +0 |

Positive latency and throughput changes indicate that the Fireworks deployment
performed better than the baseline in this report.