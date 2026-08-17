# Kimi deployment benchmark

This repository compares two Azure OpenAI-compatible deployments:

- `Kimi-K2.6`: baseline deployment
- `FW-Kimi-K2.6`: Fireworks deployment configured with Fireworks serving optimizations

The benchmark sends the same prompts and generation settings to both deployments and measures observed performance.

## Prerequisites

- Python 3.10 or newer
- Access to the Azure endpoint configured in `benchmark_kimi.py`
- Azure credentials available to `DefaultAzureCredential`
- Permission to call both deployments

Authenticate with one of the methods supported by `DefaultAzureCredential`, such as Azure CLI:

```bash
az login
```

## Microsoft Foundry setup

In Microsoft Foundry, create or select the project and Azure AI resource that
will host the deployments. Deploy the baseline Kimi model with the deployment
name `Kimi-K2.6`, and deploy the Fireworks-configured version with the exact
deployment name `FW-Kimi-K2.6`. Configure speculative decoding, adaptive
caching, and quantization on the Fireworks deployment according to your
approved serving configuration. Copy the resource's OpenAI-compatible endpoint
into `DEFAULT_ENDPOINT` or pass it with `--endpoint`, and grant the identity
used by `DefaultAzureCredential` permission to invoke both deployments.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run a benchmark

Run the benchmark with the default endpoint, three measured runs, and one warmup run:

```bash
python benchmark_kimi.py
```

For a more stable comparison, use more repetitions and save the raw report:

```bash
python benchmark_kimi.py \
  --runs 5 \
  --warmup 1 \
  --max-tokens 512 \
  --output benchmark_results.json
```

The endpoint can be overridden without editing the script:

```bash
python benchmark_kimi.py --endpoint "https://your-endpoint/openai/v1"
```

The script alternates deployment order on each measured repetition to reduce ordering bias. Warmup requests are not included in the reported metrics.

## Analyze results

The console and JSON report include results for each deployment and the following metrics:

- Median and p95 latency: lower is better
- Output tokens per second: higher is better
- Prompt, completion, and total token medians
- Valid response rate and failed request count
- Fireworks percentage changes versus the baseline

Focus on the `FW-Kimi-K2.6` section and the `Fireworks versus baseline` section. A useful result is lower median and p95 latency, higher output throughput, and comparable response validity and token usage.

The cache-oriented prompts reuse a long banking policy context and include an exact repeated task. Compare those prompt results separately when assessing prompt-prefix or adaptive caching effects. Results can vary with network conditions, service load, warmup state, and deployment configuration, so repeat the test and compare medians rather than relying on one request.

## Important limitation

Speculative decoding, adaptive caching, and quantization are configured by the serving deployment. This client measures their combined observed effect; it does not enable or isolate those features individually. To attribute a result to one feature, run controlled deployments where that feature is changed while all other settings remain the same.

## Example scripts

- `kimiwofw.py`: one request to the baseline deployment
- `kimiwfw.py`: one request to the Fireworks deployment
- `benchmark_kimi.py`: repeated comparison benchmark
