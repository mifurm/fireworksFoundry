"""Compare the baseline Kimi deployment with the Fireworks deployment.

The Fireworks optimizations are configured by the serving deployment. This
client measures their observed effect under the same request conditions; it
does not enable speculative decoding, caching, or quantization itself.
"""

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI


DEFAULT_ENDPOINT = "https://fireworksmf.services.ai.azure.com/openai/v1"
SHARED_CACHE_CONTEXT = """
You are assisting a retail bank's operations team. Use the following policy context
for every task in this benchmark. Treat it as authoritative and do not invent
exceptions. The bank serves personal and small-business customers through web,
mobile, branch, and contact-centre channels.

Customer verification requires two independent factors before account-sensitive
actions. A transaction should be held for review when its risk score is high, the
device is new, the payment destination is unusual, or the transaction pattern
differs materially from the customer's history. A held payment must not be
described as successful until review is complete. Customers must receive clear,
plain-language explanations and a next step.

Operational incidents are classified as P1 when they affect many customers or
prevent essential payments, P2 when they materially degrade a key workflow, and
P3 when a workaround exists and impact is limited. P1 incidents require immediate
escalation, an incident owner, customer-impact tracking, and updates at least
every thirty minutes. P2 incidents require an owner and updates at least hourly.

When proposing a process, distinguish confirmed facts from assumptions, identify
the customer or operational risk, and recommend a measurable next action. Keep
answers concise and structured. Never include real personal data, credentials,
account numbers, or payment details in an example.
""".strip()

DEFAULT_PROMPTS = (
    "What is the capital of France? Answer in one sentence.",
    "Explain three practical ways a retail bank can reduce payment fraud. Give a concise numbered list.",
    "Write a Python function that returns the first n Fibonacci numbers and explain its time complexity.",
    f"{SHARED_CACHE_CONTEXT}\n\nTask: Create a five-step response playbook for a customer whose card payment is held for fraud review.",
    f"{SHARED_CACHE_CONTEXT}\n\nTask: Classify this incident and state the first three actions: mobile payments are unavailable for 18% of customers, but branch payments still work.",
    f"{SHARED_CACHE_CONTEXT}\n\nTask: Produce a concise checklist for an operations analyst reviewing an unusual new-device transfer.",
    f"{SHARED_CACHE_CONTEXT}\n\nTask: Create a five-step response playbook for a customer whose card payment is held for fraud review.",
)


@dataclass
class RunResult:
    deployment: str
    prompt_number: int
    run_number: int
    latency_seconds: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    reasoning_tokens: int | None
    output_tokens_per_second: float | None
    response_chars: int | None
    valid_response: bool
    error: str | None = None


def token_value(usage: Any, name: str) -> int | None:
    value = getattr(usage, name, None)
    return int(value) if value is not None else None


def reasoning_token_value(usage: Any) -> int | None:
    details = getattr(usage, "completion_tokens_details", None)
    return token_value(details, "reasoning_tokens") if details else None


def call_model(
    client: OpenAI,
    deployment: str,
    prompt: str,
    prompt_number: int,
    run_number: int,
    max_tokens: int,
) -> RunResult:
    started = time.perf_counter()
    try:
        completion = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )
        latency = time.perf_counter() - started
        message = completion.choices[0].message
        response_text = message.content or ""
        usage = completion.usage
        completion_tokens = token_value(usage, "completion_tokens")
        tokens_per_second = (
            completion_tokens / latency
            if completion_tokens is not None and latency > 0
            else None
        )
        return RunResult(
            deployment=deployment,
            prompt_number=prompt_number,
            run_number=run_number,
            latency_seconds=latency,
            prompt_tokens=token_value(usage, "prompt_tokens"),
            completion_tokens=completion_tokens,
            total_tokens=token_value(usage, "total_tokens"),
            reasoning_tokens=reasoning_token_value(usage),
            output_tokens_per_second=tokens_per_second,
            response_chars=len(response_text),
            valid_response=bool(response_text.strip()),
        )
    except Exception as error:
        return RunResult(
            deployment=deployment,
            prompt_number=prompt_number,
            run_number=run_number,
            latency_seconds=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            reasoning_tokens=None,
            output_tokens_per_second=None,
            response_chars=None,
            valid_response=False,
            error=f"{type(error).__name__}: {error}",
        )


def percentile(values: list[float], fraction: float) -> float | None:
    return statistics.quantiles(values, n=100, method="inclusive")[int(fraction * 100) - 1] if len(values) >= 2 else (values[0] if values else None)


def summarize(results: list[RunResult], deployment: str) -> dict[str, Any]:
    deployment_results = [result for result in results if result.deployment == deployment]
    successful = [result for result in deployment_results if result.latency_seconds is not None]
    latencies = [result.latency_seconds for result in successful if result.latency_seconds is not None]
    output_rates = [result.output_tokens_per_second for result in successful if result.output_tokens_per_second is not None]
    prompt_tokens = [result.prompt_tokens for result in successful if result.prompt_tokens is not None]
    completion_tokens = [result.completion_tokens for result in successful if result.completion_tokens is not None]
    total_tokens = [result.total_tokens for result in successful if result.total_tokens is not None]
    reasoning_tokens = [result.reasoning_tokens for result in successful if result.reasoning_tokens is not None]

    return {
        "deployment": deployment,
        "successful_runs": len(successful),
        "failed_runs": len(deployment_results) - len(successful),
        "valid_response_rate": sum(result.valid_response for result in successful) / len(successful) if successful else None,
        "latency_seconds": {
            "median": statistics.median(latencies) if latencies else None,
            "p95": percentile(latencies, 0.95),
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "output_tokens_per_second_median": statistics.median(output_rates) if output_rates else None,
        "prompt_tokens_median": statistics.median(prompt_tokens) if prompt_tokens else None,
        "completion_tokens_median": statistics.median(completion_tokens) if completion_tokens else None,
        "total_tokens_median": statistics.median(total_tokens) if total_tokens else None,
        "reasoning_tokens_median": statistics.median(reasoning_tokens) if reasoning_tokens else None,
    }


def print_summary(summary: dict[str, Any]) -> None:
    latency = summary["latency_seconds"]
    print(f"\n{summary['deployment']}")
    print(f"  successful/failed: {summary['successful_runs']}/{summary['failed_runs']}")
    print(f"  valid response rate: {summary['valid_response_rate']:.1%}" if summary["valid_response_rate"] is not None else "  valid response rate: n/a")
    print(f"  latency median/p95: {latency['median']:.3f}s / {latency['p95']:.3f}s" if latency["median"] is not None else "  latency median/p95: n/a")
    print(f"  output tokens/sec median: {summary['output_tokens_per_second_median']:.2f}" if summary["output_tokens_per_second_median"] is not None else "  output tokens/sec median: n/a")
    print(f"  prompt/completion/reasoning/total tokens median: {summary['prompt_tokens_median']} / {summary['completion_tokens_median']} / {summary['reasoning_tokens_median']} / {summary['total_tokens_median']}")


def print_comparison(baseline: dict[str, Any], fireworks: dict[str, Any]) -> None:
    baseline_latency = baseline["latency_seconds"]["median"]
    fireworks_latency = fireworks["latency_seconds"]["median"]
    baseline_rate = baseline["output_tokens_per_second_median"]
    fireworks_rate = fireworks["output_tokens_per_second_median"]

    print("\nFireworks versus baseline")
    if baseline_latency and fireworks_latency:
        latency_change = (baseline_latency - fireworks_latency) / baseline_latency
        print(f"  median latency change: {latency_change:+.1%} (positive means Fireworks is faster)")
    else:
        print("  median latency change: n/a")
    if baseline_rate and fireworks_rate:
        throughput_change = (fireworks_rate - baseline_rate) / baseline_rate
        print(f"  output throughput change: {throughput_change:+.1%} (positive means Fireworks is faster)")
    else:
        print("  output throughput change: n/a")
    for label, key in (("prompt", "prompt_tokens_median"), ("completion", "completion_tokens_median"), ("reasoning", "reasoning_tokens_median"), ("total", "total_tokens_median")):
        baseline_tokens = baseline[key]
        fireworks_tokens = fireworks[key]
        if baseline_tokens is not None and fireworks_tokens is not None:
            print(f"  {label} token change: {fireworks_tokens - baseline_tokens:+g}")
        else:
            print(f"  {label} token change: n/a")


def build_markdown_summary(
    generated_at: str,
    endpoint: str,
    summaries: list[dict[str, Any]],
) -> str:
    baseline, fireworks = summaries
    baseline_latency = baseline["latency_seconds"]["median"]
    fireworks_latency = fireworks["latency_seconds"]["median"]
    latency_change = ((baseline_latency - fireworks_latency) / baseline_latency) if baseline_latency and fireworks_latency else None
    baseline_rate = baseline["output_tokens_per_second_median"]
    fireworks_rate = fireworks["output_tokens_per_second_median"]
    throughput_change = ((fireworks_rate - baseline_rate) / baseline_rate) if baseline_rate and fireworks_rate else None

    def value(item: dict[str, Any], key: str) -> str:
        result = item.get(key)
        return "n/a" if result is None else str(result)

    def percentage(result: float | None) -> str:
        return "n/a" if result is None else f"{result:+.1%}"

    lines = [
        "# Benchmark Results Summary",
        "",
        f"Generated: `{generated_at}`",
        f"Endpoint: `{endpoint}`",
        "",
        "## Deployment Results",
        "",
        "| Metric | `Kimi-K2.6` | `FW-Kimi-K2.6` |",
        "| --- | ---: | ---: |",
        f"| Successful / failed runs | {baseline['successful_runs']} / {baseline['failed_runs']} | {fireworks['successful_runs']} / {fireworks['failed_runs']} |",
        f"| Valid response rate | {value(baseline, 'valid_response_rate')} | {value(fireworks, 'valid_response_rate')} |",
        f"| Median latency | {value(baseline['latency_seconds'], 'median')}s | {value(fireworks['latency_seconds'], 'median')}s |",
        f"| P95 latency | {value(baseline['latency_seconds'], 'p95')}s | {value(fireworks['latency_seconds'], 'p95')}s |",
        f"| Median output throughput | {value(baseline, 'output_tokens_per_second_median')} tokens/sec | {value(fireworks, 'output_tokens_per_second_median')} tokens/sec |",
        f"| Median prompt tokens | {value(baseline, 'prompt_tokens_median')} | {value(fireworks, 'prompt_tokens_median')} |",
        f"| Median completion tokens | {value(baseline, 'completion_tokens_median')} | {value(fireworks, 'completion_tokens_median')} |",
        f"| Median reasoning tokens | {value(baseline, 'reasoning_tokens_median')} | {value(fireworks, 'reasoning_tokens_median')} |",
        f"| Median total tokens | {value(baseline, 'total_tokens_median')} | {value(fireworks, 'total_tokens_median')} |",
        "",
        "## Fireworks Versus Baseline",
        "",
        "| Metric | Change |",
        "| --- | ---: |",
        f"| Median latency | {percentage(latency_change)} |",
        f"| Output throughput | {percentage(throughput_change)} |",
        f"| Prompt tokens | {value(fireworks, 'prompt_tokens_median')} - {value(baseline, 'prompt_tokens_median')} |",
        f"| Completion tokens | {value(fireworks, 'completion_tokens_median')} - {value(baseline, 'completion_tokens_median')} |",
        f"| Reasoning tokens | {value(fireworks, 'reasoning_tokens_median')} - {value(baseline, 'reasoning_tokens_median')} |",
        f"| Total tokens | {value(fireworks, 'total_tokens_median')} - {value(baseline, 'total_tokens_median')} |",
        "",
        "Positive latency and throughput changes indicate that the Fireworks deployment performed better in this report.",
        "Reasoning-token values are `n/a` when the endpoint does not return completion-token details.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per prompt and deployment.")
    parser.add_argument("--warmup", type=int, default=1, help="Unmeasured warmup runs per deployment.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output", help="JSON output path; defaults to a timestamped benchmark_results file.")
    parser.add_argument("--summary", help="Markdown summary path; defaults to a timestamped benchmark_results_summary file.")
    parser.add_argument("--endpoint", default=os.getenv("AZURE_OPENAI_ENDPOINT", DEFAULT_ENDPOINT))
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0:
        parser.error("--runs must be at least 1 and --warmup cannot be negative")

    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")
    client = OpenAI(base_url=args.endpoint, api_key=token_provider)
    deployments = ("Kimi-K2.6", "FW-Kimi-K2.6")
    results: list[RunResult] = []

    print(f"Endpoint: {args.endpoint}")
    print(f"Prompts: {len(DEFAULT_PROMPTS)}, measured runs: {args.runs}, warmups: {args.warmup}")
    print("Run order alternates by repetition to reduce ordering bias.")

    for deployment in deployments:
        for _ in range(args.warmup):
            call_model(client, deployment, DEFAULT_PROMPTS[0], 0, 0, args.max_tokens)

    for run_number in range(1, args.runs + 1):
        ordered_deployments = deployments if run_number % 2 else tuple(reversed(deployments))
        for prompt_number, prompt in enumerate(DEFAULT_PROMPTS, start=1):
            for deployment in ordered_deployments:
                result = call_model(client, deployment, prompt, prompt_number, run_number, args.max_tokens)
                results.append(result)
                status = "ok" if result.error is None else "error"
                print(f"{deployment} prompt={prompt_number} run={run_number}: {status} {result.latency_seconds:.3f}s" if result.latency_seconds else f"{deployment} prompt={prompt_number} run={run_number}: {status}")

    summaries = [summarize(results, deployment) for deployment in deployments]
    for summary in summaries:
        print_summary(summary)
    print_comparison(summaries[0], summaries[1])

    print("\nInterpretation: compare median and p95 latency, output tokens/sec, token usage, and valid response rate.")
    print("These results show the observed effect of the configured Fireworks deployment; they do not isolate each serving feature independently.")

    generated_at = datetime.now(timezone.utc).isoformat()
    file_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = args.output or f"benchmark_results_{file_timestamp}.json"
    summary_path = args.summary or f"benchmark_results_summary_{file_timestamp}.md"
    report = {
        "generated_at": generated_at,
        "endpoint": args.endpoint,
        "runs_per_prompt": args.runs,
        "warmup_runs_per_deployment": args.warmup,
        "max_tokens": args.max_tokens,
        "summaries": summaries,
        "runs": [asdict(result) for result in results],
    }
    with open(json_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(build_markdown_summary(generated_at, args.endpoint, summaries))
    print(f"Raw report written to {json_path}")
    print(f"Markdown summary written to {summary_path}")


if __name__ == "__main__":
    main()
