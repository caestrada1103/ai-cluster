#!/usr/bin/env python3
"""
AI Cluster Benchmarking Tool
============================

Comprehensive benchmarking tool for testing performance of the AI cluster.

Usage:
    python benchmark.py --url http://localhost:8000 --model deepseek-7b --concurrency 10
    python benchmark.py --config benchmark_config.yaml --output results.json
    python benchmark.py --compare baseline.json new_results.json
"""

import argparse
import asyncio
import json
import time
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import aiohttp
import numpy as np
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import csv

# Try to import optional dependencies
try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

try:
    from prometheus_api_client import PrometheusConnect
    HAVE_PROMETHEUS = True
except ImportError:
    HAVE_PROMETHEUS = False


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    
    # Configuration
    model: str
    concurrency: int
    total_requests: int
    prompt_length: int
    max_tokens: int
    temperature: float
    
    # Results
    successful_requests: int
    failed_requests: int
    total_time_seconds: float
    total_tokens_generated: int
    
    # Latency stats (ms)
    latencies: List[float]
    p50_latency: float
    p95_latency: float
    p99_latency: float
    min_latency: float
    max_latency: float
    mean_latency: float
    stddev_latency: float
    
    # Throughput
    requests_per_second: float
    tokens_per_second: float
    time_to_first_token_ms: List[float]
    
    # Errors
    errors: Dict[str, int]
    
    # Metadata
    timestamp: str
    duration_seconds: float
    cluster_info: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        d = asdict(self)
        # Remove large lists for JSON output
        if "latencies" in d:
            del d["latencies"]
        if "time_to_first_token_ms" in d:
            del d["time_to_first_token_ms"]
        return d
    
    def summary(self) -> str:
        """Return a human-readable summary."""
        return f"""
Benchmark Summary
=================
Model: {self.model}
Concurrency: {self.concurrency}
Total Requests: {self.total_requests} (✓ {self.successful_requests} ✗ {self.failed_requests})

Throughput:
  Requests/sec: {self.requests_per_second:.2f}
  Tokens/sec: {self.tokens_per_second:.2f}

Latency (ms):
  Mean: {self.mean_latency:.2f}
  P50:  {self.p50_latency:.2f}
  P95:  {self.p95_latency:.2f}
  P99:  {self.p99_latency:.2f}
  Min:  {self.min_latency:.2f}
  Max:  {self.max_latency:.2f}
  StdDev: {self.stddev_latency:.2f}

Errors: {self.errors if self.errors else 'None'}

Duration: {self.duration_seconds:.2f} seconds
        """


class BenchmarkRunner:
    """Run benchmarks against the AI cluster."""
    
    def __init__(
        self,
        url: str,
        model: str = "deepseek-7b",
        concurrency: int = 1,
        total_requests: int = 100,
        prompt_file: Optional[Path] = None,
        prompt_length: int = 50,
        max_tokens: int = 100,
        temperature: float = 0.7,
        timeout: int = 60,
        warmup: int = 10,
        output_dir: Path = Path("./benchmarks"),
        prometheus_url: Optional[str] = None,
    ):
        self.base_url = url.rstrip("/")
        self.model = model
        self.concurrency = concurrency
        self.total_requests = total_requests
        self.prompt_file = prompt_file
        self.prompt_length = prompt_length
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.warmup = warmup
        self.output_dir = Path(output_dir)
        self.prometheus_url = prometheus_url
        
        # Load prompts
        self.prompts = self._load_prompts()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prometheus client
        self.prom = None
        if prometheus_url and HAVE_PROMETHEUS:
            self.prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)
    
    def _load_prompts(self) -> List[str]:
        """Load prompts from file or generate them."""
        if self.prompt_file and self.prompt_file.exists():
            with open(self.prompt_file) as f:
                prompts = [line.strip() for line in f if line.strip()]
            return prompts
        
        # Generate random prompts
        words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
                "artificial", "intelligence", "machine", "learning", "deep", "neural",
                "network", "transformer", "attention", "model", "language", "AI"]
        
        prompts = []
        for _ in range(max(self.total_requests, 100)):
            length = np.random.randint(self.prompt_length // 2, self.prompt_length)
            prompt = " ".join(np.random.choice(words, length))
            prompts.append(prompt)
        
        return prompts
    
    async def _single_request(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        request_id: int,
    ) -> Tuple[int, float, int, Optional[str]]:
        """Execute a single inference request."""
        start_time = time.time()
        
        try:
            async with session.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    latency = (time.time() - start_time) * 1000  # ms
                    tokens = data.get("tokens_generated", 0)
                    return request_id, latency, tokens, None
                else:
                    error_text = await response.text()
                    return request_id, 0, 0, f"HTTP {response.status}: {error_text}"
        
        except asyncio.TimeoutError:
            return request_id, 0, 0, "timeout"
        except Exception as e:
            return request_id, 0, 0, str(e)
    
    async def _warmup(self, session: aiohttp.ClientSession):
        """Warm up the model with a few requests."""
        print(f"Warming up with {self.warmup} requests...")
        tasks = []
        for i in range(self.warmup):
            prompt = self.prompts[i % len(self.prompts)]
            tasks.append(self._single_request(session, prompt, i))
        
        results = await asyncio.gather(*tasks)
        success = sum(1 for r in results if r[3] is None)
        print(f"Warmup complete: {success}/{self.warmup} successful")
        await asyncio.sleep(1)  # Cool down
    
    async def _collect_metrics(self, start_time: float) -> Dict[str, Any]:
        """Collect metrics from Prometheus."""
        if not self.prom:
            return {}
        
        try:
            metrics = {}
            
            # GPU metrics
            gpu_util = self.prom.custom_query(
                query='avg(rate(gpu_utilization_percent[1m]))'
            )
            if gpu_util:
                metrics["gpu_utilization"] = float(gpu_util[0]["value"][1])
            
            gpu_mem = self.prom.custom_query(
                query='avg(gpu_memory_used_bytes / gpu_memory_total_bytes) * 100'
            )
            if gpu_mem:
                metrics["gpu_memory_usage"] = float(gpu_mem[0]["value"][1])
            
            # Request metrics
            req_rate = self.prom.custom_query(
                query='sum(rate(inference_requests_total[1m]))'
            )
            if req_rate:
                metrics["request_rate"] = float(req_rate[0]["value"][1])
            
            return metrics
        except Exception as e:
            print(f"Warning: Failed to collect Prometheus metrics: {e}")
            return {}
    
    async def run(self) -> BenchmarkResult:
        """Run the benchmark."""
        print(f"\n🚀 Starting benchmark:")
        print(f"  Model: {self.model}")
        print(f"  Concurrency: {self.concurrency}")
        print(f"  Total requests: {self.total_requests}")
        print(f"  Max tokens: {self.max_tokens}")
        print()
        
        # Create connection pool
        connector = aiohttp.TCPConnector(limit=self.concurrency)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Warmup
            if self.warmup > 0:
                await self._warmup(session)
            
            # Prepare requests
            tasks = []
            for i in range(self.total_requests):
                prompt = self.prompts[i % len(self.prompts)]
                tasks.append(self._single_request(session, prompt, i))
            
            # Run benchmark
            print(f"Running benchmark...")
            start_time = time.time()
            
            # Use semaphore to control concurrency
            semaphore = asyncio.Semaphore(self.concurrency)
            
            async def bounded_request(task):
                async with semaphore:
                    return await task
            
            bounded_tasks = [bounded_request(task) for task in tasks]
            
            # Run with progress bar
            results = []
            for coro in tqdm.as_completed(bounded_tasks, total=len(bounded_tasks)):
                result = await coro
                results.append(result)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Collect metrics
            metrics = await self._collect_metrics(start_time)
        
        # Process results
        return self._process_results(results, total_time, metrics)
    
    def _process_results(
        self,
        results: List[Tuple],
        total_time: float,
        metrics: Dict,
    ) -> BenchmarkResult:
        """Process raw results into a BenchmarkResult."""
        successful = []
        failed = []
        errors = {}
        tokens_total = 0
        ttft = []  # Time to first token (not available in current implementation)
        
        for req_id, latency, tokens, error in results:
            if error is None:
                successful.append(latency)
                tokens_total += tokens
            else:
                failed.append(error)
                errors[error] = errors.get(error, 0) + 1
        
        # Calculate statistics
        if successful:
            latencies = np.array(successful)
            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)
            mean = np.mean(latencies)
            std = np.std(latencies)
            min_lat = np.min(latencies)
            max_lat = np.max(latencies)
        else:
            p50 = p95 = p99 = mean = std = min_lat = max_lat = 0
        
        # Throughput
        rps = len(successful) / total_time if total_time > 0 else 0
        tps = tokens_total / total_time if total_time > 0 else 0
        
        # Get cluster info
        cluster_info = {
            "model": self.model,
            "concurrency": self.concurrency,
            "prompt_length": self.prompt_length,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "metrics": metrics,
        }
        
        return BenchmarkResult(
            model=self.model,
            concurrency=self.concurrency,
            total_requests=self.total_requests,
            prompt_length=self.prompt_length,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            successful_requests=len(successful),
            failed_requests=len(failed),
            total_time_seconds=total_time,
            total_tokens_generated=tokens_total,
            latencies=successful,
            p50_latency=float(p50),
            p95_latency=float(p95),
            p99_latency=float(p99),
            min_latency=float(min_lat),
            max_latency=float(max_lat),
            mean_latency=float(mean),
            stddev_latency=float(std),
            requests_per_second=float(rps),
            tokens_per_second=float(tps),
            time_to_first_token_ms=ttft,
            errors=errors,
            timestamp=datetime.now().isoformat(),
            duration_seconds=total_time,
            cluster_info=cluster_info,
        )
    
    def save_results(self, result: BenchmarkResult, filename: Optional[str] = None):
        """Save benchmark results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{self.model}_{self.concurrency}req_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        print(f"\nResults saved to {output_path}")
        
        # Also save as CSV for easy analysis
        csv_path = output_path.with_suffix(".csv")
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for key, value in result.to_dict().items():
                if isinstance(value, (int, float, str)):
                    writer.writerow([key, value])
        
        return output_path
    
    def plot_results(self, result: BenchmarkResult, output_file: Optional[Path] = None):
        """Generate plots from benchmark results."""
        if not result.latencies:
            print("No successful requests to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Latency histogram
        axes[0, 0].hist(result.latencies, bins=50, alpha=0.7, color="blue", edgecolor="black")
        axes[0, 0].axvline(result.p50_latency, color="green", linestyle="--", label=f"P50: {result.p50_latency:.1f}ms")
        axes[0, 0].axvline(result.p95_latency, color="orange", linestyle="--", label=f"P95: {result.p95_latency:.1f}ms")
        axes[0, 0].axvline(result.p99_latency, color="red", linestyle="--", label=f"P99: {result.p99_latency:.1f}ms")
        axes[0, 0].set_xlabel("Latency (ms)")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Latency Distribution")
        axes[0, 0].legend()
        
        # Cumulative distribution
        sorted_latencies = np.sort(result.latencies)
        cumulative = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
        axes[0, 1].plot(sorted_latencies, cumulative * 100)
        axes[0, 1].axhline(50, color="green", linestyle="--", alpha=0.5)
        axes[0, 1].axhline(95, color="orange", linestyle="--", alpha=0.5)
        axes[0, 1].axhline(99, color="red", linestyle="--", alpha=0.5)
        axes[0, 1].set_xlabel("Latency (ms)")
        axes[0, 1].set_ylabel("Cumulative %")
        axes[0, 1].set_title("Cumulative Latency Distribution")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Throughput over time (simulated)
        times = np.linspace(0, result.duration_seconds, len(result.latencies))
        window = max(1, len(result.latencies) // 20)
        throughput = np.convolve([1] * window, result.latencies, mode="valid") / window
        times_conv = times[:len(throughput)]
        axes[1, 0].plot(times_conv, throughput)
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("Latency (ms)")
        axes[1, 0].set_title("Latency Over Time")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Error pie chart
        if result.errors:
            labels = list(result.errors.keys())
            sizes = list(result.errors.values())
            axes[1, 1].pie(sizes, labels=labels, autopct="%1.1f%%")
            axes[1, 1].set_title("Error Distribution")
        else:
            axes[1, 1].text(0.5, 0.5, "No errors", ha="center", va="center")
            axes[1, 1].set_title("Errors")
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {output_file}")
        else:
            plt.show()


def compare_results(results_files: List[Path]) -> pd.DataFrame:
    """Compare multiple benchmark results."""
    data = []
    
    for file in results_files:
        with open(file) as f:
            result = json.load(f)
        
        data.append({
            "file": file.name,
            "model": result.get("model"),
            "concurrency": result.get("concurrency"),
            "requests": result.get("successful_requests"),
            "rps": result.get("requests_per_second"),
            "tps": result.get("tokens_per_second"),
            "p50_ms": result.get("p50_latency"),
            "p95_ms": result.get("p95_latency"),
            "p99_ms": result.get("p99_latency"),
            "errors": result.get("failed_requests"),
        })
    
    df = pd.DataFrame(data)
    return df


def load_config(config_file: Path) -> Dict:
    """Load benchmark configuration from YAML file."""
    with open(config_file) as f:
        return yaml.safe_load(f)


async def run_benchmark_suite(config: Dict):
    """Run multiple benchmarks from a configuration file."""
    url = config.get("url", "http://localhost:8000")
    prometheus_url = config.get("prometheus_url")
    output_dir = Path(config.get("output_dir", "./benchmarks"))
    
    results = []
    
    for bench_config in config.get("benchmarks", []):
        print(f"\n{'='*60}")
        print(f"Running benchmark: {bench_config.get('name', 'unnamed')}")
        print(f"{'='*60}")
        
        runner = BenchmarkRunner(
            url=url,
            model=bench_config.get("model", "deepseek-7b"),
            concurrency=bench_config.get("concurrency", 1),
            total_requests=bench_config.get("total_requests", 100),
            prompt_file=Path(bench_config["prompt_file"]) if bench_config.get("prompt_file") else None,
            prompt_length=bench_config.get("prompt_length", 50),
            max_tokens=bench_config.get("max_tokens", 100),
            temperature=bench_config.get("temperature", 0.7),
            timeout=bench_config.get("timeout", 60),
            warmup=bench_config.get("warmup", 10),
            output_dir=output_dir,
            prometheus_url=prometheus_url,
        )
        
        result = await runner.run()
        print(result.summary())
        
        # Save results
        filename = bench_config.get("output_file")
        runner.save_results(result, filename)
        
        # Plot if requested
        if bench_config.get("plot", False):
            plot_file = output_dir / f"{filename.replace('.json', '.png')}"
            runner.plot_results(result, plot_file)
        
        results.append(result)
        
        # Cool down between benchmarks
        await asyncio.sleep(5)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="AI Cluster Benchmark Tool")
    
    # Run modes
    parser.add_argument("--config", "-c", type=Path, help="Benchmark configuration file")
    parser.add_argument("--url", default="http://localhost:8000", help="Coordinator URL")
    parser.add_argument("--model", default="deepseek-7b", help="Model to benchmark")
    
    # Load test parameters
    parser.add_argument("--concurrency", "-n", type=int, default=1, help="Concurrent requests")
    parser.add_argument("--requests", "-r", type=int, default=100, help="Total requests")
    parser.add_argument("--prompt-file", type=Path, help="File with prompts (one per line)")
    parser.add_argument("--prompt-length", type=int, default=50, help="Generated prompt length")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup requests")
    
    # Output options
    parser.add_argument("--output-dir", "-o", type=Path, default="./benchmarks", help="Output directory")
    parser.add_argument("--output-file", type=str, help="Output filename")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    
    # Analysis
    parser.add_argument("--compare", nargs="+", type=Path, help="Compare result files")
    parser.add_argument("--prometheus", help="Prometheus URL for metrics collection")
    
    args = parser.parse_args()
    
    # Compare mode
    if args.compare:
        df = compare_results(args.compare)
        print("\nComparison Results:")
        print(df.to_string(index=False))
        return 0
    
    # Config mode
    if args.config:
        config = load_config(args.config)
        asyncio.run(run_benchmark_suite(config))
        return 0
    
    # Single benchmark mode
    async def run_single():
        runner = BenchmarkRunner(
            url=args.url,
            model=args.model,
            concurrency=args.concurrency,
            total_requests=args.requests,
            prompt_file=args.prompt_file,
            prompt_length=args.prompt_length,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            warmup=args.warmup,
            output_dir=args.output_dir,
            prometheus_url=args.prometheus,
        )
        
        result = await runner.run()
        print(result.summary())
        
        if not args.no_save:
            runner.save_results(result, args.output_file)
        
        if args.plot:
            plot_file = args.output_dir / f"{args.output_file or 'benchmark'}.png"
            runner.plot_results(result, plot_file)
    
    asyncio.run(run_single())
    return 0


if __name__ == "__main__":
    sys.exit(main())