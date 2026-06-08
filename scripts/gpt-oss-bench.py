"""Benchmark one hardcoded GPT-OSS serving case via TokenSpeed CLI.

python scripts/gpt-oss-bench.py

with rocprofv3:

rocprofv3 --kernel-trace --output-format csv --stats -d trace -- python scripts/gpt-oss-bench.py
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkConfig:
    model: str = "amd/gpt-oss-120b-w-mxfp4-a-fp8"
    tokenizer: str | None = None
    host: str = "127.0.0.1"
    port: int = 21000
    world_size: int = 1
    random_input_len: int = 8192
    random_output_len: int = 1024
    random_range_ratio: float = 0.0
    random_prefix_len: int = 0
    num_prompts: int = 80
    max_concurrency: int = 16
    seed: int = 1
    request_id_prefix: str = "bench-"
    num_warmups: int = 16
    disable_kvstore: bool = True
    kvstore_ratio: float = 0.0
    disable_prefix_caching: bool = True
    enforce_eager: bool = False
    ignore_eos: bool = True
    server_timeout_sec: float = 1800.0
    benchmark_timeout_sec: float = 0.0
    moe_backend: str | None = "triton_kernel"


def wait_for_server_ready(
    proc: subprocess.Popen[Any], host: str, port: int, timeout_sec: float
) -> None:
    url = f"http://{host}:{port}/readiness"
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2)

    raise TimeoutError(f"server did not become ready within {timeout_sec:.0f}s")


def build_server_args(config: BenchmarkConfig) -> list[str]:
    argv = [
        "--model",
        config.model,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--world-size",
        str(config.world_size),
    ]
    if config.tokenizer:
        argv.extend(["--tokenizer-path", config.tokenizer])
    if config.disable_kvstore:
        argv.append("--disable-kvstore")
    argv.extend(["--kvstore-ratio", str(config.kvstore_ratio)])
    if config.disable_prefix_caching:
        argv.append("--no-enable-prefix-caching")
    if config.enforce_eager:
        argv.append("--enforce-eager")
    if config.moe_backend is not None:
        argv.extend(["--moe-backend", config.moe_backend])
    return argv


def build_server_cmd(config: BenchmarkConfig) -> list[str]:
    return [sys.executable, "-m", "tokenspeed.cli", "serve", *build_server_args(config)]


def start_server(config: BenchmarkConfig) -> subprocess.Popen[Any]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cmd = build_server_cmd(config)
    return subprocess.Popen(cmd, env=env, start_new_session=True)


def stop_server(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    for sig, timeout in (
        (signal.SIGINT, 30),
        (signal.SIGTERM, 15),
        (signal.SIGKILL, 5),
    ):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue


def build_benchmark_cmd(config: BenchmarkConfig) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "tokenspeed.cli",
        "bench",
        "--backend",
        "tokenspeed",
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--model",
        config.model,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(config.random_input_len),
        "--random-output-len",
        str(config.random_output_len),
        "--random-range-ratio",
        str(config.random_range_ratio),
        "--random-prefix-len",
        str(config.random_prefix_len),
        "--num-prompts",
        str(config.num_prompts),
        "--max-concurrency",
        str(config.max_concurrency),
        "--request-rate",
        "inf",
        "--num-warmups",
        str(config.num_warmups),
        "--seed",
        str(config.seed),
        "--percentile-metrics",
        "ttft,tpot,itl",
        "--disable-tqdm",
        "--request-id-prefix",
        config.request_id_prefix,
    ]
    if config.ignore_eos:
        cmd.append("--ignore-eos")
    else:
        cmd.extend(["--extra-body", json.dumps({"ignore_eos": False})])
    if config.tokenizer:
        cmd.extend(["--tokenizer", config.tokenizer])
    return cmd


def run_benchmark(config: BenchmarkConfig) -> subprocess.CompletedProcess[None]:
    return subprocess.run(
        build_benchmark_cmd(config),
        timeout=config.benchmark_timeout_sec or None,
        check=False,
    )


def main() -> int:
    config = BenchmarkConfig()
    if config.max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    if config.num_prompts <= 0:
        raise ValueError("num_prompts must be positive")

    print("Engine args:", " ".join(build_server_args(config)), flush=True)
    proc: subprocess.Popen[Any] | None = None
    try:
        proc = start_server(config)
        wait_for_server_ready(proc, config.host, config.port, config.server_timeout_sec)
        completed = run_benchmark(config)
        return completed.returncode
    finally:
        stop_server(proc)


if __name__ == "__main__":
    raise SystemExit(main())
