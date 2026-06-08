"""GPT-OSS benchmark for TokenSpeed's HTTP serving path.

Run with

```
python gpt-oss-perf.py
```

This will produce a `results.json` file in the same directory, containing
detailed results for each scenario and concurrency level, as well as a summary
of the final results. Logs and raw benchmark outputs are stored in `logs/` and
`raw/` subdirectories, respectively.

Change the `BenchmarkConfig` dataclass to customize the benchmark parameters.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkConfig:
    model: str = "amd/gpt-oss-120b-w-mxfp4-a-fp8"
    tokenizer_path: str | None = None
    scenarios: tuple[tuple[str, int, int], ...] = (
        ("1k8k", 1024, 8192),
        ("8k1k", 8192, 1024),
        ("1k1k", 1024, 1024),
        ("4k4k", 4096, 4096),
    )
    concurrencies: tuple[int, ...] = (1, 2, 4, 8, 16)
    host: str = "127.0.0.1"
    port: int = 21000
    world_size: int = 1
    hip_visible_devices: str = "0"
    random_range_ratio: float = 0.0
    random_prefix_len: int = 0
    seed: int = 1
    num_warmups: int = 5
    cooldown_sec: float = 5.0
    server_timeout_sec: float = 1800.0
    benchmark_timeout_sec: float = 0.0
    ignore_eos: bool = True
    disable_kvstore: bool = True
    kvstore_ratio: float = 0.0
    disable_prefix_caching: bool = True
    enforce_eager: bool = False
    verbose: bool = False
    output_dir: Path = Path(__file__).resolve().with_suffix("")
    moe_backend = "triton_kernel"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def wait_for_server_ready(
    proc: subprocess.Popen[Any],
    host: str,
    port: int,
    timeout_sec: float,
    log_path: Path,
) -> None:
    url = f"http://{host}:{port}/readiness"
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited with code {proc.returncode}; tail of {log_path}:\n"
                f"{tail(log_path)}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2)

    raise TimeoutError(
        f"server did not become ready within {timeout_sec:.0f}s; "
        f"tail of {log_path}:\n{tail(log_path)}"
    )


def tail(path: Path, max_bytes: int = 4000) -> str:
    if not path.exists():
        return "<missing log>"
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        return f.read().decode(errors="replace")


def start_server(config: BenchmarkConfig, log_path: Path) -> subprocess.Popen[Any]:
    env = os.environ.copy()
    env.update(
        {
            "HIP_VISIBLE_DEVICES": config.hip_visible_devices,
            "PYTHONUNBUFFERED": "1",
        }
    )
    cmd = [
        sys.executable,
        "-m",
        "tokenspeed.cli",
        "serve",
        "--model",
        config.model,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--world-size",
        str(config.world_size),
    ]
    if config.tokenizer_path:
        cmd.extend(["--tokenizer-path", config.tokenizer_path])
    if config.disable_kvstore:
        cmd.append("--disable-kvstore")
    cmd.extend(["--kvstore-ratio", str(config.kvstore_ratio)])
    if config.disable_prefix_caching:
        cmd.append("--no-enable-prefix-caching")
    if config.enforce_eager:
        cmd.append("--enforce-eager")
    if config.moe_backend is not None:
        cmd.extend(["--moe-backend", config.moe_backend])

    if config.verbose:
        log_path.write_text(
            "verbose=true: server stdout/stderr inherited by parent process\n",
            encoding="utf-8",
        )
        proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    else:
        log_file = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        # Popen does not own the file object; close this handle in the parent.
        log_file.close()
    proc.cmd = cmd  # type: ignore[attr-defined]
    return proc


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


def run_benchmark(
    config: BenchmarkConfig,
    scenario_name: str,
    input_len: int,
    output_len: int,
    concurrency: int,
    result_path: Path,
) -> subprocess.CompletedProcess[str | None]:
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
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--random-range-ratio",
        str(config.random_range_ratio),
        "--random-prefix-len",
        str(config.random_prefix_len),
        "--num-prompts",
        str(concurrency * 5),
        "--max-concurrency",
        str(concurrency),
        "--request-rate",
        "inf",
        "--num-warmups",
        str(config.num_warmups),
        "--seed",
        str(config.seed),
        "--save-result",
        "--output-file",
        str(result_path),
        "--percentile-metrics",
        "ttft,tpot,itl",
        "--disable-tqdm",
        "--request-id-prefix",
        f"bench-{scenario_name}-conc{concurrency}-",
    ]
    if config.ignore_eos:
        cmd.append("--ignore-eos")
    else:
        cmd.extend(["--extra-body", json.dumps({"ignore_eos": False})])
    if config.tokenizer_path:
        cmd.extend(["--tokenizer", config.tokenizer_path])

    timeout = config.benchmark_timeout_sec or None
    if config.verbose:
        return subprocess.run(cmd, timeout=timeout, check=False)

    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "duration": result.get("duration"),
        "completed": result.get("completed"),
        "failed": result.get("failed"),
        "total_input_tokens": result.get("total_input_tokens"),
        "total_output_tokens": result.get("total_output_tokens"),
        "request_throughput": result.get("request_throughput"),
        "output_throughput": result.get("output_throughput"),
        "peak_output_throughput": result.get("max_output_tokens_per_s"),
        "peak_concurrent_requests": result.get("max_concurrent_requests"),
        "total_token_throughput": result.get("total_token_throughput"),
        "ttft_ms": {
            "mean": result.get("mean_ttft_ms"),
            "median": result.get("median_ttft_ms"),
            "p99": result.get("p99_ttft_ms"),
        },
        "tpot_ms": {
            "mean": result.get("mean_tpot_ms"),
            "median": result.get("median_tpot_ms"),
            "p99": result.get("p99_tpot_ms"),
        },
        "itl_ms": {
            "mean": result.get("mean_itl_ms"),
            "median": result.get("median_itl_ms"),
            "p99": result.get("p99_itl_ms"),
        },
    }


def write_results(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    tmp_path.replace(path)


def format_k_tokens(tokens: int) -> str:
    if tokens % 1024 == 0:
        return f"{tokens // 1024}k"
    return str(tokens)


def print_final_results(runs: list[dict[str, Any]]) -> None:
    print(
        "{s:{c}^{n}}".format(
            s=" GPT-OSS TokenSpeed Serving Benchmark Results ", n=50, c="="
        )
    )
    for run in runs:
        label = (
            f"{format_k_tokens(run['input_len'])}/"
            f"{format_k_tokens(run['output_len'])}/"
            f"conc{run['max_concurrency']}"
        )
        print(f"{label}:")
        if run.get("status") != "ok":
            print(f"- status: {run.get('status')}")
            print(f"- error: {run.get('error', '<unknown>')}")
            continue

        metrics = run.get("metrics") or {}
        ttft = metrics.get("ttft_ms") or {}
        tpot = metrics.get("tpot_ms") or {}
        itl = metrics.get("itl_ms") or {}
        print(f"- duration (s): {metrics.get('duration', 0.0):.2f}")
        print(f"- successful requests: {metrics.get('completed', 0)}")
        print(f"- failed requests: {metrics.get('failed', 0)}")
        print(
            "- output token throughput (tok/s): "
            f"{metrics.get('output_throughput', 0.0):.2f}"
        )
        print(
            "- peak output token throughput (tok/s): "
            f"{metrics.get('peak_output_throughput', 0.0):.2f}"
        )
        print(
            "- peak concurrent requests: "
            f"{metrics.get('peak_concurrent_requests', 0)}"
        )
        print(
            "- total token throughput (tok/s): "
            f"{metrics.get('total_token_throughput', 0.0):.2f}"
        )
        print(
            "- Time to First Token (ms): "
            f"mean={ttft.get('mean', 0.0):.2f}, "
            f"median={ttft.get('median', 0.0):.2f}, "
            f"p99={ttft.get('p99', 0.0):.2f}"
        )
        print(
            "- Time per Output Token (ms): "
            f"mean={tpot.get('mean', 0.0):.2f}, "
            f"median={tpot.get('median', 0.0):.2f}, "
            f"p99={tpot.get('p99', 0.0):.2f}"
        )
        print(
            "- Inter-token Latency (ms): "
            f"mean={itl.get('mean', 0.0):.2f}, "
            f"median={itl.get('median', 0.0):.2f}, "
            f"p99={itl.get('p99', 0.0):.2f}"
        )
    print("=" * 50)


def print_subprocess_output(stdout: str | None, stderr: str | None) -> None:
    if stdout:
        print("----- benchmark stdout -----")
        print(stdout.rstrip())
    if stderr:
        print("----- benchmark stderr -----", file=sys.stderr)
        print(stderr.rstrip(), file=sys.stderr)


def run_key(
    scenario_name: str, input_len: int, output_len: int, concurrency: int
) -> tuple[str, int, int, int]:
    return (scenario_name, int(input_len), int(output_len), int(concurrency))


def run_key_from_record(run: dict[str, Any]) -> tuple[str, int, int, int] | None:
    try:
        return run_key(
            str(run["scenario"]),
            int(run["input_len"]),
            int(run["output_len"]),
            int(run["max_concurrency"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def normalize_runs(
    runs: list[dict[str, Any]],
) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    runs_by_key: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for run in runs:
        key = run_key_from_record(run)
        if key is None:
            continue
        previous = runs_by_key.get(key)
        if previous is None:
            runs_by_key[key] = run
        elif previous.get("status") != "ok" and run.get("status") == "ok":
            runs_by_key[key] = run
        elif previous.get("status") == "ok" and run.get("status") != "ok":
            continue
        else:
            runs_by_key[key] = run
    return runs_by_key


def sorted_runs(
    config: BenchmarkConfig,
    runs_by_key: dict[tuple[str, int, int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    scenario_order = {name: idx for idx, (name, _, _) in enumerate(config.scenarios)}
    concurrency_order = {value: idx for idx, value in enumerate(config.concurrencies)}

    def sort_key(run: dict[str, Any]) -> tuple[int, int, int, int, int]:
        key = run_key_from_record(run)
        if key is None:
            return (9999, 9999, 0, 0, 0)
        scenario_name, input_len, output_len, concurrency = key
        return (
            scenario_order.get(scenario_name, 9999),
            concurrency_order.get(concurrency, 9999),
            input_len,
            output_len,
            concurrency,
        )

    return sorted(runs_by_key.values(), key=sort_key)


def build_final_results(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "label": (
                f"{format_k_tokens(run['input_len'])}/"
                f"{format_k_tokens(run['output_len'])}/"
                f"conc{run['max_concurrency']}"
            ),
            "status": run.get("status"),
            "metrics": run.get("metrics"),
            "error": run.get("error"),
        }
        for run in runs
    ]


def update_payload_runs(
    payload: dict[str, Any],
    config: BenchmarkConfig,
    runs_by_key: dict[tuple[str, int, int, int], dict[str, Any]],
) -> None:
    payload["runs"] = sorted_runs(config, runs_by_key)
    payload["final_results"] = build_final_results(payload["runs"])
    payload["updated_at"] = now_iso()


def main() -> int:
    config = BenchmarkConfig()
    output_dir = config.output_dir.resolve()
    raw_dir = output_dir / "raw"
    log_dir = output_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.json"
    existing_payload: dict[str, Any] = {}
    if results_path.exists():
        existing_payload = load_json(results_path) or {}

    server_config = {
        "host": config.host,
        "port": config.port,
        "world_size": config.world_size,
        "hip_visible_devices": config.hip_visible_devices,
        "tokenizer_path": config.tokenizer_path,
        "disable_kvstore": config.disable_kvstore,
        "kvstore_ratio": config.kvstore_ratio,
        "disable_prefix_caching": config.disable_prefix_caching,
        "enforce_eager": config.enforce_eager,
        "serve_mode": "tokenspeed_smg_http",
    }
    benchmark_config = {
        "scenarios": [
            {"name": name, "input_len": isl, "output_len": osl}
            for name, isl, osl in config.scenarios
        ],
        "concurrencies": list(config.concurrencies),
        "num_prompts": "5 * max_concurrency",
        "num_warmups": config.num_warmups,
        "random_range_ratio": config.random_range_ratio,
        "random_prefix_len": config.random_prefix_len,
        "seed": config.seed,
        "ignore_eos": config.ignore_eos,
        "backend": "tokenspeed",
        "verbose": config.verbose,
    }
    config_matches = (
        existing_payload.get("server_config") == server_config
        and existing_payload.get("benchmark_config") == benchmark_config
    )
    runs_by_key = (
        normalize_runs(existing_payload.get("runs", [])) if config_matches else {}
    )

    payload: dict[str, Any] = {
        "model": config.model,
        "started_at": (
            existing_payload.get("started_at", now_iso())
            if config_matches
            else now_iso()
        ),
        "server_config": server_config,
        "benchmark_config": benchmark_config,
        "runs": [],
    }
    update_payload_runs(payload, config, runs_by_key)
    write_results(results_path, payload)

    total_runs = len(config.scenarios) * len(config.concurrencies)
    run_index = 0
    for scenario_name, input_len, output_len in config.scenarios:
        for concurrency in config.concurrencies:
            run_index += 1
            run_id = f"{scenario_name}_conc{concurrency}"
            key = run_key(scenario_name, input_len, output_len, concurrency)
            existing_run = runs_by_key.get(key)
            if existing_run and existing_run.get("status") == "ok":
                print(
                    f"[{run_index}/{total_runs}] Skipping {run_id}: "
                    "existing ok result",
                    flush=True,
                )
                continue

            print(f"[{run_index}/{total_runs}] Running {run_id}", flush=True)

            server_log_path = log_dir / f"server_{run_id}.log"
            benchmark_json_path = raw_dir / f"benchmark_{run_id}.json"
            proc: subprocess.Popen[Any] | None = None
            run_record: dict[str, Any] = {
                "run_id": run_id,
                "scenario": scenario_name,
                "input_len": input_len,
                "output_len": output_len,
                "max_concurrency": concurrency,
                "num_prompts": concurrency * 5,
                "status": "running",
                "started_at": now_iso(),
                "server_log": str(server_log_path),
                "benchmark_json": str(benchmark_json_path),
            }

            try:
                proc = start_server(config, server_log_path)
                run_record["server_cmd"] = getattr(proc, "cmd", None)
                wait_for_server_ready(
                    proc,
                    config.host,
                    config.port,
                    config.server_timeout_sec,
                    server_log_path,
                )
                if benchmark_json_path.exists():
                    benchmark_json_path.unlink()
                completed = run_benchmark(
                    config,
                    scenario_name,
                    input_len,
                    output_len,
                    concurrency,
                    benchmark_json_path,
                )
                benchmark_result = load_json(benchmark_json_path)
                run_record.update(
                    {
                        "status": "ok" if completed.returncode == 0 else "failed",
                        "benchmark_returncode": completed.returncode,
                        "benchmark_stdout": completed.stdout,
                        "benchmark_stderr": completed.stderr,
                        "metrics": extract_metrics(benchmark_result),
                        "raw_result": benchmark_result,
                    }
                )
                if completed.returncode != 0:
                    run_record["error"] = "benchmark subprocess failed"
                if not config.verbose and completed.returncode != 0:
                    print_subprocess_output(completed.stdout, completed.stderr)
            except Exception as exc:  # Keep the sweep going and preserve context.
                server_log_tail = tail(server_log_path)
                run_record.update(
                    {
                        "status": "failed",
                        "error": repr(exc),
                        "server_log_tail": server_log_tail,
                    }
                )
                print(f"Run {run_id} failed: {exc!r}", file=sys.stderr, flush=True)
                print("----- server log tail -----", file=sys.stderr)
                print(server_log_tail.rstrip(), file=sys.stderr)
            finally:
                stop_server(proc)
                run_record["finished_at"] = now_iso()
                runs_by_key[key] = run_record
                payload["finished_at"] = now_iso()
                update_payload_runs(payload, config, runs_by_key)
                write_results(results_path, payload)
                print(
                    f"[{run_index}/{total_runs}] Finished {run_id}: "
                    f"{run_record['status']}",
                    flush=True,
                )
                time.sleep(config.cooldown_sec)

    update_payload_runs(payload, config, runs_by_key)
    write_results(results_path, payload)
    print_final_results(payload["runs"])

    failed = [run for run in payload["runs"] if run.get("status") != "ok"]
    print(f"Wrote {results_path}", flush=True)
    if failed:
        print(f"{len(failed)} run(s) failed", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
