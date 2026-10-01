"""Run the bounded GPU compatibility suites and retain reproducible evidence.

Use a fresh environment for each vLLM version; see docs/compatibility.md.
Each suite has its own process because the existing GPU fixtures manage vLLM
subprocesses and terminate the pytest interpreter during teardown.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SMOKE = [
    (
        "discovery",
        [
            "vllm_lens/tests/test_layer_discovery_unit.py",
            "vllm_lens/tests/test_layer_discovery.py",
        ],
    ),
    (
        "capture",
        ["vllm_lens/tests/test_activations_offline.py", "-k", "not tp2 and not pp2"],
    ),
    ("chunked-prefill", ["vllm_lens/tests/test_activations_chunked_prefill.py"]),
    ("offline-interventions", ["vllm_lens/tests/test_steering_offline.py"]),
    (
        "async-interventions",
        ["vllm_lens/tests/test_steering.py", "-k", "not tp2 and not pp2"],
    ),
    ("http", ["tests/test_compatibility_http.py"]),
]
PARALLEL = [
    (
        "parallel-capture",
        ["vllm_lens/tests/test_activations_offline.py", "-k", "tp2 or pp2"],
    ),
    ("pipeline", ["vllm_lens/tests/test_activations_pp.py"]),
    ("parallel-steering", ["vllm_lens/tests/test_steering.py", "-k", "tp2 or pp2"]),
]


def command_output(args: list[str]) -> str:
    try:
        result = subprocess.run(
            args, cwd=ROOT, text=True, capture_output=True, timeout=30
        )
        return (result.stdout + result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)


def subprocess_environment() -> dict[str, str]:
    env = os.environ.copy()
    # Also support invoking this script directly by the environment's Python.
    # Do not resolve symlinks: venv Python may link to a system interpreter.
    env["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", os.defpath)
    )
    env["VIRTUAL_ENV"] = sys.prefix
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["smoke", "parallel"], default="smoke")
    parser.add_argument("--expected-vllm", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "compatibility-results"
    )
    args = parser.parse_args()
    if os.environ.get("SLURM_JOB_ID") or shutil.which("srun") or shutil.which("sbatch"):
        if not (os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID")):
            parser.error(
                "Run GPU tests in an srun step; submit scripts/compatibility.slurm"
            )
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            parser.error("Slurm must set CUDA_VISIBLE_DEVICES for the allocated GPUs")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "revision": command_output(["git", "rev-parse", "HEAD"]),
        "working_tree": command_output(["git", "status", "--porcelain"]),
        "python": sys.version,
        "platform": platform.platform(),
        "expected_vllm": args.expected_vllm,
        "suite": args.suite,
        "runner": "V1, eager, prefix caching enabled",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "nvidia_smi": command_output(["nvidia-smi"]),
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_STEP_ID",
                "SLURM_JOB_NODELIST",
                "SLURM_CPUS_PER_TASK",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "packages": dict(
            sorted(
                (dist.metadata["Name"], dist.version)
                for dist in importlib.metadata.distributions()
                if dist.metadata["Name"]
            )
        ),
        "results": [],
        "status": "failed",
    }
    report_path = output / "environment.json"
    try:
        import torch

        actual = importlib.metadata.version("vllm")
        if actual != args.expected_vllm:
            raise RuntimeError(
                f"Expected vLLM {args.expected_vllm}, installed {actual}"
            )
        required = 2 if args.suite == "parallel" else 1
        if torch.cuda.device_count() < required:
            raise RuntimeError(f"{args.suite} requires {required} visible CUDA GPU(s)")
        report["torch_cuda"] = torch.version.cuda
        env = subprocess_environment()
        if not shutil.which("ninja", path=env["PATH"]):
            raise RuntimeError(
                "FlashInfer requires ninja; install requirements/compatibility-tests.txt "
                "in this Python environment"
            )
        subprocess.run(["ninja", "--version"], env=env, check=True, timeout=30)
        env.pop("VLLM_LENS_DISABLE", None)
        env.pop("VLLM_LENS_LAYER_PATH", None)
        env.pop("RAY_ADDRESS", None)
        env.update(
            {
                "VLLM_LENS_STRICT_COMPATIBILITY": "1",
                "VLLM_USE_V2_MODEL_RUNNER": "0",
                "VLLM_TEST_MODEL": report["model"],
                "VLLM_TEST_TP_SIZE": "1",
                "VLLM_TEST_PP_SIZE": "1",
                "VLLM_TEST_MAX_MODEL_LEN": "2048",
                "VLLM_TEST_REUSE_SERVER": "0",
                "PYTHONUNBUFFERED": "1",
                "PYTHON_DOTENV_DISABLED": "1",
            }
        )
        for name, tests in SMOKE if args.suite == "smoke" else PARALLEL:
            junit = output / f"{name}.xml"
            junit.unlink(missing_ok=True)
            command = [
                sys.executable,
                "-m",
                "pytest",
                *tests,
                "-v",
                f"--junitxml={junit}",
            ]
            print(f"Running {name}; log: {output / (name + '.log')}", flush=True)
            with (output / f"{name}.log").open("w") as log:
                with subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                ) as process:
                    try:
                        code = process.wait(timeout=1800)
                    except subprocess.TimeoutExpired:
                        # Workers inherit this process group; killing only
                        # pytest would leave model processes holding GPU memory.
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        code = 124
            # A missing report, empty selection or skipped GPU test must never
            # turn a candidate version into a passing compatibility result.
            cases = list(ET.parse(junit).iter("testcase")) if junit.exists() else []
            passed = (
                code == 0
                and bool(cases)
                and all(
                    case.find(tag) is None
                    for case in cases
                    for tag in ("failure", "error", "skipped")
                )
            )
            report["results"].append(
                {
                    "name": name,
                    "command": command,
                    "exit_code": code,
                    "tests": len(cases),
                    "passed": passed,
                }
            )
            print(f"{name}: {'PASS' if passed else 'FAIL'}", flush=True)
            if not passed:
                # Stop before another engine starts after a timeout/failure.
                break
        expected_count = len(SMOKE if args.suite == "smoke" else PARALLEL)
        if len(report["results"]) == expected_count and all(
            item["passed"] for item in report["results"]
        ):
            report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        print(report["error"], file=sys.stderr)
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Evidence: {report_path}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
