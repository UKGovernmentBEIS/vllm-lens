"""Create a fresh vLLM test environment and run local GPU compatibility checks.

Requires Linux, Python 3, uv, and working NVIDIA GPUs. Defaults to vLLM 0.30.0
and both suites (two GPUs). Use --suite smoke on a single-GPU machine.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"\n$ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vllm-version",
        default="0.30.0",
        help="explicit stable release (default: 0.30.0)",
    )
    parser.add_argument("--suite", choices=["all", "smoke", "parallel"], default="all")
    args = parser.parse_args()
    if not re.fullmatch(r"\d+\.\d+\.\d+", args.vllm_version):
        parser.error("--vllm-version must be an explicit stable version, e.g. 0.30.0")
    if sys.platform != "linux":
        parser.error("GPU compatibility checks require Linux")
    if os.environ.get("SLURM_JOB_ID") or shutil.which("srun") or shutil.which("sbatch"):
        if not (os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID")):
            parser.error(
                "Run inside a Slurm job step: submit scripts/compatibility.slurm "
                "with sbatch, or use srun inside your allocation"
            )
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            parser.error("Slurm must set CUDA_VISIBLE_DEVICES for the allocated GPUs")
    if not shutil.which("uv"):
        parser.error(
            "uv is required; see https://docs.astral.sh/uv/getting-started/installation/"
        )
    if not shutil.which("nvidia-smi"):
        parser.error(
            "nvidia-smi is unavailable; use a machine with working NVIDIA drivers"
        )

    required_gpus = 1 if args.suite == "smoke" else 2
    try:
        # Check the host driver before downloading several GB of dependencies.
        gpu_info = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            text=True,
            capture_output=True,
        )
        gpu_names = [line for line in gpu_info.stdout.splitlines() if line.strip()]
        if len(gpu_names) < required_gpus:
            parser.error(
                f"--suite {args.suite} requires {required_gpus} GPUs, found {len(gpu_names)}; "
                "use --suite smoke for single-GPU validation"
            )
        print("GPUs: " + ", ".join(gpu_names), flush=True)

        environment_root = ROOT / ".venv-compatibility"
        environment_root.mkdir(exist_ok=True)
        # Never clear an existing environment. uv's download cache is reused,
        # while each run gets its own dependency resolution and installation.
        environment = Path(
            tempfile.mkdtemp(prefix=f"vllm-{args.vllm_version}-", dir=environment_root)
        )
        python = str(environment / "bin" / "python")
        run(["uv", "venv", "--python", "3.12", str(environment)])
        overrides = environment / "overrides.txt"
        overrides.write_text(f"vllm=={args.vllm_version}\n")
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                python,
                "--overrides",
                str(overrides),
                "-e",
                str(ROOT),
                "-r",
                str(ROOT / "requirements" / "compatibility-tests.txt"),
            ]
        )
        # An absolute Python path does not activate console tools for children.
        env = os.environ.copy()
        env["PATH"] = (
            str(environment / "bin") + os.pathsep + env.get("PATH", os.defpath)
        )
        env["VIRTUAL_ENV"] = str(environment)
        # Test conftests must not load .env over scheduler-provided settings.
        env["PYTHON_DOTENV_DISABLED"] = "1"
        env.pop("RAY_ADDRESS", None)
        # nvidia-smi lists host GPUs; torch also checks wheel/driver compatibility
        # and honors CUDA_VISIBLE_DEVICES before either suite starts an engine.
        run(
            [
                python,
                "-c",
                (
                    "import torch; "
                    f"required = {required_gpus}; "
                    "count = torch.cuda.device_count(); "
                    "assert torch.cuda.is_available() and count >= required, "
                    "f'Need {required} visible CUDA GPUs, found {count}. Check the driver, "
                    "CUDA_VISIBLE_DEVICES, and installed vLLM/PyTorch wheels.'"
                ),
            ],
            env=env,
        )

        results_root = ROOT / "compatibility-results" / args.vllm_version
        results_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path(tempfile.mkdtemp(prefix=f"{stamp}-", dir=results_root))
        print(f"\nEnvironment: {environment}\nResults: {output}", flush=True)
        suites = ["smoke", "parallel"] if args.suite == "all" else [args.suite]
        for suite in suites:
            run(
                [
                    python,
                    str(ROOT / "scripts" / "check_compatibility.py"),
                    "--expected-vllm",
                    args.vllm_version,
                    "--suite",
                    suite,
                    "--output-dir",
                    str(output / suite),
                ],
                env=env,
            )
        print(
            f"\nPassed: {', '.join(suites)} on vLLM {args.vllm_version}. Reports: {output}"
        )
        return 0
    except subprocess.CalledProcessError as error:
        if error.stdout:
            print(error.stdout, file=sys.stderr)
        if error.stderr:
            print(error.stderr, file=sys.stderr)
        print(
            f"Compatibility run stopped: {shlex.join(error.cmd)} (exit {error.returncode})",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
