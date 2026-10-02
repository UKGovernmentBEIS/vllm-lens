"""Exercise launcher failure handling without downloads or CUDA."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_compatibility.py"
    spec = importlib.util.spec_from_file_location("compatibility_setup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(
        module.shutil,
        "which",
        lambda name: None if name in ("srun", "sbatch") else f"/bin/{name}",
    )
    monkeypatch.setattr(module.sys, "platform", "linux")
    for name in ("SLURM_JOB_ID", "SLURM_STEP_ID", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    # The launcher should sanitize these only in its child environment.
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "0")
    monkeypatch.setenv("RAY_ADDRESS", "test-cluster")
    return module


def test_driver_failure_stops_before_install(launcher, monkeypatch, tmp_path):
    commands = []

    def fail(command, **kwargs):
        commands.append(command)
        raise subprocess.CalledProcessError(9, command, stderr="driver unavailable")

    monkeypatch.setattr(sys, "argv", ["run_compatibility.py"])
    monkeypatch.setattr(launcher.subprocess, "run", fail)
    assert launcher.main() == 1
    assert len(commands) == 1 and commands[0][0] == "nvidia-smi"
    assert not (tmp_path / ".venv-compatibility").exists()


@pytest.mark.parametrize("suite,gpus", [("all", 1), ("parallel", 1)])
def test_insufficient_gpus_stop_before_install(
    launcher, monkeypatch, tmp_path, suite, gpus
):
    monkeypatch.setattr(sys, "argv", ["run_compatibility.py", "--suite", suite])
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="GPU\n" * gpus
        ),
    )
    with pytest.raises(SystemExit, match="2"):
        launcher.main()
    assert not (tmp_path / ".venv-compatibility").exists()


@pytest.mark.parametrize(
    "suite,expected", [("all", ["smoke", "parallel"]), ("smoke", ["smoke"])]
)
def test_setup_runs_requested_suites_in_order(
    launcher, monkeypatch, tmp_path, suite, expected
):
    commands = []

    def succeed(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="GPU\nGPU\n")

    monkeypatch.setattr(sys, "argv", ["run_compatibility.py", "--suite", suite])
    monkeypatch.setattr(launcher.subprocess, "run", succeed)
    assert launcher.main() == 0
    suites = [
        command[command.index("--suite") + 1]
        for command in commands
        if "--suite" in command
    ]
    assert suites == expected
    install = next(
        command for command in commands if command[:3] == ["uv", "pip", "install"]
    )
    override = Path(install[install.index("--overrides") + 1])
    assert override.read_text() == "vllm==0.30.0\n"
    assert "--clear" not in [arg for command in commands for arg in command]


def test_smoke_failure_prevents_parallel_run(launcher, monkeypatch):
    commands = []

    def fail_smoke(command, **kwargs):
        commands.append(command)
        if "--suite" in command:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0, stdout="GPU\nGPU\n")

    monkeypatch.setattr(sys, "argv", ["run_compatibility.py"])
    monkeypatch.setattr(launcher.subprocess, "run", fail_smoke)
    assert launcher.main() == 1
    assert not any("parallel" in command for command in commands)


@pytest.mark.parametrize("job_id", [None, "123"])
def test_slurm_requires_job_step_before_touching_gpus(
    launcher, monkeypatch, tmp_path, job_id
):
    monkeypatch.setattr(sys, "argv", ["run_compatibility.py"])
    monkeypatch.setattr(launcher.shutil, "which", lambda name: f"/bin/{name}")
    if job_id:
        monkeypatch.setenv("SLURM_JOB_ID", job_id)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("must not run commands without allocation"),
    )
    with pytest.raises(SystemExit, match="2"):
        launcher.main()
    assert not (tmp_path / ".venv-compatibility").exists()


def test_slurm_requires_device_mask(launcher, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_compatibility.py"])
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "must not run commands without device mask"
        ),
    )
    with pytest.raises(SystemExit, match="2"):
        launcher.main()


def test_slurm_preserves_device_mask_and_ignores_dotenv(launcher, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_compatibility.py"])
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")

    def succeed(command, **kwargs):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "3,5"
        if "--suite" in command:
            env = kwargs["env"]
            assert env["CUDA_VISIBLE_DEVICES"] == "3,5"
            assert env["PYTHON_DOTENV_DISABLED"] == "1"
            assert "RAY_ADDRESS" not in env
        return subprocess.CompletedProcess(command, 0, stdout="GPU\nGPU\n")

    monkeypatch.setattr(launcher.subprocess, "run", succeed)
    assert launcher.main() == 0


def make_ninja(bin_dir):
    bin_dir.mkdir(parents=True, exist_ok=True)
    ninja = bin_dir / "ninja"
    ninja.write_text("#!/bin/sh\necho environment-ninja\n")
    ninja.chmod(0o755)


def test_launcher_children_find_environment_tools(launcher, monkeypatch, tmp_path):
    real_run = subprocess.run
    probes = []
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    monkeypatch.setattr(sys, "argv", ["run_compatibility.py"])

    def run(command, **kwargs):
        if command[:2] == ["uv", "venv"]:
            make_ninja(Path(command[-1]) / "bin")
        if command[0].endswith("/bin/python"):
            env = kwargs["env"]
            assert env["VIRTUAL_ENV"] == str(Path(command[0]).parent.parent)
            # Exercise actual child/grandchild executable lookup, as FlashInfer does.
            result = real_run(
                [
                    sys.executable,
                    "-c",
                    "import subprocess; subprocess.run(['ninja'], check=True)",
                ],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            assert result.stdout.strip() == "environment-ninja"
            probes.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="GPU\nGPU\n")

    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert launcher.main() == 0
    assert len(probes) == 3  # CUDA preflight, smoke, parallel
    assert os.environ["PATH"] == str(tmp_path / "empty-path")
    assert os.environ["RAY_ADDRESS"] == "test-cluster"


def test_direct_checker_children_find_environment_tools(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[2] / "scripts" / "check_compatibility.py"
    spec = importlib.util.spec_from_file_location("compatibility_checker", path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    bin_dir = tmp_path / "venv" / "bin"
    make_ninja(bin_dir)
    # Preserve the venv bin directory even when Python is a symlink.
    python = bin_dir / "python"
    python.symlink_to(sys.executable)
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setattr(sys, "prefix", str(bin_dir.parent))
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    env = checker.subprocess_environment()
    result = subprocess.run(
        [str(python), "-c", "import subprocess; subprocess.run(['ninja'], check=True)"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "environment-ninja"
    assert env["VIRTUAL_ENV"] == str(bin_dir.parent)


def test_checker_rejects_inaccessible_git_metadata_before_cuda(monkeypatch, tmp_path):
    import torch

    path = Path(__file__).resolve().parents[2] / "scripts" / "check_compatibility.py"
    spec = importlib.util.spec_from_file_location("compatibility_checker", path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    monkeypatch.setattr(checker.shutil, "which", lambda name: None)
    for name in ("SLURM_JOB_ID", "SLURM_STEP_ID", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        checker,
        "command_output",
        lambda args: "fatal: not a git repository" if args[0] == "git" else "",
    )
    monkeypatch.setattr(
        torch.cuda,
        "device_count",
        lambda: pytest.fail("invalid revision must fail before CUDA checks"),
    )
    output = tmp_path / "reports"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_compatibility.py",
            "--expected-vllm",
            "0.30.0",
            "--output-dir",
            str(output),
        ],
    )
    assert checker.main() == 1
    report = json.loads((output / "environment.json").read_text())
    assert report["status"] == "failed"
    assert report["results"] == []
    assert "Git revision" in report["error"]
