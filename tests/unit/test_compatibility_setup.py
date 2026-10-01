"""Exercise launcher failure handling without downloads or CUDA."""

import importlib.util
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
    monkeypatch.setattr(module.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(module.sys, "platform", "linux")
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
