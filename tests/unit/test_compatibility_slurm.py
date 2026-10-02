"""Check batch-script handoff without submitting a real Slurm job."""

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "compatibility.slurm"


def test_batch_script_requires_allocation():
    env = os.environ.copy()
    env.pop("SLURM_JOB_ID", None)
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "Submit this script with sbatch" in result.stderr


def test_batch_script_preserves_allocation_and_isolates_job(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "step.json"
    fake_srun = bin_dir / "srun"
    fake_srun.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['STEP_RECORD']).write_text(json.dumps({\n"
        "'args': sys.argv[1:],\n"
        "'env': {k: os.environ.get(k) for k in [\n"
        "'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'PYTHON_DOTENV_DISABLED',\n"
        "'RAY_ADDRESS', 'TMPDIR', 'RAY_TMPDIR', 'VLLM_TEST_PORT']}\n"
        "}))\n"
    )
    fake_srun.chmod(0o755)
    sibling = tmp_path / "another-job"
    sibling.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env.get('PATH', '')}",
            "STEP_RECORD": str(record),
            "SLURM_JOB_ID": "123",
            "SLURM_SUBMIT_DIR": str(ROOT),
            "SLURM_TMPDIR": str(tmp_path),
            "SLURM_CPUS_PER_TASK": "4",
            "CUDA_VISIBLE_DEVICES": "3,5",
            "RAY_ADDRESS": "another-job",
        }
    )
    env.pop("VLLM_TEST_PORT", None)
    result = subprocess.run(
        ["bash", str(SCRIPT), "--suite", "smoke"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    step = json.loads(record.read_text())
    assert step["args"][-4:] == [
        "python3",
        "scripts/run_compatibility.py",
        "--suite",
        "smoke",
    ]
    observed = step["env"]
    assert observed["CUDA_VISIBLE_DEVICES"] == "3,5"
    assert observed["OMP_NUM_THREADS"] == "4"
    assert observed["PYTHON_DOTENV_DISABLED"] == "1"
    assert observed["RAY_ADDRESS"] is None
    assert observed["VLLM_TEST_PORT"] == "20123"
    assert observed["RAY_TMPDIR"] == observed["TMPDIR"] + "/ray"
    assert not Path(observed["TMPDIR"]).exists()
    assert sibling.is_dir()
