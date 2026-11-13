"""GLA-007 real-kernel fallback/parallel smoke gate."""
import os
from pathlib import Path
import subprocess


def test_fp16_parallel_and_fallback_full_call():
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}:{root}:" + env.get("PYTHONPATH", "")
    run = subprocess.run(
        ["/flagos/bin/python", "agent-results/scripts/gla007_dtype_ab.py",
         "--smoke", "--samples", "2"],
        cwd=root, env=env, text=True, capture_output=True, timeout=600,
    )
    assert run.returncode == 0, run.stdout[-3000:] + "\n" + run.stderr[-3000:]

