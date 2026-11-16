"""Negative control: the A/B runner must reject a wrong current-stream device.

Loads ``msa003_ab`` with a stubbed current stream on GCU 2; no GPU kernel of
the sweep runs. Exit 0 here means the runner correctly returned non-zero.
"""
import os
import sys

os.environ.setdefault("FLAG_ATTN_TEST_DEVICE", "gcu:2")
os.environ.setdefault("S60_TEST_DEVICE", "2")
ROOT = "/workspace/agent-workspaces/msa-group16-formal-5"
os.environ["MSA_AB_OUT"] = os.path.join(
    ROOT, "agent-results", "raw", "MSA-005", "negative", "ab_neg_stream.jsonl"
)
sys.path.insert(0, os.path.join(ROOT, "agent-results", "scripts"))
sys.argv = ["msa003_ab.py"]

import torch
import torch_gcu  # noqa: F401

import msa003_ab as ab

real_stream = torch.gcu.current_stream


class Wrong:
    device = torch.device("gcu:5")


torch.gcu.current_stream = lambda *a, **k: Wrong()
try:
    code = ab.main()
finally:
    torch.gcu.current_stream = real_stream
print("NEGATIVE_STREAM_RUNNER_EXIT", code)
raise SystemExit(0 if code != 0 else 1)
