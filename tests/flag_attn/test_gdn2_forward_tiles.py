"""GDN2-003/004: S60 per-call tile selection for the fused Enflame forward.

The fused GDN-2 inference kernels used a fixed BK=BV=256 tile even when K or V
was only 64, so most lanes of every dot were masked.  GDN2-003 clamps the tile
to the real K/V extent (never below 64, never above the tuned 256).  GDN2-004
splits the K2 (state + output) V tile in two when ``128 < V <= 256``: that
kernel is serial over chunks and parallel only over (V-tile, batch*head), so
one 256-wide tile leaves half the machine idle.  These tests pin both tile
maps, show that the clamps change only launch geometry, and compare the tuned
launch with the legacy fixed-256 launch on device.

Nothing here exercises a backward pass: ``chunk_gdn2`` is inference-only, so
device gradients remain UNTESTED and no CPU FP64 adjoint stands in for them.
"""

import importlib
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "tests" / "flag_attn") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests" / "flag_attn"))

gdn2_impl = importlib.import_module(
    "flag_attn.runtime.backend._enflame.FLA.gdn2.chunk_gdn2"
)
from flag_attn.testing import backend as test_backend  # noqa: E402
from gdn2_backward_ref import gdn2_forward  # noqa: E402

pytestmark = pytest.mark.skipif(
    not test_backend.supports_operator("chunk_gdn2"),
    reason="chunk_gdn2 is unavailable on the active backend",
)

TILE_CASES = [
    # (extent, limit, expected tile)
    (1, 256, 64), (32, 256, 64), (64, 256, 64), (65, 256, 128),
    (128, 256, 128), (129, 256, 256), (255, 256, 256), (256, 256, 256),
    (64, 128, 64), (64, 64, 64), (256, 128, 128),
]

# (V, expected K2 state V tile): unchanged below/at 128, halved for 129..256,
# back to 256 above that (a blanket 128 cap regressed V=512).
STATE_BV_CASES = [
    (1, 64), (8, 64), (64, 64), (96, 128), (128, 128),
    (129, 128), (160, 128), (192, 128), (224, 128), (256, 128),
    (257, 256), (320, 256), (384, 256), (512, 256),
]


@pytest.mark.parametrize("extent,limit,expected", TILE_CASES)
def test_s60_tile_is_power_of_two_and_covers_extent(extent, limit, expected):
    tile = gdn2_impl._s60_tile(extent, limit)
    assert tile == expected
    assert 64 <= tile <= 256
    assert tile & (tile - 1) == 0, "tile must stay a power of two"
    assert tile >= min(extent, limit), "tile must cover the logical extent"


def test_s60_tile_respects_limit_and_legacy_switch(monkeypatch):
    assert gdn2_impl._s60_tile(256, 128) == 128
    monkeypatch.setattr(gdn2_impl, "_S60_TILES_ADAPT", False)
    assert gdn2_impl._s60_tile(64, 256) == 256, "legacy switch must restore the fixed tile"


@pytest.mark.parametrize("extent,expected", STATE_BV_CASES)
def test_s60_state_bv_splits_only_the_middle_range(extent, expected):
    tile = gdn2_impl._s60_state_bv(extent)
    assert tile == expected
    assert tile & (tile - 1) == 0, "tile must stay a power of two"
    assert 64 <= tile <= 256
    if 128 < extent <= 256:
        # The split fires: K2 gets exactly two V tiles and each tile is
        # narrower than the logical extent -- that is the point of the split.
        assert -(-extent // tile) == 2
    else:
        assert tile >= min(extent, 256), "tile must cover the logical V extent"


def test_s60_state_bv_switches_restore_the_accepted_rule(monkeypatch):
    """Both kill switches must reproduce the GDN2-003 K2 V tile exactly."""
    assert gdn2_impl._s60_state_bv(160) == 128, "split must be the default"
    monkeypatch.setattr(gdn2_impl, "_S60_STATE_BV_SPLIT_ENABLED", False)
    for extent, _ in STATE_BV_CASES:
        assert gdn2_impl._s60_state_bv(extent) == gdn2_impl._s60_tile(
            extent, 256), "FLAG_ATTN_GDN2_STATE_BV_SPLIT=0 must restore 256"
    monkeypatch.setattr(gdn2_impl, "_S60_TILES_ADAPT", False)
    assert gdn2_impl._s60_state_bv(160) == 256, "legacy tile switch must win too"


def _make(B, T, H, K, V, dtype, device):
    torch.manual_seed(42)
    q = (torch.randn(B, T, H, K) / math.sqrt(K)).to(dtype)
    k = (torch.randn(B, T, H, K) / math.sqrt(K)).to(dtype)
    v = torch.randn(B, T, H, V).to(dtype)
    g = (-torch.rand(B, T, H, K) * 0.1).to(dtype)
    b = torch.rand(B, T, H, K).to(dtype)
    w = torch.rand(B, T, H, V).to(dtype)
    h0 = torch.randn(B, H, K, V) * 0.01
    return [x.to(device) for x in (q, k, v, g, b, w, h0)]


def _capture_launches(monkeypatch):
    from triton.runtime.jit import JITFunction

    seen = []
    original = JITFunction.run

    def traced(self, *args, **kwargs):
        seen.append({
            "kernel": getattr(self.fn, "__name__", str(self.fn)),
            "BK": kwargs.get("BK"),
            "BV": kwargs.get("BV"),
            "BT": kwargs.get("BT"),
        })
        return original(self, *args, **kwargs)

    monkeypatch.setattr(JITFunction, "run", traced)
    return seen


def _last_tiles(seen):
    """Tiles of the last launch of each kernel (ignoring repeat launches)."""
    by_name = {}
    for item in seen:
        by_name[item["kernel"]] = item
    return by_name


@pytest.mark.chunk_gdn2
@pytest.mark.parametrize(
    "shape,tiles",
    [
        ((2, 130, 2, 64, 64), {"intra": (64, 64), "state": (64, 64)}),
        ((2, 130, 2, 128, 128), {"intra": (128, 128), "state": (128, 128)}),
        ((1, 130, 1, 256, 256), {"intra": (256, 256), "state": (256, 128)}),
        # GDN2-004: K != V and non-power-of-two K/V, all inside the K<=256 guard.
        ((2, 65, 2, 64, 128), {"intra": (64, 128), "state": (64, 128)}),
        ((2, 65, 2, 128, 64), {"intra": (128, 64), "state": (128, 64)}),
        ((2, 130, 2, 96, 96), {"intra": (128, 128), "state": (128, 128)}),
        ((2, 130, 2, 96, 160), {"intra": (128, 256), "state": (128, 128)}),
        ((2, 65, 1, 160, 64), {"intra": (256, 64), "state": (256, 64)}),
    ],
    ids=["k64v64", "k128v128", "k256v256", "k64v128", "k128v64",
         "k96v96", "k96v160", "k160v64"],
)
@torch.inference_mode()
def test_adaptive_tiles_are_launched_and_output_is_unchanged(shape, tiles, monkeypatch):
    from flag_attn import chunk_gdn2

    device = test_backend.device
    B, T, H, K, V = shape
    args = _make(B, T, H, K, V, torch.bfloat16, device)
    scale = K ** -0.5
    kwargs = dict(scale=scale, initial_state=args[6], output_final_state=True, chunk_size=64)

    monkeypatch.setattr(gdn2_impl, "_S60_TILES_ADAPT", True)
    monkeypatch.setattr(gdn2_impl, "_S60_STATE_BV_SPLIT_ENABLED", True)
    seen = _capture_launches(monkeypatch)
    o_new, ht_new = chunk_gdn2(*args[:6], **kwargs)
    torch.gcu.synchronize()
    assert seen, "no Triton kernel was launched"
    by_name = _last_tiles(seen)
    intra = by_name["_chunk_gdn2_fwd_intra_infer_kernel"]
    state = by_name["_chunk_gdn2_fwd_h_o_infer_kernel"]
    assert (intra["BK"], intra["BV"]) == tiles["intra"]
    assert (state["BK"], state["BV"]) == tiles["state"]

    monkeypatch.setattr(gdn2_impl, "_S60_TILES_ADAPT", False)
    o_ref, ht_ref = chunk_gdn2(*args[:6], **kwargs)
    torch.gcu.synchronize()

    assert torch.equal(o_new, o_ref), "adaptive tile changed the output"
    assert torch.equal(ht_new, ht_ref), "adaptive tile changed the final state"
    assert torch.isfinite(o_new).all() and torch.isfinite(ht_new).all()


@pytest.mark.chunk_gdn2
@pytest.mark.parametrize(
    "shape",
    [
        (2, 1, 1, 96, 96),
        (2, 64, 1, 96, 64),
        (2, 65, 1, 64, 96),
        (2, 129, 1, 96, 128),
        (2, 130, 1, 128, 96),
        (2, 200, 1, 96, 96),
        (1, 65, 1, 160, 64),
        (2, 65, 2, 64, 128),
        (3, 65, 2, 128, 64),
        (2, 65, 1, 63, 65),
    ],
    ids=["t1-k96v96", "t64-k96v64", "t65-k64v96", "t129-k96v128",
         "t130-k128v96", "t200-k96v96", "k160v64", "k64v128",
         "b3-k128v64", "k63v65"],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16],
                         ids=["fp16", "bf16"])
@torch.inference_mode()
def test_irregular_shapes_match_cpu_fp64_reference(shape, dtype):
    """K != V and non-power-of-two K/V against an independent FP64 recurrence.

    Gate A (strict) is asserted for FP16; BF16 is held to Gate B because its
    final state scales with |ref|max rather than ref RMS (a pre-existing dtype
    property, identical before and after the GDN2-004 tile change).
    """
    from flag_attn import chunk_gdn2

    B, T, H, K, V = shape
    args = _make(B, T, H, K, V, dtype, test_backend.device)
    ref_o, ref_ht = gdn2_forward(*[x.cpu() for x in args[:6]], K ** -0.5, args[6].cpu())
    o, ht = chunk_gdn2(*args[:6], scale=K ** -0.5, initial_state=args[6],
                       output_final_state=True, chunk_size=64)
    torch.gcu.synchronize()

    assert o.shape == (B, T, H, V)
    assert ht.shape == (B, H, K, V)
    assert torch.isfinite(o).all() and torch.isfinite(ht).all()

    for label, ref, act in (("o", ref_o, o), ("ht", ref_ht, ht)):
        e = ref.double().flatten()
        a = act.double().flatten().cpu()
        diff = (e - a).abs()
        rms = diff.square().mean().sqrt().item()
        ref_rms = e.square().mean().sqrt().item()
        rms_rel = rms / (ref_rms + 1e-12)
        max_abs = diff.max().item()
        gate_a = max_abs <= 1e-3 + 2e-2 * ref_rms and rms_rel <= 2e-2
        gate_b = max_abs <= 1e-3 + 2e-2 * e.abs().max().item() and rms_rel <= 2e-2
        ok = gate_a if dtype == torch.float16 else gate_b
        assert ok, (f"{label} {shape} {dtype}: max_abs={max_abs:.3e} "
                    f"rms_rel={rms_rel:.3e} gateA={gate_a} gateB={gate_b}")


@pytest.mark.chunk_gdn2
@pytest.mark.parametrize(
    "V", [96, 129, 160, 192, 256, 257, 512],
    ids=lambda v: f"v{v}",
)
@torch.inference_mode()
def test_state_bv_split_is_bit_identical(V):
    """The split only changes launch geometry: outputs must be bit-identical."""
    from flag_attn import chunk_gdn2

    device = test_backend.device
    K = 64
    args = _make(2, 65, 2, K, V, torch.float16, device)
    kwargs = dict(scale=K ** -0.5, initial_state=args[6],
                  output_final_state=True, chunk_size=64)

    old = gdn2_impl._S60_STATE_BV_SPLIT_ENABLED
    try:
        gdn2_impl._S60_STATE_BV_SPLIT_ENABLED = True
        o_split, ht_split = chunk_gdn2(*args[:6], **kwargs)
        gdn2_impl._S60_STATE_BV_SPLIT_ENABLED = False
        o_whole, ht_whole = chunk_gdn2(*args[:6], **kwargs)
        torch.gcu.synchronize()
    finally:
        gdn2_impl._S60_STATE_BV_SPLIT_ENABLED = old

    assert torch.equal(o_split, o_whole), f"V={V}: output changed"
    assert torch.equal(ht_split, ht_whole), f"V={V}: final state changed"
    assert torch.isfinite(o_split).all() and torch.isfinite(ht_split).all()

# --- PROFILEGDN2-0001: opt-in K2 final-state allocation --------------------
#
# K2's ``final_state`` is zero-filled with ``new_zeros`` before the launch, but
# the kernel stores every element of it, so the fill is a device memset that is
# immediately overwritten.  The experiment allocates it with ``new_empty`` when
# FLAG_ATTN_GDN2_STATE_EMPTY_HT=1; the default must stay ``new_zeros``.


def test_empty_ht_switch_is_off_by_default():
    """The shipped operator must keep the zero-filled allocation."""
    assert gdn2_impl._S60_STATE_EMPTY_HT_ENABLED is False


def test_empty_ht_switch_reads_the_environment(monkeypatch):
    import importlib as _importlib

    monkeypatch.setenv("FLAG_ATTN_GDN2_STATE_EMPTY_HT", "1")
    reloaded = _importlib.reload(gdn2_impl)
    try:
        assert reloaded._S60_STATE_EMPTY_HT_ENABLED is True
    finally:
        monkeypatch.delenv("FLAG_ATTN_GDN2_STATE_EMPTY_HT", raising=False)
        _importlib.reload(gdn2_impl)


@pytest.mark.chunk_gdn2
@pytest.mark.parametrize(
    "shape,dtype",
    [((2, 65, 2, 64, 96), torch.float16),
     ((2, 130, 3, 128, 160), torch.bfloat16),
     ((1, 65, 2, 96, 128), torch.float16),
     ((2, 65, 2, 256, 256), torch.bfloat16)],
    ids=lambda x: str(x),
)
@torch.inference_mode()
def test_empty_ht_allocation_is_bit_identical(shape, dtype):
    """Skipping the zero fill cannot change any output bit."""
    from flag_attn import chunk_gdn2

    B, T, H, K, V = shape
    args = _make(B, T, H, K, V, dtype, test_backend.device)
    kwargs = dict(scale=K ** -0.5, initial_state=args[6],
                  output_final_state=True, chunk_size=64)

    old = gdn2_impl._S60_STATE_EMPTY_HT_ENABLED
    try:
        gdn2_impl._S60_STATE_EMPTY_HT_ENABLED = False
        o_zeros, ht_zeros = chunk_gdn2(*args[:6], **kwargs)
        gdn2_impl._S60_STATE_EMPTY_HT_ENABLED = True
        o_empty, ht_empty = chunk_gdn2(*args[:6], **kwargs)
        torch.gcu.synchronize()
    finally:
        gdn2_impl._S60_STATE_EMPTY_HT_ENABLED = old

    assert torch.equal(o_zeros, o_empty), f"{shape}: output changed"
    assert torch.equal(ht_zeros, ht_empty), f"{shape}: final state changed"
    assert torch.isfinite(o_empty).all() and torch.isfinite(ht_empty).all()


@pytest.mark.chunk_gdn2
@torch.inference_mode()
def test_empty_ht_falls_back_when_bk_does_not_cover_k(monkeypatch):
    """The guard must refuse the experiment if K is beyond the BK chain.

    ``K <= 4 * state_bk`` is what makes the kernel write the whole state.  With
    an artificially small BK ladder the guard has to fall back to the zeroed
    allocation rather than return uninitialised memory.
    """
    from flag_attn import chunk_gdn2

    monkeypatch.setattr(gdn2_impl, "_S60_STATE_BK_SMALL", 16)
    monkeypatch.setattr(gdn2_impl, "_S60_STATE_BK_MEDIUM", 16)
    monkeypatch.setattr(gdn2_impl, "_S60_STATE_BK_LARGE", 16)
    monkeypatch.setattr(gdn2_impl, "_S60_STATE_EMPTY_HT_ENABLED", True)

    args = _make(1, 65, 2, 256, 64, torch.float16, test_backend.device)
    o, ht = chunk_gdn2(*args[:6], scale=256 ** -0.5, initial_state=args[6],
                       output_final_state=True, chunk_size=64)
    torch.gcu.synchronize()
    # K=256 > 4*16 means the guarded path stays on new_zeros; the state must be
    # fully finite and equal to the default run.
    assert torch.isfinite(ht).all()
    gdn2_impl._S60_STATE_EMPTY_HT_ENABLED = False
    o_ref, ht_ref = chunk_gdn2(*args[:6], scale=256 ** -0.5, initial_state=args[6],
                               output_final_state=True, chunk_size=64)
    assert torch.equal(o, o_ref) and torch.equal(ht, ht_ref)
