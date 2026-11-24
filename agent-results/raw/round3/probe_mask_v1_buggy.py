#!/usr/bin/env python
"""Probe 10: falsifiable per-row causal-mask check.

In this case prefix_len + (qlen-1) == seq_len, so *every* position below seq_len
is legal for at least one row; a whole-tensor poison test cannot distinguish
anything.  The falsifiable form is per row:

  for a sampled row r, poison every position in [prefix_len + r + 1, seq_len)
  -- i.e. exactly the future positions row r must not read -- then require
  row r's output to be bit-identical to the clean run.

Those positions are read legally by later rows, so their outputs may change;
only row r is asserted.  A leaked causal mask moves row r by O(1e3); a correct
mask leaves it bit-identical.  Negative control: poison row r's own visible
window [0, prefix_len + r] and require the output to change, proving the poison
actually reaches the kernel.
"""
import os, sys, json
REPO = os.environ["MSA_REPO"]
sys.path.insert(0, os.path.join(REPO, "src")); sys.path.insert(0, REPO)
import torch


def poisoned_run(b, data, tk, qlen, kvh, pages, offs_by_page):
    """Run the operator over a copy of the KV cache with the given positions set."""
    kv = data.kv_cache.clone()
    big = torch.tensor(30000.0, dtype=kv.dtype, device=kv.device)
    n = 0
    for page, offs in offs_by_page.items():
        if not offs:
            continue
        o = torch.tensor(sorted(offs), device=kv.device)
        kv[page, :, o, :] = big
        n += len(offs)
    saved = data.kv_cache
    data.kv_cache = kv
    try:
        out = torch.empty_like(data.q)
        b._call_sparse(b.minimax_m3_sparse_attn, data, tk, out, decode=False,
                       max_query_len=qlen, num_kv_heads=kvh, decode_qlen=1)
        torch.gcu.synchronize()
    finally:
        data.kv_cache = saved
    return out, n


def main():
    from benchmark import test_minimax_sparse_attention_benchmark as b
    torch.gcu.set_device(5)
    out = {}
    for tag, idx in (("main", 6), ("bypass", 1)):
        batch, qlen, seq, kvh, nh = b.PREFILL_SHAPES[idx]
        gen = torch.Generator(device="cpu").manual_seed(idx)
        data = b.make_data(batch, seq, kvh, nh, torch.device("gcu"), "bf16", decode=False,
                           decode_qlen=1, query_len=qlen, randomize_pages=True, generator=gen)
        tk = b.minimax_m3_index_topk(b.minimax_m3_index_score(
            data.idx_q, data.index_kv_cache, data.block_table, data.cu_q, data.seq_lens,
            data.prefix_lens, qlen, seq, kvh), data.cu_q, data.prefix_lens, qlen, 16, 1, 2)
        clean = torch.empty_like(data.q)
        b._call_sparse(b.minimax_m3_sparse_attn, data, tk, clean, decode=False,
                       max_query_len=qlen, num_kv_heads=kvh, decode_qlen=1)
        torch.gcu.synchronize()
        prefix_len = int(data.prefix_lens[0].item())
        bt = data.block_table.cpu()
        pages = {int(bt[0, int(blk)]) for blk in tk[0].reshape(-1).tolist() if int(blk) >= 0}

        def windows(lo_fn, hi_fn):
            d = {p: set() for p in pages}
            for p in pages:
                base = p * 128
                for o in range(128):
                    pos = base + o
                    if lo_fn(pos) <= pos < hi_fn(pos):
                        d[p].add(o)
            return d

        rows = sorted({0, 1, qlen // 2, qlen - 1})
        cases = {}
        for r in rows:
            q_abs = prefix_len + r
            future = windows(lambda pos, q=q_abs: q + 1, lambda pos: seq)
            dirty, n_f = poisoned_run(b, data, tk, qlen, kvh, pages, future)
            changed_f = int((dirty[r] != clean[r]).sum())
            d_f = (dirty[r].float() - clean[r].float()).abs().max().item()

            # negative control: poison only row r's own visible window
            visible = windows(lambda pos: 0, lambda pos, q=q_abs: q + 1)
            ctrl, n_v = poisoned_run(b, data, tk, qlen, kvh, pages, visible)
            changed_v = int((ctrl[r] != clean[r]).sum())
            d_v = (ctrl[r].float() - clean[r].float()).abs().max().item()

            cases[f"row{r}"] = dict(q_abs=q_abs,
                                    future_positions=n_f, future_changed_row=changed_f,
                                    future_max_delta_row=d_f, future_mask_enforced=(changed_f == 0),
                                    control_positions=n_v, control_changed_row=changed_v,
                                    control_max_delta_row=d_v, control_reaches_kernel=(changed_v > 0))
            print(f"[mask:{tag} row {r:4d}] future poisoned={n_f:5d} -> changed={changed_f} "
                  f"max_d={d_f:.4g} | control poisoned={n_v:5d} -> changed={changed_v} "
                  f"max_d={d_v:.4g} -> "
                  f"{'MASK ENFORCED' if changed_f == 0 and changed_v > 0 else 'INCONCLUSIVE/LEAK'}")
        out[tag] = dict(seq_len=seq, qlen=qlen, prefix_len=prefix_len, rows=cases)
    json.dump(out, open("/tmp/probe_mask.json", "w"), indent=1)


if __name__ == "__main__":
    main()
