#!/usr/bin/env python
"""Probe 12: does the grouped8 row-1 causal-mask leak exist at pristine 984069d?

Same per-row poison test as probe_mask, but the whole-vector poison is applied
only to positions strictly after q_abs, and the asserted row is compared against
that same build's clean output.  Run twice: pristine worktree and candidate.
"""
import os, sys, json
REPO = os.environ["MSA_REPO"]
sys.path.insert(0, os.path.join(REPO, "src")); sys.path.insert(0, REPO)
import torch


def main():
    from benchmark import test_minimax_sparse_attention_benchmark as b
    torch.gcu.set_device(5)
    idx = int(os.environ.get("MSA_SHAPE_INDEX", "1"))
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
    out = {}
    for r in sorted({int(x) for x in os.environ.get("MSA_ROWS", "0,1,7,8,15,16").split(",")}):
        q_abs = prefix_len + r
        kv = data.kv_cache.clone()
        big = torch.tensor(float(os.environ.get("MSA_POISON", "30000")), dtype=kv.dtype, device=kv.device)
        n = 0
        for p in pages:
            base = p * 128
            offs = [o for o in range(128) if q_abs < base + o < seq]
            if offs:
                kv[p, :, torch.tensor(offs, device=kv.device), :] = big
                n += len(offs)
        saved = data.kv_cache
        data.kv_cache = kv
        try:
            dirty = torch.empty_like(data.q)
            b._call_sparse(b.minimax_m3_sparse_attn, data, tk, dirty, decode=False,
                           max_query_len=qlen, num_kv_heads=kvh, decode_qlen=1)
            torch.gcu.synchronize()
        finally:
            data.kv_cache = saved
        ch = int((dirty[r] != clean[r]).sum())
        d = (dirty[r].float() - clean[r].float()).abs().max().item()
        out[f"row{r}"] = dict(q_abs=q_abs, poisoned=n, changed=ch, max_delta=d)
        print(f"[{os.environ.get('MSA_LABEL','?')} row {r:4d}] q_abs={q_abs} poisoned={n} "
              f"changed={ch} max_d={d:.4g} -> {'mask ok' if ch == 0 else 'MASK LEAK'}")
    json.dump(out, open(os.environ.get("MSA_OUT", "/tmp/probe_mask_pristine.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
