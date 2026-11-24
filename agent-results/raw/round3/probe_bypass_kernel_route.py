#!/usr/bin/env python
"""Probe 11: which kernel serves the B1/Q512/KV4096 bypass shape, and does the
row-1 mask leak also exist on the pristine baseline?"""
import os, sys, json
REPO = os.environ["MSA_REPO"]
sys.path.insert(0, os.path.join(REPO, "src")); sys.path.insert(0, REPO)
import torch

def main():
    from benchmark import test_minimax_sparse_attention_benchmark as b
    from flag_attn.runtime.backend._enflame.msa import sparse_attn as sa
    torch.gcu.set_device(5)
    res = {}
    for tag, idx in (("bypass", 1), ("main", 6)):
        batch, qlen, seq, kvh, nh = b.PREFILL_SHAPES[idx]
        gen = torch.Generator(device="cpu").manual_seed(idx)
        data = b.make_data(batch, seq, kvh, nh, torch.device("gcu"), "bf16", decode=False,
                           decode_qlen=1, query_len=qlen, randomize_pages=True, generator=gen)
        tk = b.minimax_m3_index_topk(b.minimax_m3_index_score(
            data.idx_q, data.index_kv_cache, data.block_table, data.cu_q, data.seq_lens,
            data.prefix_lens, qlen, seq, kvh), data.cu_q, data.prefix_lens, qlen, 16, 1, 2)
        names = ["_gqa_sparse_common_prefix", "_gqa_sparse_fwd_direct",
                 "_gqa_sparse_fwd_grouped8", "_gqa_sparse_decode_kernel_enflame"]
        orig = {n: getattr(sa, n) for n in names}
        log = []
        for n in names:
            fn = orig[n]
            class W:
                def __init__(self, fn, n): self.fn, self.n = fn, n
                def __getitem__(self, grid):
                    outer = self
                    class L:
                        def __call__(inner, *a, **k):
                            log.append({"kernel": outer.n, "grid": list(grid)})
                            return outer.fn[grid](*a, **k)
                    return L()
            setattr(sa, n, W(fn, n))
        try:
            out = torch.empty_like(data.q)
            b._call_sparse(b.minimax_m3_sparse_attn, data, tk, out, decode=False,
                           max_query_len=qlen, num_kv_heads=kvh, decode_qlen=1)
            torch.gcu.synchronize()
        finally:
            for n in names: setattr(sa, n, orig[n])
        res[tag] = dict(block_table_width=int(data.block_table.shape[1]),
                        topk=int(tk.shape[-1]), launches=log)
        print(f"[{tag}] block_table_width={data.block_table.shape[1]} topk={tk.shape[-1]} "
              f"kernels={[ (x['kernel'], x['grid']) for x in log]}")
    json.dump(res, open("/tmp/probe_bypass.json", "w"), indent=1)

if __name__ == "__main__":
    main()
