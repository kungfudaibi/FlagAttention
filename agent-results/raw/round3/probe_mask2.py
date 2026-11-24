#!/usr/bin/env python
"""Probe 13 (corrected): per-row causal-mask check using LOGICAL positions.

Earlier probes poisoned by physical page offset, but the kernels mask on the
*logical* position blk*BLOCK_SIZE_K + o (blk = index into the page table).  The
physical page is a random permutation of blk, so that probe was meaningless.

Here the poison set is built from the logical position and then written into the
physical page the block table maps it to:

  future  : { (blk, o) : blk*128 + o > q_abs(row r) }   -> row r must not change
  control : { (blk, o) : blk*128 + o <= q_abs(row r) }  -> row r must change

Only row r is asserted, because the poisoned pages are legitimately visible to
other rows.
"""
import os, sys, json
REPO = os.environ["MSA_REPO"]
sys.path.insert(0, os.path.join(REPO, "src")); sys.path.insert(0, REPO)
import torch

BLOCK = 128


def run(b, data, tk, qlen, kvh, edits):
    kv = data.kv_cache.clone()
    big = torch.tensor(float(os.environ.get("MSA_POISON", "30000")), dtype=kv.dtype, device=kv.device)
    n = 0
    for page, offs in edits.items():
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
        blocks = sorted({int(x) for x in tk[0].reshape(-1).tolist() if int(x) >= 0})
        page_of = {blk: int(bt[0, blk]) for blk in blocks}

        rows = sorted({0, 1, 7, 15, 16, qlen // 4, qlen // 2, qlen - 1})
        cases = {}
        for r in rows:
            q_abs = prefix_len + r
            fut, vis = {}, {}
            for blk in blocks:
                page = page_of[blk]
                base = blk * BLOCK
                f = [o for o in range(BLOCK) if base + o > q_abs]
                v = [o for o in range(BLOCK) if base + o <= q_abs]
                if f:
                    fut[page] = f
                if v:
                    vis[page] = v
            dirty, n_f = run(b, data, tk, qlen, kvh, fut)
            ch_f = int((dirty[r] != clean[r]).sum())
            d_f = (dirty[r].float() - clean[r].float()).abs().max().item()
            ctrl, n_v = run(b, data, tk, qlen, kvh, vis)
            ch_v = int((ctrl[r] != clean[r]).sum())
            d_v = (ctrl[r].float() - clean[r].float()).abs().max().item()
            cases[f"row{r}"] = dict(q_abs=q_abs, future_positions=n_f, future_changed=ch_f,
                                    future_max_delta=d_f, control_positions=n_v,
                                    control_changed=ch_v, control_max_delta=d_v,
                                    mask_enforced=(ch_f == 0 and ch_v > 0))
            print(f"[mask:{tag} r={r:4d}] q_abs={q_abs} future n={n_f:5d} ch={ch_f:4d} "
                  f"d={d_f:.4g} | control n={n_v:5d} ch={ch_v:4d} d={d_v:.4g} -> "
                  f"{'MASK ENFORCED' if ch_f == 0 and ch_v > 0 else 'LEAK/INCONCLUSIVE'}")
        out[tag] = dict(seq_len=seq, qlen=qlen, prefix_len=prefix_len, rows=cases)
    json.dump(out, open(os.environ.get("MSA_OUT", "/tmp/probe_mask2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
