import json
# Observed ceilings from this session
RES = {(256,128,64):77.81,(128,128,128):98.00}          # resident dot, 96 CTA
TILE= {(256,128,64):37.61,(128,128,128):39.22}          # tile-major feed, 96 CTA
def model(B,H,T,D,BM,BN,lat_ms,cta,resident,tile):
    iters = T//BN
    # per-iter bytes actually loaded by the shipped kernel per CTA
    k_bytes = BM*BN*2   # K tile FP16 (pre-dequantized) per CTA per iter
    v_bytes = BN*D*2
    q_bytes = BM*D*2
    out_bytes = BM*D*2
    flops_iter = 2*BM*BN*D*2  # qk + pv
    total_bytes = cta*(iters*(k_bytes+v_bytes) + q_bytes + out_bytes)
    total_flops = B*H*T*T*D*4
    return dict(shape=[B,H,T,D],block_m=BM,block_n=BN,iters=iters,n_cta=cta,
        waves=cta/24.0,
        flops_gflop=total_flops/1e9,
        bytes_mib=total_bytes/(1<<20),
        arithmetic_intensity_flop_per_byte=total_flops/total_bytes,
        latency_ms=lat_ms,
        achieved_tflops=total_flops/lat_ms/1e9,
        resident_ceiling_tflops=resident, tile_ceiling_tflops=tile,
        frac_of_tile_ceiling=total_flops/lat_ms/1e9/tile)
rows=[]
rows.append(model(1,2,4096,64,256,128,1.0431,32,RES[(256,128,64)],TILE[(256,128,64)]))
rows.append(model(4,32,1024,128,128,128,3.9062,1024,RES[(128,128,128)],TILE[(128,128,128)]))
for r in rows: print(json.dumps(r))
