import torch, torch_gcu, json, os, sys, glob
sys.path.insert(0,"/workspace/agent-workspaces/sage-fused-sdpa-study-5/src")
from flag_attn import sage_attention
from flag_attn.sage_attention import per_block_int8
from flag_attn.runtime.backend._enflame.sage_attention.attn_qk_int8_per_block import _attn_fwd
from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import quant_per_block_int8_kernel
torch.gcu.set_device(5)
q=torch.randn((1,2,4096,64),device="gcu:5",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
pre=per_block_int8(q,k)
out=sage_attention(pre[0],pre[2],v,pre[1],pre[3],output_dtype=torch.float16)[0]
torch.gcu.synchronize()
for name,fn in (("attn",_attn_fwd),("quant",quant_per_block_int8_kernel)):
    print("====",name)
    for dev,cache in fn.device_caches.items():
        print("dev",dev,"len",len(cache))
        for key,val in cache.items():
            print("key",str(key)[:200])
            print("type",type(val).__name__)
            for attr in ("n_regs","n_spills","n_shared","name","metadata","num_warps","num_stages"):
                try: print(" ",attr,getattr(val,attr))
                except Exception as e: print(" ",attr,"ERR",type(e).__name__)
            try: print("  asm keys",list(val.asm.keys()))
            except Exception as e: print("  asm ERR",e)
            try: print("  packed",len(val.asm["cubin"]) if "cubin" in val.asm else None)
            except Exception as e: print("  packed ERR",e)
