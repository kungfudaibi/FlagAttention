import sys, time, torch, torch_gcu
sys.path.insert(0,"/workspace/agent-workspaces/sage-fused-sdpa-study-5/src")
from flag_attn import sage_attention
from flag_attn.sage_attention import per_block_int8
torch.gcu.set_device(5)
mode=sys.argv[1]; shape=tuple(int(x) for x in sys.argv[2].split("x")); dur=float(sys.argv[3])
q=torch.randn(shape,device="gcu:5",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
if mode.startswith("sage"):
    p=per_block_int8(q,k)
def native(): return torch.ops.aten._scaled_dot_product_flash_attention(q,k,v,0.0,False,False)[0]
def full():
    p=per_block_int8(q,k); return sage_attention(p[0],p[2],v,p[1],p[3],output_dtype=torch.float16)[0]
def attn():
    return sage_attention(p[0],p[2],v,p[1],p[3],output_dtype=torch.float16)[0]
fn={"native":native,"full":full,"attn":attn}[mode]
for _ in range(5): fn()
torch.gcu.synchronize()
n=0; t0=time.perf_counter()
while time.perf_counter()-t0 < dur:
    fn(); n+=1
torch.gcu.synchronize()
el=time.perf_counter()-t0
print("MODE",mode,"SHAPE",shape,"CALLS",n,"SECONDS",round(el,4),"MS_PER_CALL",round(el/n*1e3,5))
