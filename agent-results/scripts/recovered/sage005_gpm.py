import torch, torch_gcu, time, json, sys
torch.gcu.set_device(5)
from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import per_block_int8
from flag_attn.runtime.backend._enflame.sage_attention import attn_qk_int8_per_block as A
PHASES=[]
def run_phase(name, fn, secs=3.0):
    t0=time.time(); n=0
    while time.time()-t0 < secs:
        fn(); n+=1
        if n%20==0: torch.gcu.synchronize()
    torch.gcu.synchronize()
    dt=time.time()-t0
    PHASES.append({"phase":name,"t_start":t0,"t_end":time.time(),"calls":n,"ms_per_call":dt/n*1e3})
    print(f"PHASE {name} calls={n} ms/call={dt/n*1e3:.3f}", flush=True)

d64=(1,2,4096,64); d128=(4,32,1024,128)
q64=torch.randn(d64,device="gcu:5",dtype=torch.float16); k64=torch.randn_like(q64); v64=torch.randn_like(q64)
q12=torch.randn(d128,device="gcu:5",dtype=torch.float16); k12=torch.randn_like(q12); v12=torch.randn_like(q12)
qi64,qs64,ki64,ks64=per_block_int8(q64,k64,sm_scale=64**-0.5)
qi12,qs12,ki12,ks12=per_block_int8(q12,k12,sm_scale=128**-0.5)
torch.gcu.synchronize()
def nat64(): torch.ops.aten._scaled_dot_product_flash_attention(q64,k64,v64,0.0,False,False)
def nat12(): torch.ops.aten._scaled_dot_product_flash_attention(q12,k12,v12,0.0,False,False)
def sag64(): A.forward(qi64,ki64,v64,qs64,ks64,output_dtype=torch.float16)
def sag12(): A.forward(qi12,ki12,v12,qs12,ks12,output_dtype=torch.float16)
for f in (nat64,nat12,sag64,sag12): f()
torch.gcu.synchronize(); time.sleep(0.5)
run_phase("idle", lambda: time.sleep(0.05), 3.0)
run_phase("native_d64", nat64, 4.0)
run_phase("sage_d64", sag64, 4.0)
run_phase("native_d128", nat12, 4.0)
run_phase("sage_d128", sag12, 4.0)
json.dump(PHASES, open("/tmp/gpm_phases.json","w"), indent=1)
print("DONE", flush=True)
