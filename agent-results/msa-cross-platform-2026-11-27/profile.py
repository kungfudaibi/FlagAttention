import argparse
import hashlib
import statistics
import torch,triton
from flag_attn import minimax_m3_sparse_attn
from flag_attn.minimax_sparse_attention import minimax_m3_index_score,minimax_m3_index_topk
import flag_attn

parser = argparse.ArgumentParser(description="Profile the fixed BF16 MiniMax MSA prefill case.")
parser.add_argument("--trace-only", action="store_true", help="Run one priming call and 12 steady-state chains for vendor tracing.")
parser.add_argument("--score-split-ab", action="store_true", help="Override the ZW810E Score launch grid while preserving its arithmetic.")
parser.add_argument("--common-score", action="store_true", help="Replay the same Score kernel and launch parameters on either platform.")
args = parser.parse_args()


def sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]
torch.cuda.set_device(0)
print("DEVICE",torch.cuda.get_device_name(0),torch.__version__,triton.__version__,flag_attn.runtime.device.vendor_name,flush=True)
print("TARGET",triton.runtime.driver.active.get_current_target(),flush=True)
print("SOURCE",flag_attn.__file__,flush=True)
g=torch.Generator(device="cpu").manual_seed(6)
D=128; Q=1024; S=16384; H=8; P=128; K=16
def make(shape):return (torch.randn(shape,generator=g,dtype=torch.bfloat16)*0.5).contiguous()
q0=make((Q,H,D)); iq0=make((Q,1,D)); kv0=make((128,1,P,2*D)); ikv0=make((128,P,D))
order=torch.randperm(128,generator=g)
bt0=order.to(torch.int32)[None,:]
kv0=kv0[order.argsort()].contiguous();ikv0=ikv0[order.argsort()].contiguous()
q,iq,kv,ikv,bt=(x.to("cuda") for x in (q0,iq0,kv0,ikv0,bt0))
cu=torch.tensor([0,Q],dtype=torch.int32).to("cuda")
se=torch.tensor([S],dtype=torch.int32).to("cuda")
pr=torch.tensor([S-Q],dtype=torch.int32).to("cuda")
out=torch.empty_like(q)
print("INPUT",sha(q0),sha(iq0),sha(kv0),sha(ikv0),sha(bt0),flush=True)
def score(): return minimax_m3_index_score(iq,ikv,bt,cu,se,pr,Q,S,1)
def top(x): return minimax_m3_index_topk(x,cu,pr,Q,K,1,2)
def attend(t):
    return minimax_m3_sparse_attn(q=q,kv_cache=kv,topk_idx=t,output=out,
                                                block_table=bt,cu_seqlens_q=cu,seq_lens=se,prefix_lens=pr,max_query_len=Q,num_kv_heads=1,sm_scale=D**-0.5)
def chain():
    attend(top(score()))
sc=score(); ix=top(sc);attend(ix);torch.cuda.synchronize()
if args.trace_only:
    for _ in range(12):
        chain()
    torch.cuda.synchronize()
    print('NATIVE_MSA_PASS', flush=True)
    raise SystemExit(0)
indices=ix.cpu()[0]; actual=out.cpu().float()
errs=[]
for i in (0,17,257,769,1023):
    sel=indices[i];sel=sel[sel>=0].long()
    physical=bt0[0,sel].long()
    store=kv0[physical,0].reshape(-1,2*D).float()
    tok=(sel[:,None]*P+torch.arange(P)).flatten()
    logits=(q0[i].float()@store[:,:D].T)*(D**-0.5)
    logits[:,tok>S-Q+i]=float("-inf")
    expected=logits.softmax(dim=-1)@store[:,D:]
    errs.append((actual[i]-expected).abs().max().item())
assert torch.isfinite(actual).all() and max(errs)<0.01,errs
print("CORRECT",max(errs),"INDEX",sha(ix),"OUTPUT",sha(out),flush=True)
for name, f in (("score",score),("topk",lambda:top(sc)),("attention",lambda:attend(ix)),("chain",chain)):
    for _ in range(4): f()
    torch.cuda.synchronize()
    samples=[]
    for _ in range(9):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(8): f()
        b.record();b.synchronize()
        samples.append(a.elapsed_time(b)/8)
    print("EVENT",name,statistics.median(samples),min(samples),max(samples),flush=True)
# Two hardware-path controls with identical shapes on both devices.
def bf(shape,value):return torch.full(shape,value,dtype=torch.bfloat16).to("cuda")
a=bf((2048,2048),.125); b=bf((2048,2048),.25); c=torch.empty_like(a)
ma=bf((1024,8,128),.125);mb=bf((1024,128,128),.25);mc=torch.empty((1024,8,128),dtype=torch.bfloat16,device="cuda")
for name,f in (("dense",lambda:torch.mm(a,b,out=c)),("batch_m8",lambda:torch.bmm(ma,mb,out=mc))):
    samples=[]
    for _ in range(3):f()
    torch.cuda.synchronize()
    for _ in range(7):
        x=torch.cuda.Event(enable_timing=True);y=torch.cuda.Event(enable_timing=True)
        x.record()
        for _ in range(16): f()
        y.record();y.synchronize()
        samples.append(x.elapsed_time(y)/16)
    print("CONTROL_EVENT",name,statistics.median(samples),min(samples),max(samples),flush=True)
if args.score_split_ab:
    if flag_attn.runtime.device.vendor_name != "thead":
        parser.error("--score-split-ab expects the 810E (thead) backend")
    original=triton.runtime.jit.JITFunction.run
    override=None
    configurations={}
    def hook(kernel,*args,**kw):
        if kernel.__name__=="_index_block_score_kernel":
            grid=kw["grid"]
            configurations.setdefault(str(override),[list(grid),kw["NUM_KV_SPLITS"]])
            if override is not None:
                kw["grid"]=(grid[0],grid[1],override)
                kw["NUM_KV_SPLITS"]=override
        return original(kernel,*args,**kw)
    def timings(action):
        for _ in range(5):action()
        torch.cuda.synchronize()
        samples=[]
        for _ in range(11):
            a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            for _ in range(12):action()
            b.record();b.synchronize()
            samples.append(a.elapsed_time(b)/12)
        return statistics.median(samples)
    triton.runtime.jit.JITFunction.run=hook
    try:
        for choice in (None,1,2,4,8,16,None):
            override=choice
            got=score()
            mask=(torch.arange(128)[None,:] <= ((S-Q+torch.arange(Q))//P)[:,None]).to("cuda")
            assert torch.equal(sc[0][mask],got[0][mask]),choice
            assert torch.equal(ix,top(got)),choice
            print("SPLIT_AB",choice,"grid_default",configurations[str(choice)],"score_ms",timings(score),"chain_ms",timings(chain),flush=True)
    finally:
        triton.runtime.jit.JITFunction.run=original
if args.common_score:
    from common_score import _index_block_score_kernel as common_kernel
    out_common=torch.empty_like(sc)
    valid=(torch.arange(128)[None,:] <= ((S-Q+torch.arange(Q))//P)[:,None]).to("cuda")
    def shared_score(splits):
        common_kernel[(16,1,splits)](iq,ikv,out_common,bt,cu,se,pr,1,D,
                    *iq.stride(),*ikv.stride(),*out_common.stride(),bt.stride(0),
                    BLOCK_SIZE_Q=64,BLOCK_SIZE_K=128,NUM_KV_SPLITS=splits,
                    num_warps=4,num_stages=1)
    for split in (1,2,4,8,16):
        try:
            for _ in range(4):shared_score(split)
            torch.cuda.synchronize()
            err=(sc[0][valid]-out_common[0][valid]).abs().max().item()
            same_topk=torch.equal(ix,top(out_common))
            assert err<0.002 and same_topk,(split,err,same_topk)
            vals=[]
            for _ in range(9):
                a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                a.record()
                for _ in range(12):shared_score(split)
                b.record();b.synchronize()
                vals.append(a.elapsed_time(b)/12)
            print("MATCHED_SCORE",split,statistics.median(vals),min(vals),max(vals),"max_err",err,"topk_match",same_topk,flush=True)
        except Exception as exc:
            print("MATCHED_FAIL",split,type(exc).__name__,str(exc)[-2800:],flush=True)

print("PASS", flush=True)
