"""Bounded stock-FA2 / pinned-author interface investigation, not a model experiment."""
import argparse,ast,hashlib,importlib.util,json,os,subprocess,sys,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
P=Path('@PROJECT_ROOT@')
GUARD='529772667f1f2063e1f37fc9693b6b82c5130fa994c6dd84e82d5cfd44331275'
ADDITION=P/'route/diagnostics/qwen_ipcv_fa2_dependency_20261006_r5_a1/dependency/ADDITION_RECEIPT.json'
ADDITION_SHA='045b522c4a2b34232f82801f782a0af17e71169dc1f8b9722872e509ab7da68e'
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def main():
    out=HERE/'FA2_INTERFACE_GPU';out.mkdir(exist_ok=False);(out/'retained_tmp').mkdir()
    (out/'guard.jsonl').touch(exist_ok=False)
    os.environ.update(STAGE_GUARD_ROOT=str(out),STAGE_GUARD_LOG=str(out/'guard.jsonl'),TMPDIR=str(out/'retained_tmp'))
    report=dict(state='running',formal_launch=False,full_model=False,image_validation='skipped by user request')
    try:
        addon=load(P/'releases/fa2_dependency_cp311_offline_20261006_r5/fa2_addon.py','fa2_existing_addon')
        addon.start_guard(GUARD);report['dependency']=addon.activate(ADDITION,ADDITION_SHA,GUARD)
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy'
        import torch,flash_attn
        from torch import nn
        from flash_attn import flash_attn_varlen_func
        assert torch.cuda.get_device_capability()[0]>=8
        report.update(gpu=torch.cuda.get_device_name(),flash_attn=flash_attn.__version__)
        native=load(HERE/'native_release_20261012_r5/qwen_native/illava_native_source.py','fa2_pinned_source')
        tree=ast.parse(native.AUTHOR_PATH.read_bytes())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VisionFlashAttention2')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        fn.name='author_flash_forward'
        namespace=dict(torch=torch,flash_attn_varlen_func=flash_attn_varlen_func,
            apply_rotary_pos_emb_vision=native.AUTHOR_EAGER_FORWARD.__globals__['apply_rotary_pos_emb_vision'])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(native.AUTHOR_PATH),'exec'),namespace)
        class Attention(nn.Module):
            def __init__(self):
                super().__init__();self.num_heads=4;self.qkv=nn.Linear(32,96);self.proj=nn.Linear(32,32)
            def forward(self,*a,**kw):return namespace['author_flash_forward'](self,*a,**kw)
        class Block(nn.Module):
            def __init__(self):
                super().__init__();self.norm1=nn.LayerNorm(32);self.norm2=nn.LayerNorm(32)
                self.attn=Attention();self.mlp=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,32))
        cells=[]
        with torch.inference_mode():
            for count in (1,5):
                torch.manual_seed(12012);b=Block().cuda().half().eval()
                x=torch.randn(count*16,32,device='cuda',dtype=torch.float16)
                cu=torch.arange(count+1,device='cuda',dtype=torch.int32)*16;rotary=torch.randn(count*16,4,device='cuda')
                q=torch.randn(count*16,4,8,device='cuda',dtype=torch.float16)
                result=flash_attn_varlen_func(q,q,q,cu,cu,16,16,return_attn_probs=True)
                shapes=[list(t.shape) for t in result]
                # Native author code deliberately takes tuple slot 1; do not substitute slot 2.
                y,newcu,newrot=native.AUTHOR_BLOCK_FORWARD(b,x.clone(),cu,rotary,perform_self_selection=True,reduce_tokens=0.25)
                assert y.shape==(count*12,32) and y.dtype==torch.float16 and torch.isfinite(y).all()
                assert newcu.tolist()==[i*12 for i in range(count+1)]
                cells.append(dict(segments=count,state='passed',fa2_return_shapes=shapes,
                    author_uses='tuple[1], stock FA2 softmax_lse',retained_per_image=12))
        torch.cuda.synchronize();report.update(state='passed',cells=cells,author_half_preserved=True,
            interpretation='Native author FA2 code executes; its score is stock FA2 tuple[1] softmax_lse, not eager incoming-attention probability. Fidelity interpretation requires cross-review; no full-model or accuracy claim.')
    except BaseException:
        report.update(state='failed',error=traceback.format_exc());raise
    finally:
        (out/'RESULT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(dict(state=report['state'],gpu=report.get('gpu'),cells=report.get('cells'),error=report.get('error'))),flush=True)
if __name__=='__main__':main()
