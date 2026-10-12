"""Author rounding and scatter compatibility; no old-candidate identity gate."""
import argparse, importlib.util, json, sys
from pathlib import Path
import torch

def run(release, device):
    path=Path(release)/'smol/smol_merge_ports.py'
    spec=importlib.util.spec_from_file_location('smol_half_test',path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    rows=[]
    for dtype in (torch.float16,torch.float32):
        for crops in (1,5):
            x=torch.linspace(-1.1,2.3,32,device=device).reshape(1,8,4).repeat(crops,1,1).to(dtype)
            columns=torch.arange(1,9,device=device,dtype=dtype)/36
            attention=columns.expand(crops,2,8,8).clone()
            step=mod.illava_step(x,attention,2)
            # Increasing incoming attention chooses slots 0,1,2; the last is the recipient.
            weights=torch.tensor([3,2,1],device=device,dtype=torch.float32)
            expected=torch.stack([weights @ row[:3].float()/6 for row in x]).half().to(dtype)
            if dtype==torch.float32:
                unrounded=torch.stack([weights @ row[:3].float()/6 for row in x])
                assert not torch.equal(expected,unrounded),'test must distinguish omitted FP16 rounding'
            assert step.hidden.dtype==dtype and step.hidden.shape==(crops,6,4)
            torch.testing.assert_close(step.hidden[:,0],expected,rtol=0,atol=0)
            torch.testing.assert_close(step.hidden[:,1:],x[:,3:],rtol=0,atol=0)
            assert step.parent.tolist()==[[0,0,0,1,2,3,4,5]]*crops
            empty=mod.illava_step(x,attention,0)
            assert empty.hidden is x and empty.parent.tolist()==[list(range(8))]*crops
            rows.append(dict(dtype=str(dtype),crops=crops,state='passed',author_rounding=True,scatter_compatible=True))
    if device=='cuda':torch.cuda.synchronize()
    return dict(state='passed',device=device,cells=rows,skipped=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',required=True,type=Path);p.add_argument('--device',choices=['cpu','cuda'],required=True)
    a=p.parse_args();print(json.dumps(run(a.release,a.device)))
