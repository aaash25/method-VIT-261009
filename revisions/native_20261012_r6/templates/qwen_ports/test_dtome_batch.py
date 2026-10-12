import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from peer_port_core import make_segments, update_state
from dtome_batch import merge_question_segments

SOURCE=Path(__file__).resolve().parent/'author_visual_source.py'
spec=importlib.util.spec_from_file_location('dtome_batch_oracle',SOURCE)
author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
merge=author.load_dtome_functions(deterministic_scatter=False)['merge_tokens']
torch.set_num_threads(2)

@pytest.mark.parametrize('lengths',[(16,16),(15,9),(18,7),(3,2),(2,2),(1,5),(1,1)])
@pytest.mark.parametrize('threshold',[-.5,.4,2.])
@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
@pytest.mark.parametrize('device',['cpu']+(['cuda'] if torch.cuda.is_available() else []))
def test_two_layers_match_literal_author_batch(lengths,threshold,dtype,device):
    # Exercise the author's native precision. Numerical closeness tests arithmetic;
    # parent assertions below check this controlled oracle's structural mapping.
    for seed in range(6):
        torch.manual_seed(seed)
        m=max(lengths);b=len(lengths);x=torch.randn(b,m,8,device=device).to(dtype)
        size=torch.zeros(b,m,1,device=device,dtype=dtype);mask=torch.ones(b,m,device=device,dtype=dtype);pos=torch.zeros(b,m,m,device=device,dtype=dtype)
        for i,n in enumerate(lengths):
            size[i,:n]=1;mask[i,:n]=0;pos[i,:n,:n]=torch.eye(n,device=device)
        states=make_segments(torch.cat([x[i,:n].to(dtype) for i,n in enumerate(lengths)]),list(lengths))
        holder=SimpleNamespace(_tome_info={'merge_mode':'batch_level','size':size,'trace_source':False,'class_token':False,'distill_token':False,'max_r_per_instance_ratio':None},training=False,update_threshold=False,specified_threshold=threshold)
        for layer in range(2):
            result=merge_question_segments(states,[s.hidden for s in states],threshold=threshold)
            x,mask,pos=merge(holder,x.clone(),x.shape[1],x,padding_mask=mask.clone(),pos_tracking=pos)
            x=x.to(dtype)
            for i,state in enumerate(states):
                update_state(state,result[i]);n=int((mask[i]<.5).sum())
                assert state.hidden.shape[0]==n
                torch.testing.assert_close(state.hidden,x[i,:n],rtol=2e-3 if dtype==torch.float16 else 2e-6,atol=2e-3 if dtype==torch.float16 else 2e-6)
                torch.testing.assert_close(state.size,holder._tome_info['size'][i,:n])
                assert torch.equal(state.original_to_current,pos[i,:n,:lengths[i]].argmax(dim=0))
                assert int(state.original_positions.min())>=sum(lengths[:i])
                assert int(state.original_positions.max())<sum(lengths[:i+1])
