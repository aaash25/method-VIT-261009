"""DToMe's native B>1 packing for all visual segments of one question.

Qwen supplies ragged segments. The padded batch width is retained for the
author's padding-mask argsort, while attention and pair scores remain local to
each image. No features are shared with another question. Threshold selection
with no per-instance cap selects every above-threshold edge, so it does not
require materializing a batch of quadratic score matrices.
"""
import torch
import torch.nn.functional as F
from peer_port_core import MergeResult, dtome_merge


@torch.no_grad()
def merge_question_segments(states, metrics, *, threshold):
    if not states or len(states) != len(metrics):
        raise ValueError('one metric per nonempty visual segment required')
    if len(states) == 1:
        s=states[0]
        return [dtome_merge(s.hidden,s.original_positions,s.size,metrics[0],threshold=threshold)]
    width=max(s.hidden.shape[0] for s in states)
    if width < 2:
        return [MergeResult(s.hidden,s.original_positions,torch.arange(len(s.hidden),device=s.hidden.device),s.size,0) for s in states]
    device=states[0].hidden.device
    even_width=(width+1)//2
    mask=torch.ones((len(states),width),device=device,dtype=states[0].hidden.dtype)
    selections=[]
    for row,(s,metric) in enumerate(zip(states,metrics)):
        n=len(s.hidden);ids=torch.arange(n,device=device);evens=ids[::2];odds=ids[1::2]
        if metric.shape[0] != n: raise ValueError('metric/segment length mismatch')
        combined=torch.where(ids.remainder(2)==0,ids//2,even_width+ids//2)
        mask[row,combined]=0
        if len(odds):
            normalized=metric / metric.norm(dim=-1,keepdim=True)
            maximum,destination=(normalized[evens] @ normalized[odds].T).max(dim=-1)
            selected=torch.where(maximum>threshold)[0]
            sources=evens[selected];targets=destination[selected]
        else:
            sources=ids[:0];targets=ids[:0]
        mask[row,sources//2]=1
        selections.append((ids,odds,combined,sources,targets))
    # The author's default (unstable) argsort is intentional: token order affects
    # the next layer's even/odd partition. Do not replace it with stable packing.
    order=torch.argsort(mask,dim=1)
    results=[]
    for row,(s,selection) in enumerate(zip(states,selections)):
        ids,odds,combined,sources,targets=selection
        n=len(ids);kept=n-len(sources);packed=order[row,:kept]
        inverse=torch.empty(width,device=device,dtype=torch.long)
        inverse[order[row]]=torch.arange(width,device=device)
        parent_slots=combined.clone()
        parent_slots[sources]=even_width+targets
        sizes=s.size
        accumulation=s.hidden.float() if s.hidden.dtype==torch.float16 else s.hidden
        weighted=accumulation*sizes
        sums=torch.zeros((width,s.hidden.shape[-1]),device=device,dtype=weighted.dtype)
        counts=torch.zeros((width,1),device=device,dtype=s.size.dtype)
        sums[combined]=weighted;counts[combined]=sizes
        if len(sources):
            sums[even_width:even_width+len(odds)].scatter_add_(
                0,targets[:,None].expand(-1,s.hidden.shape[-1]),weighted[sources])
            counts[even_width:even_width+len(odds)].scatter_add_(0,targets[:,None],sizes[sources])
            sums[sources//2]=0;counts[sources//2]=0
        positions=torch.full((width,),-1,device=device,dtype=torch.long)
        positions[combined]=s.original_positions
        # Preserve the author's batch denominator, including unmerged tokens.
        # FP16 weighted sums need FP32 range; the native denominator and
        # selection/packing dtype are unchanged. No deterministic sorting.
        out=(sums[packed]/(counts[packed]+1e-4)).to(s.hidden.dtype)
        results.append(MergeResult(out,positions[packed],inverse[parent_slots],counts[packed].to(s.size.dtype),len(sources)))
    return results
