"""Unmodified author DToMe visual-core definitions, extracted from DyMU.

Origin: https://github.com/MikeWangWZHL/dymu
Commit: 3770d6add8fd9b972ae611f3f3b16a552dea13cc
See AUTHOR_ORIGIN.json and the retained full upstream files. The only glue is
imports and the exact reshape-only einops pattern below. No algorithm patches.
"""
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple
import math
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.functional import _in_projection_packed
from torch.utils.checkpoint import checkpoint


def rearrange(value, pattern):
    # The single einops use in the selected author core; no optional dependency.
    if pattern != "b i j -> (b i) j" or value.ndim != 3:
        raise ValueError("unsupported author rearrange pattern")
    return value.reshape(value.shape[0] * value.shape[1], value.shape[2])

def do_nothing(x, mode=None):
    return x

def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
    specified_threshold: Optional[float] = None,
    max_r_per_instance: Optional[int] = None
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)

        # If a threshold is provided, adjust r per instance based on token scores.
        if specified_threshold is not None:
            assert metric.shape[0], "batch size have to be 1"
            # Count, per instance, how many token-pairs exceed the threshold.
            merge_counts = (node_max > specified_threshold).sum(dim=-1)
            # Use the minimum count across the batch as the effective number to merge.
            r_effective = int(merge_counts[0].item())
            # r = min(r, r_effective)
            r = min(r_effective, (t - protected) // 2)
        
            if r <= 0:
                return do_nothing, do_nothing
        
        if max_r_per_instance is not None:
            r = min(r, max_r_per_instance)

        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge

def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None, pos_tracking: torch.Tensor = None # (b, s, s_ori)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])
    
    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size

    if pos_tracking is not None:
        pos_tracking = merge(pos_tracking, mode="sum")

    return x, size, pos_tracking

def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source

def batch_level_bipartite_soft_matching(
    metric: torch.Tensor, # (b, s, c)
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
    padding_mask: Optional[torch.Tensor] = None, # (b, s) # 0 for non padding, 1 for padding
    max_r_per_instance: int = None,
    specified_threshold: float = None
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    bsz, seq_len, hdim = metric.shape

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing, None

    with torch.no_grad():
        
        # compute scores within instance
        metric = metric / metric.norm(dim=-1, keepdim=True) # (b, s, c)
        # print(metric)

        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2) # (b, s//2, s//2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        # add padding mask
        if padding_mask is not None:
            # padding_mask is 0 for non-padding and 1 for padding
            mask_a = padding_mask[..., ::2].unsqueeze(2).bool()  # Shape: (b, s//2, 1)
            mask_b = padding_mask[..., 1::2].unsqueeze(1).bool()  # Shape: (b, 1, s//2)

            # Combine masks to identify where either 'a' or 'b' has padding
            combined_mask = mask_a | mask_b  # Shape: (b, s//2, s//2)

            # Set scores at padding positions to -inf
            # scores = scores.masked_fill(combined_mask, -math.inf)
            scores.masked_fill_(combined_mask, -math.inf)

        if max_r_per_instance is not None:
            node_max_instance, node_idx_instance = scores.max(dim=-1)
            edge_idx_instance = node_max_instance.argsort(dim=-1, descending=True)[..., None]
            unm_idx_instance = edge_idx_instance[..., max_r_per_instance:, :] # keep tokens beyond r_max unmerged
            unm_idx_instance_expanded = unm_idx_instance.expand(-1, -1, scores.size(-1))
            batch_indices = torch.arange(bsz).view(-1, 1, 1).expand_as(unm_idx_instance_expanded)
            scores[batch_indices, unm_idx_instance_expanded, :] = -math.inf

        # flatten across batch
        scores = rearrange(scores, 'b i j -> (b i) j')

        # get the best matching over the batch
        node_max, node_idx = scores.max(dim=-1) # (b * s // 2)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        if specified_threshold is not None:
            # rb = sum(node_max > specified_threshold) # merge all tokens over the specified_threshold
            rb = int((node_max > specified_threshold).sum().item())
        else:
            rb = r * bsz

        unm_idx = edge_idx[rb:, :]  # Unmerged Tokens (unmerged_token_num, 1)
        src_idx = edge_idx[:rb, :]  # Merged Tokens (rb, 1)
        dst_idx = node_idx.gather(dim=0, index=src_idx.squeeze()) # (rb,)

        if specified_threshold is not None:
            batch_threshold = None
        else:
            # keep track of batch level threshold for this layer
            j = rb if rb < len(edge_idx) else len(edge_idx) - 1
            batch_threshold = node_max[edge_idx[j, 0]]
            batch_threshold = max(batch_threshold, torch.zeros_like(batch_threshold)) # should be non-negative
        # print(scores)
        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def update_dst(dst_flat, src_elements, index_expanded, reduce='sum', include_self=True):
        if reduce == 'sum':
            # Use scatter_add_ for sum reduction
            dst_flat.scatter_add_(
                dim=0,
                index=index_expanded,
                src=src_elements
            )
        elif reduce == 'mean':
            # For mean reduction, we'll need to keep track of counts
            counts = torch.zeros_like(dst_flat)
            ones = torch.ones_like(src_elements)

            # Sum the src_elements into dst_flat
            sum_dst_flat = torch.zeros_like(dst_flat)
            sum_dst_flat.scatter_add_(
                dim=0,
                index=index_expanded,
                src=src_elements
            )

            # Count the number of times each index is updated
            counts.scatter_add_(
                dim=0,
                index=index_expanded,
                src=ones
            )

            if include_self:
                # Include original dst values in the mean calculation
                sum_dst_flat += dst_flat
                # counts += (dst_flat != 0).float()
                counts += (dst_flat != 0).to(counts.dtype)

            # Avoid division by zero
            counts = counts.clamp(min=1)

            # Compute the mean
            dst_flat = sum_dst_flat / counts
        else:
            raise ValueError("Unsupported reduction type. Use 'sum' or 'mean'.")

        return dst_flat


    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]  # Shape: (b, s//2, c)

        # src = rearrange(src, 'b s c -> (b s) c') # (b * s // 2, c)
        t1 = src.shape[1]
        unm_b = unm_idx.squeeze(-1) // t1
        unm_s = unm_idx.squeeze(-1) % t1
        src_b = src_idx.squeeze(-1) // t1
        src_s = src_idx.squeeze(-1) % t1

        src_tokens = src[src_b, src_s, :]  # Shape: (b * r, c)
        dst_b = src_b
        dst_seq_len = dst.size(1)
        dst_flat_indices = dst_b * dst_seq_len + dst_idx
        dst_flat = dst.reshape(-1, dst.size(-1)).clone()
        index_expanded = dst_flat_indices.unsqueeze(-1).expand(-1, dst_flat.size(-1))
        
        # 
        # dst_new = dst_flat.clone()
        dst_new = update_dst(dst_flat, src_tokens, index_expanded, reduce=mode, include_self=True)
        dst_new = dst_new.reshape(dst.size())
        # print("dst tokens merged:", dst_new)

        # # construct new x
        # x_new = x.clone()
        # x[..., :src.size(1), :] = src
        # x[..., src.size(1):, :] = dst_new
        x_new = torch.cat([src, dst_new], dim=1)
        # x_new[src_b, src_s, :] = torch.zeros_like(src[src_b, src_s, :])
        x_new[src_b, src_s, :] = 0

        if padding_mask is not None:
            # padding_mask_src, padding_mask_dst = padding_mask[..., ::2].clone(), padding_mask[..., 1::2].clone()
            padding_mask_src, padding_mask_dst = padding_mask[..., ::2], padding_mask[..., 1::2]
            padding_mask_src[src_b, src_s] = 1
            new_padding_mask = torch.cat([padding_mask_src, padding_mask_dst], dim=1)
        else:
            new_padding_mask = torch.zeros((bsz, seq_len), device=x.device, dtype=x.dtype)
            new_padding_mask_a = new_padding_mask[..., ::2].clone()
            new_padding_mask_a[src_b, src_s] = 1
            new_padding_mask[..., :src.size(1)] = new_padding_mask_a
        
        # # construct padding masking: (b, s); 0 for non-padding, 1 for padding; fill in 1 where x_new is zero
        # padding_mask = torch.all(x_new == 0, dim=-1).int().to(x.device)
        return x_new, new_padding_mask


    def unmerge(x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Unmerge not implemented yet.")

    return merge, unmerge, batch_threshold

def batch_level_merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None, pos_tracking: torch.Tensor = None, # (b, s, s_ori)
    cls_token: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x, padding_mask = merge(x * size, mode="sum")
    size, _ = merge(size, mode="sum")

    if pos_tracking is not None:
        pos_tracking, _ = merge(pos_tracking, mode="sum")

    assert padding_mask is not None
    if cls_token:
        if x.size(1) > 1:
            # Separate the cls token (first token)
            cls_token_x = x[:, :1, :]
            cls_token_padding = padding_mask[:, :1]
            cls_token_size = size[:, :1, :]
            if pos_tracking is not None:
                cls_token_pos = pos_tracking[:, :1, :]
            # Process the rest of the tokens (from index 1 onward)
            rest_x = x[:, 1:]
            rest_padding_mask = padding_mask[:, 1:]
            rest_size = size[:, 1:]
            if pos_tracking is not None:
                rest_pos_tracking = pos_tracking[:, 1:]
            # Sort only the rest tokens based on the padding mask
            sort_indices = torch.argsort(rest_padding_mask, dim=1)
            rest_x = rest_x.gather(1, sort_indices.unsqueeze(-1).expand(-1, -1, x.size(2)))
            rest_padding_mask = rest_padding_mask.gather(1, sort_indices)
            rest_size = rest_size.gather(1, sort_indices.unsqueeze(-1).expand(-1, -1, size.size(2)))
            if pos_tracking is not None:
                rest_pos_tracking = rest_pos_tracking.gather(
                    1, sort_indices.unsqueeze(-1).expand(-1, -1, pos_tracking.size(2))
                )
            # Recombine the unchanged cls token with the sorted tokens
            x = torch.cat([cls_token_x, rest_x], dim=1)
            padding_mask = torch.cat([cls_token_padding, rest_padding_mask], dim=1)
            size = torch.cat([cls_token_size, rest_size], dim=1)
            if pos_tracking is not None:
                pos_tracking = torch.cat([cls_token_pos, rest_pos_tracking], dim=1)
        else:
            # if there is only one token, do nothing
            pass
    else:
        # Rearrange x, padding_mask, and size so that non-padding instances are at the front
        # padding_mask is 0 for non-padding and 1 for padding
        sort_indices = torch.argsort(padding_mask, dim=1)
        # Use gather to rearrange x, padding_mask, and size according to sort_indices
        x = x.gather(1, sort_indices.unsqueeze(-1).expand(-1, -1, x.size(2)))
        padding_mask = padding_mask.gather(1, sort_indices)
        size = size.gather(1, sort_indices.unsqueeze(-1).expand(-1, -1, size.size(2))) # (b, s, 1)
        if pos_tracking is not None:
            pos_tracking = pos_tracking.gather(1, sort_indices.unsqueeze(-1).expand(-1, -1, pos_tracking.size(2)))

    x = x / (size+1e-4)

    # Truncate to the maximum length
    max_len = int((padding_mask < 0.5).to(torch.int64).sum(dim=-1).max().item()) # 0 for non-padding, 1 for padding

    x = x[:, :max_len]
    padding_mask = padding_mask[:, :max_len]
    size = size[:, :max_len]
    if pos_tracking is not None:
        pos_tracking = pos_tracking[:, :max_len]
    return x, size, padding_mask, pos_tracking

def repeat_merged_tokens_w_pos_tracking(merged_tokens, pos_tracking=None):
    """
    Args:
        merged_tokens (Tensor): shape (B, merged_num, hidden_size)
        pos_tracking (Tensor, optional): shape (B, merged_num, target_len)
            For example, suppose merged token num is 2 and target_len is 4;
            pos_tracking might be:
                [[1, 0, 0, 1],
                 [0, 1, 1, 0]]
            meaning token 0 should be repeated in positions 0 and 3, 
            and token 1 in positions 1 and 2.
            
    Returns:
        Tensor: shape (B, target_len, hidden_size)
            Each target position is filled with the corresponding merged token.
    """
    if pos_tracking is None:
        return merged_tokens
    else:
        # Ensure pos_tracking is of float type (in case it is provided as a boolean tensor)
        pos_tracking = pos_tracking.to(merged_tokens.dtype)
        # Transpose pos_tracking to shape (B, target_len, merged_num)
        # Then use batch matrix multiplication to "gather" the merged tokens to the target positions
        repeated_tokens = torch.bmm(pos_tracking.transpose(1, 2), merged_tokens)
        return repeated_tokens

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class ToMEMultiheadAttention(nn.MultiheadAttention):
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        size:Optional[Tensor]=None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None):
        assert self.batch_first
        B = query.shape[0]
        N = query.shape[1]
        full_bias = None
        if size is not None:
            size_bias_log = size.log()[:, :, 0] # (b, src_len, 1) -> (b, src_len)
            size_bias_log = size_bias_log.unsqueeze(1).unsqueeze(1).expand(B, self.num_heads, N, N) # (b, src_len) -> (b, num_heads, 1, src_len)
            full_bias = size_bias_log
        
        # apply attention mask before softmax (-inf for masked tokens)
        if attn_mask is not None:
            if attn_mask.size() != (B, 1, N, N):
                raise ValueError(
                    f"Attention mask should be of size {(B, 1, N, N)}, but is {attn_mask.size()}"
                )
            # Expand the attn_mask to [B, num_heads, N, N]
            expanded_mask = attn_mask.expand(B, self.num_heads, N, N)
            if full_bias is None:
                full_bias = expanded_mask
            else:
                full_bias = full_bias + expanded_mask

        # If a bias mask was created, reshape it to merge the batch and head dimensions.
        if full_bias is not None:
            full_bias = full_bias.reshape(B * self.num_heads, N, N)      

        # compute the metric
        q, k, v = _in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)
        assert k.shape[0] == B, k.shape[1] == N
        k = k.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3) # (B, h, N, d_h)
        metric = k.mean(1)

        return super().forward(query=query, key=key, value=value, attn_mask=full_bias, need_weights=need_weights)[0], metric

class ToMEResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
            batch_first: bool = True,
            # TOME args
            trace_source: bool = False,
            prop_attn: bool = True,
            cls_token: bool = True,
            r: int = 0,
            merge_mode: str = "instance_level",
            max_r_per_instance_ratio: float = None,
            update_threshold: bool = False,
            specified_threshold: float = None

    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = ToMEMultiheadAttention(d_model, n_head, batch_first=batch_first)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        # ToMe configs
        self._tome_info = {
            "r": r, # number of tokens to remove
            "size": None,
            "source": None,
            "trace_source": trace_source,
            "prop_attn": prop_attn,
            "class_token": cls_token,
            "distill_token": False,
            "merge_mode": merge_mode,
            "max_r_per_instance_ratio": max_r_per_instance_ratio,
        }
        if max_r_per_instance_ratio is not None:
            print("setting max r per instance to: ", int(self._tome_info["max_r_per_instance_ratio"] * r))

        self.update_threshold = update_threshold
        # if r>0:
        self.register_buffer('threshold', torch.tensor(1.0)) # default to be no merging
        # self.threshold = torch.tensor(1.0)
        self.threshold_count = 1.0
        self.specified_threshold = specified_threshold

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        attn_out, metric = self.attn(
            q_x, k_x, v_x, need_weights=False, size=attn_size, attn_mask=attn_mask
        )
        return attn_out, metric

    def threshold_running_avg(self, new_value):
        if new_value is not None:
            with torch.no_grad():
                if torch.all(self.threshold == 1.0):
                    self.threshold = new_value
                else:
                    if new_value.device != self.threshold.device:
                        new_value = new_value.to(self.threshold.device)
                    # self.threshold = (1-self.momentum) * self.threshold + self.momentum * new_value
                    self.threshold = (self.threshold*self.threshold_count + new_value)/(self.threshold_count+1)
                    self.threshold_count+=1
                    # print(f'New threshold: {self.threshold}')
                    if dist.is_initialized() and dist.get_world_size() > 1:
                        dist.all_reduce(self.threshold, op=dist.ReduceOp.AVG)
                            
    def merge_tokens(self, metric, r, hidden_states, padding_mask=None, pos_tracking=None):
        if self._tome_info["merge_mode"] == "instance_level":
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, hidden_states, self._tome_info["source"]
                )
            hidden_states, self._tome_info["size"], pos_tracking = merge_wavg(
                merge, hidden_states, self._tome_info["size"], pos_tracking=pos_tracking
            )

        elif self._tome_info["merge_mode"] == "batch_level":

            if not self.training and not self.update_threshold:
                if self.specified_threshold is not None:
                    specified_threshold = self.specified_threshold
                else:
                    specified_threshold = self.threshold
            else:
                specified_threshold = None

            if self._tome_info["max_r_per_instance_ratio"] is None:
                max_r_per_instance = None
            else:
                max_r_per_instance = int(self._tome_info["max_r_per_instance_ratio"] * r)

            B = hidden_states.shape[0]
            if specified_threshold is not None and B == 1:
                # inference time; use efficient instance-level with threshold version
                merge, _ = bipartite_soft_matching(
                    metric,
                    r,
                    self._tome_info["class_token"],
                    self._tome_info["distill_token"],
                    specified_threshold=specified_threshold,
                    max_r_per_instance=max_r_per_instance
                )
                hidden_states, self._tome_info["size"], pos_tracking = merge_wavg(
                    merge, hidden_states, self._tome_info["size"], pos_tracking=pos_tracking
                )
            else:
                merge, _, batch_threshold = batch_level_bipartite_soft_matching(
                    metric,
                    r,
                    self._tome_info["class_token"],
                    self._tome_info["distill_token"],
                    padding_mask = padding_mask,
                    max_r_per_instance = max_r_per_instance,
                    specified_threshold = specified_threshold
                )
                if merge != do_nothing:
                    hidden_states, self._tome_info["size"], padding_mask, pos_tracking = batch_level_merge_wavg(
                        merge, hidden_states, self._tome_info["size"], pos_tracking=pos_tracking, cls_token=self._tome_info["class_token"]
                    )
                    if self.training or self.update_threshold:
                        self.threshold_running_avg(batch_threshold)
            
        return hidden_states, padding_mask, pos_tracking

    def _get_attn_mask_from_padding_mask(self, padding_mask, dtype):
        """
            input: padding mask: (b, s): 0 for non-padding, 1 for padding
            output: attention mask: (b, 1, s, s): 0 for non-padding, -inf for padding
        """
        # Expand padding mask to match attention mask shape
        attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # Shape: (b, 1, 1, s)
        attn_mask = attn_mask.expand(-1, 1, padding_mask.size(1), -1)  # Shape: (b, 1, s, s)
        # Convert padding positions to -inf
        attn_mask = attn_mask * torch.finfo(dtype).min
        return attn_mask


    def forward(
            self,
            q_x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
            padding_mask: Optional[torch.Tensor] = None,
            pos_tracking: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        dtype = q_x.dtype
        assert attn_mask is None
        if padding_mask is not None:
            attn_mask = self._get_attn_mask_from_padding_mask(padding_mask, dtype)
        else:
            attn_mask = None
        attn_out, metric = self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        x = q_x + self.ls_1(attn_out)
        
        r = self._tome_info["r"]
        if r > 0:
            x, padding_mask, pos_tracking = self.merge_tokens(metric, r, x, padding_mask=padding_mask, pos_tracking=pos_tracking)
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x, padding_mask, pos_tracking

class ToMEOpenAITransformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            batch_first: bool = True,
            # tome args
            merge_mode: str = "batch_level", # merge mode: instance_level or batch_level
            r_total: int = 0, # total number of tokens to remove
            r_schedule: str = "constant", # r schedule: constant, linear, reverse_linear
            max_r_per_instance_ratio: float = None, # 1.0 => rever to fixed r for each instance; > 1.0 => dynamic r
            update_threshold: bool = False, # whether to post-hoc update threshold after training
            specified_thresholds: List[float] = None, # specified threshold for each layer
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.batch_first = batch_first
        self.grad_checkpointing = False
        ### ToMe configs ###
        self.rs = self._get_rs(layers, r_total, r_schedule)
        print("set total avg remove token nums each layer as: ", self.rs)
        print("merge mode: ", merge_mode)
        assert merge_mode in ["instance_level", "batch_level"]
        self._tome_info = {
            "size": None,
            "source": None,
            "trace_source": False,
            "prop_attn": True,
            "class_token": True,
            "distill_token": False,
            "merge_mode": merge_mode,
            "max_r_per_instance_ratio": max_r_per_instance_ratio,
            "update_threshold": update_threshold,
            "specified_thresholds": specified_thresholds,
        }
        self.resblocks = nn.ModuleList([
            ToMEResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                batch_first=batch_first,
                ## ToMe args
                trace_source = False,
                prop_attn = True,
                cls_token = True,
                r = self.rs[i],
                merge_mode = merge_mode,
                max_r_per_instance_ratio = max_r_per_instance_ratio,
                update_threshold = update_threshold,
                specified_threshold = specified_thresholds[i] if specified_thresholds is not None else None
            )
            for i in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def _get_rs(self, num_layers, r_total, r_schedule="constant"):
        
        if r_total == 0:
            return [0] * num_layers
        
        if r_schedule == "constant":
            if r_total % num_layers == 0:
                r = r_total // num_layers
                return [r] * num_layers
            else:
                # Distribute as evenly as possible, but account for remainders
                base_r = r_total // num_layers
                remainder = r_total % num_layers
                # Create a distribution list starting with the base value
                distribution = [base_r] * num_layers
                # Distribute the remainder across the first few layers
                for i in range(remainder):
                    distribution[i] += 1
                return distribution

        elif r_schedule in ["linear", "reverse_linear"]:
            # approximate a linear schedule with the last layer has no reduction
            M = r_total
            N = num_layers
            r0 = (2*M) // N
            step = r0 / N
            s = []
            while sum(s) + int(r0 - len(s)*step) < M:
                s.append(int(r0 - len(s)*step))
            if sum(s) < M:
                s.append(M - sum(s))
            while len(s) < N:
                s.append(0)
            assert sum(s) == M
            assert len(s) == N
            if r_schedule == "linear":
                return s
            else:
                return s[::-1]
        else:
            raise ValueError(f"Invalid r_schedule: {r_schedule}")


    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # for enable getting intermediate features for llava-style model
        self._tome_info["size"] = None
        self._tome_info["source"] = None
        assert self.batch_first, "only support batch_first"
        # if not self.batch_first:
        #     x = x.transpose(0, 1).contiguous()    # NLD -> LND

        hidden_states = []
        padding_masks = []
        sizes = []
        pos_trackings = []
        self.output_stats = {}
        padding_mask = None
        B, N = x.shape[:2]
        pos_tracking = torch.eye(N, dtype=torch.int32, device=x.device).unsqueeze(0).expand(B, -1, -1)
        for idx, r in enumerate(self.resblocks):
            r._tome_info["size"] = self._tome_info["size"]
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x, padding_mask, pos_tracking = checkpoint(r, x, None, None, attn_mask, padding_mask, pos_tracking)
            else:
                x, padding_mask, pos_tracking = r(x, attn_mask=attn_mask, padding_mask=padding_mask, pos_tracking=pos_tracking)
            self._tome_info["size"] = r._tome_info["size"]
            hidden_states.append(x)
            padding_masks.append(padding_mask)
            pos_trackings.append(pos_tracking)
            sizes.append(self._tome_info["size"])
            if padding_mask is not None:
                ntoks = (padding_mask<0.5).float().sum(-1)
                ntoks = ntoks.detach().tolist()
            else:
                ntoks = [x.shape[1]]*x.shape[0]
            self.output_stats[f"block_{idx}_ntoks"] = ntoks
        final_size = self._tome_info["size"]

        # if not self.batch_first:
        #     x = x.transpose(0, 1)    # LND -> NLD
        return x, padding_mask, final_size, {"hidden_states": hidden_states, "padding_masks": padding_masks, "sizes": sizes, "pos_trackings": pos_trackings}

