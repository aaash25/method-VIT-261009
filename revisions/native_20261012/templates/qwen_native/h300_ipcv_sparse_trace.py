"""Diagnostic hooks for actual IPCV work; never used for formal timing."""

from bisect import bisect_right


class VisionTrace:
    def __init__(self, visual, inputs, mode):
        self.visual, self.mode = visual, mode
        self.native = int(inputs["pixel_values"].shape[0])
        self.original_cu = [0]
        for temporal, height, width in inputs["image_grid_thw"].detach().cpu().tolist():
            for _ in range(temporal):
                self.original_cu.append(self.original_cu[-1] + height * width)
        if self.original_cu[-1] != self.native:
            raise ValueError("grid/token count differs")
        self.expected_rotary = visual.rot_pos_emb(inputs["image_grid_thw"])
        self.rows, self.handles, self.saved, self.output = {}, [], None, None
        self.visual_calls = 0

    def __enter__(self):
        def begin(_module, _args):
            self.visual_calls += 1

        def finish(_module, _args, output):
            import torch
            self.output = {"shape": list(output.shape), "finite": bool(torch.isfinite(output).all())}

        self.handles += [self.visual.register_forward_pre_hook(begin), self.visual.register_forward_hook(finish)]
        for block in self.visual.blocks:
            layer = block.layer_idx

            def attention(_module, args, kwargs, layer=layer):
                import torch
                if layer in self.rows:
                    raise ValueError("vision layer executed more than once")
                saved = getattr(self.visual, "_sparse_vit_saved", None)
                if saved is not None and self.saved is None:
                    kept = saved["keep_indexs"].detach().cpu().tolist()
                    removed = saved["removed_indices"].detach().cpu().tolist()
                    links = saved["rem_to_kept_idx"].detach().cpu().tolist()
                    counts = [sum(start <= index < end for index in kept)
                              for start, end in zip(self.original_cu, self.original_cu[1:])]
                    cu = [0]
                    for count in counts:
                        cu.append(cu[-1] + count)
                    cross = sum(bisect_right(self.original_cu, removed_index) !=
                                bisect_right(self.original_cu, kept[neighbor])
                                for removed_index, neighbors in zip(removed, links) for neighbor in neighbors)
                    self.saved = {"kept": kept, "removed_count": len(removed), "pruned_cu": cu,
                                  "per_image_kept": counts, "cross_image_neighbor_links": cross,
                                  "neighbor_rule": "author global nearest neighbors preserved",
                                  "retained_boundary_tokens": [index for index in self.original_cu[1:-1] if index in kept]}
                config = self.visual.config.Sparse_config
                pruned_attention = saved is not None and layer >= config["vit_pruned_layer"] + config["AS_layer"]
                expected_rotary = self.expected_rotary[saved["keep_indexs"]] if pruned_attention else self.expected_rotary
                self.rows[layer] = {"layer": layer, "attention_tokens": len(args[0]),
                                    "cu": kwargs["cu_seqlens"].detach().cpu().tolist(),
                                    "rotary_tokens": len(kwargs["rotary_pos_emb"]),
                                    "rotary_exact": torch.equal(kwargs["rotary_pos_emb"], expected_rotary)}

            def mlp(_module, args, layer=layer):
                self.rows[layer]["mlp_tokens"] = len(args[0])

            self.handles += [block.attn.register_forward_pre_hook(attention, with_kwargs=True),
                             block.mlp.register_forward_pre_hook(mlp)]
        return self

    def __exit__(self, *_error):
        for handle in self.handles:
            handle.remove()

    def audit(self):
        config = self.visual.config.Sparse_config
        sparse = self.mode != "dense"
        if bool(config["vit_Sparse"]) != sparse:
            raise ValueError("requested sparse mode did not activate")
        if self.visual_calls != 1 or set(self.rows) != set(range(len(self.visual.blocks))):
            raise ValueError("incomplete visual execution")
        if hasattr(self.visual, "_sparse_vit_saved"):
            raise ValueError("stale sparse state after visual execution")
        if not self.output or not self.output["finite"]:
            raise ValueError("missing/nonfinite visual output")
        expected_shape = [self.native // (self.visual.spatial_merge_size ** 2), self.visual.config.hidden_size]
        if self.output["shape"] != expected_shape:
            raise ValueError("visual output was not restored to complete placeholder length")
        if sparse:
            expected_kept = int(self.native * (1 - config["vit_reduction_ratio"]))
            if not self.saved or len(self.saved["kept"]) != expected_kept or not 0 < expected_kept < self.native:
                raise ValueError("IPCV retention count differs")
            kept = self.saved["kept"]
            if kept != sorted(set(kept)) or kept[0] < 0 or kept[-1] >= self.native:
                raise ValueError("IPCV retained indices invalid")
        elif self.saved is not None:
            raise ValueError("Dense unexpectedly pruned")
        for layer, row in self.rows.items():
            pruned = sparse and layer >= config["vit_pruned_layer"]
            pruned_attention = pruned and layer >= config["vit_pruned_layer"] + config["AS_layer"]
            count = len(self.saved["kept"]) if pruned else self.native
            attention_count = count if pruned_attention else self.native
            cu = self.saved["pruned_cu"] if pruned_attention else self.original_cu
            if row["attention_tokens"] != attention_count or row["mlp_tokens"] != count:
                raise ValueError(f"actual IPCV work differs at layer {layer}")
            if row["cu"] != cu or row["rotary_tokens"] != attention_count or not row["rotary_exact"]:
                raise ValueError(f"actual IPCV image/rotary segments differ at layer {layer}")
        return {"state": "actual_work_verified", "mode": self.mode, "native_tokens": self.native,
                "original_cu": self.original_cu, "pruning": self.saved,
                "trace": list(self.rows.values()), "output": self.output,
                "instrumentation_scope": "diagnostic only; excluded from formal performance"}
