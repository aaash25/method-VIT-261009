# Native FP16 memory adaptation

The failed r3 completed POPE in all four modes, then ran out of memory during
HRBench4K/0 Dense warmup in native text SDPA (5.76 GiB allocation). V100 torch
2.5.1 selects the math kernel for mask-free GQA; its independent KV head groups
can be evaluated sequentially with the same scaling, causal mask, dtype and
native math operation. No images, crops, tokens or ViT method are changed.

Only V100 (compute capability 7.0), torch 2.5.x, no-gradient inference, zero
dropout, no explicit attention mask and prefill >=1024 tokens are adapted.
Decode, short requests and masked calls retain native dispatch. Both Dense
and every candidate use this same scoped text adaptation. Registry restored
on success and error; no installed library or frozen source is edited.

FP32 and FP16 GPU component tests require PyTorch's standard dtype-specific
assert_close tolerance, including 4097-token prefill and actual Qwen2 model
binding. Sequential head groups preserve mathematical operations but may change
BLAS kernel scheduling and floating rounding. R4 strict bitwise admission failed
and correctly stopped before model work: V100 maximum difference was 4.8828125e-4
for FP16 and 4.76837158203125e-7 for FP32 in 4097-token prefill. Actual tiny Qwen2
FP16 logits were bitwise equal; FP32 maximum difference was 7.45e-8. These are
not claims of universal bitwise equivalence. All raw failed tests are retained.

The user's clarification is authoritative: fidelity means preserving the method,
not forcing elementwise agreement or rewriting faithful outputs to match history.
R6 records complete-token and text comparison with pre-adaptation r3 POPE
Dense/disabled/restored and PiToMe candidate as observations, not admission gates.
Component closeness checks only catch implementation errors in the same
mathematical attention operation; no candidate-vs-Dense elementwise gate exists. The longer r3
HRBench request had no complete output because of OOM, so no full-token identity
claim is possible there. New Dense identity across all methods and restored
mode, complete EOS, journal/export and native input gates remain strict.

Timing is simultaneous; the text dispatch and shape counters are included in
E2E for every mode. Pure ViT boundary is unchanged. This bounded diagnostic
does not establish full-split accuracy or speedup. All previous failures remain.

Primary version-specific source: https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/torch/nn/functional.py
