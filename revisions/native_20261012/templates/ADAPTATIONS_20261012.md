# Native-method revision, 2026-10-12

This revision prioritizes paper-level logical fidelity and native author operations.
Candidate predictions are not forced to match earlier migration outputs. It is a
bounded regression release; full-split accuracy and speedup are not inferred from it.

| Path | Change and remaining adaptation |
|---|---|
| Qwen DToMe / PiToMe | Remove extra deterministic destination grouping, stable ranking and FP32 routing. Use native matching dtype and author scatter rules. FP16 merge accumulation alone uses FP32 because the real POPE/0 PiToMe request overflowed in layer 32's intermediate mean sum (finite input maximum 15936, infinite output); the same finite weighted-mean failure is reproduced for DToMe. Keep per-image attention intervals, rotary positions, ragged batch representation and original-slot restoration required by Qwen's host interface. These targets have no corresponding author-native Qwen implementation. |
| Smol DToMe | Replace the migration's sorted FP32 sums with the author's instance scatter-reduce / batch scatter-add. Keep threshold selection, numeric padding and restoration. Default reporting is deferred; optional diagnostic checks are separate from timed requests. |
| LLaVA DToMe | Execute the pinned author's OpenAI CLIP visual core with the checkpoint's HF weights, norms and MLP. Defer CPU layer-count reporting until both timers stop. Keep penultimate-layer selection, CLS handling and restoration required by the vision-only host boundary. |
| OneVision DToMe | Execute hash-pinned native author SigLIP functions; remove forced deterministic scatter dispatch. Keep native epsilon, batch ranking, padding and zero-edge reordering. Restore original slots and defer CPU reporting. |
| OneVision PiToMe | Remove the forced deterministic-algorithm wrapper. Preserve author's merge arithmetic and ranking. |
| Qwen IPCV | Move diagnostic VisionTrace to an unmeasured warmup admission. The complete timed request runs the method without diagnostic tensor copies/equality checks. |

Necessary compatibility adaptations retained:

- Qwen FP16 text QK upcast protects against observed overflow (explicitly accepted by user).
- Smol text attention memory handling and key-hook cleanup address the real five-image failure; no image is dropped and no ViT work is shared between questions.
- OneVision V100 sequential text KV-head groups avoid a measured Dense prefill OOM. They preserve the mathematical SDPA operation but can change floating-point scheduling; the same context applies to Dense and candidates. This is not a ViT optimization claim.
- iLLaVA FP32 uses the actual input dtype instead of an incompatible FP16 allocation; author method decisions remain unchanged.
- Existing environment/signature, model-boundary and restoration glue remains. Upstream source commits and hashes remain in the frozen base releases and local author loaders.

`run.py` is the unified bounded regression entry for the four models and both precisions.
It reuses existing offline model assets and the named frozen base runtimes. These
dependencies are intentional; this server overlay is not a standalone model distribution.
The earlier per-bundle launchers are retained as historical material and are not the new entry.
`JOB_MATRIX.json` freezes the 26 worker cells (Smol runs all methods within each precision).
Actual launcher dry-run output must agree with that matrix before any GPU work.

Every timed request retains all official images, runs one question at a time, and
records simultaneous pure-ViT and E2E timings including the observer overhead.
Image-file validation is skipped at the user's request. Dense restoration, full
natural-EOS generation, method activation, source seals, journal CRC and export
agreement remain gates. Native compressed-output variation is respected.

Status at source freeze: local component checks passed; exact-release GPU results
are pending and will be recorded in a separate receipt without rewriting this source.
