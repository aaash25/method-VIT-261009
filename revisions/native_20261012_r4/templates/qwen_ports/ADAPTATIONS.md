# Qwen DToMe within-question batch repair

Status: CPU candidate validated; exact-release GPU smoke pending. Historical
releases and outputs remain unchanged.

The previous Qwen port applied the author's B=1 branch independently to every
image. The author's B>1 branch instead packs even/odd slots using the padding
mask's default argsort and uses size + 1e-4 in the weighted denominator. These
differences change the next layer's bipartite partition; the frozen author-only
characterization found different final partitions in 2 of 12 two-layer cases.
This repair follows the B>1 branch for all segments of one original question.

Qwen has ragged image grids. Segments are conceptually padded to the maximum
current length for the author's packing order. Pair scores and attention stay
within each segment. With a specified threshold and no per-instance removal
cap, selecting every edge above threshold is equivalent to the author's global
sort-and-count selection. We do not batch different questions. We do not change
the three frozen thresholds or layer schedules.

The mapping back to original slots uses parent indices instead of a quadratic
one-hot membership matrix. Rotary positions follow surviving target tokens.
Native-dtype similarity, native ranking and native scatter dispatch are retained.
Only FP16 merge accumulation is promoted to FP32: a real POPE/0 request produced
an infinite mean at layer 32 despite finite inputs (maximum 15936), and a bounded
weighted-mean case reproduces the same overflow for DToMe. No extra stable
sorting or deterministic group reductions remain in these methods. Outputs retain
the requested model dtype. The author's batch denominator epsilon is preserved
at both precisions. The B=1 branch is unchanged. When the padded width is one,
the author's early no-op is preserved.

An active B>1 method with zero removed tokens can still reorder and rescale
tokens under the author's rules. It must not be mislabeled Dense identity.
Disabled-mode, restored-mode and cross-method Dense gates still require exact
complete tokens. The no-removal method semantics are checked against the literal
author CPU oracle, including two successive layers.

CPU coverage: six seeds, seven equal/ragged/odd/minimal length pairs, three
thresholds including no removal, two precisions, two layers; 42 parameterized
oracle tests plus two real Transformers visual-tower tests covering disabled
and restored Dense identity, independent requests, and cleanup after failure.
The real model checkpoint, remote stack and GPU execution are still pending.
