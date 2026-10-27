# DRAFT FINAL record for C5 (not registered; the coordinator registers it with `labbook hypothesis --dir ledger/paper`)

Open before registering: (1) the engine image is not protocol.json's engine_image (needs Dylan's protocol change);
(2) the `final` map source (the release ships none; proposal: the release's own bf16 source, sha256-pinned below);
(3) identity arm (not expressible in the engine format; proposed "not run"); (4) random-rows control (proposal in
c5/PLAN.md: "not run", or the class-matched rarity-shuffled substitute). Edit the arms line to what is decided.
Fill the two `<commit>` slots with the c5-kit commit the run uses.

**id:** FINAL-FLASHNEXT-C5

**claim:**
[FINAL-FLASHNEXT-C5] cell qwen3.8-flash-next-c5 (flashnext, tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ at
6fe3e7464c18), engine tcclaviger/vllm:29.08.10 (image sha256:809657ddb1533db10a0c08267c3ee6fde180b839e379713414efa121415914a3,
RidgeFill built in: vllm/v1/ced.py, vllm/models/qwen4_exp/amd/ced.py; eager, TP2 on 2x R9700), engine tier, code tag
c5-kit <commit>, locked manifest 21d4fa8ca15c8c8e. Claim C5 (Dylan, 2026-10-08): on Flash-Next, RidgeFill with the GDN
terminal-state correction and class56 exact rows at 25% (the graded arm, ridgefill+corr+prot25) is non-inferior to exact
on both perplexity and filled-region retrieval while prefilling at least 1.25x faster at 16K. Projector: the public
release TheCacheIsALie/RidgeFill-Projector-Qwen3.8-Flash-Next-i8 at 0b6f2ce017fe912198c697bcc813a1578bc67ce1 (manifest
sha256 795363fa153f0e7e16454af6acabd644709e703d84de0dd9535565226bb51faa; split 24; int8 maps dequantised exactly to the
engine's bf16 format, correction split per TP rank, rowsel score as the engine's unigram table: c5/convert.py, checked
through the engine's own loaders, c5/engine_check.py), converted files SHA256SUMS sha256 <sha of projector-tcc/SHA256SUMS>;
`final` map (read only for approximated rows' logits and MTP, neither used here) verbatim from the release's bf16 source
sha256 c9db6064eb275d3166df8139239959dfdeb64f2778707a2f42215164999cb9cd (the manifest's fit.proj_sha256). Tail T = 2048
exact rows (DYLUHN_KVA_EXACT_TAIL=1, prefix caching off; real tails read back per request). Data: the locked long-prompt
documents of nominal length <= 32K (90 documents, 6 domains x 15; continuation 512 tokens, novel tokens primary); RULER's
4 NIAH tasks at 8K/16K/32K (300 rows) plus niah_single_1 at 64K (25 rows), raw format, generation caps of protocol 1.1.2,
filled-region rows (every scored needle ends at or before N - 2048): 219 eligible; TTFT at 9,216 / 16,384 / 32,768 tokens
(prefixes of long-prompts/chat_32768_0.txt), 1 warm-up + 5 timed, median, arm and same-boot exact path interleaved per
round. Arms (one server boot each): exact (with exact-rerun on the same boot: all perplexity documents and the 8K+16K
RULER rows), ridgefill+corr+prot25 (graded; plus 6 documents on its exact path as the positive control), ridgefill+corr,
ridgefill, ridgefill+corr+prot10 / prot50 / prot100 (knob curve: perplexity and speed only). Not run: identity (the
engine has no hyper-connection identity fill), all-rows control (no engine rule; the exact-path control stands in),
random rows (no engine rule) [edit if decided]. Declared: declared_share = 25, margins {ppl_ratio_upper_pct: 1.0,
ruler_points_lower: -2.0, ttft_min_speedup: 1.25, ttft_length: 16384}, split_layer = 24, tail = 2048.
| pins: kit c5-kit <commit>; projector-tcc SHA256SUMS sha256=<...>; release manifest sha256=795363fa153f0e7e16454af6acabd644709e703d84de0dd9535565226bb51faa

**predict:**
(a) Perplexity: the per-document paired novel-token NLL difference ridgefill+corr+prot25 - exact has a one-sided 95%
upper bound <= log(1.01) (document-weighted; BCa, 5,000 resamples of documents within domain;
finals/stats.py noninferiority, upper margin). (b) RULER: the per-row paired filled-region accuracy difference
ridgefill+corr+prot25 - exact, in points, has a one-sided 95% lower bound >= -2 (sample-weighted, rows within task,
unit range (-100, 100), the exact-binomial floor applies: 219 eligible rows give a zero-miss floor of about -1.36 points).
(c) Speed: median TTFT of exact over ridgefill+corr+prot25 at 16,384 tokens >= 1.25 on the same boot (exact path
interleaved per round), and reported at 9K and 32K with the exact boot's own TTFT as the cross-check. Reported alongside,
not graded: the same-boot exact-rerun differences (perplexity, RULER 8K+16K) as the noise floor next to (a) and (b);
ridgefill+corr (speed mode) and ridgefill vs exact on perplexity and speed; the knob curve (share of prefix rows run
exactly, from the engine log, vs perplexity delta and TTFT); RULER all-needles accuracy; the positive control's
difference to exact. Determinism: if the two exact passes on one boot are not identical, protocol.json
engine_tier.final_requires (the deterministic top-k fix) is unmet, the result is reported only under
engine_tier.prefix_label and supports no claim.

**kill:**
C5 fails if (a) or (b) fails, i.e. the perplexity upper bound exceeds +1% or the RULER lower bound is below -2 points,
or if (c) is below 1.25x at 16K. A failure of (a) or (b) alone does not show harm (non-inferiority, not equivalence);
the paper then states the measured bound and drops C5's non-inferiority wording for the failing metric. If the
same-boot exact-rerun bound itself lies outside a margin, the run cannot resolve that margin and C5 is reported as
inconclusive for it, with the noise floor.
