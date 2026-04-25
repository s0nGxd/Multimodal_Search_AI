# 002 — Remove BLIP image captions and the BM25 keyword channel

## Context

The original retrieval pipeline ran three channels in parallel and fused them with Reciprocal Rank Fusion (RRF):

1. **Image vector** — SigLIP visual similarity
2. **Caption vector** — SigLIP semantic match against BLIP-generated captions
3. **BM25 keyword** — exact keyword match on the same captions

The intent was that caption-driven channels would catch queries SigLIP missed, and that the three signals together would be more robust than any one alone.

In practice the caption channel introduced more noise than signal. BLIP-generated captions for surveillance-style images were vague ("a person walking on a road") and frequently lost the salient details we wanted to match against. RRF doesn't down-weight a noisy channel; it just averages ranks. So an image with a misleading caption could still get pushed into the top-K by the BM25 + caption-vector channels.

The team judged that captions were "wrong too often" to be load-bearing.

## Decision

Removed the BLIP caption generation step from ingestion entirely. Removed the BM25 channel. Removed `caption_vector` from the LanceDB schema. Search now runs a single channel: SigLIP image-vector vs. SigLIP text-query.

## Consequences

- Pipeline simplified. One channel instead of three. No more RRF fusion logic. No BLIP model loaded. Ingestion is faster.
- Lost the small benefit BM25 provided for queries that matched a literal keyword in a caption.
- Created a precision gap that needed to be filled — pure SigLIP scoring on its own wasn't strong enough to maintain the previous quality. This gap directly motivated [ADR 003](003-owl-vit-detection-rerank.md).
- Schema change required existing data to be re-ingested. Documented as a deployment gotcha in `CLAUDE.md`.
