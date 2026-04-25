# 001 — Use SigLIP instead of CLIP as the vision-language encoder

## Context

The first working version of the search engine used `openai/clip-vit-base-patch32` (512-dim) as the cross-modal encoder. CLIP is the canonical choice for image-text matching and we adopted it as a sensible default.

In testing, however, raw cosine similarity between query text embeddings and image embeddings clustered tightly in the 0.20-0.30 range across both relevant and irrelevant images. This made score calibration difficult — the gap between "definitely matches" and "looks vaguely related" was too narrow to threshold on. CLIP is trained with a softmax contrastive loss over an entire batch, which optimises for relative ranking but produces poorly-calibrated absolute similarities for one-shot retrieval.

## Decision

Switched to `google/siglip-base-patch16-256` (768-dim). SigLIP replaces softmax with a per-pair sigmoid loss, which produces better-calibrated absolute similarities and is empirically stronger on cross-modal retrieval at the same parameter count.

## Consequences

- Score distribution widened. Genuine matches now cluster around 0.06-0.10 raw cosine — narrower in absolute terms, but the *gap* from non-matches widened, making thresholding viable.
- Embedding dimension changed from 512 → 768. The existing LanceDB index had to be rebuilt because dimension mismatch crashes vector search at boot.
- Discovered a SigLIP-specific gotcha: the text encoder requires `padding="max_length"` (fixed 64 tokens), not `padding=True`. Padding to longest gave near-random embeddings for single queries. Documented in `CLAUDE.md`.
- All downstream calibration constants in `search_service.py` (`SIM_FLOOR`, `SIM_CEIL`) were tuned for SigLIP's narrower absolute range.
