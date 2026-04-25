# 003 — Add OWL-ViT detection as a second-stage re-ranker

## Context

After removing the caption channel ([ADR 002](002-removing-blip-captions-and-bm25.md)), pure SigLIP scoring was the only signal driving ranking. SigLIP cosine similarity for genuine matches on our dataset topped out around 0.06-0.10. That meant a strong match displayed at ~70%, a weak match at ~30%, and a non-match at ~0% — a respectable distribution but with no second signal to disambiguate when SigLIP was uncertain.

The result: queries like "ambulance" returned the right ambulance video, but on harder queries SigLIP would surface vaguely-related photos at scores indistinguishable from genuine matches.

## Decision

Added [OWL-ViT](https://huggingface.co/google/owlvit-base-patch32) — Google's open-vocabulary object detection model — as a second-stage re-ranker. The pipeline now is:

1. SigLIP retrieves top-K candidates by vector similarity.
2. The top-N candidates (default N=5) are passed through OWL-ViT with the query as the open-vocabulary detection prompt.
3. Detection confidence is folded into the final score: high-confidence detections boost the candidate; absent or low-confidence detections do not.

OWL-ViT was chosen because it accepts arbitrary text prompts as detection classes — meaning we don't need to retrain it for new query categories.

## Consequences

- Recovered most of the precision lost when captions were dropped. False positives in the top-3 became rare on object-noun queries.
- Adds ~700-900ms per detection pass on the HuggingFace `cpu-basic` tier. To stay within latency budget we cap re-ranks at top-5 and skip re-rank when SigLIP is already very confident (`rescaled >= 0.65`). Constants live in `search_service.py`.
- For attribute-heavy clauses, we *force* the re-rank (no skip) and weight detection confidence more heavily — OWL-ViT binds attributes more reliably than SigLIP at base size.
- OWL-ViT's open-vocabulary nature is partial — it does well on simple noun phrases ("car", "person") and degrades on long descriptive phrases ("child in a red shirt and black cap"). The frontend now extracts the bare object from the parsed query for detection ([ADR 004](004-gemini-parser-for-compositional-queries.md)).
