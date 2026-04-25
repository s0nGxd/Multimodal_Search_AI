# 005 — Add Gemini Flash multimodal as a final VLM re-ranker

## Context

After [ADR 003](003-owl-vit-detection-rerank.md) and [ADR 004](004-gemini-parser-for-compositional-queries.md), the system handled object queries and compositional queries well. It still failed on a category we hadn't accounted for: **actions and interactions.**

The breaking example: the query *"a guy getting in a car"* returned a photo of a man sitting on a couch at 58% match. Both SigLIP and OWL-ViT see "guy" in the couch image and have no way to verify whether the person is *getting in a car* or merely *visible*. SigLIP encodes objects, not verbs. OWL-ViT detects objects, not actions. Decomposing the query into "guy AND car" via the parser was a tempting fix, but it loses the action semantics — a guy *standing next to* a car would also satisfy "guy AND car" and should not.

The root problem is model capability: small vision encoders cannot reason about actions. No amount of prompt engineering or score calibration fixes this.

## Decision

Added Gemini Flash multimodal as a final re-ranking stage. After SigLIP + OWL-ViT produce the top candidates, each candidate image is sent to Gemini Flash with the original natural-language query and a strict scoring prompt. Gemini reasons about whether the image actually depicts what the query describes — including actions, attributes, and relationships — and returns a 0.0-1.0 score with a short reason.

Implementation: `client/app/api/vlm-rerank/route.ts`. Called from the frontend after the backend returns candidates. Top-8 candidates are scored in parallel. Image bytes are downloaded and resized server-side (to ~768px JPEG) before being sent to Vertex — Vertex's URL-fetch path was unreliable on the original full-size images.

Gemini calls run with `temperature: 0` for deterministic scoring. The VLM score replaces the SigLIP-derived score on the frontend; the user-visible threshold filters on the final VLM score, not the intermediate one.

## Consequences

- The couch-guy false positive dropped from 58% to under 20%, correctly classified as "no car visible, person sitting on couch."
- End-to-end latency increased: warm queries are 3-5s, cold queries up to 8s. Mitigated by parallel calls, server-side image resize, and warmup on page mount.
- Per-query API cost is now non-zero but negligible (~$0.0005 in Gemini calls per search).
- Gemini cannot fetch HuggingFace Space URLs reliably; the server-side fetch+resize step is mandatory, not optional.
- This is the single change with the largest perceived quality improvement for the operator-facing demo. It is also what positions the product against the broader category of "object-class video analytics" — competitors detect object classes; this product evaluates whether the image actually depicts what the user described.
