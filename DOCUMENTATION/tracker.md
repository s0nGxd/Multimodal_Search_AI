# Brief Requirements Tracker

Maps the original client brief ([Project Description_ Intelligent Multimodal Search Engine.pdf](./Project%20Description_%20Intelligent%20Multimodal%20Search%20Engine.pdf)) against what we delivered. Use this to verify scope coverage before submission.

## Status

| Brief requirement | Status | Delivered as |
|---|---|---|
| Multimodal natural-language → image search | ✅ Done | Core product behaviour |
| VLM (CLIP suggested) for shared text/image space | ✅ Done | SigLIP — see [ADR 001](./decisions/001-siglip-over-clip.md) for the upgrade rationale |
| Public datasets (Unsplash / COCO / Flickr30k) | ✅ Done | Unsplash imagery present in current index |
| LanceDB for vector storage | ✅ Done | Production storage layer |
| Phase 1: Data Acquisition & Setup | ✅ Done | Ingestion pipeline + admin upload |
| Phase 2: Embedding Generation | ✅ Done | SigLIP image + text embeddings |
| Phase 3: Database Indexing | ✅ Done | LanceDB cosine vector index |
| Phase 4: Search Engine + web UI (Streamlit/Flask) | ✅ Done | Next.js + FastAPI (upgraded from suggested stack) |
| Phase 5: Evaluation — Precision@K | ⚠️ Missing | No formal Precision@K evaluation yet |
| Phase 5: Evaluation — Latency | ✅ Done | Latency numbers in [PLAN/low_level.md](../PLAN/low_level.md) |
| Phase 5: Evaluation — Usability | ⚠️ Informal | No formal usability study; team-internal testing only |
| Functional prototype | ✅ Done | Live on Vercel + HuggingFace Space |
| Documented codebase | ✅ Done | [PLAN/](../PLAN/) + [ADRs](./decisions/) + READMEs |
| Project report | ✅ Done | [Int-Grp-20.md](./Int-Grp-20.md) / [Int-Grp-20.pdf](./Int-Grp-20.pdf) |
| Performance metrics | ⚠️ Partial | Latency yes; Precision@K outstanding |
| Presentation | ↗ External - ✅ | Separate deliverable, not in repo |

## Beyond the brief — capabilities added during the project

The brief asked for single-query semantic image search using CLIP. We delivered that and added the following capabilities, each documented as an ADR:

- **Video frame search and jump-to-best-moment** — searching across video frames as well as still images.
- **Object detection re-ranking** — recovers precision lost when removing the captioning channel. See [ADR 003](./decisions/003-owl-vit-detection-rerank.md).
- **Compositional queries** (OR / AND / NOT) — boolean structure that a single embedding cannot represent. See [ADR 004](./decisions/004-gemini-parser-for-compositional-queries.md).
- **Action-aware retrieval via VLM re-rank** — verbs and interactions that object encoders cannot resolve. See [ADR 005](./decisions/005-vlm-rerank-for-action-understanding.md). The brief's own example query (*"a dog catching a frisbee in a sunny park"*) is itself an action query, so this addition directly serves the spirit of the brief.
- **Live deployment** — the brief asked for a prototype; we delivered a hosted production-grade system.

## Outstanding before submission — in order of priority

Features stack on fundamentals. Anything in this list past item 1 is only worth doing once the foundations underneath it are solid.

### 1. Make it functional and push it to Vercel

The system has to actually work end-to-end before anything else. Until this is true, no evaluation, no usability study, no fancy capability adds are worth attempting. The day-of-demo failures all traced to layers failing in seams with no graceful degradation.

- ~~**Upgrade Hugging Face Space to a GPU tier** (T4 small or L4) so SigLIP, OWL-ViT and ingestion stop competing for two CPU cores. Single biggest reliability win available; one-click change in the Space settings.~~ **Not being done — out of scope (cost).** The project must stay on the free `cpu-basic` tier; paid GPU tiers and PRO-gated ZeroGPU are not options, and a Community GPU Grant is at HF's discretion and not something we can rely on. Consequences carried into the live demo:
    - Cold start stays at ~79s after 48h idle. Mitigated by UptimeRobot pings, but the first query after a true cold start will still feel slow.
    - Warm search stays at ~1.1s, dominated by OWL-ViT re-rank on 15 candidates running on 2 vCPUs.
    - Ingestion remains ~3–5s per image; bulk uploads of 100+ images during the demo are not viable.
    - Concurrent requests serialise on the 2 vCPUs, so two judges querying at the same time will queue. The demo plan must keep queries sequential.
    - Under sustained CPU saturation, FastAPI request timeouts cascade and surface as `0 results` — this is the failure mode that dominated demo day, and is the reason the remaining items in this section (retries, fallbacks, circuit breaker, caching, health check, pre-warm, error UI) are worth doing even without a GPU. Search *quality* (embeddings, scores, ranking) is identical on CPU vs GPU, so the constraint costs us speed and headroom, not correctness. To be acknowledged in the project report as a known limitation.
- **Add retries with exponential backoff** at every external call: Gemini Vertex, HF image fetch, Vercel function fetches. Today a single transient failure surfaces as `0 results`.
- **Add fallback paths.** If the VLM re-rank fails, fall back to the SigLIP+OWL-ViT score rather than dropping the candidate. If the parser fails, fall back to single-mode raw query. Never let an outer layer silently swallow an inner-layer error.
- **Add a circuit breaker on the VLM stage.** If two consecutive Gemini calls error within a session, skip VLM re-rank for the rest of the session and surface a banner explaining the system is in degraded mode.
- **Aggressive caching:** parsed plans, VLM scores per (query, photo_id), search results. Reduces redundant external calls and makes the demo path repeatable.
- **End-to-end pipeline health check.** Today `/ready` only proves the models loaded. We need a check that runs an actual canned query through parser → search → VLM → results and verifies a known-good outcome.
- **Pre-warm everything** on container start and on page mount. Cold-start failures kill demos.
- **Better error UI.** "Search failed" with no detail leaves the user guessing. Show what stage failed, offer to retry.
- **Push the stable build to Vercel production** and verify a fresh incognito session can run the canned demo queries without intervention.

### 2. Real automated tests

- Snapshot tests for the parser (input query → expected plan JSON). Pin Gemini behaviour and fail CI on drift.
- End-to-end integration test: known query → expected top result.
- Smoke test of every API endpoint with realistic payloads.

Without these, every change is tested in production.

### 3. Run Precision@K evaluation (Phase 5 of the brief)

Once the system is stable, pick 10–20 representative queries spanning object, action, and attribute types; hand-label relevance over the current index; compute P@1, P@5, P@10. Add the results to the project report and to [PLAN/low_level.md](../PLAN/low_level.md).

### 4. Document usability findings

Even informally — which team members ran which queries, what they observed, what they reported.

### 5. Cross-link the project report to the ADRs and PLAN docs

So a reader of `Int-Grp-20` can trace any claim back to its supporting evidence.

---

## Bonus / future work — only after the above is done

The items below would meaningfully extend the product but do not move the marker grade against the brief. Pursue only if everything above is shipped.

### Pick a more robust VLM than Gemini Flash

Gemini Flash works but its failure modes (cold starts, rate limits, prompt sensitivity) were the source of most of our day-of-demo regressions. Worth a short bake-off if time permits.

| Option | Strengths | Tradeoffs |
|---|---|---|
| **Claude 3.5 Sonnet (Anthropic API)** | Strongest multimodal reasoning currently available; deterministic; well-documented prompt behaviour | Paid; rate limits |
| **GPT-4o (OpenAI)** | Reliable; well-tested; established ecosystem | Paid; not always best on action grading |
| **Self-hosted Qwen2-VL or LLaVA-1.6** on a small GPU | No API dependency; no cold starts; full prompt control; per-call cost ~zero | Requires GPU infra; slower per-call than API VLMs at small scale |
| **Stay on Gemini Flash + harden** | No migration cost | Same failure modes |

### Differentiation moves (system-level, not API integrations)

Each of these is buildable in days, not months. None require new training. They are how the system stops being "yet another CLIP demo" and starts being a tool an operator could actually use:

- **Query expansion.** Rephrase the query in 3–5 ways via the parser LLM, run SigLIP for each, fuse via RRF.
- **Multi-prompt VLM grading.** Replace the single score with weighted sub-scores for object presence, action, and attribute match.
- **Relevance feedback.** Operator clicks "good" / "wrong" on a few results; system learns an offset to the query embedding and re-ranks. Standard in visual e-commerce, missing from CCTV analytics.
- **Temporal video queries.** "Find when X enters then Y happens." No commercial product offers open-vocabulary temporal video search.
- **Cross-modal query.** Operator uploads a reference image *and* text ("more like this person, wearing different clothing"). Direct fit for forensic use case.
- **Confidence calibration.** Surface uncertainty explicitly ("3 strong matches, 4 possible") rather than a flat ranked list — important when results are evidence.
