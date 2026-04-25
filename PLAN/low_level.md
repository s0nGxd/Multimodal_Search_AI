# Production Prep

Post-demo snapshot. Everything a future contributor (or future-you) needs to pick this up cold: where the code lives, what's broken, what to ship next, and what the product is actually for.

---

## 1. Live state at demo close (2026-04-24)

| Surface | URL | Status |
|---|---|---|
| Frontend | https://segp-siglip-frontend.vercel.app | OVERWATCH redesign, red accent, VLM pipeline live |
| Backend | https://aariz-s-segp-siglip-search.hf.space | Running on `cpu-basic`, serving `/search`, `/search/complex`, `/detect`, `/track` |
| Dataset | https://huggingface.co/datasets/aariz-s/segp-siglip-data | Reverted to commit `bd9338c` (before the 732-image poisoning). Current index: **125 rows** (20 verified-good, 105 legacy external) |

Frontend deploy: `vercel --prod --yes` from `client/` (prod alias live on `segp-siglip-frontend.vercel.app`).
Backend deploy: manual rsync into sibling `hf-space` clone + `git push` (see [CLAUDE.md](CLAUDE.md)).

---

## 2. GitHub state — **do this first**

The demo-session work is **not in GitHub**. Everything currently running in prod lives in the uncommitted working tree on branch `demo`.

- `origin/SigLip-RRF-Scoring` (canonical working branch): stale at `e59ab9f` — does NOT include the VLM pipeline, redesign, compositional search, parser, attribute handling, or any fix from today.
- `origin/final-version`: docs-only parallel branch.
- `origin/main`: stale, not used.

### What needs to be committed

Frontend (client/):
- `app/page.tsx` — OVERWATCH rewrite + VLM re-rank wiring + detection-phrase extraction
- `app/layout.tsx`, `app/globals.css` — redesign
- `app/api/parse-query/route.ts` — Gemini parser route (new)
- `app/api/vlm-rerank/route.ts` — VLM re-rank route (new)
- `lib/api.ts` — `parseQuery`, `searchComplex`, `vlmRerank` helpers
- `lib/vertex.ts` — shared Vertex client + `withQuotaRetry` (new)
- `package.json`, `package-lock.json` — new deps: `ai`, `@ai-sdk/google-vertex`, `zod`, `sharp`

Backend (server/):
- `app/routers/search.py` — `/search/complex`, clause cache, thread pool, DATA_DIR fix on detect/track, URL rewriter helpers
- `app/services/search_service.py` — `force_rerank`, `boost_weight`, `rerank_top`, `rerank_size`, `dedup_videos` flags; filter broken-vector rows (dist >= 0.98)

Docs:
- `DESIGN.md` (already tracked untouched)
- `CLAUDE.md` (untracked — was always a local doc, decide whether to commit or gitignore permanently)
- `DOCUMENTATION/gemini_vertex.md` (untracked reference)
- Design reference: `client/design-preview*.html` (decide keep or delete)

### Recommended push strategy

```bash
cd /Users/aarizsajan/Desktop/unm-segp-group-20-autumn-2025-online-indexing
git checkout -b production-2026-04-24
git add client/app client/lib client/package.json client/package-lock.json \
        server/app production_prep.md
git commit -m "post-demo snapshot: VLM re-rank + OVERWATCH UI + compositional search"
git push -u origin production-2026-04-24
```

Open a PR against `SigLip-RRF-Scoring` (or make `production-2026-04-24` the new canonical branch and archive the old one).

**Do NOT commit `.env`, `.vercel/project.json`, `/data`, `/lancedb_db`, or the design-preview html files.**

---

## 3. What shipped this cycle

| Feature | Where | Notes |
|---|---|---|
| Compositional parser (OR/AND/NOT) | `client/app/api/parse-query/route.ts` | Gemini Flash via Vertex, Zod-typed output |
| Attribute-aware clause search | `server/app/routers/search.py` `/search/complex` | Parallel thread pool, 60s clause cache |
| VLM re-rank | `client/app/api/vlm-rerank/route.ts` | Gemini Flash multimodal, temp=0, `sharp` resize to 768px |
| Detection phrase extraction | `client/app/page.tsx` | Uses `plan.clauses[0].object` for OWL-ViT (not raw query) |
| Per-video dedup after VLM | `client/app/page.tsx` | Backend returns multiple frames; frontend picks best VLM score |
| Broken-vector filter | `server/app/services/search_service.py` | Drops rows with distance >= 0.98 |
| OVERWATCH redesign | `client/app/*`, `globals.css` | Red accent (#D92D20), JetBrains Mono + Archivo |
| Perf: parallel clauses, top-3@512px, clause cache | `server/app/routers/search.py` | 2-clause attribute query: 3.4s cold → 1.0s warm |
| VLM + Gemini warmup on mount | `client/app/page.tsx` | Cuts first-query cold start |
| DATA_DIR fix for detect/track | `server/app/routers/search.py` | Was hardcoded "data" — matched CLAUDE.md gotcha note |

---

## 4. Known broken or fragile

1. **URL-ingestion stores placeholder vectors.** The main production bug. When images are ingested by URL (not file upload), the ingestion path writes zero/junk vectors into LanceDB. Those rows match every query with distance ~0 and poison top-K. Workaround deployed: server-side filter `_distance < 0.98`. Real fix: download + SigLIP-embed inside the ingest route. **This is the #1 item to fix.**

2. **Admin deletes are not durable across Space restart.** `sync_to_repo()` is async. If the Space restarts before sync completes, HF dataset repo restore brings deleted rows back. Happened today. Fix: either block on sync after delete, or manipulate the dataset repo directly from the delete handler.

3. **Browser cache eats Vercel deploys.** Hard-refresh didn't reliably pick up new builds today. Use incognito for demos. Production fix: add `Cache-Control: no-cache` on `/` response, or fingerprint client chunks more aggressively.

4. **Tracking jitter on cpu-basic.** OWL-ViT at ~800ms/frame caps tracking at ~1-2 fps. Norfair stitches, but it looks jumpy. Fix: T4 GPU tier, or de-scope tracking to single-frame bbox at `best_timestamp`.

5. **VLM cold start ~10s.** Mitigated by warmup on mount but Vercel function cold starts can still hurt. Consider pre-warming from a cron.

6. **Parser is under-tested.** Gemini prompt was tuned 4+ times today. Brittle to Gemini version bumps. Needs snapshot tests (input query → expected plan JSON) before any further prompt changes.

7. **No automated tests anywhere.** Every change is manually tested in prod.

8. **Vercel deploy uses a different project locally.** `client/.vercel/project.json` can drift. Per-deploy, re-run `vercel link --yes --project segp-siglip-frontend` before `vercel --prod` if unsure.

9. **Local `demo` branch diverges from origin.** 30+ unpushed commits represent real prod state — see section 2.

---

## 5. Production-readiness gaps

| Gap | Severity | Effort |
|---|---|---|
| No auth on `/api/admin/*`, `/api/clear-db`, `/api/ingest/*` | **High** — anyone can nuke the DB | Hours |
| Admin password is literal `"admin"` | **High** | Minutes |
| No rate limiting | Medium — HF Space will fall over at 10 RPS | Hours |
| No observability (Sentry, structured logs) | Medium | Half-day |
| No CI/CD; all deploys manual | Medium | Half-day |
| Space cold start 79s | Medium — ops impact only | N/A until GPU |
| `cpu-basic` saturates at ~5 concurrent searches | High for scaling | Upgrade |
| No automated tests | High — can't refactor safely | Days |
| Single-tenant (no org/user isolation) | Depends on target | Weeks |
| No secrets rotation plan | Medium | Hours |
| No health SLO tracking | Low at current scale | N/A |

---

## 6. Roadmap

### Short (1-2 weeks) — unblock next ingestion + shore up basics
- [ ] Fix URL-ingestion path: download → SigLIP embed → write real vector. Don't accept an ingest request that can't verify the vector is non-zero.
- [ ] Blocking `sync_to_repo()` after admin delete (or direct dataset-repo write).
- [ ] Parser snapshot tests — pin Gemini output for the five demo queries. Fail CI on drift.
- [ ] Basic auth gate on admin routes (env-based shared secret is fine).
- [ ] Cache-control: make sure new Vercel deploys are visible on refresh without incognito.
- [ ] CI/CD: GitHub Actions on push to `main` → Vercel prod + HF Space rebuild (via dispatch).
- [ ] Evaluate T4 GPU tier ($0.60/hr) for ingestion speed, tracking, search throughput.

### Medium (1 month) — features that make the product
- [ ] **Alert system:** save a query; receive a notification when new footage matches. This is what converts "search tool" into "surveillance product."
- [ ] **Incident timeline view:** group matches from the same video on a single timeline with bboxes at each hit.
- [ ] **Saved searches / query history** (per-user).
- [ ] **Export clips:** matched frames → downloadable evidence package with chain-of-custody metadata.
- [ ] **Multi-tenant:** org-scoped indexes so different customers don't see each other's footage.
- [ ] **Retry-aware ingestion:** resumable bulk uploads with per-file status.

### Long (3+ months) — moat
- [ ] **Live monitoring:** VLM continuously scans a small set of cameras for a registered subject. Different from archive search. Probably needs a dedicated GPU worker pool.
- [ ] **Cross-camera re-ID:** track a person across multiple cameras/feeds (not just within one video).
- [ ] **Multi-modal queries:** upload a reference image + a text description (e.g. "this person, but leaving the building").
- [ ] **Audio semantic search:** gunshots, glass breaks, raised voices. Huge signal for incident detection.
- [ ] **Query-as-policy:** "alert me if anyone enters the west door after 22:00 not wearing a uniform" — compositional query + schedule + camera + action.

### Final step — expose this as an MCP server
- [ ] **MCP server wrapping the search API.** Once the product is stable, expose `search`, `search_complex`, `vlm_rerank`, `detect`, `track`, `list_images`, and `ingest` as Model Context Protocol tools. This turns the surveillance archive into a tool any AI agent (Claude Desktop, Cursor, custom agents) can call directly — investigators stop using a UI and instead ask their assistant "find the moment from yesterday's car park footage where someone got in the white van", and the agent calls the MCP tool, gets back the frame + bbox + timestamp, and renders it inline.

  **Why this is the endgame, not a feature:** the UI is one consumption surface. MCP makes it infrastructure. Every downstream AI workflow — incident reports auto-drafted from search results, investigator chatbots, multi-step forensic agents — becomes possible without rebuilding clients. It's also the cleanest moat: the search engine becomes the back end behind every AI-native security tool, instead of competing with the next prettier UI.

  **Scope of work:** ~1 week. TypeScript MCP SDK; one tool per existing endpoint; auth via env-var token; deploy as a published npm package or HTTP MCP endpoint. Reference: `@modelcontextprotocol/sdk`. Test against Claude Desktop config first, then expand.

---

## 7. Applicability — what this product is (and isn't)

### Positioning
**Forensic visual retrieval for surveillance archives.**
Post-incident investigators describe what they're looking for in plain English; the system returns the matching frames in seconds instead of hours.

### Real use cases
- **Insurance claims** — vehicle theft, property damage, slip-and-fall. Evidence in seconds, filed before lunch.
- **Unauthorized access** — server rooms, restricted zones, after-hours entries. IT/security incident response.
- **Emergency response audit** — EMS arrival verification, dispatch accuracy, fraud investigation.
- **Missing persons** — description-based CCTV search ("child in red shirt, black cap, last seen near food court").
- **Retail loss prevention** — post-shoplifting investigation, repeat-offender identification.
- **Litigation discovery** — search hours of footage for events matching a legal description.
- **HR / harassment investigation** — locating specific interactions in office CCTV.

### What it is NOT
- **Not real-time monitoring.** On `cpu-basic` it cannot watch live feeds 24/7. The demo is reactive (query → result), not proactive (event → alert).
- **Not an alerting system.** No subscriptions, no push notifications (yet — see roadmap).
- **Not a video analytics suite.** Doesn't do counting, heatmaps, dwell-time, people-counting.
- **Not multi-camera live.** One video at a time.

### Market position
| Category | Competitor example | Our edge |
|---|---|---|
| Traditional NVR | Hikvision, Dahua | Adds natural-language semantic retrieval |
| Motion analytics | Axis, Avigilon | Knows what subjects are doing, not just that pixels moved |
| Object-class analytics | Genetec, Milestone | Distinguishes "red car" from "blue car", "getting in" from "standing next to" — action-aware |
| Manual scrubbing | (most of the market) | 3 seconds vs. 8 hours |

### Pitch in one paragraph
> Security cameras record 24/7. A mid-sized facility has 40 cameras — 960 hours of footage per day. When an incident happens, investigators scrub timestamp bars looking for the right 3 seconds. We let them type a description instead. "A guy getting in a car" returns the right frame in seconds, drawn with a bounding box on the subject. Forensic retrieval for the archive every camera already produces.

### Rehearsed demo queries (known-good on current dataset)
1. `a guy getting in a car` → Car_Park_Break_In video, VLM correctly promotes the action moment.
2. `a child in a mall wearing a black cap and a red shirt` → Shopping_Mall_Security_Video frame, VLM scores 0.95.
3. `ambulance` → 5977704 ambulance video frame.

Do not freestyle queries in front of an audience. Every semantic search system has embarrassing false positives.

---

## 8. Immediate cleanup before walking away

```bash
# stray scratch files
rm -f /tmp/drop2.txt /tmp/to_delete_ids.txt

# sibling clones (keep or delete — they're useful for deploys)
# ls ../hf-space ../hf-dataset

# verify prod is serving what you think it is
curl -sS -o /dev/null -w 'frontend %{http_code}\n' https://segp-siglip-frontend.vercel.app
curl -sS -o /dev/null -w 'backend  %{http_code}\n' https://aariz-s-segp-siglip-search.hf.space/ready
```

---

## 9. Open questions for the team / supervisor

- Who owns the HF Space + dataset repos long-term? Currently all on `aariz-s`.
- Is GPU spend approved? T4 small is ~$0.60/hr, L4 is ~$0.80/hr.
- Target market cut: forensic/post-incident vs. live-monitoring? They need different architectures.
- If WyseTime wants this, under what licensing — is this a product or a research prototype handoff?
- Do we keep the `origin/final-version` docs branch, or fold it in?
