# Presentation Script — Group 20

**Total time: 10 minutes (+ 5 min Q&A)**

Speaker assignments are based on subsystem ownership. Transitions are marked with `→`.

---

## SLIDE 1 — Title (Aariz · ~30s)

Good morning/afternoon everyone. We are Group 20, and our project is an Intelligent Multimodal Search Engine, built for WyseTime Technologies.

In short — you type a description in natural language, and the system finds matching images using AI. No filenames, no manual tags. Just describe what you are looking for.

The system is live right now — you can try it at the URL on screen.

→ Let me start by explaining the problem we were asked to solve.

---

## SLIDE 2 — The Problem (Aariz · ~50s)

WyseTime Technologies builds AI computer vision tools for security and surveillance. Their challenge is simple but painful: they have a growing library of images, and the only way to find anything is by filename or manually applied tags.

There are three core issues with this. First, keyword search fails — if no one tagged an image with the right words, it is invisible. Second, manual tagging does not scale — it is slow, inconsistent, and incomplete. Third, there is no conceptual understanding — you cannot search for "a man wearing glasses" or "crowd near an entrance" because traditional systems have no idea what is actually in the image.

The client asked us to build an MVP — a proof of concept — demonstrating that intelligent, real-time image search is possible using CLIP and LanceDB.

→ So here is what we built.

---

## SLIDE 3 — Our Solution (Aariz · ~50s)

Our system has four key capabilities.

First, natural language search. You type a description — "elderly man with glasses", "red trolley on a city street" — and the engine finds matching images by understanding meaning, not matching keywords.

Second, auto-captioning. Every image that gets uploaded is automatically described by an AI model called BLIP. These captions are displayed in the results and are themselves searchable.

Third, hybrid search. For every query, we actually run two separate searches — one against the image itself, one against its caption — and merge the results. This gives us better recall than either search alone.

Fourth, online ingestion. You can upload images through the admin panel — by file, by URL, or in bulk via CSV. Images become searchable in about 13 seconds. No command line required.

→ Karl and Song will now explain the AI models that power this.

---

## SLIDE 4 — Architecture (Karl · ~30s)

Before we get into the models, here is the high-level architecture. The system has three layers.

The frontend is built with Next.js and React, hosted on Vercel. It handles the search interface and the admin panel.

The backend is a FastAPI server running inside a Docker container on Hugging Face Spaces. This is where the AI models live — CLIP for embeddings and BLIP for captioning.

The data layer is LanceDB, an embedded vector database. It stores the vectors and metadata locally, and syncs everything to a Hugging Face Dataset repository in the background so data survives container restarts.

→ Now let me explain how CLIP and BLIP actually work.

---

## SLIDE 5 — AI Pipeline (Karl & Song · ~1m 15s)

**Karl:**

We use two AI models together. The first is CLIP, by OpenAI. CLIP converts both images and text into 512-dimensional vectors in a shared mathematical space. This is what makes text-to-image search possible — the text query and the image end up in the same space, so we can measure how close they are.

CLIP is about 400 megabytes and loads at server startup. We use the ViT-B/32 variant — a Vision Transformer with a 32-by-32 patch size.

**Song:**

The second model is BLIP, by Salesforce. BLIP looks at an image and generates a natural language caption — for example, "a couple of elephants walking down a dirt road." This caption gives us human-readable context and a second search pathway.

BLIP is about 990 megabytes and is lazy-loaded — it only initialises the first time a caption is requested. This cuts cold start time significantly.

**Karl:**

Every uploaded image goes through a three-step pipeline. Step one: CLIP encodes the image into a 512-dimensional vector. Step two: BLIP generates a text caption. Step three: CLIP encodes that caption into a second vector. Both vectors and the caption are stored in the database — giving us two search pathways per image.

→ Bessie and Zheng will explain how the search actually works.

---

## SLIDE 6 — How Search Works (Bessie & Zheng · ~1m 15s)

**Bessie:**

When a user types a query — say "elephant crossing" — CLIP encodes it into a 512-dimensional vector. We then run two searches against LanceDB using cosine distance.

Search one compares the query vector against the image vectors. This is cross-modal matching — text versus image. Because they are different modalities, the maximum similarity here is typically around 0.35.

Search two compares the same query vector against the caption vectors. This is same-modal matching — text versus text. Because both are from CLIP's text encoder, similarity can reach 1.0 for exact matches.

**Zheng:**

After both searches complete, we merge the results. For each image, we keep whichever search gave the better — meaning lower distance — score. Then we filter by the user's threshold and sort by distance.

One important detail: the raw cosine similarity numbers are not intuitive. A "good" match might show as 22% — which makes the system look broken. So we rescale the output. The actual useful range — 0.40 to 1.00 — is mapped to 0 to 100 percent for display. This is why results show clean percentage scores.

→ Back to Aariz for the evaluation results.

---

## SLIDE 7 — Results & Accuracy (Aariz · ~50s)

We evaluated the system with 15 test queries against 30 indexed images.

Precision at 1 — meaning the correct image is the number one result — is 73.3 percent. Precision at 10 is 86.7 percent. The Mean Reciprocal Rank is 0.84.

All three misses were abstract or semantic queries. "Wildlife in Africa" ranked 6th because the BLIP caption says "elephants on a dirt road" — no mention of Africa. "Agricultural scene with livestock" ranked 3rd because BLIP misidentified the image — it generated "a man feeding a goat" when the image actually shows a man petting a dog. Both the image vector and the incorrect caption vector fail to match the query well. This is a BLIP captioning error, not a semantic distance problem."Fashion portrait, minimal background" ranked 10th because those are abstract descriptors CLIP does not strongly associate with a woman in a red dress.

The pattern is clear: the system excels at concrete descriptions and struggles with abstract or geographic concepts. This is a known limitation of CLIP's training data.

→ Here are the performance numbers.

---

## SLIDE 8 — Performance (Aariz · ~45s)

On the production deployment — Hugging Face Spaces free tier, CPU only, no GPU — search takes about 1.1 seconds on a warm server. Single image ingestion takes about 13 seconds. Bulk ingestion runs at roughly 24 images per minute. Cold start after the container sleeps is about 79 seconds.

We applied four key optimisations. An LRU cache for text embeddings — 128 entries, so repeated queries return instantly. Lazy BLIP loading — the 990 megabyte model only loads when needed, cutting cold start from 60 seconds to 30 for search-only sessions. Batch CLIP embedding — bulk ingestion processes 16 images per forward pass with 8 concurrent download threads. And an IVF-PQ index that builds automatically at 256 rows, partitioning the vector space so search becomes sub-linear.

→ Now for what we learned.

---

## SLIDE 9 — Lessons Learned (Karl or Song + Aariz · ~1m)

**Karl or Song:**

On the technical side — we tested replacing BLIP with Google Gemini Flash Lite for captioning. Gemini produced significantly richer, more accurate captions. But overall search accuracy actually dropped.

The reason: CLIP's text encoder was trained on short, simple captions — the kind BLIP produces. Gemini's long, detailed captions created diluted embeddings. The lesson is that upgrading one model in a tightly coupled pipeline can degrade the whole system. To benefit from Gemini, we would also need to upgrade the text encoder.

**Aariz:**

On the process side — we ended the project with 19 unmerged commits on the main branch. The final working version lived on a separate branch. When we tried to merge, we hit a conflict — just in the README, nothing catastrophic — but under deadline pressure, we made the pragmatic call to leave it.

The root cause: we never established a branching strategy. Most of the development was done by one person, so regular merges felt unnecessary. That assumption broke down when the rest of the team needed to access the codebase for Open Day. The lesson: process debt compounds just like code debt. Small, frequent merges beat large, painful ones.

→ To wrap up.

---

## SLIDE 10 — Research Extension: SigLIP + RRF + OWL-ViT (Song · ~1m 30s)

While the deployed system uses CLIP and a two-channel hybrid search, I investigated whether we could do better — and built a research branch to find out.

The first change is the embedding model. I replaced CLIP with SigLIP — Google's updated vision-language model. The key difference is the loss function. CLIP uses softmax, which compares every image-text pair against the entire training batch. SigLIP uses sigmoid loss, which treats each pair independently — this scales better and produces stronger cross-modal alignment. I also found that prepending "a photo of" to every query — so "elephant crossing" becomes "a photo of elephant crossing" — meaningfully improves SigLIP's match accuracy for short queries.

The second change is how search results are merged. Instead of just taking the best score from each channel, I implemented Reciprocal Rank Fusion across three channels: image vectors, caption vectors, and BM25 keyword search on the caption text. RRF assigns each result a score of 1 divided by 60 plus its rank, summed across all channels it appears in. A result that ranks well in all three channels scores higher than one that dominates just one. This is more principled than the max-score approach.

The third layer is OWL-ViT — Open World Localization with Vision Transformers, also from Google. After RRF produces a ranked list, the top candidates are physically re-examined. OWL-ViT looks inside each image and tries to locate the object described in the query. If it finds a strong match, it boosts that result's score. If not, the RRF ranking holds. The system goes from "these vectors are similar" to "this object is actually present in this image."

One more thing worth noting about BLIP in this system. In the CLIP version, BLIP carries significant weight — CLIP's cross-modal alignment is weaker, so the caption channel compensates. In the SigLIP version, the image vector is already strong enough that BLIP becomes less critical. A concrete example: on the same image where BLIP generates a wrong caption, CLIP's image vector scores 76% — borderline. SigLIP's image vector scores 96% on that same image, from pixels alone, no caption needed. BLIP still contributes for keyword-only queries like "veteran" or "Africa" that have no visual signature — but for everything else, SigLIP has largely made it redundant.

The branch is not yet merged into production — but the results show meaningful accuracy improvements, particularly on abstract and geographic queries that the base CLIP system struggles with.

→ Back to Aariz for the conclusion.

---

## SLIDE 11 — Conclusion & Q&A (Aariz · ~30s)

To summarise what we delivered: a working, deployed semantic image search engine with natural language queries, hybrid dual-vector search, auto-captioning, online ingestion, a persistent cloud deployment, and solid accuracy benchmarks.

On the right you can see where we would take it next — a larger CLIP model, pairing Gemini with a matching text encoder, image-to-image reverse search, and proper authentication.

The system is live. We are happy to take any questions.

*[Open Q&A — 5 minutes]*

---

## Timing Summary

| Slide | Speaker | Time |
|:------|:--------|:-----|
| 1 — Title | Aariz | ~30s |
| 2 — Problem | Aariz | ~50s |
| 3 — Solution | Aariz | ~50s |
| 4 — Architecture | Karl | ~30s |
| 5 — AI Pipeline | Karl + Song | ~1m 15s |
| 6 — Search | Bessie + Zheng | ~1m 15s |
| 7 — Results | Aariz | ~50s |
| 8 — Performance | Aariz | ~45s |
| 9 — Lessons | Karl + Aariz | ~1m |
| 10 — Research Extension | Song | ~1m 30s |
| 11 — Conclusion | Aariz | ~30s |
| **Total** | | **~10 min 35s** |

This leaves a short buffer before the 5-minute Q&A. Also resolves the previously unassigned speaker on slide 9 — Karl takes the technical reflection, Aariz takes the process reflection.

---

## Q&A Preparation — Person-Specific

> These are directed questions likely aimed at each person based on their role and what the examiners know from the interim report. The interim report is their most recent knowledge of the project — they will probe the gaps between what was promised and what was delivered.

---

### Aariz — UI / Systems Engineer / Quality Evaluation

**"Your interim report said no cloud infrastructure was permitted. You're running on Hugging Face Spaces. How do you justify that?"**
The interim constraint was written at a point where we anticipated local deployment only. As the system matured, it became clear that a live demonstrable product required a deployment environment. HF Spaces is a free, open-source platform — not a paid cloud service like AWS or Azure. It sits closer to a self-hosted solution than to commercial cloud infrastructure, and it was the only way to make the live demo possible without local machine dependency.

**"You moved from Streamlit to Next.js. The interim report spent a full section explaining why Streamlit had problems. Was this planned or reactive?"**
It was partly anticipated. The interim report identified the core limitations of Streamlit — no front-end control, poor separation of concerns, impossible to achieve a professional UI. Those findings directly justified the switch. By the time the interim was submitted, the decision to move to a decoupled JS frontend was already forming. Next.js was chosen specifically because it gives full React control, clean API separation, and zero-config Vercel deployment.

**"Why Next.js specifically and not just plain React or Vue?"**
Next.js adds file-based routing and Vercel deployment integration with zero configuration. The admin panel at `/admin` and the search page at `/` are naturally separate routes. It also gives us SSR as an option if we ever need it, though we didn't use it here since everything is API-driven.

**"How did you evaluate the system? 15 queries feels like a small sample."**
15 queries against 30 images was a deliberate, bounded benchmark — not a statistical study. Every query had one pre-assigned target image chosen before evaluation so there was no post-hoc cherry-picking. The corpus was kept small so every failure could be manually investigated and explained. Precision@K and MRR are standard IR metrics — the values are comparable to published CLIP-based retrieval benchmarks on small corpora. For a production system you'd want thousands of queries, but for an MVP proof of concept this is appropriate.

**"Your Precision@1 is 73.3%. How does that compare to the state of the art?"**
CLIP ViT-B/32 on zero-shot retrieval tasks typically achieves 65–80% Precision@1 on small, diverse corpora. We're in that range running on CPU-only free-tier hardware with no fine-tuning. It's a solid baseline. The three misses were all abstract or geographic queries — a known limitation of CLIP's training data, not a system bug.

**"What does the score percentage in the UI actually represent?"**
Raw cosine similarity for a good cross-modal match only reaches 0.35–0.70 — which looks broken to users. We rescale the useful range, 0.40 to 1.00, onto 0 to 100% for display. A 96% badge means the best possible match. A 0% badge means it's at the floor of useful similarity. This is purely a display transformation — the underlying search uses raw cosine distance.

**"What happens to the system when the container restarts on HF Spaces?"**
The container is ephemeral — all local data would be lost. We solved this with a HF Dataset repository. Every write to LanceDB triggers an async sync to the dataset repo via the HF Hub API. On boot, the server downloads the dataset and restores the database before accepting any requests. The 79-second cold start includes this restore step.

---

### Karl — Ingestion / Offline Indexing

**"The interim described an offline indexing pipeline. The final system has online ingestion. How did that transition happen?"**
The interim planned an offline batch process — pre-embed a fixed dataset and serve it. The client's actual requirement was for images to become searchable without any command-line intervention. Online ingestion — upload a file, it's embedded and searchable in 13 seconds — directly addresses that. The offline pipeline still exists in the bulk CSV ingest path, but the primary flow became online.

**"Walk me through exactly what happens when an image is uploaded."**
Three steps. First, CLIP encodes the image into a 512-dimensional vector. Second, BLIP generates a natural language caption — for example, "a couple of elephants walking down a dirt road." Third, CLIP encodes that caption into a second 512-dimensional vector. Both vectors plus the caption are stored as a single record in LanceDB. This gives us two search pathways per image.

**"The interim mentioned embedding generation was slow on laptops without GPUs. How did you address that in the final system?"**
Three things. Batch embedding — bulk ingestion processes 16 images per CLIP forward pass, which is more efficient than one at a time. Concurrent downloads — 8 threads download images in parallel before embedding, so the GPU (or CPU) is never waiting on network. And lazy BLIP loading — BLIP is 990MB and only initialises the first time a caption is requested, cutting cold start significantly for search-only sessions.

**"How does the IVF-PQ index work and what parameters did you use?"**
IVF partitions the vector space into clusters. When searching, instead of comparing against every vector, you only check the nearest clusters. PQ compresses each vector by splitting it into sub-vectors and quantizing each one. The index builds automatically when the table reaches 256 rows — we use LanceDB's defaults for the partition count and PQ parameters. Below 256 rows, LanceDB falls back to brute-force which is still fast at that scale.

**"Why cosine distance rather than Euclidean?"**
Cosine measures the angle between vectors, not their magnitude. CLIP embeddings are normalised to unit length, so cosine distance and Euclidean distance are equivalent for normalised vectors — but cosine is the convention for embedding similarity and is what CLIP was trained to optimise.

---

### Song — Ingestion / SigLIP + RRF + OWL-ViT Research

**"The interim report listed your initial research area as embedded systems optimisation. How did you end up building SigLIP+RRF+OWL-ViT?"**
The initial scope changed in Meeting 3 when the client redefined the project. Once the core CLIP system was working, I investigated whether we could improve on it. The natural direction was the embedding model — SigLIP is Google's direct successor to CLIP for this exact use case. RRF and OWL-ViT followed from wanting a more principled fusion strategy and a physical verification layer.

**"Why SigLIP over CLIP specifically? What is the practical difference?"**
CLIP uses softmax loss — during training, every image competes against every other image in the batch to match its text pair. This gets noisy at large batch sizes. SigLIP uses sigmoid loss — each image-text pair is treated independently as a binary yes/no match. The result is tighter cross-modal alignment. On the same image where CLIP scores 76%, SigLIP scores 96% from the image vector alone.

**"Explain the RRF formula."**
Each image gets a score of 1 divided by 60 plus its rank in each channel it appears in. You sum this across all channels. So an image ranked 1st in image search, 2nd in caption search, and 3rd in BM25 gets 1/61 + 1/62 + 1/63, which is higher than an image ranked 1st in only one channel. The 60 is a constant that dampens the difference between top ranks — rank 1 and rank 5 score similarly, but rank 5 and rank 50 differ significantly.

**"Why three channels? Why add BM25?"**
Image vectors catch visual similarity. Caption vectors catch semantic similarity. BM25 catches exact keyword matches — queries like "veteran" or "Africa" that have no strong visual signature but might appear word-for-word in a BLIP caption. Three channels give you coverage across all three failure modes.

**"What does OWL-ViT actually do in your pipeline?"**
After RRF produces a ranked list, I pass the top candidates to OWL-ViT along with the query text. OWL-ViT tries to locate a bounding box for the described object inside each image. If it finds one with confidence above 0.15, that image's RRF score gets boosted. It's a physical verification layer — the system goes from "these vectors are similar" to "this object is actually present in this image."

**"Why is the branch not merged into production?"**
Three deployment bugs needed fixing first — the Dockerfile had stale environment variables pointing to CLIP instead of SigLIP, and there was a hardcoded vector dimension of 512 in the ingestion service that would crash with SigLIP's 768-dim output. Those are fixed now and the SigLIP system is deployed separately for comparison.

---

### Bessie (XU BINDAN) — Query / Retrieval / Similarity Search

**"The interim report mentioned FAISS for approximate nearest neighbour search. The final system uses LanceDB's built-in IVF-PQ. Why the change?"**
FAISS is a standalone library that would require managing a separate index alongside LanceDB. LanceDB natively supports IVF-PQ indexing — it's built in and integrates with the same table that stores the vectors and metadata. Adding FAISS would have added complexity with no benefit given our scale.

**"How does the two-channel hybrid search merge work exactly?"**
For a given query, we run two vector searches against LanceDB — one against the image embedding column, one against the caption embedding column. Each returns a list of images with distance scores. We iterate over all results and for each unique photo_id, we keep whichever search gave the lower distance — that is, the better match. Then we filter by the threshold and sort ascending by distance.

**"Why does cross-modal similarity cap at around 0.35?"**
Because image and text are different modalities. Even a perfect semantic match between a query like "elephants on a road" and an elephant photo will only align the vectors to about 0.35 cosine similarity — they're encoded by different encoder paths. Text-to-text matching, where both the query and the caption go through CLIP's text encoder, can reach 1.0 because they're in the same sub-space.

**"How did you choose the similarity threshold?"**
The threshold controls what gets filtered out of results. We set it to 0.9 cosine distance (which corresponds to 0.1 similarity) by default — essentially showing everything that has any signal. The UI exposes this as a slider mapped to the 0.40–1.00 useful similarity range so users can tighten it interactively.

**"What is MRR and why is 0.84 a good score?"**
Mean Reciprocal Rank is the average of 1/rank across all queries. If the correct image is always #1, MRR is 1.0. If it's always #2, MRR is 0.5. Our MRR of 0.84 means that on average, the correct image is just barely below #1 — even on the queries where it misses the top spot, it's very close.

---

### Zheng — Query / Retrieval / Similarity Search

**"The interim planned Text Query Encoding using CLIP's text encoder to get a 512-dimensional vector. Is that exactly what the final system does?"**
Yes, that's precisely what happens. The user's query goes through CLIP's text encoder and produces a 512-dimensional vector in the same embedding space as the stored image vectors. The text encoder was not changed or fine-tuned — it's the pretrained OpenAI ViT-B/32 text encoder loaded via HuggingFace Transformers.

**"Your interim timeline showed similarity calculation using FAISS. The final system doesn't use FAISS. Did you implement the FAISS path first?"**
FAISS was considered during the planning phase. When we integrated LanceDB more deeply, we found its native IVF-PQ index performed equivalently for our scale and removed the need for a separate FAISS dependency. No separate FAISS implementation was built for production.

**"How does the LRU cache improve performance?"**
CLIP text encoding is deterministic — the same query string always produces the same 512-dim vector. So we cache the result with a 128-entry LRU cache. If a user searches "elephants" twice, the second search skips CLIP entirely and returns the cached vector instantly. This is particularly useful for repeated demo queries.

**"What happens when a query doesn't match anything well — are bad results still shown?"**
The threshold filter handles this. Results below the similarity threshold are discarded. If nothing clears the threshold, an empty result set is returned. The UI shows a "no results found" state rather than forcing low-quality matches onto the user.

**"The interim mentioned post-processing of results. What does that involve in the final system?"**
Three things: merging the two result lists by taking the best distance per photo_id, applying the threshold filter, and rescaling the raw cosine similarity to a 0–100% display score. The raw scores are not shown directly because cross-modal similarities look confusingly low — the rescaling maps the useful range to an intuitive percentage.
