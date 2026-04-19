# Evaluation Metrics — SEGP Semantic Search

## Search Accuracy (15 test queries, 30 indexed images)

| Metric | Value | Description |
|---|---|---|
| Precision@1 | **73.3%** (11/15) | Correct image is the #1 result |
| Precision@3 | **80.0%** (12/15) | Correct image is in the top 3 results |
| Precision@5 | **80.0%** (12/15) | Correct image is in the top 5 results |
| Precision@10 | **86.7%** (13/15) | Correct image is in the top 10 results |
| Mean Reciprocal Rank (MRR) | **0.84** | Average of 1/rank across all queries |
| Avg Cosine Similarity | **0.7946** | Average similarity score for correct matches |

> **Note:** Precision@1 and @3 are slightly lower than the 10-image benchmark (80.0% and 86.7% respectively) because the index now contains 30 images, introducing more competition in the result rankings. This is expected behaviour — accuracy metrics decrease as the corpus grows.

### Test Queries & Results

| # | Query | Expected Image | Rank | Cosine Sim | P@1 |
|---|---|---|---|---|---|
| 1 | elephants walking on a road | mario-scheibl | 1 | 0.836 | Hit |
| 2 | women wearing traditional saris | sabesh-photography | 1 | 0.841 | Hit |
| 3 | person cutting a cake | land-o-lakes-m9cB | 1 | 0.823 | Hit |
| 4 | man feeding a goat on a farm | land-o-lakes-TXdL | 1 | 0.828 | Hit |
| 5 | elderly man wearing glasses | land-o-lakes-ydAy | 1 | 0.807 | Hit |
| 6 | computer desk with neon sign | remy_loz | 1 | 0.842 | Hit |
| 7 | two people sitting on railroad tracks | land-o-lakes-lAB4 | 1 | 0.819 | Hit |
| 8 | mountain landscape with clouds | roberto-shumski | 1 | 0.831 | Hit |
| 9 | red trolley on a city street | land-o-lakes-1w3t | 1 | 0.816 | Hit |
| 10 | woman in a red dress | hoi-an-photographer | 1 | 0.845 | Hit |
| 11 | wildlife in Africa | mario-scheibl | 6 | 0.683 | Miss |
| 12 | office setup with RGB lighting | remy_loz | 1 | 0.791 | Hit |
| 13 | veteran with glasses | land-o-lakes-ydAy | 1 | 0.760 | Hit |
| 14 | agricultural scene with livestock | land-o-lakes-TXdL | 3 | 0.692 | Miss |
| 15 | fashion portrait against minimal background | hoi-an-photographer | 10 | 0.654 | Miss |

### Failure Analysis

All 3 misses are **abstract/semantic queries** that require reasoning beyond literal visual or caption matching:

- **"wildlife in Africa"** — BLIP caption is "a herd of elephants walking across a dirt road," which doesn't mention Africa or wildlife. CLIP image embedding lacks geographic context. Ranked 6th.
- **"agricultural scene with livestock"** — BLIP misidentified the image entirely. The image shows a man petting a dog; BLIP generated the caption "a man feeding a goat" — a hallucination. Both the image vector and the (incorrect) caption vector therefore fail to match "agricultural scene with livestock" well. This is a captioning error, not a semantic distance problem. Ranked 3rd.
- **"fashion portrait against minimal background"** — BLIP caption is "a woman in a red dress standing in front of a white wall." The concepts "fashion portrait" and "minimal background" are abstract descriptors that CLIP doesn't strongly associate with the literal caption. Ranked 10th.

## System Performance (Production — HF Spaces)

| Metric | Value |
|---|---|
| Search query response time (warm) | **1,100 - 1,200ms** avg |
| Search query response time (first query after model load) | **~2,700ms** |
| Cold start time (container restart to healthy) | **~79s** |
| Single image upload-to-searchable | **~13s** |
| Bulk ingestion throughput | **~23.7 images/min** (20 images in 50.7s) |

## Hardware Specifications

### Production (HF Spaces Free Tier)

| Spec | Value |
|---|---|
| Platform | Hugging Face Spaces (Docker) |
| CPU | 2x vCPU |
| RAM | 16 GB |
| GPU | None (CPU-only inference) |
| Storage | Ephemeral (persisted via HF Hub sync) |
| PyTorch | CPU-only build |

### Models

| Model | Purpose | Parameters | Embedding Dim |
|---|---|---|---|
| openai/clip-vit-base-patch32 | Image + text embedding | 151M | 512 |
| Salesforce/blip-image-captioning-base | Auto-captioning | 247M | — |

### Database

| Spec | Value |
|---|---|
| Engine | LanceDB (open source) |
| Index type | IVF-PQ (when >= 256 rows) |
| Distance metric | Cosine |
| Vectors per image | 2 (image embedding + caption embedding) |

## Captioning Model Comparison: BLIP vs Gemini 3.1 Flash Lite

A comparative evaluation was conducted to assess whether replacing the local BLIP captioning model with Google's Gemini 3.1 Flash Lite (via Vertex AI) would improve search accuracy.

> **Note on corpus size:** This comparison was run against a **10-image index** — an earlier, smaller benchmark used specifically to isolate the captioning model variable. The main evaluation above uses 30 images. BLIP's Precision@1 of 80.0% here reflects the 10-image corpus; the same system scores 73.3% on the 30-image corpus, which is expected — accuracy decreases as the index grows and result competition increases.

### Results

| Metric | BLIP (10-image corpus) | Gemini 3.1 Flash Lite (10-image corpus) | Delta |
|---|---|---|---|
| Precision@1 | **80.0%** | 73.3% | -6.7pp |
| Precision@3 | **86.7%** | 86.7% | 0 |
| MRR | **0.84** | 0.81 | -0.03 |
| Avg Cosine Similarity | **0.79** | 0.37 | -0.42 |
| Caption time (10 images) | ~1.5s (local) | ~15s (API) | 10x slower |
| Cost per image | $0 (local) | ~$0.0001 | — |

### Analysis

Gemini produces significantly richer, more accurate captions (e.g., "a man wearing a hat" vs "An older Black man with a graying beard, wearing a dark blue t-shirt and a black Veteran baseball cap with gold embroidery"). However, **CLIP's text encoder was trained on short, simple captions**. Long, detailed Gemini captions produce diluted embeddings that reduce search precision for simple queries.

**Gemini improved** abstract/semantic queries (e.g., "wildlife in Africa" jumped from rank 6 to rank 1), but **regressed** on literal queries (e.g., "person cutting a cake" dropped from rank 1 to rank 6).

### Conclusion

BLIP remains the better choice for the current CLIP-based architecture. Upgrading the captioning model alone does not improve overall search accuracy due to the text encoder bottleneck. A captioning upgrade would need to be paired with a text encoder upgrade (e.g., switching from CLIP ViT-B/32 to a model trained on longer descriptions) to realize the benefit.
