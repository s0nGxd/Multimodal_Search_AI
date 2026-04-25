# Architecture Decision Records

This folder documents the major architectural decisions made during the SEGP Group 20 project. Each record captures the **context** that triggered the change, the **decision** we made, and the **consequences** of that change.

Decisions are numbered in the order they were made. Later decisions sometimes build on earlier ones — read in order to follow the evolution of the system.

The format is a lightweight ADR (Architecture Decision Record), based on the convention popularized by Michael Nygard in 2011. The intent is that a future contributor — or a marker — can understand not just *what* the system looks like today, but *why* it ended up that way.

## Index

| # | Decision | Trigger |
|---|---|---|
| [001](001-siglip-over-clip.md) | Use SigLIP instead of CLIP as the vision-language encoder | CLIP's softmax-trained embeddings underperformed on cross-modal cosine similarity for our dataset |
| [002](002-removing-blip-captions-and-bm25.md) | Remove BLIP image captions and BM25 keyword channel | Generated captions introduced more noise than signal on hard queries |
| [003](003-owl-vit-detection-rerank.md) | Add OWL-ViT detection as a second-stage re-ranker | Pure SigLIP scoring lost too much precision after dropping the caption channel |
| [004](004-gemini-parser-for-compositional-queries.md) | Add a Gemini-based query parser for compositional queries | A single embedding cannot represent boolean structure ("X or Y", "X and not Y") |
| [005](005-vlm-rerank-for-action-understanding.md) | Add Gemini Flash multimodal as a final VLM re-ranker | SigLIP + OWL-ViT understand objects but not actions — false positives on relational queries |

## How these documents complement the rest of the repo

- **The git log** shows *when* each change happened.
- **The code** shows *what* the system does today.
- **These ADRs** show *why* each change was made — the failure that triggered it and the tradeoff we accepted.
