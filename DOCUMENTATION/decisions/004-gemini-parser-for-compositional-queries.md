# 004 — Add a Gemini-based query parser for compositional queries

## Context

Surveillance investigators often want to combine concepts in a single query:

- "dog **or** cat" — disjunction
- "woman **and** child" — conjunction
- "person **not** wearing a hat" — negation

A single SigLIP embedding cannot represent boolean structure. Embedding "dog or cat" produces a vector in some interpolated region of feature space — close to neither "dog" nor "cat" specifically — and the search returns nothing satisfying. The system was effectively limited to single-concept queries.

## Decision

Added a query-parsing layer powered by Google's Gemini Flash, called via Vertex AI from a Next.js API route (`client/app/api/parse-query/route.ts`). The parser converts the natural-language query into a structured plan:

```json
{
  "mode": "OR" | "AND" | "SINGLE",
  "clauses": [
    {"object": "dog", "attributes": [], "negated": false},
    {"object": "cat", "attributes": [], "negated": false}
  ]
}
```

A new backend endpoint `/api/search/complex` consumes this plan and executes one search per clause in parallel, then combines results via set operations:

- **OR** → union, score = max across clauses
- **AND** → intersection, score = min across clauses
- **Negated** clauses → subtract their frame set from the final result

The parser uses structured output (Zod schema) so the response is always a valid plan. If the Gemini call times out or fails, the system falls back to SINGLE mode with the raw query — degraded but functional.

## Consequences

- Compositional queries now work end-to-end. "Dog or cat" returns the union; "person and chair" returns the intersection.
- Adds ~300ms-1s of latency per query (Gemini call). Mitigated by warming the Gemini connection on page mount.
- Introduces an external dependency on Google Vertex AI — credential management (`GOOGLE_PROJECT_ID`, `GOOGLE_CLIENT_EMAIL`, `GOOGLE_PRIVATE_KEY`) is now part of the deploy story.
- Parser-prompt tuning is non-trivial. Several iterations were needed to find the right balance between aggressive splitting (broke action queries like "guy getting in a car") and conservative splitting (didn't help compositional queries). Snapshot tests are listed as future work in `PLAN/low_level.md`.
- The parser also produces an `attributes` array per clause, which downstream code uses to extract a simplified detection phrase for OWL-ViT (per [ADR 003](003-owl-vit-detection-rerank.md)).
