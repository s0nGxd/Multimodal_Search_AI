# Follow-Up Questions for Next Meeting

These are NOT shared with the team. They have their 10 prep questions. These are the deeper questions you will ask to verify they actually understood the system, not just memorised answers.

---

## System 1 Questions (Karl & Song) — 25 Questions

### CLIP Deep Dive
1. CLIP was trained on 400 million image-text pairs from the internet. How does that training data affect what kinds of queries work well and what kinds fail?
2. What does "contrastive" mean in "Contrastive Language-Image Pre-training"? What is the model being contrasted against?
3. If I wanted to switch from `clip-vit-base-patch32` to `clip-vit-large-patch14`, what would change in the system? What would break?
4. Why is the embedding dimension 512 and not 256 or 1024? Can we change it?
5. The processor does tokenisation for text and pixel normalisation for images. What happens if you skip the processor and feed raw pixels to the model?
6. What is the difference between `model.get_image_features()` and just running `model(images=...)`?
7. Why do we convert the output to `float32` specifically? What would happen with `float16`?
8. If someone searches "a photo of a cat" vs just "cat" — which gets better results and why?

### BLIP Deep Dive
9. BLIP uses autoregressive generation. What does that mean? How is each word of the caption produced?
10. What is the maximum caption length BLIP can generate? What happens if the image is very complex?
11. We found that BLIP sometimes produces repetition loops like "sari sari sari sari." Why does this happen in autoregressive models?
12. If BLIP generates a bad caption, how does that affect search results? Give a specific example.
13. Why can't we just use BLIP for everything — both captioning AND embedding? Why do we still need CLIP?
14. The BLIP model is ~990MB. Where does it get stored when it downloads? What happens on first run vs subsequent runs?
15. If the server crashes while BLIP is generating a caption, what happens to the image upload?

### Ingestion Pipeline
16. Why does the pipeline go CLIP → BLIP → CLIP (three steps) instead of CLIP → BLIP (two steps)?
17. What exactly is in a "zero vector" and why does hybrid search ignore it? Prove it mathematically.
18. If bulk ingestion fails halfway through (say, 50 out of 100 images downloaded), what happens to the 50 that succeeded? Are they in the database?
19. Why is `ThreadPoolExecutor` used instead of `asyncio` for downloading? When would asyncio be better?
20. The batch size for CLIP is 16. What happens if you set it to 1? What happens if you set it to 1000?
21. Why do we call `sync_to_repo()` after every single upload but also after bulk ingestion? Isn't that redundant for bulk?
22. What would happen if two users uploaded images at the exact same time? Is there a race condition?

### The Gemini Comparison
23. We tested Gemini 3.1 Flash Lite as a replacement for BLIP. Gemini captions were better but search accuracy dropped. Explain WHY in technical terms.
24. What would we need to change in the system to make Gemini captions work well? (Hint: it is not just a caption model swap.)
25. The Gemini API added 50x latency to ingestion. Name two architectural approaches that could mitigate this without switching back to BLIP.

---

## System 2 Questions (Bessie & Zheng) — 25 Questions

### LanceDB Deep Dive
1. LanceDB is "embedded." What does that mean compared to PostgreSQL or MongoDB? What are the advantages and disadvantages?
2. The database is stored as files on disk. What file format does LanceDB use internally? (Hint: it is in the name.)
3. What happens if the LanceDB files get corrupted? How would you detect this? How would you recover?
4. Why do we call `lancedb.connect()` in `_insert_records` instead of keeping a persistent connection?
5. What is the difference between `db.create_table()` and `table.add()`? When is each called?
6. If the "images" table has 10,000 rows, approximately how much disk space does it use? (512 floats × 4 bytes × 2 vectors × 10,000 rows = ?)
7. Could we use SQLite instead of LanceDB? What would we lose?

### Cosine Distance
8. Two normalised vectors have a dot product of 0.8. What is their cosine distance? What would the rescaled score be?
9. What is the cosine distance between a vector and itself? Between a vector and its negative?
10. Why does L2 normalisation make cosine distance equivalent to the dot product? Prove it.
11. If a vector is NOT normalised, does our search still work? What breaks?
12. Euclidean distance and cosine distance rank results differently. Give an example where they disagree and explain which is correct for our use case.

### Hybrid Search
13. If Search 1 returns images [A, B, C] and Search 2 returns images [B, D, E], what does the merged result look like? Walk through the merge logic step by step.
14. Could we do a weighted average of the two distances instead of taking the minimum? What are the pros and cons?
15. What happens if the same image appears in both search results but with very different distances? Which one wins and why?
16. The threshold default is 0.9 (cosine distance). What percentage of the similarity range does this let through? Is it strict or permissive?
17. If all images have zero vectors for `caption_vector` (no captions), does hybrid search degrade gracefully? What does Search 2 return?
18. Why do we sort by `_distance` ascending and not descending?
19. The merge loop uses `df.iterrows()` on a pandas DataFrame. Is this efficient? What would be faster for 100,000 results?

### Score Rescaling
20. The floor changed from 0.10 to 0.40 when we added hybrid search. Explain why the floor had to increase.
21. If we set CLIP_SIM_FLOOR to 0.00 and CLIP_SIM_CEIL to 1.00 (the full range), what would happen to the displayed scores?
22. A user sets the similarity slider to 75%. Walk me through the exact math: what distance threshold gets sent to the backend?
23. The rescaling is linear. Could a non-linear rescaling (e.g., logarithmic) produce better-looking scores? Why or why not?

### IVF-PQ Index
24. After building an IVF-PQ index, is the search result guaranteed to be the same as brute-force? Why or why not?
25. We use 16 sub-vectors for a 512-dim space. That means each sub-vector covers 32 dimensions. If we used 32 sub-vectors (16 dims each), search would be faster but less accurate. Explain why.
