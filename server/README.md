---
title: SEGP Semantic Search API
emoji: 🔍
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
---

# SEGP Semantic Search Engine — Backend API

A multimodal semantic image search engine powered by **CLIP** and **LanceDB**, with auto-captioning via **BLIP**.

## Features
- **Text-to-Image Search**: Natural language queries matched against image embeddings
- **Image Upload & Indexing**: Real-time online ingestion via file upload or URL
- **Auto-Captioning**: BLIP generates descriptions for uploaded images
- **Vector Database**: LanceDB stores and queries 512-dimensional CLIP embeddings

## API Endpoints
- `POST /api/search` — Search images by text query
- `POST /api/upload` — Upload and index a new image
- `POST /api/ingest/url` — Index an image from a URL
- `POST /api/ingest/bulk` — Bulk index from CSV
- `GET /health` — Health check
