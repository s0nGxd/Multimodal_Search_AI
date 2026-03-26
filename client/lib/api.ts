const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

export interface IngestResult {
    id: string;
    url: string;
    description: string;
    status: string;
}

export interface SearchResult {
    photo_id: string;
    photo_image_url: string;
    description?: string;
    score?: number;
}

export async function searchImages(query: string, k: number = 20, threshold: number = 0.9): Promise<SearchResult[]> {
    const res = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, k, threshold }),
    });
    if (!res.ok) throw new Error('Search failed');
    return res.json();
}

export async function uploadImage(file: File) {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
    });

    if (!res.ok) {
        const errorText = await res.text();
        throw new Error(errorText || 'Upload failed');
    }

    return res.json();
}

export async function ingestViaUrl(url: string, photo_id?: string) {
    const res = await fetch(`${API_BASE}/ingest/url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, photo_id }),
    });
    if (!res.ok) {
        const errorText = await res.text();
        throw new Error(errorText || 'URL Ingestion failed');
    }
    return res.json();
}

export async function bulkIngest(limit: number = 50) {
    const res = await fetch(`${API_BASE}/ingest/bulk`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ limit }),
    });

    if (!res.ok) {
        const errorText = await res.text();
        throw new Error(errorText || 'Bulk ingestion failed');
    }

    return res.json();
}
