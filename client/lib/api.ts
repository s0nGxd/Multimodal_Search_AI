const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';
const HEALTH_URL = API_BASE.replace('/api', '/health');

export interface IngestResult {
    id: string;
    url: string;
    description: string;
    status: string;
}

export interface SearchResult {
    photo_id: string;
    photo_image_url: string;
    video_url?: string;
    timestamp?: number;
    best_timestamp?: number;
    description?: string;
    score?: number;
}

export interface DetectResult {
    box: [number, number, number, number] | null;
    score: number | null;
}

export interface TrackResult {
    track_id: number;
    bbox: [number, number, number, number];
    score: number;
}

export interface VideoFrame {
    timestamp: number;
    description: string;
}

export async function checkBackendHealth(): Promise<boolean> {
    try {
        const res = await fetch(HEALTH_URL, { signal: AbortSignal.timeout(5000) });
        if (!res.ok) return false;
        const data = await res.json();
        return data?.status === 'ok';
    } catch {
        return false;
    }
}

export async function waitForBackend(onAttempt?: (attempt: number) => void): Promise<void> {
    for (let i = 1; i <= 30; i++) {
        onAttempt?.(i);
        const healthy = await checkBackendHealth();
        if (healthy) return;
        await new Promise(r => setTimeout(r, 3000));
    }
    throw new Error('Backend did not respond after 90 seconds');
}

export async function searchImages(query: string, k: number = 20, threshold: number = 0.20): Promise<SearchResult[]> {
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

export async function getVideoFrames(videoUrl: string): Promise<VideoFrame[]> {
    const res = await fetch(`${API_BASE}/video/frames?url=${encodeURIComponent(videoUrl)}`);
    if (!res.ok) {
        throw new Error("Failed to fetch video frames");
    }
    return res.json();
}

export async function listAllImages(): Promise<SearchResult[]> {
    const res = await fetch(`${API_BASE}/images/all`);
    if (!res.ok) throw new Error('Failed to fetch images');
    return res.json();
}

export async function deleteImage(photo_id: string) {
    const res = await fetch(`${API_BASE}/images/${photo_id}`, {
        method: 'DELETE',
    });
    if (!res.ok) throw new Error('Delete failed');
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

export async function detectObject(photo_image_url: string, query: string, base64_image?: string): Promise<DetectResult> {
    const res = await fetch(`${API_BASE}/detect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ photo_image_url, query, base64_image })
    });
    if (!res.ok) throw new Error('Detection failed');
    return res.json();
}

export async function trackObject(
    photo_image_url: string,
    query: string,
    video_url?: string,
    video_id?: string,
    base64_image?: string
): Promise<{ tracks: TrackResult[] }> {
    const res = await fetch(`${API_BASE}/track`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ photo_image_url, video_url, query, video_id, base64_image })
    });
    if (!res.ok) throw new Error('Tracking failed');
    return res.json();
}
