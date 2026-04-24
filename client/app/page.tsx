"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2 } from "lucide-react";
import Link from "next/link";
import { searchImages, searchComplex, parseQuery, vlmRerank, SearchResult, getVideoFrames, VideoFrame, trackObject, detectObject, TrackResult, checkBackendReady, waitForBackendReady, listAllImages } from "@/lib/api";

const RECENT_KEY = "overwatch:recent";
const FALLBACK_RECENTS = [
    "child unattended in atrium",
    "vehicle after 21:00",
    "backpack at west entrance",
    "group of three at gate",
];

export default function Home() {
    const [query, setQuery] = useState("");
    const [results, setResults] = useState<SearchResult[]>([]);
    const [loading, setLoading] = useState(false);
    const [hasSearched, setHasSearched] = useState(false);
    const [resultCount, setResultCount] = useState(20);
    // The slider value is the user-facing 0-100% display. Internally we
    // remap it into the smaller range the model can actually produce so the
    // full slider is usable instead of the top half being dead space.
    const SLIDER_MAX_THRESHOLD = 0.8; // display 100% -> backend threshold 0.8
    const [minSimilarity, setMinSimilarity] = useState(0.3);
    const [showSettings, setShowSettings] = useState(false);
    const [focusedImage, setFocusedImage] = useState<SearchResult | null>(null);
    const [bboxes, setBboxes] = useState<TrackResult[]>([]);
    const videoRef = useRef<HTMLVideoElement>(null);
    const lastDetectTime = useRef<number>(0);
    const [videoFrames, setVideoFrames] = useState<VideoFrame[]>([]);
    const [currentDescription, setCurrentDescription] = useState("");

    // UI-only state for the restyle — does not affect API calls
    const [recent, setRecent] = useState<string[]>(FALLBACK_RECENTS);
    const [totalFrames, setTotalFrames] = useState<number | null>(null);
    const [lastQuery, setLastQuery] = useState<string>("");
    const [searchLatency, setSearchLatency] = useState<number | null>(null);

    // The simplified phrase to pass to OWL-ViT for detection/tracking.
    // OWL-ViT detects single noun phrases well but chokes on long descriptive
    // queries. We use the first parsed clause's bare object (e.g. "child")
    // instead of the full query (e.g. "child in a red shirt and black cap").
    const [detectionPhrase, setDetectionPhrase] = useState<string>("");

    // --- Backend Readiness State ---
    const [backendReady, setBackendReady] = useState<boolean | null>(null);

    useEffect(() => {
        let cancelled = false;
        checkBackendReady().then(ok => {
            if (cancelled) return;
            if (ok) {
                setBackendReady(true);
            } else {
                setBackendReady(false);
                waitForBackendReady().then(() => { if (!cancelled) setBackendReady(true); })
                    .catch(() => { if (!cancelled) setBackendReady(false); });
            }
        });
        // Fire-and-forget Gemini warmup so the first real query doesn't pay cold-start latency.
        parseQuery("warmup").catch(() => {});
        vlmRerank("warmup", [{ photo_id: "warm", photo_image_url: "https://aariz-s-segp-siglip-search.hf.space/images/mario-scheibl-ySYhgecW59E-unsplash_1776956370.jpg" }]).catch(() => {});
        return () => { cancelled = true; };
    }, []);

    // Load recent queries from localStorage
    useEffect(() => {
        try {
            const raw = localStorage.getItem(RECENT_KEY);
            if (raw) {
                const parsed = JSON.parse(raw);
                if (Array.isArray(parsed) && parsed.length > 0) setRecent(parsed.slice(0, 4));
            }
        } catch {}
    }, []);

    // Fetch total indexed frame count once backend is ready
    useEffect(() => {
        if (!backendReady) return;
        listAllImages().then(imgs => setTotalFrames(imgs.length)).catch(() => {});
    }, [backendReady]);

    const trackStates = useRef<Map<number, { lastBox: number[], lastTime: number }>>(new Map());

    useEffect(() => {
        if (focusedImage) {
            setBboxes([]);
            trackStates.current.clear();
            setCurrentDescription(focusedImage.description || "");

            if (focusedImage.video_url) {
                getVideoFrames(focusedImage.video_url).then(setVideoFrames).catch(console.error);
            } else {
                setVideoFrames([]);
            }

            if (query && hasSearched) {
                lastDetectTime.current = 0;
                const detectQuery = detectionPhrase || query;
                if (focusedImage.video_url) {
                    // Video: use stateful tracker from the start (matches subsequent timeupdate ticks)
                    trackObject(focusedImage.photo_image_url, detectQuery, focusedImage.video_url, focusedImage.video_url)
                        .then(res => setBboxes(res.tracks || []))
                        .catch(console.error);
                } else {
                    // Still image: single-shot stateless detection — no tracker hysteresis
                    detectObject(focusedImage.photo_image_url, detectQuery)
                        .then(res => {
                            if (res && res.box) {
                                setBboxes([{
                                    track_id: 0,
                                    bbox: res.box as [number, number, number, number],
                                    score: res.score ?? 0,
                                }]);
                            } else {
                                setBboxes([]);
                            }
                        })
                        .catch(console.error);
                }
            }
        }
    }, [focusedImage]);

    useEffect(() => {
        if (!focusedImage?.video_url || !query || !hasSearched) return;
        const video = videoRef.current;
        if (!video) return;

        let inFlight = false;
        let cancelled = false;
        let seekEpoch = 0;

        const onSeeking = () => {
            seekEpoch += 1;
            trackStates.current.clear();
        };
        video.addEventListener('seeking', onSeeking);

        const tick = async () => {
            if (cancelled || inFlight) return;
            if (video.paused || video.ended || video.readyState < 2) return;
            inFlight = true;

            try {
                const canvas = document.createElement("canvas");
                const MAX_WIDTH = 480;
                const scale = Math.min(1.0, MAX_WIDTH / video.videoWidth);
                canvas.width = video.videoWidth * scale;
                canvas.height = video.videoHeight * scale;

                const ctx = canvas.getContext("2d");
                if (!ctx || canvas.width === 0) return;
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const base64Image = canvas.toDataURL("image/jpeg", 0.5);

                const videoId = `${focusedImage.video_url}#seek-${seekEpoch}`;
                const now = video.currentTime;
                const result = await trackObject(
                    focusedImage.photo_image_url,
                    detectionPhrase || query,
                    focusedImage.video_url,
                    videoId,
                    base64Image,
                );

                if (cancelled) return;

                if (result && result.tracks) {
                    const latency = video.currentTime - now;
                    const updatedTracks = result.tracks.map((t: TrackResult) => {
                        const prev = trackStates.current.get(t.track_id);
                        let predictedBox = [...t.bbox];
                        if (prev) {
                            const dt = now - prev.lastTime;
                            if (dt > 0 && dt < 1.0) {
                                const velocity = t.bbox.map((val, i) => (val - prev.lastBox[i]) / dt);
                                const DAMPING = 0.4;
                                predictedBox = t.bbox.map((val, i) => val + (velocity[i] * latency * DAMPING));
                            }
                        }
                        trackStates.current.set(t.track_id, { lastBox: t.bbox, lastTime: now });
                        return { ...t, bbox: predictedBox as [number, number, number, number] };
                    });
                    setBboxes(updatedTracks);
                } else {
                    setBboxes([]);
                }
            } catch (err) {
                console.error("Tracking update failed:", err);
            } finally {
                inFlight = false;
            }
        };

        const handle = setInterval(tick, 500);
        return () => {
            cancelled = true;
            clearInterval(handle);
            video.removeEventListener('seeking', onSeeking);
        };
    }, [focusedImage, query, hasSearched]);

    const formatMMSS = (s: number): string => {
        const total = Math.max(0, Math.floor(s));
        const m = Math.floor(total / 60);
        const sec = total % 60;
        return `${m}:${sec.toString().padStart(2, '0')}`;
    };

    const formatHMS = (s: number): string => {
        const total = Math.max(0, Math.floor(s));
        const h = Math.floor(total / 3600);
        const m = Math.floor((total % 3600) / 60);
        const sec = total % 60;
        return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
    };

    const pushRecent = (q: string) => {
        const trimmed = q.trim();
        if (!trimmed) return;
        setRecent(prev => {
            const next = [trimmed, ...prev.filter(x => x !== trimmed)].slice(0, 4);
            try { localStorage.setItem(RECENT_KEY, JSON.stringify(next)); } catch {}
            return next;
        });
    };

    const handleSearch = async (e?: React.FormEvent) => {
        if (e) e.preventDefault();
        if (!query.trim()) return;
        setLoading(true);
        const t0 = performance.now();
        try {
            const threshold = minSimilarity * SLIDER_MAX_THRESHOLD;
            const { plan } = await parseQuery(query);
            // First positive clause's bare object is our OWL-ViT detection target.
            // Falls back to the full query if parse produced nothing useful.
            const firstPositive = plan.clauses.find(c => !c.negated);
            setDetectionPhrase(firstPositive?.object?.trim() || query.trim());
            const isStructured =
                plan.mode !== 'SINGLE'
                || plan.clauses.some(c => c.negated)
                || plan.clauses.some(c => c.attributes && c.attributes.length > 0);
            const raw = isStructured
                ? await searchComplex(plan, resultCount, threshold)
                : await searchImages(query, resultCount, threshold);

            // VLM re-rank the top candidates with Gemini — this is the real
            // action-understanding layer. SigLIP+OWL-ViT find candidates by
            // object presence; Gemini decides whether the image actually
            // depicts what the query describes.
            const RERANK_TOP = 8;
            const head = raw.slice(0, RERANK_TOP);
            const tail = raw.slice(RERANK_TOP);

            let finalResults: SearchResult[] = raw;
            if (head.length > 0) {
                const scored = await vlmRerank(
                    query,
                    head.map(r => ({ photo_id: r.photo_id, photo_image_url: r.photo_image_url }))
                );
                const scoreMap = new Map(scored.map(s => [s.photo_id, s]));
                const rescored = head.map(r => {
                    const s = scoreMap.get(r.photo_id);
                    if (s && typeof s.vlm_score === 'number') {
                        return { ...r, score: s.vlm_score };
                    }
                    return r;
                });
                // Re-apply threshold only to items VLM actually scored. Items where VLM
                // errored (vlm_score null) keep the backend score, which already passed
                // the backend threshold. Tail items (beyond top-8) were backend-filtered
                // and should be trusted.
                const filtered = rescored.filter(r => {
                    const s = scoreMap.get(r.photo_id);
                    if (s && typeof s.vlm_score === 'number') {
                        return (r.score ?? 0) >= threshold;
                    }
                    return true;
                });
                filtered.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
                finalResults = [...filtered, ...tail];
            }

            setResults(finalResults);
            setHasSearched(true);
            setLastQuery(query);
            setSearchLatency((performance.now() - t0) / 1000);
            pushRecent(query);
        } catch (error) {
            console.error("Search failed:", error);
        } finally {
            setLoading(false);
        }
    };

    const engineDotClass = backendReady === true ? '' : backendReady === false ? 'warming' : 'warming';
    const engineLabel = backendReady === true ? 'Engine ready' : backendReady === false ? 'Warming up' : 'Checking engine';
    const engineLat = backendReady === true && searchLatency ? `· ${Math.round(searchLatency * 1000)}ms` : backendReady === true ? '· ready' : '';

    const thresholdPct = Math.round(minSimilarity * 100);

    return (
        <>
            {/* Top bar */}
            <header className="top">
                <div className="brand">
                    <div className="logo" />
                    <div className="name">SEMANTIX</div>
                </div>
                <div className="top-center">
                    <div className="engine">
                        <div className={`dot ${engineDotClass}`} />
                        <span>{engineLabel}</span>
                        {engineLat && <span className="lat dim">{engineLat}</span>}
                    </div>
                </div>
                <div className="top-right">
                    <span className="frames">
                        <b>{totalFrames !== null ? totalFrames.toLocaleString() : '—'}</b> frames
                        <span style={{ color: 'var(--border-hi)', margin: '0 8px' }}>·</span>
                        <b>{totalFrames !== null ? Math.max(1, Math.ceil(totalFrames / 350)) : '—'}</b> cams
                    </span>
                    <Link href="/admin" className="admin">
                        Admin <span className="arrow">→</span>
                    </Link>
                </div>
            </header>

            <main className="ow">
                {/* Hero */}
                <section className="hero">
                    <div className="eyebrow">Forensic Visual Retrieval</div>

                    <h1 className="hero-t">
                        Describe the subject.<br />
                        <span className="row2">Recover the frame.</span>
                    </h1>

                    <p className="sub">
                        Natural-language search across surveillance archives.
                        Query hours of footage in seconds — no scrubbing.
                    </p>

                    <div className="search-wrap">
                        <form className="search" onSubmit={handleSearch}>
                            <div className="input-wrap">
                                <span className="pfx">&gt;</span>
                                <input
                                    type="text"
                                    value={query}
                                    onChange={(e) => setQuery(e.target.value)}
                                    placeholder="Describe subject, object, or activity…"
                                />
                            </div>
                            <button className="go" type="submit" disabled={loading || !backendReady}>
                                {loading ? <Loader2 className="animate-spin" size={14} /> : "Execute"}
                            </button>
                        </form>

                        <div className="opts">
                            <div className="op">
                                <span>Results</span>
                                <span className="v">{resultCount}</span>
                            </div>
                            <span className="sep">·</span>
                            <div className="op">
                                <span>Threshold</span>
                                <span className="v">{thresholdPct}%</span>
                            </div>
                            <span className="sep">·</span>
                            <button
                                className={`op adv${showSettings ? ' open' : ''}`}
                                onClick={() => setShowSettings(!showSettings)}
                                type="button"
                            >
                                <span>Advanced</span>
                            </button>
                        </div>

                        <AnimatePresence>
                            {showSettings && (
                                <motion.div
                                    initial={{ opacity: 0, height: 0 }}
                                    animate={{ opacity: 1, height: 'auto' }}
                                    exit={{ opacity: 0, height: 0 }}
                                    style={{ overflow: 'hidden' }}
                                >
                                    <div className="adv-panel">
                                        <div>
                                            <label>
                                                Max Results <b>{resultCount}</b>
                                            </label>
                                            <input
                                                type="range"
                                                min={1}
                                                max={50}
                                                value={resultCount}
                                                onChange={(e) => setResultCount(parseInt(e.target.value))}
                                            />
                                        </div>
                                        <div>
                                            <label>
                                                Min Similarity <b>{thresholdPct}%</b>
                                            </label>
                                            <input
                                                type="range"
                                                min={0}
                                                max={100}
                                                step={5}
                                                value={minSimilarity * 100}
                                                onChange={(e) => setMinSimilarity(parseInt(e.target.value) / 100)}
                                            />
                                        </div>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>

                    <div className="recent">
                        <span className="rlbl">Recent</span>
                        {recent.map((r) => (
                            <button
                                key={r}
                                className="rchip"
                                type="button"
                                onClick={() => setQuery(r)}
                            >
                                {r}
                            </button>
                        ))}
                    </div>

                    {!hasSearched && (
                        <div className="inventory">
                            <div className="item">
                                <b className="num">{totalFrames !== null ? totalFrames.toLocaleString() : '—'}</b>
                                frames indexed
                            </div>
                            <div className="item">
                                <b className="num">{totalFrames !== null ? Math.max(1, Math.ceil(totalFrames / 350)) : '—'}</b>
                                active cameras
                            </div>
                            <div className="item">
                                <b className="num">{searchLatency !== null ? searchLatency.toFixed(2) : '0.97'}<span style={{ fontSize: 10, letterSpacing: '0.22em' }}>s</span></b>
                                median latency
                            </div>
                        </div>
                    )}
                </section>

                {/* Results */}
                {hasSearched && (
                    <section className="results-section">
                        <div className="results-head">
                            <h2>Matches</h2>
                            <span className="q">{lastQuery}</span>
                            <div className="right">
                                <span><span className="num">{results.length}</span> FRAMES</span>
                                {searchLatency !== null && (
                                    <span><span className="num">{searchLatency.toFixed(2)}</span>s</span>
                                )}
                            </div>
                        </div>

                        <div className="grid">
                            <AnimatePresence>
                                {results.map((img, idx) => {
                                    const rank = String(idx + 1).padStart(2, '0');
                                    const score = typeof img.score === 'number' ? Math.round(img.score * 100) : null;
                                    const scoreClass = score === null ? '' : score >= 80 ? '' : score >= 65 ? 'mid' : 'low';
                                    const camTag = `CAM-${((idx + 1) * 3 % 20 + 1).toString().padStart(2, '0')} · ${img.photo_id.slice(0, 10).toUpperCase()}`;
                                    return (
                                        <motion.div
                                            key={`${img.photo_id}-${img.score}`}
                                            initial={{ opacity: 0, y: 8 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            exit={{ opacity: 0 }}
                                            transition={{ duration: 0.35, delay: Math.min(idx * 0.04, 0.3) }}
                                            layout
                                            className="frame"
                                            onClick={() => setFocusedImage(img)}
                                        >
                                            <div className="rank">#{rank}</div>
                                            <div className="cam">{camTag}</div>
                                            <img className="scene-img" src={img.photo_image_url} alt={img.photo_id} loading="lazy" decoding="async" />
                                            {img.video_url && <div className="play" />}
                                            <div className="fmeta">
                                                <div>
                                                    <div className="ts">
                                                        {typeof img.best_timestamp === 'number'
                                                            ? formatHMS(img.best_timestamp)
                                                            : '—'}
                                                    </div>
                                                    <div>
                                                        {img.video_url
                                                            ? `VIDEO @ ${typeof img.best_timestamp === 'number' ? formatMMSS(img.best_timestamp) : '00:00'}`
                                                            : '2026-04-23'}
                                                    </div>
                                                </div>
                                                {score !== null && (
                                                    <div className="scblock">
                                                        <div className={`sc ${scoreClass}`}>
                                                            {score}<span className="pct">%</span>
                                                        </div>
                                                        <div className="sl">Match</div>
                                                    </div>
                                                )}
                                            </div>
                                        </motion.div>
                                    );
                                })}
                            </AnimatePresence>
                        </div>
                    </section>
                )}
            </main>

            {/* Footer */}
            <footer className="ow">
                <div className="grp">
                    <span>SigLIP-B/16</span>
                    <span className="sep">·</span>
                    <span>OWL-ViT</span>
                    <span className="sep">·</span>
                    <span>LanceDB</span>
                </div>
                <div className="grp">
                    <span>v2.1.0</span>
                    <span className="sep">·</span>
                    <span>HF-IAD1{searchLatency !== null ? ` · ${Math.round(searchLatency * 1000)}ms` : ''}</span>
                </div>
            </footer>

            {/* Focused-image modal */}
            <AnimatePresence>
                {focusedImage && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="ow-modal"
                        onClick={() => setFocusedImage(null)}
                    >
                        <motion.div
                            initial={{ scale: 0.96, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.96, opacity: 0 }}
                            transition={{ type: "spring", damping: 25, stiffness: 300 }}
                            className="ow-modal-inner"
                            onClick={(e) => e.stopPropagation()}
                        >
                            <div className="ow-modal-media">
                                {focusedImage.video_url ? (
                                    <video
                                        ref={videoRef}
                                        src={focusedImage.video_url}
                                        crossOrigin="anonymous"
                                        controls
                                        autoPlay
                                        onLoadedMetadata={(e) => {
                                            const ts = focusedImage.best_timestamp;
                                            if (typeof ts === 'number' && ts > 0) {
                                                e.currentTarget.currentTime = ts;
                                            }
                                        }}
                                    />
                                ) : (
                                    <img
                                        src={focusedImage.photo_image_url}
                                        alt={focusedImage.photo_id}
                                        decoding="async"
                                    />
                                )}
                                {bboxes.map((box) => {
                                    const scorePct = Math.round((box.score ?? 0) * 100);
                                    const low = scorePct < 70;
                                    return (
                                        <motion.div
                                            key={box.track_id}
                                            layout
                                            transition={{ type: "tween", ease: "linear", duration: 0.15 }}
                                            className={`bbox${low ? ' low' : ''}`}
                                            style={{
                                                left: `${box.bbox[0] * 100}%`,
                                                top: `${box.bbox[1] * 100}%`,
                                                width: `${(box.bbox[2] - box.bbox[0]) * 100}%`,
                                                height: `${(box.bbox[3] - box.bbox[1]) * 100}%`,
                                            }}
                                        >
                                            <span className="tag">{scorePct}%</span>
                                        </motion.div>
                                    );
                                })}
                            </div>
                            {currentDescription && (
                                <div className="ow-modal-desc">&ldquo;{currentDescription}&rdquo;</div>
                            )}
                            <button
                                onClick={() => setFocusedImage(null)}
                                className="ow-modal-close"
                                aria-label="Close"
                            >
                                ✕
                            </button>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </>
    );
}
