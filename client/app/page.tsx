"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Search, Image as ImageIcon, Sparkles, Loader2, ArrowRight, Play } from "lucide-react";
import Link from "next/link";
import { searchImages, SearchResult, getVideoFrames, VideoFrame, trackObject, TrackResult, preloadVideoTracks, PreloadKeyframe, checkBackendHealth, waitForBackend } from "@/lib/api";

export default function Home() {
    const [query, setQuery] = useState("");
    const [results, setResults] = useState<SearchResult[]>([]);
    const [loading, setLoading] = useState(false);
    const [hasSearched, setHasSearched] = useState(false);
    const [resultCount, setResultCount] = useState(20);
    const [minSimilarity, setMinSimilarity] = useState(0.8);
    const [deepSearch, setDeepSearch] = useState(false);
    const [showSettings, setShowSettings] = useState(false);
    const [focusedImage, setFocusedImage] = useState<SearchResult | null>(null);
    const [bboxes, setBboxes] = useState<TrackResult[]>([]);
    const [bboxLoading, setBboxLoading] = useState(false);
    const videoRef = useRef<HTMLVideoElement>(null);
    const [videoFrames, setVideoFrames] = useState<VideoFrame[]>([]);
    const [currentDescription, setCurrentDescription] = useState("");

    // --- Video preload tracking state ---
    const [videoPreloading, setVideoPreloading] = useState(false);
    const preloadedKeyframes = useRef<PreloadKeyframe[]>([]);

    // --- Backend Readiness State ---
    const [backendReady, setBackendReady] = useState<boolean | null>(null);

    useEffect(() => {
        let cancelled = false;
        checkBackendHealth().then(ok => {
            if (cancelled) return;
            if (ok) {
                setBackendReady(true);
            } else {
                setBackendReady(false);
                waitForBackend().then(() => { if (!cancelled) setBackendReady(true); })
                    .catch(() => { if (!cancelled) setBackendReady(false); });
            }
        });
        return () => { cancelled = true; };
    }, []);

    useEffect(() => {
        const video = videoRef.current;
        if (video) {
            // Pause while preloading, play once preloaded
            if (videoPreloading) {
                video.pause();
            } else if (focusedImage?.video_url) {
                video.play().catch(e => console.log("Playback prevented:", e));
            }
        }
    }, [videoPreloading, focusedImage]);

    // ── When user clicks an image/video result ──────────────────────────
    useEffect(() => {
        if (focusedImage) {
            setBboxes([]);
            preloadedKeyframes.current = [];
            setCurrentDescription(focusedImage.description || "");

            if (focusedImage.video_url) {
                // Fetch frame descriptions for the timeline
                getVideoFrames(focusedImage.video_url).then(setVideoFrames).catch(console.error);

                // ── PRELOAD tracking: run GDINO on all keyframes upfront ──
                if (query && hasSearched) {
                    setVideoPreloading(true);
                    preloadVideoTracks(focusedImage.video_url, query)
                        .then(res => {
                            preloadedKeyframes.current = res.keyframes;
                            setVideoPreloading(false);
                            console.log(`[Preload] Received ${res.keyframes.length} keyframes for ${res.duration}s video`);

                            // Show initial bbox from first keyframe that has detections
                            const firstHit = res.keyframes.find(kf => kf.tracks.length > 0);
                            if (firstHit) {
                                setBboxes(firstHit.tracks);
                            }
                        })
                        .catch(err => {
                            console.error("Video preload failed:", err);
                            setVideoPreloading(false);
                        });
                }
            } else {
                setVideoFrames([]);

                // Image bbox detection (unchanged)
                if (query && hasSearched) {
                    if (focusedImage.florence_box) {
                        const [x1, y1, x2, y2] = focusedImage.florence_box;
                        setBboxes([{ track_id: 0, bbox: [x1, y1, x2, y2], score: focusedImage.score ?? 1 }]);
                    } else {
                        setBboxLoading(true);
                        trackObject(focusedImage.photo_image_url, query, undefined, focusedImage.photo_image_url)
                            .then(res => {
                                setBboxes(res.tracks || []);
                                setBboxLoading(false);
                            })
                            .catch(err => {
                                console.error(err);
                                setBboxLoading(false);
                            });
                    }
                }
            }
        }
    }, [focusedImage]);

    // ── Video time update: interpolate between preloaded keyframes ───────
    // No network calls! Pure client-side interpolation for smooth tracking.
    const handleTimeUpdate = () => {
        const video = videoRef.current;
        if (!video || !focusedImage) return;

        const now = video.currentTime;
        const kfs = preloadedKeyframes.current;

        // ── Interpolate bounding boxes from preloaded keyframes ──────
        if (kfs.length > 0) {
            // Binary search for the surrounding keyframes
            let lo = 0, hi = kfs.length - 1;
            while (lo < hi - 1) {
                const mid = Math.floor((lo + hi) / 2);
                if (kfs[mid].time <= now) lo = mid;
                else hi = mid;
            }

            const prev = kfs[lo];
            const next = kfs[hi];

            if (prev.tracks.length > 0 && next.tracks.length > 0 && prev.time !== next.time) {
                // Interpolation factor (0..1 between prev and next keyframe)
                const t = Math.max(0, Math.min(1, (now - prev.time) / (next.time - prev.time)));

                const interpolated: TrackResult[] = prev.tracks.map((prevTrack, i) => {
                    // Find matching track in next keyframe by track_id
                    const nextTrack = next.tracks.find(nt => nt.track_id === prevTrack.track_id)
                        || next.tracks[i]
                        || prevTrack;

                    const bbox: [number, number, number, number] = [
                        prevTrack.bbox[0] + (nextTrack.bbox[0] - prevTrack.bbox[0]) * t,
                        prevTrack.bbox[1] + (nextTrack.bbox[1] - prevTrack.bbox[1]) * t,
                        prevTrack.bbox[2] + (nextTrack.bbox[2] - prevTrack.bbox[2]) * t,
                        prevTrack.bbox[3] + (nextTrack.bbox[3] - prevTrack.bbox[3]) * t,
                    ];

                    return { ...prevTrack, bbox };
                });

                setBboxes(interpolated);
            } else if (prev.tracks.length > 0) {
                setBboxes(prev.tracks);
            } else if (next.tracks.length > 0) {
                setBboxes(next.tracks);
            } else {
                setBboxes([]);
            }
        }

        // ── Update frame description ─────────────────────────────────
        if (videoFrames.length > 0) {
            let closest = videoFrames[0];
            for (const f of videoFrames) {
                if (f.timestamp <= now) closest = f;
                else break;
            }
            if (closest && closest.description !== currentDescription) {
                setCurrentDescription(closest.description);
            }
        }
    };

    const handleSearch = async (e?: React.FormEvent) => {
        if (e) e.preventDefault();
        if (!query.trim()) return;
        setLoading(true);
        try {
            const data = await searchImages(query, resultCount, minSimilarity, deepSearch);
            setResults(data);
            setHasSearched(true);
        } catch (error) {
            console.error("Search failed:", error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <main className="min-h-screen bg-black text-white selection:bg-purple-500/30">
            <div className="fixed inset-0 overflow-hidden pointer-events-none -z-10">
                <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-purple-900/10 rounded-full blur-[120px]" />
                <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-900/10 rounded-full blur-[120px]" />
            </div>

            <div className="absolute top-6 right-6">
                <Link
                    href="/admin"
                    className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 text-xs font-medium hover:bg-white/10 transition-all text-gray-400 hover:text-white"
                >
                    Admin Access
                    <ArrowRight className="w-3 h-3" />
                </Link>
            </div>

            <div className={`transition-all duration-700 ease-in-out ${hasSearched ? 'pt-12' : 'pt-[25vh]'}`}>
                <div className="max-w-4xl mx-auto px-6 text-center">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="mb-8"
                    >
                        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-400 text-xs font-medium mb-6">
                            <Sparkles className="w-3 h-3" />
                            Next-Gen Semantic Vision
                        </div>

                        <div className="mb-6 flex justify-center">
                            {backendReady === null && (
                                <motion.div
                                    initial={{ opacity: 0, y: -10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-gray-500/10 border border-gray-500/20 text-gray-400 text-xs font-medium"
                                >
                                    <Loader2 className="w-3 h-3 animate-spin" />
                                    Checking engine status...
                                </motion.div>
                            )}
                            {backendReady === false && (
                                <motion.div
                                    initial={{ opacity: 0, y: -10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-yellow-500/10 border border-yellow-500/20 text-yellow-400 text-xs font-medium"
                                >
                                    <Loader2 className="w-3 h-3 animate-spin" />
                                    Waking up search engine — this takes about 30 seconds...
                                </motion.div>
                            )}
                            {backendReady === true && (
                                <motion.div
                                    initial={{ opacity: 0, y: -10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-green-500/10 border border-green-500/20 text-green-400 text-xs font-medium"
                                >
                                    <span className="w-2 h-2 rounded-full bg-green-400" />
                                    Search engine ready
                                </motion.div>
                            )}
                        </div>

                        <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-4 pb-2 bg-clip-text text-transparent bg-gradient-to-b from-white to-gray-500 leading-tight">
                            Find anything <br /> in your images.
                        </h1>
                        <p className="text-gray-400 text-lg max-w-xl mx-auto">
                            Natural language search powered by SigLIP & LanceDB.
                            Search by concepts, emotions, or objects.
                        </p>
                    </motion.div>

                    <div className="relative max-w-2xl mx-auto group">
                        <form onSubmit={handleSearch} className="relative z-10">
                            <div className="absolute inset-0 bg-gradient-to-r from-purple-600/20 to-blue-600/20 rounded-2xl blur-xl opacity-0 group-focus-within:opacity-100 transition-opacity" />
                            <div className="relative flex items-center bg-white/5 border border-white/10 rounded-2xl backdrop-blur-xl transition-all">
                                <Search className="ml-4 w-5 h-5 text-gray-500" />
                                <input
                                    type="text"
                                    value={query}
                                    onChange={(e) => setQuery(e.target.value)}
                                    placeholder="Describe an image... (e.g. 'a person smiling' or 'sunset over mountains')"
                                    className="w-full bg-transparent border-none focus:ring-0 focus:outline-none py-5 px-4 text-white placeholder-gray-600"
                                />
                                <button
                                    type="submit"
                                    disabled={loading || !backendReady}
                                    className="mr-2 px-6 py-2.5 bg-white text-black font-bold rounded-xl hover:bg-gray-200 transition-all active:scale-95 disabled:opacity-50 whitespace-nowrap"
                                >
                                    {loading
                                        ? deepSearch
                                            ? <span className="flex items-center gap-2"><Loader2 className="animate-spin w-4 h-4" />Deep Analyzing…</span>
                                            : <Loader2 className="animate-spin w-5 h-5" />
                                        : "Search"
                                    }
                                </button>
                            </div>
                        </form>

                        <div className="mt-4 flex justify-center">
                            <button
                                onClick={() => setShowSettings(!showSettings)}
                                className="text-xs text-gray-500 hover:text-white flex items-center gap-1 transition-colors"
                            >
                                {showSettings ? "Hide Options" : "Advanced Search Options"}
                                <span className={`transform transition-transform ${showSettings ? 'rotate-180' : ''}`}>▼</span>
                            </button>
                        </div>

                        <AnimatePresence>
                            {showSettings && (
                                <motion.div
                                    initial={{ opacity: 0, height: 0 }}
                                    animate={{ opacity: 1, height: 'auto' }}
                                    exit={{ opacity: 0, height: 0 }}
                                    className="overflow-hidden"
                                >
                                    <div className="bg-white/5 border border-white/5 rounded-xl p-4 mt-2 grid grid-cols-2 gap-6 text-left">
                                        <div>
                                            <label className="block text-xs text-gray-400 mb-2 font-medium">
                                                Max Results: <span className="text-white">{resultCount}</span>
                                            </label>
                                            <input
                                                type="range"
                                                min="1"
                                                max="50"
                                                value={resultCount}
                                                onChange={(e) => setResultCount(parseInt(e.target.value))}
                                                className="w-full h-1.5 bg-white/10 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white"
                                            />
                                            <div className="flex justify-between text-[10px] text-gray-600 mt-1">
                                                <span>1</span>
                                                <span>50</span>
                                            </div>
                                        </div>
                                        <div>
                                            <label className="block text-xs text-gray-400 mb-2 font-medium">
                                                Min Similarity: <span className="text-white">{Math.round(minSimilarity * 100)}%</span>
                                            </label>
                                            <input
                                                type="range"
                                                min="0"
                                                max="100"
                                                step="5"
                                                value={minSimilarity * 100}
                                                onChange={(e) => setMinSimilarity(parseInt(e.target.value) / 100)}
                                                className="w-full h-1.5 bg-white/10 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white"
                                            />
                                            <div className="flex justify-between text-[10px] text-gray-600 mt-1">
                                                <span>0%</span>
                                                <span>Strict</span>
                                            </div>
                                        </div>
                                        {/* Deep Search toggle — spans full width */}
                                        <div className="col-span-2 flex items-start gap-3 pt-2 border-t border-white/5">
                                            <button
                                                id="deep-search-toggle"
                                                type="button"
                                                onClick={() => setDeepSearch(v => !v)}
                                                className={`relative inline-flex h-5 w-9 flex-shrink-0 cursor-pointer rounded-full border-2 transition-colors duration-200 ${
                                                    deepSearch ? 'bg-purple-500 border-purple-500' : 'bg-white/10 border-white/20'
                                                }`}
                                            >
                                                <span className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform duration-200 ${
                                                    deepSearch ? 'translate-x-4' : 'translate-x-0'
                                                }`} />
                                            </button>
                                            <div>
                                                <p className="text-xs text-white font-medium flex items-center gap-1.5">
                                                    <Sparkles className="w-3 h-3 text-purple-400" />
                                                    Deep Search
                                                    <span className="text-[10px] bg-purple-500/20 text-purple-300 px-1.5 py-0.5 rounded-full border border-purple-500/30">~30s</span>
                                                </p>
                                                <p className="text-[10px] text-gray-500 mt-0.5">
                                                    Uses Florence-2 to verify adjectives, actions &amp; object identity on top results.
                                                    Results with a verified box are shown as ✦ AI Verified.
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                </div>

                <div className="max-w-7xl mx-auto px-6 mt-20 pb-20">
                    <div className="columns-1 sm:columns-2 md:columns-3 lg:columns-4 gap-6 space-y-6">
                        <AnimatePresence>
                            {results.map((img) => (
                                <motion.div
                                    key={`${img.photo_id}-${img.score}`}
                                    initial={{ opacity: 0, scale: 0.9 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    exit={{ opacity: 0 }}
                                    layout
                                    className="break-inside-avoid relative group rounded-xl overflow-hidden cursor-pointer"
                                    onClick={() => setFocusedImage(img)}
                                >
                                    <img
                                        src={img.photo_image_url}
                                        alt={img.photo_id}
                                        className="w-full h-auto object-cover transform transition-transform duration-500 group-hover:scale-110"
                                    />
                                    {img.video_url && (
                                        <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
                                            <div className="w-12 h-12 bg-black/50 rounded-full flex items-center justify-center backdrop-blur-md border border-white/10 shadow-xl">
                                                <Play className="w-5 h-5 text-white ml-1 fill-white" />
                                            </div>
                                        </div>
                                    )}
                                    {/* AI Verified badge — top-left corner, always visible */}
                                    {img.verified && (
                                        <div className="absolute top-2 left-2 z-20">
                                            <span className="inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded-full bg-amber-500/90 text-black backdrop-blur-sm shadow-lg">
                                                ✦ AI Verified
                                            </span>
                                        </div>
                                    )}
                                    <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col justify-end p-4">
                                        <div className="flex justify-between items-center mb-1.5">
                                            <p className="text-[10px] text-gray-400 font-mono truncate">{img.photo_id}</p>
                                            {img.score && (
                                                <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium border ${
                                                    img.verified
                                                        ? 'bg-amber-500/30 border-amber-500/40 text-amber-300'
                                                        : 'bg-purple-500/30 border-purple-500/30 text-purple-300'
                                                }`}>
                                                    {Math.round(img.score * 100)}% Match
                                                </span>
                                            )}
                                        </div>
                                        {img.description && (
                                            <p className="text-[11px] text-white/90 line-clamp-2 leading-tight bg-black/40 p-2 rounded-lg border border-white/5 backdrop-blur-md italic">
                                                "{img.description}"
                                            </p>
                                        )}
                                    </div>
                                </motion.div>
                            ))}
                        </AnimatePresence>
                    </div>
                </div>
            </div>

            <AnimatePresence>
                {focusedImage && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-6"
                        onClick={() => setFocusedImage(null)}
                    >
                        <motion.div
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.9, opacity: 0 }}
                            transition={{ type: "spring", damping: 25, stiffness: 300 }}
                            className="relative max-w-4xl max-h-[85vh] w-full flex flex-col items-center"
                            onClick={(e) => e.stopPropagation()}
                        >
                            {/* Image / Video container — boxes are absolute within this */}
                            <div className="relative inline-flex items-start justify-center max-h-[70vh] rounded-2xl overflow-hidden shadow-2xl bg-black/50">
                                {focusedImage.video_url ? (
                                    <div className="relative">
                                        <video
                                            ref={videoRef}
                                            src={focusedImage.video_url}
                                            crossOrigin="anonymous"
                                            controls
                                            className="max-h-[70vh] w-auto block"
                                            onLoadedMetadata={(e) => {
                                                if (focusedImage.timestamp) {
                                                    e.currentTarget.currentTime = focusedImage.timestamp;
                                                }
                                            }}
                                            onTimeUpdate={handleTimeUpdate}
                                        />
                                        {/* Preloading overlay */}
                                        {videoPreloading && (
                                            <div className="absolute inset-0 flex items-center justify-center bg-black/60 z-20 rounded-2xl">
                                                <div className="flex flex-col items-center gap-3">
                                                    <svg className="animate-spin w-8 h-8 text-purple-400" fill="none" viewBox="0 0 24 24">
                                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
                                                    </svg>
                                                    <span className="text-sm text-white/80 font-medium">Preloading object tracking...</span>
                                                </div>
                                            </div>
                                        )}
                                        {bboxes.map((box) => (
                                            <motion.div
                                                key={box.track_id}
                                                layout
                                                transition={{ type: "tween", ease: "linear", duration: 0.15 }}
                                                className="absolute border-[3px] pointer-events-none rounded"
                                                style={{
                                                    left: `${box.bbox[0] * 100}%`,
                                                    top: `${box.bbox[1] * 100}%`,
                                                    width: `${(box.bbox[2] - box.bbox[0]) * 100}%`,
                                                    height: `${(box.bbox[3] - box.bbox[1]) * 100}%`,
                                                    backgroundColor: `hsla(${(box.track_id * 60) % 360}, 80%, 55%, 0.1)`,
                                                    boxShadow: `0 0 12px hsla(${(box.track_id * 60) % 360}, 80%, 55%, 0.5)`,
                                                    borderColor: `hsl(${(box.track_id * 60) % 360}, 80%, 55%)`
                                                }}
                                            >
                                                <span
                                                    className="absolute -top-6 left-0 text-[10px] font-bold px-1.5 py-0.5 rounded whitespace-nowrap"
                                                    style={{ color: `hsl(${(box.track_id * 60) % 360}, 80%, 55%)`, backgroundColor: "rgba(0,0,0,0.75)" }}
                                                >
                                                    {query}
                                                </span>
                                            </motion.div>
                                        ))}
                                    </div>
                                ) : (
                                    /* inline-block sizes exactly to the image, making absolute bboxes accurate */
                                    <div className="relative inline-block">
                                        <img
                                            src={focusedImage.photo_image_url}
                                            alt={focusedImage.photo_id}
                                            className="max-h-[70vh] w-auto block"
                                        />
                                        {bboxLoading && (
                                            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                                                <div className="bg-black/60 rounded-full p-2">
                                                    <svg className="animate-spin w-6 h-6 text-purple-400" fill="none" viewBox="0 0 24 24">
                                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
                                                    </svg>
                                                </div>
                                            </div>
                                        )}
                                        {bboxes.map((box) => (
                                            <motion.div
                                                key={box.track_id}
                                                initial={{ opacity: 0 }}
                                                animate={{ opacity: 1 }}
                                                className="absolute border-[3px] pointer-events-none rounded"
                                                style={{
                                                    left: `${box.bbox[0] * 100}%`,
                                                    top: `${box.bbox[1] * 100}%`,
                                                    width: `${(box.bbox[2] - box.bbox[0]) * 100}%`,
                                                    height: `${(box.bbox[3] - box.bbox[1]) * 100}%`,
                                                    backgroundColor: `hsla(${(box.track_id * 60) % 360}, 80%, 55%, 0.1)`,
                                                    boxShadow: `0 0 12px hsla(${(box.track_id * 60) % 360}, 80%, 55%, 0.5)`,
                                                    borderColor: `hsl(${(box.track_id * 60) % 360}, 80%, 55%)`
                                                }}
                                            >
                                                <span
                                                    className="absolute -top-6 left-0 text-[10px] font-bold px-1.5 py-0.5 rounded whitespace-nowrap"
                                                    style={{ color: `hsl(${(box.track_id * 60) % 360}, 80%, 55%)`, backgroundColor: "rgba(0,0,0,0.75)" }}
                                                >
                                                    {query}
                                                </span>
                                            </motion.div>
                                        ))}
                                    </div>
                                )}

                            </div>
                            <div className="mt-4 w-full max-w-2xl text-center space-y-2 px-4">
                                {currentDescription && (
                                    <p className="text-sm text-white/90 bg-white/5 border border-white/10 rounded-xl px-4 py-3 italic">
                                        "{currentDescription}"
                                    </p>
                                )}
                            </div>
                            <button
                                onClick={() => setFocusedImage(null)}
                                className="absolute top-2 right-2 w-8 h-8 flex items-center justify-center rounded-full bg-white/10 text-white hover:bg-white/20 transition-all text-sm"
                            >
                                ✕
                            </button>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </main>
    );
}
