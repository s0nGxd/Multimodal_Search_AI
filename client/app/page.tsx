"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Search, Image as ImageIcon, Sparkles, Loader2, ArrowRight, Play } from "lucide-react";
import Link from "next/link";
import { searchImages, SearchResult, getVideoFrames, VideoFrame, trackObject, detectObject, TrackResult, checkBackendHealth, waitForBackend, checkBackendReady, waitForBackendReady } from "@/lib/api";

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
        return () => { cancelled = true; };
    }, []);

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
                if (focusedImage.video_url) {
                    // Video: use stateful tracker from the start (matches subsequent timeupdate ticks)
                    trackObject(focusedImage.photo_image_url, query, focusedImage.video_url, focusedImage.video_url)
                        .then(res => setBboxes(res.tracks || []))
                        .catch(console.error);
                } else {
                    // Still image: single-shot stateless detection — no tracker hysteresis
                    detectObject(focusedImage.photo_image_url, query)
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
                    query,
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

    const handleSearch = async (e?: React.FormEvent) => {
        if (e) e.preventDefault();
        if (!query.trim()) return;
        setLoading(true);
        try {
            // Map the user-facing 0-1 slider value into the backend threshold's
            // 0..SLIDER_MAX_THRESHOLD band so the full slider is meaningful.
            const data = await searchImages(query, resultCount, minSimilarity * SLIDER_MAX_THRESHOLD);
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
                                    className="mr-2 px-6 py-2.5 bg-white text-black font-bold rounded-xl hover:bg-gray-200 transition-all active:scale-95 disabled:opacity-50"
                                >
                                    {loading ? <Loader2 className="animate-spin w-5 h-5" /> : "Search"}
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
                                        loading="lazy"
                                        decoding="async"
                                    />
                                    {img.video_url && (
                                        <>
                                            <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
                                                <div className="w-12 h-12 bg-black/50 rounded-full flex items-center justify-center backdrop-blur-md border border-white/10 shadow-xl">
                                                    <Play className="w-5 h-5 text-white ml-1 fill-white" />
                                                </div>
                                            </div>
                                            <div className="absolute top-2 left-2 z-10 text-[10px] px-2 py-1 rounded-full bg-black/60 text-white/90 border border-white/10 backdrop-blur-md font-mono">
                                                {typeof img.best_timestamp === 'number'
                                                    ? `video @ ${formatMMSS(img.best_timestamp)}`
                                                    : 'video'}
                                            </div>
                                        </>
                                    )}
                                    <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col justify-end p-4">
                                        <div className="flex justify-between items-center mb-1.5">
                                            <p className="text-[10px] text-gray-400 font-mono truncate">{img.photo_id}</p>
                                            {img.score && (
                                                <span className="text-[10px] bg-purple-500/30 border border-purple-500/30 px-1.5 py-0.5 rounded text-purple-300 font-medium">
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
                            <div className="relative inline-flex items-center justify-center max-h-[70vh] rounded-2xl overflow-hidden shadow-2xl bg-black/50">
                                {focusedImage.video_url ? (
                                    <video
                                        ref={videoRef}
                                        src={focusedImage.video_url}
                                        crossOrigin="anonymous"
                                        controls
                                        autoPlay
                                        className="max-h-[70vh] w-auto"
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
                                        className="max-h-[70vh] w-auto"
                                        decoding="async"
                                    />
                                )}
                                {bboxes.map((box) => (
                                    <motion.div
                                        key={box.track_id}
                                        layout
                                        transition={{ type: "tween", ease: "linear", duration: 0.15 }}
                                        className="absolute border-[3px] bg-red-500/10 pointer-events-none rounded sm:border-4 transition-all"
                                        style={{
                                            left: `${box.bbox[0] * 100}%`,
                                            top: `${box.bbox[1] * 100}%`,
                                            width: `${(box.bbox[2] - box.bbox[0]) * 100}%`,
                                            height: `${(box.bbox[3] - box.bbox[1]) * 100}%`,
                                            boxShadow: "0 0 15px rgba(239, 68, 68, 0.5)",
                                            borderColor: `hsl(${(box.track_id * 60) % 360}, 80%, 55%)`
                                        }}
                                    >
                                        <span 
                                            className="absolute -top-6 left-0 text-[10px] font-bold px-1 rounded"
                                            style={{ color: `hsl(${(box.track_id * 60) % 360}, 80%, 55%)`, backgroundColor: "rgba(0,0,0,0.7)" }}
                                        >
                                            #{box.track_id}
                                        </span>
                                    </motion.div>
                                ))}
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
