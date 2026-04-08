"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Search, Image as ImageIcon, Sparkles, Loader2, ArrowRight, Play } from "lucide-react";
import Link from "next/link";
import { searchImages, SearchResult, detectObject, getVideoFrames, VideoFrame, trackObject, TrackResult } from "@/lib/api";

export default function Home() {
    const [query, setQuery] = useState("");
    const [results, setResults] = useState<SearchResult[]>([]);
    const [loading, setLoading] = useState(false);
    const [hasSearched, setHasSearched] = useState(false);
    const [resultCount, setResultCount] = useState(20);
    const [minSimilarity, setMinSimilarity] = useState(0.2);
    const [showSettings, setShowSettings] = useState(false);
    const [focusedImage, setFocusedImage] = useState<SearchResult | null>(null);
    const [bboxes, setBboxes] = useState<TrackResult[]>([]);
    const videoRef = useRef<HTMLVideoElement>(null);
    const lastDetectTime = useRef<number>(0);
    const [isDetecting, setIsDetecting] = useState(false);
    const [videoFrames, setVideoFrames] = useState<VideoFrame[]>([]);
    const [currentDescription, setCurrentDescription] = useState("");

    useEffect(() => {
        if (focusedImage) {
            setBboxes([]);
            setCurrentDescription(focusedImage.description || "");
            
            if (focusedImage.video_url) {
                getVideoFrames(focusedImage.video_url).then(setVideoFrames).catch(console.error);
            } else {
                setVideoFrames([]);
            }

            if (query && hasSearched) {
                lastDetectTime.current = 0;
                const videoId = focusedImage.video_url || undefined;
                trackObject(focusedImage.photo_image_url, query, focusedImage.video_url, videoId)
                    .then(res => setBboxes(res.tracks || []))
                    .catch(console.error);
            }
        }
    }, [focusedImage, query, hasSearched]);

    const handleTimeUpdate = async () => {
        const video = videoRef.current;
        if (!video) return;

        const now = video.currentTime;

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

        if (isDetecting || !query || !focusedImage) return;

        if (Math.abs(now - lastDetectTime.current) >= 0.04) {
            lastDetectTime.current = now;
            setIsDetecting(true);

            try {
                const canvas = document.createElement("canvas");
                const MAX_WIDTH = 480;
                const scale = Math.min(1.0, MAX_WIDTH / video.videoWidth);
                canvas.width = video.videoWidth * scale;
                canvas.height = video.videoHeight * scale;
                
                const ctx = canvas.getContext("2d");
                if (ctx && canvas.width > 0) {
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    const base64Image = canvas.toDataURL("image/jpeg", 0.3);
                    const videoId = focusedImage.video_url || undefined;
                    const result = await trackObject(
                        focusedImage.photo_image_url,
                        query,
                        focusedImage.video_url,
                        videoId,
                        base64Image
                    );
                    if (result && result.tracks) {
                        setBboxes(result.tracks);
                    } else {
                        setBboxes([]);
                    }
                }
            } catch (err) {
                console.error("Live tracking failed:", err);
            } finally {
                setIsDetecting(false);
            }
        }
    };

    const handleSearch = async (e?: React.FormEvent) => {
        if (e) e.preventDefault();
        if (!query.trim()) return;

        setLoading(true);
        try {
            const data = await searchImages(query, resultCount, minSimilarity);
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
                        <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-4 pb-2 bg-clip-text text-transparent bg-gradient-to-b from-white to-gray-500 leading-tight">
                            Find anything <br /> in your images.
                        </h1>
                        <p className="text-gray-400 text-lg max-w-xl mx-auto">
                            Natural language search powered by SigLIP & LanceDB.
                            Search by concepts, emotions, or objects.
                        </p>
                    </motion.div>

                    <div className="relative max-w-2xl mx-auto group">
                        <form
                            onSubmit={handleSearch}
                            className="relative z-10"
                        >
                            <div className="absolute inset-0 bg-gradient-to-r from-purple-600/20 to-blue-600/20 rounded-2xl blur-xl opacity-0 group-focus-within:opacity-100 transition-opacity" />
                            <div className="relative flex items-center bg-white/5 border border-white/10 rounded-2xl backdrop-blur-xl transition-all">
                                <Search className="ml-4 w-5 h-5 text-gray-500" />
                                <input
                                    type="text"
                                    value={query}
                                    onChange={(e) => setQuery(e.target.value)}
                                    placeholder="Describe an image..."
                                    className="w-full bg-transparent border-none focus:ring-0 focus:outline-none py-5 px-4 text-white placeholder-gray-600"
                                />
                                <button
                                    type="submit"
                                    disabled={loading}
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
                                            <div className="w-12 h-12 bg-black/50 rounded-full flex items-center justify-center backdrop-blur-md">
                                                <Play className="w-5 h-5 text-white ml-1" />
                                            </div>
                                        </div>
                                    )}
                                    <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col justify-end p-4">
                                        <div className="flex justify-between items-center mb-1.5">
                                            <p className="text-[10px] text-gray-400 font-mono truncate">{img.photo_id}</p>
                                            {img.score && (
                                                <span className="text-[10px] bg-purple-500/30 border border-purple-500/30 px-1.5 py-0.5 rounded text-purple-300 font-medium whitespace-nowrap">
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
                                            if (focusedImage.timestamp) {
                                                e.currentTarget.currentTime = focusedImage.timestamp;
                                                lastDetectTime.current = focusedImage.timestamp;
                                            }
                                        }}
                                        onTimeUpdate={handleTimeUpdate}
                                    />
                                ) : (
                                    <img
                                        src={focusedImage.photo_image_url}
                                        alt={focusedImage.photo_id}
                                        className="max-h-[70vh] w-auto"
                                    />
                                )}
                                {bboxes.map((box, idx) => (
                                    <div
                                        key={box.track_id || idx}
                                        className="absolute border-[3px] bg-red-500/10 pointer-events-none rounded sm:border-4 transition-all duration-300 ease-out"
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
                                    </div>
                                ))}
                            </div>
                            <div className="mt-4 w-full max-w-2xl text-center space-y-2">
                                <div className="flex items-center justify-center gap-3">
                                    <p className="text-xs text-gray-500 font-mono">{focusedImage.photo_id}</p>
                                    {focusedImage.score && (
                                        <span className="text-xs bg-purple-500/30 border border-purple-500/30 px-2 py-0.5 rounded text-purple-300 font-medium">
                                            {Math.round(focusedImage.score * 100)}% Match
                                        </span>
                                    )}
                                </div>
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
