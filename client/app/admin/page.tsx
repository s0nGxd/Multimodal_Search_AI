"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, CheckCircle, AlertCircle, Loader2, Database, ArrowLeft, Image as ImageIcon } from "lucide-react";
import Link from "next/link";
import { uploadImage, bulkIngest, ingestViaUrl, listAllImages, SearchResult } from "@/lib/api";

export default function AdminPage() {
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [password, setPassword] = useState("");

    const [file, setFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);
    const [status, setStatus] = useState<"idle" | "success" | "error">("idle");
    const [message, setMessage] = useState("");
    const [isBulkIndexing, setIsBulkIndexing] = useState(false);

    const [activeTab, setActiveTab] = useState<"file" | "url" | "bulk" | "gallery">("file");
    const [imageUrl, setImageUrl] = useState("");
    const [allImages, setAllImages] = useState<SearchResult[]>([]);
    const [loadingImages, setLoadingImages] = useState(false);
    const [focusedImage, setFocusedImage] = useState<SearchResult | null>(null);

    const fetchAllImages = async () => {
        setLoadingImages(true);
        try {
            const images = await listAllImages();
            setAllImages(images);
        } catch (e: any) {
            setStatus("error");
            setMessage(e.message || "Failed to fetch images");
        } finally {
            setLoadingImages(false);
        }
    };

    const handleLogin = (e: React.FormEvent) => {
        e.preventDefault();
        if (password === "admin") {
            setIsAuthenticated(true);
        } else {
            alert("Invalid password");
        }
    };

    const handleUpload = async () => {
        if (!file) return;
        setUploading(true);
        setStatus("idle");
        try {
            const result = await uploadImage(file);
            setStatus("success");
            setMessage(`Indexed: ${file.name}. Description: ${result.description}`);
            setFile(null);
        } catch (e: any) {
            setStatus("error");
            setMessage(e.message || "Upload failed");
        } finally {
            setUploading(false);
        }
    };

    const handleUrlIngest = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!imageUrl) return;
        setUploading(true);
        setStatus("idle");
        try {
            const result = await ingestViaUrl(imageUrl);
            setStatus("success");
            setMessage(`URL Indexed. Description: ${result.description}`);
            setImageUrl("");
        } catch (e: any) {
            setStatus("error");
            setMessage(e.message || "URL ingestion failed");
        } finally {
            setUploading(false);
        }
    };

    const handleBulkIngest = async () => {
        setIsBulkIndexing(true);
        setStatus("idle");
        try {
            const result = await bulkIngest(10); // Batch of 10 for speed
            setStatus("success");
            setMessage(`Bulk ingestion complete: Processed ${result.processed} images.`);
        } catch (e: any) {
            setStatus("error");
            setMessage(e.message || "Bulk ingestion failed");
        } finally {
            setIsBulkIndexing(false);
        }
    };

    if (!isAuthenticated) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-black">
                <div className="absolute top-0 left-0 w-full h-full overflow-hidden -z-10 pointer-events-none opacity-50">
                    <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] bg-purple-900/10 rounded-full blur-[120px]" />
                    <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-blue-900/10 rounded-full blur-[120px]" />
                </div>
                <div className="w-full max-w-sm p-8 rounded-3xl bg-white/5 border border-white/10 backdrop-blur-xl shadow-2xl">
                    <h2 className="text-2xl font-bold text-white mb-6 text-center tracking-tight">Admin Portal</h2>
                    <form onSubmit={handleLogin} className="space-y-4">
                        <input
                            type="password"
                            value={password}
                            onChange={e => setPassword(e.target.value)}
                            placeholder="Access Password"
                            className="w-full bg-black/50 border border-white/10 rounded-xl p-4 text-white placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-purple-500/50 transition-all font-mono"
                        />
                        <button type="submit" className="w-full bg-white text-black font-semibold py-4 rounded-xl hover:bg-gray-200 transition-all active:scale-[0.98]">
                            Enter Dashboard
                        </button>
                    </form>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen p-6 md:p-12 bg-black text-white selection:bg-purple-500/30">
            <div className="max-w-5xl mx-auto">
                <div className="mb-6">
                    <Link
                        href="/"
                        className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 text-sm font-medium hover:bg-white/10 transition-all text-gray-400 hover:text-white"
                    >
                        <ArrowLeft className="w-4 h-4" />
                        Back to Search
                    </Link>
                </div>
                <header className="flex flex-col md:flex-row justify-between items-start md:items-center mb-12 gap-4">
                    <div>
                        <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white via-purple-300 to-blue-300">
                            Control Center
                        </h1>
                        <p className="text-gray-500 mt-2">Manage your semantic indexed dataset</p>
                    </div>
                    <div className="px-4 py-2 rounded-full bg-white/5 border border-white/10 text-xs font-mono text-gray-400">
                        STATUS: <span className="text-green-400">OPERATIONAL</span>
                    </div>
                </header>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Main Actions Area */}
                    <div className="lg:col-span-2 space-y-8">

                        {/* Tabs */}
                        <div className="flex p-1 bg-white/5 border border-white/5 rounded-2xl w-fit">
                            {[
                                { id: "file", label: "File Upload", icon: Upload },
                                { id: "url", label: "Remote URL", icon: Database },
                                { id: "bulk", label: "Bulk Ingest", icon: Database },
                                { id: "gallery", label: "All Images", icon: ImageIcon }
                            ].map((tab) => (
                                <button
                                    key={tab.id}
                                    onClick={() => setActiveTab(tab.id as any)}
                                    className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-medium transition-all ${activeTab === tab.id
                                        ? "bg-white/10 text-white shadow-lg"
                                        : "text-gray-500 hover:text-gray-300"
                                        }`}
                                >
                                    <tab.icon className="w-4 h-4" />
                                    {tab.label}
                                </button>
                            ))}
                        </div>

                        <div className="p-8 rounded-3xl bg-white/5 border border-white/10 backdrop-blur-md">
                            {activeTab === "file" && (
                                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                                    <h3 className="text-lg font-medium mb-6">Upload Local Assets</h3>
                                    <div
                                        className={`group border-2 border-dashed rounded-2xl p-16 text-center cursor-pointer transition-all ${file ? "border-purple-500 bg-purple-500/5" : "border-white/10 hover:border-white/20 hover:bg-white/[0.02]"
                                            }`}
                                    >
                                        <input
                                            type="file"
                                            accept="image/*"
                                            className="hidden"
                                            id="file-upload"
                                            onChange={(e) => {
                                                if (e.target.files?.[0]) {
                                                    setFile(e.target.files[0]);
                                                    setStatus("idle");
                                                }
                                            }}
                                        />
                                        <label htmlFor="file-upload" className="w-full h-full block cursor-pointer">
                                            {file ? (
                                                <div className="flex flex-col items-center gap-3">
                                                    <div className="p-4 bg-purple-500/20 rounded-full text-purple-400">
                                                        <Upload className="w-8 h-8" />
                                                    </div>
                                                    <span className="text-purple-300 font-medium">{file.name}</span>
                                                    <span className="text-gray-500 text-xs">{(file.size / 1024).toFixed(1)} KB</span>
                                                </div>
                                            ) : (
                                                <div className="flex flex-col items-center gap-3">
                                                    <div className="p-4 bg-white/5 rounded-full text-gray-400 group-hover:text-gray-300 transition-colors">
                                                        <Upload className="w-8 h-8" />
                                                    </div>
                                                    <span className="text-gray-400">Drop your image or <span className="text-purple-400">browse</span></span>
                                                    <span className="text-gray-600 text-xs">JPG, PNG, GIF up to 10MB</span>
                                                </div>
                                            )}
                                        </label>
                                    </div>
                                    <button
                                        onClick={handleUpload}
                                        disabled={!file || uploading}
                                        className="w-full mt-6 py-4 bg-gradient-to-r from-purple-600 to-blue-600 rounded-xl font-semibold disabled:opacity-50 disabled:grayscale transition-all active:scale-[0.99] shadow-lg shadow-purple-500/20"
                                    >
                                        {uploading ? <Loader2 className="animate-spin w-5 h-5 mx-auto" /> : "Index File"}
                                    </button>
                                </motion.div>
                            )}

                            {activeTab === "url" && (
                                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                                    <h3 className="text-lg font-medium mb-6">Index via Remote URL</h3>
                                    <form onSubmit={handleUrlIngest} className="space-y-6">
                                        <div>
                                            <label className="text-sm text-gray-500 mb-2 block">Image Address</label>
                                            <input
                                                type="url"
                                                required
                                                value={imageUrl}
                                                onChange={e => setImageUrl(e.target.value)}
                                                placeholder="https://example.com/image.jpg"
                                                className="w-full bg-black/40 border border-white/10 rounded-xl p-4 focus:ring-2 focus:ring-blue-500/50 outline-none transition-all"
                                            />
                                        </div>
                                        <button
                                            type="submit"
                                            disabled={!imageUrl || uploading}
                                            className="w-full py-4 bg-gradient-to-r from-blue-600 to-cyan-600 rounded-xl font-semibold disabled:opacity-50 transition-all active:scale-[0.99]"
                                        >
                                            {uploading ? <Loader2 className="animate-spin w-5 h-5 mx-auto" /> : "Fetch & Index"}
                                        </button>
                                    </form>
                                </motion.div>
                            )}

                            {activeTab === "bulk" && (
                                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
                                    <div className="flex flex-col gap-2">
                                        <h3 className="text-lg font-medium">Bulk Dataset Import</h3>
                                        <p className="text-sm text-gray-500">Sync records seamlessly from your reference dataset.</p>
                                    </div>
                                    <div className="p-6 rounded-2xl bg-blue-500/5 border border-blue-500/10 flex items-start gap-4">
                                        <Database className="w-5 h-5 text-blue-400 mt-1" />
                                        <div className="text-sm leading-relaxed text-gray-400">
                                            This process will download imagery and generate CLIP vectors for each record. Performance depends on network speed.
                                        </div>
                                    </div>
                                    <button
                                        onClick={handleBulkIngest}
                                        disabled={isBulkIndexing}
                                        className="w-full py-4 bg-white text-black rounded-xl font-bold flex items-center justify-center gap-3 disabled:opacity-50 transition-all"
                                    >
                                        {isBulkIndexing ? <Loader2 className="animate-spin w-5 h-5" /> : <><Database className="w-5 h-5" /> Start Bulk Sync</>}
                                    </button>
                                </motion.div>
                            )}

                            {activeTab === "gallery" && (
                                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
                                    <div className="flex items-center justify-between">
                                        <div>
                                            <h3 className="text-lg font-medium">All Indexed Images</h3>
                                            <p className="text-sm text-gray-500">{allImages.length} images in database</p>
                                        </div>
                                        <button
                                            onClick={fetchAllImages}
                                            disabled={loadingImages}
                                            className="px-4 py-2 bg-white/10 border border-white/10 rounded-xl text-sm font-medium hover:bg-white/15 transition-all disabled:opacity-50"
                                        >
                                            {loadingImages ? <Loader2 className="animate-spin w-4 h-4" /> : "Refresh"}
                                        </button>
                                    </div>
                                    {allImages.length === 0 && !loadingImages && (
                                        <div className="text-center py-12">
                                            <ImageIcon className="w-10 h-10 text-gray-700 mx-auto mb-3" />
                                            <p className="text-gray-500 text-sm">Click Refresh to load indexed images</p>
                                        </div>
                                    )}
                                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4 max-h-[60vh] overflow-y-auto pr-2">
                                        {allImages.map((img) => (
                                            <div
                                                key={img.photo_id}
                                                className="group relative rounded-xl overflow-hidden border border-white/5 bg-white/5 cursor-pointer"
                                                onClick={() => setFocusedImage(img)}
                                            >
                                                <img
                                                    src={img.photo_image_url}
                                                    alt={img.photo_id}
                                                    className="w-full h-40 object-cover"
                                                    loading="lazy"
                                                />
                                                <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex flex-col justify-end p-3">
                                                    <p className="text-[10px] text-gray-400 font-mono truncate">{img.photo_id}</p>
                                                    {img.description && (
                                                        <p className="text-[11px] text-white/80 line-clamp-2 mt-1 italic">"{img.description}"</p>
                                                    )}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </motion.div>
                            )}

                            {/* Status Feed */}
                            <AnimatePresence>
                                {status !== "idle" && (
                                    <motion.div
                                        initial={{ opacity: 0, height: 0 }}
                                        animate={{ opacity: 1, height: "auto" }}
                                        exit={{ opacity: 0, height: 0 }}
                                        className={`mt-6 p-4 rounded-xl border flex items-center gap-3 ${status === "success" ? "bg-green-500/5 border-green-500/20 text-green-400" : "bg-red-500/5 border-red-500/20 text-red-400"
                                            }`}
                                    >
                                        {status === "success" ? <CheckCircle className="w-5 h-5" /> : <AlertCircle className="w-5 h-5" />}
                                        <span className="text-sm font-medium">{message}</span>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>
                    </div>

                    {/* Sidebar Stats */}
                    <div className="space-y-8">
                        <section className="p-8 rounded-3xl bg-white/5 border border-white/10">
                            <h2 className="text-xl font-semibold mb-6">Engine Heartbeat</h2>
                            <div className="space-y-6">
                                <div className="flex justify-between items-center group">
                                    <span className="text-gray-500 group-hover:text-gray-400 transition-colors">Core Model</span>
                                    <span className="text-sm font-mono bg-white/5 px-2 py-1 rounded">CLIP-ViT</span>
                                </div>
                                <div className="flex justify-between items-center group">
                                    <span className="text-gray-500 group-hover:text-gray-400 transition-colors">Vector Store</span>
                                    <span className="text-sm font-mono text-purple-400">LanceDB</span>
                                </div>
                                <div className="flex justify-between items-center group">
                                    <span className="text-gray-500 group-hover:text-gray-400 transition-colors">Dimension</span>
                                    <span className="text-sm font-mono">512px v</span>
                                </div>
                            </div>

                            <div className="mt-10 pt-8 border-t border-white/5">
                                <div className="flex items-center justify-between mb-4">
                                    <h4 className="text-xs uppercase tracking-widest text-gray-600 font-bold">Health</h4>
                                    <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                                </div>
                                <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                                    <div className="h-full w-[94%] bg-gradient-to-r from-blue-500 to-purple-500" />
                                </div>
                            </div>
                        </section>

                        <section className="p-8 rounded-3xl bg-gradient-to-br from-purple-900/20 to-blue-900/20 border border-white/5 shadow-xl shadow-purple-900/10">
                            <h2 className="text-lg font-semibold mb-4 text-purple-200">System Insights</h2>
                            <p className="text-sm text-gray-400 leading-relaxed italic">
                                "The semantic engine understands concepts. You don't need exact filenames. Try indexing images of diverse subjects to test the limits of your search."
                            </p>
                        </section>
                    </div>
                </div>
            </div>

            {/* Image Lightbox */}
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
                            <img
                                src={focusedImage.photo_image_url}
                                alt={focusedImage.photo_id}
                                className="max-h-[70vh] w-auto rounded-2xl object-contain shadow-2xl"
                            />
                            <div className="mt-4 w-full max-w-2xl text-center space-y-2">
                                <p className="text-xs text-gray-500 font-mono">{focusedImage.photo_id}</p>
                                {focusedImage.description && (
                                    <p className="text-sm text-white/90 bg-white/5 border border-white/10 rounded-xl px-4 py-3 italic">
                                        "{focusedImage.description}"
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
        </div>
    );
}
