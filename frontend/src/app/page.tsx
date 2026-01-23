"use client";

import { useState, useCallback } from "react";
import ModeSelector from "@/components/ModeSelector";
import FileUpload from "@/components/FileUpload";
import AnalysisPanel from "@/components/AnalysisPanel";
import SearchPanel from "@/components/SearchPanel";

type Mode = "document" | "code" | "research" | "legal";
type Tab = "analyze" | "search";

interface AnalysisResult {
    analysis_id: string;
    document_id: string;
    mode: string;
    final_output: any;
    total_tokens: number;
    total_time_ms: number;
}

export default function Home() {
    const [selectedMode, setSelectedMode] = useState<Mode>("document");
    const [activeTab, setActiveTab] = useState<Tab>("analyze");
    const [uploadedDoc, setUploadedDoc] = useState<{
        id: string;
        filename: string;
    } | null>(null);
    const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleUploadSuccess = useCallback((docId: string, filename: string) => {
        setUploadedDoc({ id: docId, filename });
        setAnalysisResult(null);
        setError(null);
    }, []);

    const handleAnalyze = async () => {
        if (!uploadedDoc) return;

        setIsAnalyzing(true);
        setError(null);

        try {
            const response = await fetch(`/api/analysis/${uploadedDoc.id}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ mode: selectedMode }),
            });

            if (!response.ok) {
                throw new Error("Analysis failed");
            }

            const result = await response.json();
            setAnalysisResult(result);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Analysis failed");
        } finally {
            setIsAnalyzing(false);
        }
    };

    return (
        <main className="min-h-screen p-6 md:p-10">
            {/* Header */}
            <header className="mb-10">
                <div className="flex items-center gap-4 mb-2">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-2xl">
                        🧠
                    </div>
                    <div>
                        <h1 className="text-3xl font-bold gradient-text">SynapseAI</h1>
                        <p className="text-white/60 text-sm">
                            Multi-Mode Multi-Agent Decision Support System
                        </p>
                    </div>
                </div>
            </header>

            {/* Tab Navigation */}
            <div className="flex gap-4 mb-8">
                <button
                    onClick={() => setActiveTab("analyze")}
                    className={`px-6 py-3 rounded-xl font-medium transition-all ${activeTab === "analyze"
                            ? "bg-white/10 text-white"
                            : "text-white/50 hover:text-white/80"
                        }`}
                >
                    📊 Analyze
                </button>
                <button
                    onClick={() => setActiveTab("search")}
                    className={`px-6 py-3 rounded-xl font-medium transition-all ${activeTab === "search"
                            ? "bg-white/10 text-white"
                            : "text-white/50 hover:text-white/80"
                        }`}
                >
                    🔍 Search
                </button>
            </div>

            {activeTab === "analyze" ? (
                <div className="grid lg:grid-cols-[1fr,1.5fr] gap-8">
                    {/* Left Column - Controls */}
                    <div className="space-y-6">
                        {/* Mode Selector */}
                        <section className="glass-card p-6">
                            <h2 className="text-lg font-semibold mb-4 text-white/90">
                                Analysis Mode
                            </h2>
                            <ModeSelector
                                selectedMode={selectedMode}
                                onModeChange={setSelectedMode}
                            />
                        </section>

                        {/* File Upload */}
                        <section className="glass-card p-6">
                            <h2 className="text-lg font-semibold mb-4 text-white/90">
                                Upload Document
                            </h2>
                            <FileUpload
                                mode={selectedMode}
                                onUploadSuccess={handleUploadSuccess}
                            />
                            {uploadedDoc && (
                                <div className="mt-4 p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
                                    <p className="text-green-400 text-sm">
                                        ✓ Uploaded: {uploadedDoc.filename}
                                    </p>
                                </div>
                            )}
                        </section>

                        {/* Analyze Button */}
                        <button
                            onClick={handleAnalyze}
                            disabled={!uploadedDoc || isAnalyzing}
                            className="btn-primary w-full flex items-center justify-center gap-2"
                        >
                            {isAnalyzing ? (
                                <>
                                    <span className="spinner" />
                                    <span>Analyzing with AI Agents...</span>
                                </>
                            ) : (
                                <>
                                    <span>🚀</span>
                                    <span>Run Multi-Agent Analysis</span>
                                </>
                            )}
                        </button>

                        {error && (
                            <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl">
                                <p className="text-red-400">{error}</p>
                            </div>
                        )}
                    </div>

                    {/* Right Column - Results */}
                    <div>
                        <AnalysisPanel result={analysisResult} isLoading={isAnalyzing} />
                    </div>
                </div>
            ) : (
                <SearchPanel />
            )}
        </main>
    );
}
