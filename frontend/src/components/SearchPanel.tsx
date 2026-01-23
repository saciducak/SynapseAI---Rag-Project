"use client";

import { useState } from "react";

interface SearchResult {
    content: string;
    filename: string;
    similarity_score: number;
    document_id: string;
}

interface AskResult {
    question: string;
    answer: string;
    sources: SearchResult[];
    tokens_used: number;
}

export default function SearchPanel() {
    const [query, setQuery] = useState("");
    const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
    const [askResult, setAskResult] = useState<AskResult | null>(null);
    const [isSearching, setIsSearching] = useState(false);
    const [isAsking, setIsAsking] = useState(false);
    const [activeMode, setActiveMode] = useState<"search" | "ask">("search");

    const handleSearch = async () => {
        if (!query.trim()) return;
        setIsSearching(true);
        setSearchResults([]);
        setAskResult(null);

        try {
            const response = await fetch("/api/search/semantic", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query, n_results: 10 }),
            });

            if (response.ok) {
                const data = await response.json();
                setSearchResults(data.results);
            }
        } catch (err) {
            console.error("Search failed:", err);
        } finally {
            setIsSearching(false);
        }
    };

    const handleAsk = async () => {
        if (!query.trim()) return;
        setIsAsking(true);
        setSearchResults([]);
        setAskResult(null);

        try {
            const response = await fetch("/api/search/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: query }),
            });

            if (response.ok) {
                const data = await response.json();
                setAskResult(data);
            }
        } catch (err) {
            console.error("Ask failed:", err);
        } finally {
            setIsAsking(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto">
            {/* Search Input */}
            <div className="glass-card p-6 mb-6">
                <div className="flex gap-4 mb-4">
                    <button
                        onClick={() => setActiveMode("search")}
                        className={`px-4 py-2 rounded-lg transition-all ${activeMode === "search"
                                ? "bg-indigo-500/30 text-indigo-300"
                                : "text-white/50 hover:text-white/80"
                            }`}
                    >
                        🔍 Semantic Search
                    </button>
                    <button
                        onClick={() => setActiveMode("ask")}
                        className={`px-4 py-2 rounded-lg transition-all ${activeMode === "ask"
                                ? "bg-indigo-500/30 text-indigo-300"
                                : "text-white/50 hover:text-white/80"
                            }`}
                    >
                        💬 Ask Question (RAG)
                    </button>
                </div>

                <div className="flex gap-3">
                    <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === "Enter") {
                                activeMode === "search" ? handleSearch() : handleAsk();
                            }
                        }}
                        placeholder={
                            activeMode === "search"
                                ? "Search across all documents..."
                                : "Ask a question about your documents..."
                        }
                        className="flex-1 px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-white/40 focus:outline-none focus:border-indigo-500/50"
                    />
                    <button
                        onClick={activeMode === "search" ? handleSearch : handleAsk}
                        disabled={isSearching || isAsking || !query.trim()}
                        className="btn-primary"
                    >
                        {isSearching || isAsking ? (
                            <span className="spinner" />
                        ) : activeMode === "search" ? (
                            "Search"
                        ) : (
                            "Ask"
                        )}
                    </button>
                </div>
            </div>

            {/* Ask Result */}
            {askResult && (
                <div className="glass-card p-6 mb-6">
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        <span className="text-xl">💡</span> Answer
                    </h3>
                    <div className="p-4 bg-gradient-to-r from-indigo-500/10 to-purple-500/10 rounded-lg border border-indigo-500/20 mb-4">
                        <p className="text-white/90 whitespace-pre-wrap">{askResult.answer}</p>
                    </div>
                    <div className="text-white/40 text-sm mb-4">
                        {askResult.tokens_used} tokens used
                    </div>
                    {askResult.sources.length > 0 && (
                        <div>
                            <h4 className="text-sm font-medium text-white/60 mb-2">Sources:</h4>
                            <div className="space-y-2">
                                {askResult.sources.map((source, i) => (
                                    <div key={i} className="p-3 bg-white/5 rounded-lg text-sm">
                                        <div className="flex justify-between mb-1">
                                            <span className="text-indigo-300">{source.filename}</span>
                                            <span className="text-white/40">
                                                {(source.similarity_score * 100).toFixed(0)}% match
                                            </span>
                                        </div>
                                        <p className="text-white/60 line-clamp-2">{source.content}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Search Results */}
            {searchResults.length > 0 && (
                <div className="glass-card p-6">
                    <h3 className="text-lg font-semibold mb-4">
                        Found {searchResults.length} results
                    </h3>
                    <div className="space-y-4">
                        {searchResults.map((result, i) => (
                            <div
                                key={i}
                                className="p-4 bg-white/5 rounded-lg border border-white/10 hover:border-indigo-500/30 transition-all"
                            >
                                <div className="flex justify-between items-start mb-2">
                                    <span className="text-indigo-300 font-medium">
                                        {result.filename}
                                    </span>
                                    <span className="text-xs px-2 py-1 bg-indigo-500/20 text-indigo-300 rounded">
                                        {(result.similarity_score * 100).toFixed(0)}% match
                                    </span>
                                </div>
                                <p className="text-white/70 text-sm line-clamp-3">
                                    {result.content}
                                </p>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Empty State */}
            {!searchResults.length && !askResult && !isSearching && !isAsking && (
                <div className="glass-card p-12 text-center">
                    <div className="text-6xl mb-4 opacity-50">🔍</div>
                    <h3 className="text-xl font-semibold text-white/70 mb-2">
                        Search Your Documents
                    </h3>
                    <p className="text-white/40 max-w-md mx-auto">
                        Use semantic search to find relevant content or ask questions with RAG-powered Q&A
                    </p>
                </div>
            )}
        </div>
    );
}
