"use client";

interface AnalysisPanelProps {
    result: any | null;
    isLoading: boolean;
}

export default function AnalysisPanel({ result, isLoading }: AnalysisPanelProps) {
    if (isLoading) {
        return (
            <div className="glass-card p-8 h-full flex flex-col items-center justify-center min-h-[400px]">
                <div className="animate-pulse-glow w-20 h-20 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-3xl mb-6">
                    🧠
                </div>
                <h3 className="text-xl font-semibold mb-2">AI Agents Working...</h3>
                <p className="text-white/50 text-center">
                    Analyzer → Summarizer → Recommender
                </p>
                <div className="mt-6 flex gap-2">
                    <div className="w-3 h-3 rounded-full bg-indigo-500 animate-bounce" style={{ animationDelay: "0ms" }} />
                    <div className="w-3 h-3 rounded-full bg-purple-500 animate-bounce" style={{ animationDelay: "150ms" }} />
                    <div className="w-3 h-3 rounded-full bg-pink-500 animate-bounce" style={{ animationDelay: "300ms" }} />
                </div>
            </div>
        );
    }

    if (!result) {
        return (
            <div className="glass-card p-8 h-full flex flex-col items-center justify-center min-h-[400px] text-center">
                <div className="text-6xl mb-4 opacity-50">📊</div>
                <h3 className="text-xl font-semibold text-white/70 mb-2">
                    No Analysis Yet
                </h3>
                <p className="text-white/40 max-w-md">
                    Upload a document and click "Run Multi-Agent Analysis" to see AI-powered insights
                </p>
            </div>
        );
    }

    const { final_output, total_tokens, total_time_ms } = result;
    const analyzer = final_output?.analyzer || {};
    const summarizer = final_output?.summarizer || {};
    const recommender = final_output?.recommender || {};

    return (
        <div className="glass-card p-6 space-y-6">
            {/* Stats Bar */}
            <div className="flex gap-4 pb-4 border-b border-white/10">
                <div className="px-3 py-1 bg-green-500/20 rounded-full text-green-400 text-sm">
                    ✓ Analysis Complete
                </div>
                <div className="text-white/40 text-sm">
                    {total_tokens} tokens • {(total_time_ms / 1000).toFixed(1)}s
                </div>
            </div>

            {/* Analysis Section */}
            {analyzer && !analyzer.parse_error && (
                <section>
                    <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                        <span className="text-xl">🔍</span> Analysis
                    </h3>
                    <div className="space-y-3">
                        {analyzer.document_type && (
                            <div className="p-3 bg-white/5 rounded-lg">
                                <span className="text-white/50 text-sm">Document Type:</span>
                                <span className="ml-2 text-white">{analyzer.document_type}</span>
                            </div>
                        )}
                        {analyzer.main_topics && (
                            <div className="p-3 bg-white/5 rounded-lg">
                                <span className="text-white/50 text-sm block mb-2">Main Topics:</span>
                                <div className="flex flex-wrap gap-2">
                                    {analyzer.main_topics.map((topic: string, i: number) => (
                                        <span key={i} className="px-2 py-1 bg-indigo-500/20 text-indigo-300 rounded text-sm">
                                            {topic}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        )}
                        {analyzer.key_points && (
                            <div className="p-3 bg-white/5 rounded-lg">
                                <span className="text-white/50 text-sm block mb-2">Key Points:</span>
                                <ul className="space-y-1">
                                    {analyzer.key_points.slice(0, 5).map((point: string, i: number) => (
                                        <li key={i} className="text-white/80 text-sm flex gap-2">
                                            <span className="text-indigo-400">•</span>
                                            {point}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>
                </section>
            )}

            {/* Summary Section */}
            {summarizer && !summarizer.parse_error && (
                <section>
                    <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                        <span className="text-xl">📝</span> Summary
                    </h3>
                    <div className="space-y-3">
                        {summarizer.executive_summary && (
                            <div className="p-4 bg-gradient-to-r from-indigo-500/10 to-purple-500/10 rounded-lg border border-indigo-500/20">
                                <span className="text-indigo-300 text-sm font-medium block mb-2">Executive Summary</span>
                                <p className="text-white/90">{summarizer.executive_summary}</p>
                            </div>
                        )}
                        {summarizer.key_takeaways && (
                            <div className="p-3 bg-white/5 rounded-lg">
                                <span className="text-white/50 text-sm block mb-2">Key Takeaways:</span>
                                <ul className="space-y-2">
                                    {summarizer.key_takeaways.slice(0, 5).map((item: string, i: number) => (
                                        <li key={i} className="text-white/80 text-sm flex gap-2">
                                            <span className="text-green-400">✓</span>
                                            {item}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>
                </section>
            )}

            {/* Recommendations Section */}
            {recommender && !recommender.parse_error && recommender.action_items && (
                <section>
                    <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                        <span className="text-xl">🎯</span> Action Items
                    </h3>
                    <div className="space-y-2">
                        {recommender.action_items.slice(0, 5).map((item: any, i: number) => (
                            <div
                                key={i}
                                className={`p-3 rounded-lg border ${item.priority === "high"
                                        ? "bg-red-500/10 border-red-500/30"
                                        : item.priority === "medium"
                                            ? "bg-yellow-500/10 border-yellow-500/30"
                                            : "bg-white/5 border-white/10"
                                    }`}
                            >
                                <div className="flex items-start gap-3">
                                    <span
                                        className={`text-xs px-2 py-0.5 rounded ${item.priority === "high"
                                                ? "bg-red-500/30 text-red-300"
                                                : item.priority === "medium"
                                                    ? "bg-yellow-500/30 text-yellow-300"
                                                    : "bg-white/20 text-white/60"
                                            }`}
                                    >
                                        {item.priority}
                                    </span>
                                    <p className="text-white/90 text-sm flex-1">{item.action}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </section>
            )}

            {/* Quick Wins */}
            {recommender?.quick_wins && recommender.quick_wins.length > 0 && (
                <section>
                    <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                        <span className="text-xl">⚡</span> Quick Wins
                    </h3>
                    <div className="flex flex-wrap gap-2">
                        {recommender.quick_wins.map((win: string, i: number) => (
                            <span
                                key={i}
                                className="px-3 py-1 bg-green-500/20 text-green-300 rounded-lg text-sm"
                            >
                                {win}
                            </span>
                        ))}
                    </div>
                </section>
            )}
        </div>
    );
}
