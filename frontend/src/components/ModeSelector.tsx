"use client";

type Mode = "document" | "code" | "research" | "legal";

interface ModeSelectorProps {
    selectedMode: Mode;
    onModeChange: (mode: Mode) => void;
}

const modes = [
    {
        id: "document" as Mode,
        name: "Document",
        icon: "📄",
        description: "General document analysis, summarization, action extraction",
    },
    {
        id: "code" as Mode,
        name: "Code Review",
        icon: "💻",
        description: "Code quality, bug detection, improvement suggestions",
    },
    {
        id: "research" as Mode,
        name: "Research",
        icon: "📚",
        description: "Academic paper analysis, methodology, citations",
    },
    {
        id: "legal" as Mode,
        name: "Legal",
        icon: "⚖️",
        description: "Contract analysis, risk assessment, obligations",
    },
];

export default function ModeSelector({
    selectedMode,
    onModeChange,
}: ModeSelectorProps) {
    return (
        <div className="grid grid-cols-2 gap-3">
            {modes.map((mode) => (
                <button
                    key={mode.id}
                    onClick={() => onModeChange(mode.id)}
                    className={`p-4 rounded-xl border transition-all text-left ${selectedMode === mode.id
                            ? "bg-indigo-500/20 border-indigo-500/50"
                            : "bg-white/5 border-white/10 hover:bg-white/10"
                        }`}
                >
                    <div className="text-2xl mb-2">{mode.icon}</div>
                    <div className="font-medium text-white/90">{mode.name}</div>
                    <div className="text-xs text-white/50 mt-1 line-clamp-2">
                        {mode.description}
                    </div>
                </button>
            ))}
        </div>
    );
}
