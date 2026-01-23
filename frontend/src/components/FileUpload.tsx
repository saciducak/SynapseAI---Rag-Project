"use client";

import { useCallback, useState } from "react";

interface FileUploadProps {
    mode: string;
    onUploadSuccess: (docId: string, filename: string) => void;
}

export default function FileUpload({ mode, onUploadSuccess }: FileUploadProps) {
    const [isDragging, setIsDragging] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleUpload = async (file: File) => {
        setIsUploading(true);
        setError(null);

        const formData = new FormData();
        formData.append("file", file);
        formData.append("mode", mode);

        try {
            const response = await fetch("/api/documents/upload", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || "Upload failed");
            }

            const result = await response.json();
            onUploadSuccess(result.document_id, result.filename);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Upload failed");
        } finally {
            setIsUploading(false);
        }
    };

    const handleDrop = useCallback(
        (e: React.DragEvent) => {
            e.preventDefault();
            setIsDragging(false);

            const file = e.dataTransfer.files[0];
            if (file) {
                handleUpload(file);
            }
        },
        [mode]
    );

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            handleUpload(file);
        }
    };

    return (
        <div>
            <div
                onDragOver={(e) => {
                    e.preventDefault();
                    setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${isDragging
                        ? "border-indigo-500 bg-indigo-500/10"
                        : "border-white/20 hover:border-white/40"
                    }`}
            >
                {isUploading ? (
                    <div className="flex flex-col items-center gap-3">
                        <span className="spinner" />
                        <p className="text-white/60">Processing document...</p>
                    </div>
                ) : (
                    <>
                        <div className="text-4xl mb-3">📁</div>
                        <p className="text-white/80 mb-2">
                            Drag & drop your file here
                        </p>
                        <p className="text-white/40 text-sm mb-4">
                            or click to browse
                        </p>
                        <input
                            type="file"
                            onChange={handleFileSelect}
                            accept=".pdf,.docx,.doc,.txt,.md,.py,.js,.ts,.jsx,.tsx"
                            className="hidden"
                            id="file-upload"
                        />
                        <label
                            htmlFor="file-upload"
                            className="inline-block px-4 py-2 bg-white/10 rounded-lg cursor-pointer hover:bg-white/20 transition-all"
                        >
                            Browse Files
                        </label>
                        <p className="text-white/30 text-xs mt-4">
                            Supports: PDF, DOCX, TXT, MD, Python, JavaScript
                        </p>
                    </>
                )}
            </div>
            {error && (
                <p className="text-red-400 text-sm mt-3">{error}</p>
            )}
        </div>
    );
}
