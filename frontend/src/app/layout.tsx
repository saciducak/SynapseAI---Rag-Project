import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
    title: "SynapseAI - Multi-Agent Decision Support",
    description: "Multi-Mode Multi-Agent Decision Support System with RAG",
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en">
            <body className="antialiased">
                {/* Background gradient effect */}
                <div className="fixed inset-0 -z-10">
                    <div className="absolute top-0 -left-40 w-80 h-80 bg-purple-600/20 rounded-full blur-[100px]" />
                    <div className="absolute top-40 right-0 w-96 h-96 bg-blue-600/20 rounded-full blur-[120px]" />
                    <div className="absolute bottom-20 left-1/3 w-72 h-72 bg-indigo-600/15 rounded-full blur-[100px]" />
                </div>
                {children}
            </body>
        </html>
    );
}
