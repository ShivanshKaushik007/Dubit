import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Dubit — AI Video Dubbing Studio",
  description: "GPU-accelerated transcription, speaker diarization, and dubbing workbench",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased dark">
      <body className="min-h-full bg-zinc-950 text-zinc-100">{children}</body>
    </html>
  );
}
