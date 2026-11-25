export function formatTimestamp(seconds?: number | null): string {
  if (seconds === undefined || seconds === null || Number.isNaN(seconds)) {
    return "00:00.0";
  }
  const safe = Math.max(0, seconds);
  const hours = Math.floor(safe / 3600);
  const mins = Math.floor((safe % 3600) / 60);
  const secs = Math.floor(safe % 60);
  const tenths = Math.floor((safe % 1) * 10);

  if (hours > 0) {
    return `${String(hours).padStart(2, "0")}:${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}.${tenths}`;
  }
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}.${tenths}`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  return `${mb.toFixed(1)} MB`;
}

const SPEAKER_BADGE_CLASSES = [
  "bg-sky-500/15 text-sky-300 border-sky-500/30",
  "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  "bg-amber-500/15 text-amber-300 border-amber-500/30",
  "bg-violet-500/15 text-violet-300 border-violet-500/30",
  "bg-rose-500/15 text-rose-300 border-rose-500/30",
  "bg-cyan-500/15 text-cyan-300 border-cyan-500/30",
];

export function speakerBadgeClass(speaker: string, speakers: string[]): string {
  const index = speakers.indexOf(speaker);
  const safeIndex = index >= 0 ? index : 0;
  return SPEAKER_BADGE_CLASSES[safeIndex % SPEAKER_BADGE_CLASSES.length];
}
