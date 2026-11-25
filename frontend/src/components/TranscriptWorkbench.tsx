"use client";

import { useMemo, useRef, useState } from "react";
import { getDownloadUrl, getMediaUrl } from "@/lib/api";
import { formatTimestamp, speakerBadgeClass } from "@/lib/time";
import type { ExportFormat, JobPatchPayload, JobResponse, TranscriptSegment } from "@/types/dubit";

interface TranscriptWorkbenchProps {
  job: JobResponse;
  onSaveEdits: (jobId: string, payload: JobPatchPayload) => Promise<void>;
}

const EXPORT_LABELS: Record<ExportFormat, string> = {
  webvtt: "WebVTT (.vtt)",
  srt: "SubRip (.srt)",
  json: "JSON (.json)",
  csv: "CSV (.csv)",
};

export function TranscriptWorkbench({ job, onSaveEdits }: TranscriptWorkbenchProps) {
  const mediaRef = useRef<HTMLVideoElement | HTMLAudioElement | null>(null);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [draftEdits, setDraftEdits] = useState<Record<number, string>>({});
  const [speakerRenames, setSpeakerRenames] = useState<Record<string, string>>({});
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [saveNotice, setSaveNotice] = useState<string | null>(null);

  const segments: TranscriptSegment[] = useMemo(
    () => job.data?.segments ?? [],
    [job.data]
  );

  const speakers = useMemo(() => {
    const unique = new Set<string>();
    for (const seg of segments) {
      if (seg.speaker) {
        unique.add(seg.speaker);
      }
    }
    return Array.from(unique);
  }, [segments]);

  const activeIndex = useMemo(() => {
    return segments.findIndex((seg) => {
      if (seg.start === undefined || seg.end === undefined) return false;
      return currentTime >= seg.start && currentTime <= seg.end;
    });
  }, [segments, currentTime]);

  const filteredEntries = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    return segments
      .map((seg, index) => ({ seg, index }))
      .filter(({ seg, index }) => {
        if (!query) return true;
        const text = (draftEdits[index] ?? seg.text ?? "").toLowerCase();
        const speaker = (seg.speaker ?? "").toLowerCase();
        return text.includes(query) || speaker.includes(query);
      });
  }, [segments, searchQuery, draftEdits]);

  const hasUnsavedChanges = useMemo(() => {
    const hasTextChanges = Object.entries(draftEdits).some(([idxStr, val]) => {
      const idx = Number(idxStr);
      return (segments[idx]?.text ?? "").trim() !== val.trim();
    });
    const hasSpeakerChanges = Object.entries(speakerRenames).some(
      ([oldName, newName]) => newName.trim() !== "" && newName.trim() !== oldName
    );
    return hasTextChanges || hasSpeakerChanges;
  }, [draftEdits, speakerRenames, segments]);

  const seekTo = (time?: number) => {
    if (time === undefined || !mediaRef.current) return;
    mediaRef.current.currentTime = time;
    void mediaRef.current.play().catch(() => {
      // Ignore autoplay blocks when user seeks before clicking play.
    });
  };

  const handleSave = async () => {
    if (!hasUnsavedChanges || isSaving) return;
    setIsSaving(true);
    setSaveNotice(null);

    try {
      const segmentUpdates = Object.entries(draftEdits)
        .filter(([idxStr, val]) => {
          const idx = Number(idxStr);
          return (segments[idx]?.text ?? "").trim() !== val.trim();
        })
        .map(([idxStr, text]) => ({
          index: Number(idxStr),
          text,
        }));

      const speakerMap: Record<string, string> = {};
      for (const [oldName, newName] of Object.entries(speakerRenames)) {
        const trimmed = newName.trim();
        if (trimmed && trimmed !== oldName) {
          speakerMap[oldName] = trimmed;
        }
      }

      await onSaveEdits(job.job_id, {
        segments: segmentUpdates,
        speaker_map: speakerMap,
      });
      setDraftEdits({});
      setSpeakerRenames({});
      setSaveNotice("Saved transcript edits and regenerated export files.");
    } finally {
      setIsSaving(false);
    }
  };

  const mediaUrl = getMediaUrl(job.job_id);

  return (
    <div className="space-y-5">
      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-5 shadow-xl">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-zinc-800 pb-3">
          <div>
            <h2 className="text-base font-semibold text-zinc-100">
              3. Studio Workbench — {job.filename}
            </h2>
            <p className="text-xs text-zinc-400">
              Language: <span className="text-zinc-200">{job.language ?? "unknown"}</span>
              {" · "}Model: <span className="text-zinc-200">{job.model_size}</span>
              {" · "}Segments: <span className="text-zinc-200">{segments.length}</span>
              {job.data?.metadata?.device ? ` · ${job.data.metadata.device} (${job.data.metadata.compute_type})` : ""}
            </p>
          </div>

          {job.status === "completed" && (
            <div className="flex flex-wrap items-center gap-2">
              {job.available_formats.map((fmt) => (
                <a
                  key={fmt}
                  href={getDownloadUrl(job.job_id, fmt)}
                  download
                  className="rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-xs font-medium text-zinc-100 transition hover:border-indigo-400 hover:bg-indigo-500/20"
                >
                  Download {EXPORT_LABELS[fmt] ?? fmt.toUpperCase()}
                </a>
              ))}
            </div>
          )}
        </div>

        {job.warnings.length > 0 && (
          <div className="mb-4 rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-200">
            <p className="font-semibold">Pipeline Notes</p>
            <ul className="mt-1 list-disc space-y-0.5 pl-4">
              {job.warnings.map((warning, i) => (
                <li key={i}>{warning}</li>
              ))}
            </ul>
          </div>
        )}

        <div className="mb-4 overflow-hidden rounded-xl border border-zinc-800 bg-black">
          {job.has_video ? (
            <video
              ref={(el) => {
                mediaRef.current = el;
              }}
              src={mediaUrl}
              controls
              onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
              className="max-h-72 w-full bg-black"
            />
          ) : (
            <div className="p-4">
              <audio
                ref={(el) => {
                  mediaRef.current = el;
                }}
                src={mediaUrl}
                controls
                onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
                className="w-full"
              />
            </div>
          )}
        </div>

        {job.status !== "completed" ? (
          <div className="rounded-xl border border-zinc-800 bg-zinc-950/60 p-6 text-center">
            {job.status === "failed" ? (
              <p className="text-sm text-rose-400">{job.error ?? "Job failed."}</p>
            ) : (
              <div className="space-y-2">
                <p className="text-sm font-medium text-indigo-300">{job.message}</p>
                <div className="mx-auto h-2 max-w-md overflow-hidden rounded-full bg-zinc-800">
                  <div
                    className="h-full bg-indigo-500 transition-all duration-300"
                    style={{ width: `${Math.max(5, job.progress)}%` }}
                  />
                </div>
                <p className="text-xs text-zinc-500">{Math.round(job.progress)}% complete</p>
              </div>
            )}
          </div>
        ) : (
          <>
            {speakers.length > 0 && (
              <div className="mb-4 rounded-xl border border-zinc-800 bg-zinc-950/60 p-3">
                <p className="mb-2 text-xs font-medium text-zinc-300">
                  Rename Diarized Speakers
                </p>
                <div className="flex flex-wrap gap-2">
                  {speakers.map((spk) => (
                    <div
                      key={spk}
                      className="flex items-center gap-1.5 rounded-lg border border-zinc-800 bg-zinc-900 px-2 py-1"
                    >
                      <span
                        className={`rounded border px-1.5 py-0.5 text-[11px] font-medium ${speakerBadgeClass(spk, speakers)}`}
                      >
                        {spk}
                      </span>
                      <span className="text-xs text-zinc-500">→</span>
                      <input
                        type="text"
                        placeholder="Custom name"
                        value={speakerRenames[spk] ?? ""}
                        onChange={(e) =>
                          setSpeakerRenames((prev) => ({
                            ...prev,
                            [spk]: e.target.value,
                          }))
                        }
                        className="w-28 bg-transparent text-xs text-zinc-100 placeholder-zinc-600 focus:outline-none"
                      />
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <input
                type="search"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Filter segments by text or speaker..."
                className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-sm text-zinc-100 placeholder-zinc-500 focus:border-indigo-500 focus:outline-none sm:max-w-xs"
              />

              <div className="flex items-center gap-2">
                {saveNotice && (
                  <span className="text-xs text-emerald-400">{saveNotice}</span>
                )}
                <button
                  type="button"
                  disabled={!hasUnsavedChanges || isSaving}
                  onClick={handleSave}
                  className="rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {isSaving ? "Saving..." : "Save Edits & Rebuild Exports"}
                </button>
              </div>
            </div>

            {filteredEntries.length === 0 ? (
              <p className="py-8 text-center text-sm text-zinc-500">
                No matching transcript segments.
              </p>
            ) : (
              <div className="max-h-[480px] space-y-2 overflow-y-auto pr-1">
                {filteredEntries.map(({ seg, index }) => {
                  const isActive = index === activeIndex;
                  const value = draftEdits[index] ?? seg.text ?? "";

                  return (
                    <div
                      key={index}
                      className={`rounded-xl border p-3 transition ${
                        isActive
                          ? "border-indigo-500 bg-indigo-500/10"
                          : "border-zinc-800 bg-zinc-950/60"
                      }`}
                    >
                      <div className="mb-1.5 flex flex-wrap items-center justify-between gap-2">
                        <button
                          type="button"
                          onClick={() => seekTo(seg.start)}
                          className="rounded-md bg-zinc-800 px-2 py-0.5 font-mono text-[11px] text-indigo-300 hover:bg-indigo-600 hover:text-white"
                        >
                          {formatTimestamp(seg.start)} → {formatTimestamp(seg.end)}
                        </button>

                        {seg.speaker && (
                          <span
                            className={`rounded border px-2 py-0.5 text-[11px] font-medium ${speakerBadgeClass(seg.speaker, speakers)}`}
                          >
                            {seg.speaker}
                          </span>
                        )}
                      </div>

                      <textarea
                        rows={2}
                        value={value}
                        onChange={(e) => {
                          setSaveNotice(null);
                          setDraftEdits((prev) => ({
                            ...prev,
                            [index]: e.target.value,
                          }));
                        }}
                        className="w-full resize-y rounded-lg border border-transparent bg-transparent p-1 text-sm leading-relaxed text-zinc-100 focus:border-zinc-700 focus:bg-zinc-900 focus:outline-none"
                      />
                    </div>
                  );
                })}
              </div>
            )}
          </>
        )}
      </div>

      <div className="rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-4">
        <p className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
          Upcoming Dubbing Pipeline Stages (Phases 3–4)
        </p>
        <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-zinc-400 sm:grid-cols-4">
          <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-2 text-emerald-300">
            1. ASR + Alignment + Diarization (Ready)
          </div>
          <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-2">
            2. Vocal / BGM Stem Separation (Planned)
          </div>
          <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-2">
            3. Timing-Aware Translation (Planned)
          </div>
          <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-2">
            4. Voice Cloning &amp; Video Mux (Planned)
          </div>
        </div>
      </div>
    </div>
  );
}
