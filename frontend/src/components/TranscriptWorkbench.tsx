"use client";

import { useMemo, useRef, useState } from "react";
import { getDownloadUrl, getDubbedMediaUrl, getMediaUrl } from "@/lib/api";
import { formatTimestamp, speakerBadgeClass } from "@/lib/time";
import type {
  DubJobPayload,
  ExportFormat,
  HealthResponse,
  JobPatchPayload,
  JobResponse,
  TranscriptSegment,
} from "@/types/dubit";

interface TranscriptWorkbenchProps {
  job: JobResponse;
  health: HealthResponse | null;
  onSaveEdits: (jobId: string, payload: JobPatchPayload) => Promise<void>;
  onDubJob: (jobId: string, payload: DubJobPayload) => Promise<void>;
}

const EXPORT_LABELS: Record<ExportFormat, string> = {
  webvtt: "WebVTT (.vtt)",
  srt: "SubRip (.srt)",
  json: "JSON (.json)",
  csv: "CSV (.csv)",
  dubbed_vtt: "Dubbed VTT",
  dubbed_srt: "Dubbed SRT",
  dubbed_audio: "Dubbed WAV",
  dubbed_video: "Dubbed MP4",
  vocals: "Vocals Stem",
  background: "BGM Stem",
};

export function TranscriptWorkbench({
  job,
  health,
  onSaveEdits,
  onDubJob,
}: TranscriptWorkbenchProps) {
  const mediaRef = useRef<HTMLVideoElement | HTMLAudioElement | null>(null);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [draftEdits, setDraftEdits] = useState<Record<number, string>>({});
  const [draftTranslations, setDraftTranslations] = useState<Record<number, string>>({});
  const [speakerRenames, setSpeakerRenames] = useState<Record<string, string>>({});
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [saveNotice, setSaveNotice] = useState<string | null>(null);

  const hasDubbedOutput = job.has_dubbed_video || job.has_dubbed_audio;
  const [playerTrack, setPlayerTrack] = useState<"original" | "dubbed">(
    hasDubbedOutput ? "dubbed" : "original"
  );

  const translationLanguages = health?.translation_languages ?? {
    en: "English",
    es: "Spanish",
    fr: "French",
    de: "German",
    hi: "Hindi",
    ja: "Japanese",
  };
  const [dubTargetLang, setDubTargetLang] = useState<string>(
    job.target_language ?? (job.language === "en" ? "es" : "en")
  );
  const [preserveBackground, setPreserveBackground] = useState<boolean>(true);
  const [bgmVolume, setBgmVolume] = useState<number>(0.28);
  const [maxSpeed, setMaxSpeed] = useState<number>(1.35);
  const [voiceOverrides, setVoiceOverrides] = useState<Record<string, string>>(
    job.dubbed_voices ?? {}
  );
  const [isStartingDub, setIsStartingDub] = useState<boolean>(false);

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

  const speakerKeysForVoices = useMemo(
    () => (speakers.length > 0 ? speakers : ["DEFAULT"]),
    [speakers]
  );

  const availableVoices = useMemo(() => {
    const map = health?.dubbing_voices ?? {};
    return map[dubTargetLang] ?? map["en"] ?? ["en-US-JennyNeural", "en-US-GuyNeural"];
  }, [health, dubTargetLang]);

  const hasTranslatedSegments = useMemo(
    () => segments.some((seg) => typeof seg.translated_text === "string"),
    [segments]
  );

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
        const trans = (draftTranslations[index] ?? seg.translated_text ?? "").toLowerCase();
        const speaker = (seg.speaker ?? "").toLowerCase();
        return text.includes(query) || trans.includes(query) || speaker.includes(query);
      });
  }, [segments, searchQuery, draftEdits, draftTranslations]);

  const hasUnsavedChanges = useMemo(() => {
    const hasTextChanges = Object.entries(draftEdits).some(([idxStr, val]) => {
      const idx = Number(idxStr);
      return (segments[idx]?.text ?? "").trim() !== val.trim();
    });
    const hasTranslationChanges = Object.entries(draftTranslations).some(([idxStr, val]) => {
      const idx = Number(idxStr);
      return (segments[idx]?.translated_text ?? "").trim() !== val.trim();
    });
    const hasSpeakerChanges = Object.entries(speakerRenames).some(
      ([oldName, newName]) => newName.trim() !== "" && newName.trim() !== oldName
    );
    return hasTextChanges || hasTranslationChanges || hasSpeakerChanges;
  }, [draftEdits, draftTranslations, speakerRenames, segments]);

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
      const changedIndices = new Set<number>([
        ...Object.keys(draftEdits).map(Number),
        ...Object.keys(draftTranslations).map(Number),
      ]);

      const segmentUpdates = Array.from(changedIndices)
        .map((idx) => {
          const origSeg = segments[idx];
          const nextText = draftEdits[idx];
          const nextTrans = draftTranslations[idx];
          const textChanged =
            nextText !== undefined && (origSeg?.text ?? "").trim() !== nextText.trim();
          const transChanged =
            nextTrans !== undefined &&
            (origSeg?.translated_text ?? "").trim() !== nextTrans.trim();

          if (!textChanged && !transChanged) return null;
          return {
            index: idx,
            ...(textChanged ? { text: nextText } : {}),
            ...(transChanged ? { translated_text: nextTrans } : {}),
          };
        })
        .filter((item): item is NonNullable<typeof item> => item !== null);

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
      setDraftTranslations({});
      setSpeakerRenames({});
      setSaveNotice("Saved edits and updated export files.");
    } finally {
      setIsSaving(false);
    }
  };

  const triggerDub = async (skipTranslation: boolean) => {
    if (isStartingDub) return;
    if (hasUnsavedChanges) {
      await handleSave();
    }
    setIsStartingDub(true);
    try {
      await onDubJob(job.job_id, {
        target_language: dubTargetLang,
        preserve_background: preserveBackground,
        bgm_volume: bgmVolume,
        max_speech_speed: maxSpeed,
        speaker_voice_map: voiceOverrides,
        skip_translation: skipTranslation,
      });
    } finally {
      setIsStartingDub(false);
    }
  };

  const activeMediaSrc =
    playerTrack === "dubbed" && hasDubbedOutput
      ? getDubbedMediaUrl(job.job_id)
      : getMediaUrl(job.job_id);

  const isVideoPlayer =
    playerTrack === "dubbed" && hasDubbedOutput ? job.has_dubbed_video : job.has_video;

  const transcriptDownloads = job.available_formats.filter((fmt) =>
    ["webvtt", "srt", "json", "csv"].includes(fmt)
  );
  const dubbedDownloads = job.available_formats.filter((fmt) =>
    ["dubbed_video", "dubbed_audio", "dubbed_vtt", "dubbed_srt", "vocals", "background"].includes(fmt)
  );

  return (
    <div className="space-y-5">
      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-5 shadow-xl">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-zinc-800 pb-3">
          <div>
            <h2 className="text-base font-semibold text-zinc-100">
              3. Studio Workbench — {job.filename}
            </h2>
            <p className="text-xs text-zinc-400">
              Source: <span className="text-zinc-200">{job.language ?? "unknown"}</span>
              {job.target_language && (
                <>
                  {" → Dub: "}
                  <span className="text-indigo-300">{job.target_language}</span>
                </>
              )}
              {" · "}Model: <span className="text-zinc-200">{job.model_size}</span>
              {" · "}Segments: <span className="text-zinc-200">{segments.length}</span>
            </p>
          </div>

          {hasDubbedOutput && (
            <div className="inline-flex rounded-xl border border-zinc-700 bg-zinc-950 p-1 text-xs">
              <button
                type="button"
                onClick={() => setPlayerTrack("original")}
                className={`rounded-lg px-3 py-1 font-medium transition ${
                  playerTrack === "original"
                    ? "bg-zinc-800 text-white"
                    : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                Original
              </button>
              <button
                type="button"
                onClick={() => setPlayerTrack("dubbed")}
                className={`rounded-lg px-3 py-1 font-medium transition ${
                  playerTrack === "dubbed"
                    ? "bg-indigo-600 text-white"
                    : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                Dubbed ({job.target_language})
              </button>
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
          {isVideoPlayer ? (
            <video
              key={activeMediaSrc}
              ref={(el) => {
                mediaRef.current = el;
              }}
              src={activeMediaSrc}
              controls
              onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
              className="max-h-72 w-full bg-black"
            />
          ) : (
            <div className="p-4">
              <audio
                key={activeMediaSrc}
                ref={(el) => {
                  mediaRef.current = el;
                }}
                src={activeMediaSrc}
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
            {(transcriptDownloads.length > 0 || dubbedDownloads.length > 0) && (
              <div className="mb-4 space-y-2 rounded-xl border border-zinc-800 bg-zinc-950/60 p-3">
                {dubbedDownloads.length > 0 && (
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-[11px] font-semibold uppercase tracking-wide text-indigo-300">
                      Dubbed Outputs:
                    </span>
                    {dubbedDownloads.map((fmt) => (
                      <a
                        key={fmt}
                        href={getDownloadUrl(job.job_id, fmt)}
                        download
                        className="rounded-lg border border-indigo-500/40 bg-indigo-500/15 px-2.5 py-1 text-xs font-medium text-indigo-200 transition hover:bg-indigo-500/30"
                      >
                        {EXPORT_LABELS[fmt] ?? fmt}
                      </a>
                    ))}
                  </div>
                )}

                {transcriptDownloads.length > 0 && (
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-[11px] font-semibold uppercase tracking-wide text-zinc-400">
                      Transcripts:
                    </span>
                    {transcriptDownloads.map((fmt) => (
                      <a
                        key={fmt}
                        href={getDownloadUrl(job.job_id, fmt)}
                        download
                        className="rounded-lg border border-zinc-700 bg-zinc-800 px-2.5 py-1 text-xs font-medium text-zinc-100 transition hover:border-zinc-500"
                      >
                        {EXPORT_LABELS[fmt] ?? fmt}
                      </a>
                    ))}
                  </div>
                )}
              </div>
            )}

            <div className="mb-4 rounded-xl border border-indigo-500/30 bg-indigo-950/20 p-4">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h3 className="text-sm font-semibold text-indigo-200">
                    AI Dubbing Studio (Stems + Translation + Neural TTS)
                  </h3>
                  <p className="text-[11px] text-zinc-400">
                    Separates vocals/BGM with Demucs, translates via local M2M100, and mixes time-fitted speech.
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                <div>
                  <label className="mb-1 block text-xs text-zinc-300">
                    Target Language
                  </label>
                  <select
                    value={dubTargetLang}
                    onChange={(e) => {
                      setDubTargetLang(e.target.value);
                      setVoiceOverrides({});
                    }}
                    className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-100 focus:border-indigo-500 focus:outline-none"
                  >
                    {Object.entries(translationLanguages).map(([code, name]) => (
                      <option key={code} value={code}>
                        {name} ({code})
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="mb-1 block text-xs text-zinc-300">
                    BGM Level ({Math.round(bgmVolume * 100)}%)
                  </label>
                  <input
                    type="range"
                    min={0}
                    max={0.8}
                    step={0.05}
                    value={bgmVolume}
                    onChange={(e) => setBgmVolume(Number(e.target.value))}
                    className="mt-2 w-full accent-indigo-500"
                  />
                </div>

                <div>
                  <label className="mb-1 block text-xs text-zinc-300">
                    Max Timing Speed-Up ({maxSpeed.toFixed(2)}x)
                  </label>
                  <input
                    type="range"
                    min={1.0}
                    max={1.6}
                    step={0.05}
                    value={maxSpeed}
                    onChange={(e) => setMaxSpeed(Number(e.target.value))}
                    className="mt-2 w-full accent-indigo-500"
                  />
                </div>
              </div>

              <div className="mt-3 flex flex-wrap items-center gap-4">
                <label className="flex cursor-pointer items-center gap-2 text-xs text-zinc-200">
                  <input
                    type="checkbox"
                    checked={preserveBackground}
                    onChange={(e) => setPreserveBackground(e.target.checked)}
                    className="h-4 w-4 rounded border-zinc-600 accent-indigo-500"
                  />
                  Separate &amp; Duck Original BGM/SFX
                </label>
              </div>

              <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
                {speakerKeysForVoices.map((spk, idx) => (
                  <div key={spk} className="flex items-center gap-2">
                    <span className="w-24 truncate text-xs text-zinc-300">
                      {spk === "DEFAULT" ? "Voice" : spk}:
                    </span>
                    <select
                      value={
                        voiceOverrides[spk] ??
                        availableVoices[idx % availableVoices.length]
                      }
                      onChange={(e) =>
                        setVoiceOverrides((prev) => ({
                          ...prev,
                          [spk]: e.target.value,
                        }))
                      }
                      className="flex-1 rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-100 focus:border-indigo-500 focus:outline-none"
                    >
                      {availableVoices.map((voice) => (
                        <option key={voice} value={voice}>
                          {voice}
                        </option>
                      ))}
                    </select>
                  </div>
                ))}
              </div>

              <div className="mt-4 flex flex-wrap gap-2">
                <button
                  type="button"
                  disabled={isStartingDub || segments.length === 0}
                  onClick={() => void triggerDub(false)}
                  className="rounded-xl bg-indigo-600 px-4 py-2 text-xs font-semibold text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {isStartingDub
                    ? "Queueing Dub..."
                    : hasDubbedOutput
                      ? `Re-Translate & Dub (${dubTargetLang.toUpperCase()})`
                      : `Translate & Dub (${dubTargetLang.toUpperCase()})`}
                </button>

                {hasTranslatedSegments && (
                  <button
                    type="button"
                    disabled={isStartingDub || segments.length === 0}
                    onClick={() => void triggerDub(true)}
                    className="rounded-xl border border-indigo-500/50 bg-indigo-500/15 px-4 py-2 text-xs font-semibold text-indigo-200 transition hover:bg-indigo-500/25 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    Re-Synthesize Edited Translations
                  </button>
                )}
              </div>
            </div>

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
                placeholder="Filter segments, translations, or speakers..."
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
                  const origValue = draftEdits[index] ?? seg.text ?? "";
                  const transValue =
                    draftTranslations[index] ?? seg.translated_text ?? "";

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
                        value={origValue}
                        onChange={(e) => {
                          setSaveNotice(null);
                          setDraftEdits((prev) => ({
                            ...prev,
                            [index]: e.target.value,
                          }));
                        }}
                        className="w-full resize-y rounded-lg border border-transparent bg-transparent p-1 text-sm leading-relaxed text-zinc-100 focus:border-zinc-700 focus:bg-zinc-900 focus:outline-none"
                      />

                      {(typeof seg.translated_text === "string" ||
                        draftTranslations[index] !== undefined) && (
                        <div className="mt-1.5 border-t border-zinc-800/80 pt-1.5">
                          <span className="block px-1 text-[10px] font-semibold uppercase tracking-wider text-indigo-400">
                            Dubbed Translation ({job.target_language ?? dubTargetLang})
                          </span>
                          <textarea
                            rows={2}
                            value={transValue}
                            onChange={(e) => {
                              setSaveNotice(null);
                              setDraftTranslations((prev) => ({
                                ...prev,
                                [index]: e.target.value,
                              }));
                            }}
                            className="w-full resize-y rounded-lg border border-transparent bg-transparent p-1 text-sm leading-relaxed text-indigo-200 focus:border-indigo-500/50 focus:bg-zinc-900 focus:outline-none"
                          />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
