"use client";

import { useRef, useState } from "react";
import { formatBytes } from "@/lib/time";
import type {
  ComputeType,
  CreateJobOptions,
  ExportFormat,
  HealthResponse,
  TTSEngine,
  WhisperModelSize,
} from "@/types/dubit";

interface UploadPanelProps {
  health: HealthResponse | null;
  isSubmitting: boolean;
  onSubmit: (options: CreateJobOptions) => Promise<void>;
}

const MODEL_NOTES: Record<WhisperModelSize, string> = {
  tiny: "Fastest (~1 GB VRAM)",
  base: "Balanced speed & accuracy (~1.2 GB VRAM)",
  small: "Recommended for 6 GB GPU (~2.2 GB VRAM)",
  medium: "Higher accuracy (~4.5 GB VRAM with int8_float16)",
  "large-v3": "Best accuracy (tight on 6 GB VRAM; use int8_float16)",
  "large-v3-turbo": "Fast large model (~4–5 GB VRAM with int8_float16)",
};

const SOURCE_LANGUAGES = [
  { code: "", label: "Auto-detect" },
  { code: "en", label: "English (en)" },
  { code: "es", label: "Spanish (es)" },
  { code: "fr", label: "French (fr)" },
  { code: "de", label: "German (de)" },
  { code: "hi", label: "Hindi (hi)" },
  { code: "ja", label: "Japanese (ja)" },
  { code: "ko", label: "Korean (ko)" },
  { code: "zh", label: "Chinese (zh)" },
  { code: "pt", label: "Portuguese (pt)" },
  { code: "ar", label: "Arabic (ar)" },
];

export function UploadPanel({ health, isSubmitting, onSubmit }: UploadPanelProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [modelSize, setModelSize] = useState<WhisperModelSize>("base");
  const [computeType, setComputeType] = useState<ComputeType>("auto");
  const [language, setLanguage] = useState<string>("");
  const [outputFormat, setOutputFormat] = useState<ExportFormat>("webvtt");
  const [doDiarization, setDoDiarization] = useState<boolean>(false);
  const [minSpeakers, setMinSpeakers] = useState<string>("");
  const [maxSpeakers, setMaxSpeakers] = useState<string>("");
  const [targetLanguage, setTargetLanguage] = useState<string>("");
  const [preserveBackground, setPreserveBackground] = useState<boolean>(true);
  const [ttsEngine, setTtsEngine] = useState<TTSEngine>("auto");

  const models = health?.supported_models ?? [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "large-v3-turbo",
  ];
  const computeTypes = health?.supported_compute_types ?? [
    "auto",
    "int8_float16",
    "float16",
    "int8",
    "float32",
  ];
  const translationLanguages = health?.translation_languages ?? {
    en: "English",
    es: "Spanish",
    fr: "French",
    de: "German",
    hi: "Hindi",
    ja: "Japanese",
  };

  const handleFileSelected = (selected: File | undefined) => {
    if (selected) {
      setFile(selected);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || isSubmitting) return;
    await onSubmit({
      file,
      outputFormat,
      modelSize,
      computeType,
      language,
      doDiarization,
      minSpeakers,
      maxSpeakers,
      targetLanguage,
      preserveBackground,
      ttsEngine,
    });
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-5 shadow-xl"
    >
      <div className="mb-4 flex items-center justify-between border-b border-zinc-800 pb-3">
        <div>
          <h2 className="text-base font-semibold text-zinc-100">
            1. New Studio Job
          </h2>
          <p className="text-xs text-zinc-400">
            Transcribe first to review, or pick a target language to auto-dub.
          </p>
        </div>
        {health && (
          <span className="rounded-full border border-zinc-700 bg-zinc-800/80 px-2.5 py-1 text-[11px] text-zinc-300">
            Max {health.max_upload_mb} MB
          </span>
        )}
      </div>

      <div
        role="button"
        tabIndex={0}
        onClick={() => fileInputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            fileInputRef.current?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          handleFileSelected(e.dataTransfer.files?.[0]);
        }}
        className={`mb-4 cursor-pointer rounded-xl border-2 border-dashed p-5 text-center transition ${
          isDragging
            ? "border-indigo-400 bg-indigo-500/10"
            : "border-zinc-700 bg-zinc-950/60 hover:border-zinc-500"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*,audio/*"
          onChange={(e) => handleFileSelected(e.target.files?.[0])}
          className="hidden"
        />
        {file ? (
          <div className="space-y-1">
            <p className="truncate text-sm font-medium text-indigo-300">
              {file.name}
            </p>
            <p className="text-xs text-zinc-400">
              {formatBytes(file.size)} · Click or drop to replace
            </p>
          </div>
        ) : (
          <div className="space-y-1">
            <p className="text-sm font-medium text-zinc-200">
              Drop video/audio here or browse
            </p>
            <p className="text-xs text-zinc-500">
              MP4, MKV, MOV, WebM, MP3, WAV, M4A, FLAC, MPEG
            </p>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div>
          <label className="mb-1 block text-xs font-medium text-zinc-300">
            Whisper Model
          </label>
          <select
            value={modelSize}
            onChange={(e) => setModelSize(e.target.value as WhisperModelSize)}
            className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-indigo-500 focus:outline-none"
          >
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
          <p className="mt-1 text-[11px] text-zinc-500">{MODEL_NOTES[modelSize]}</p>
        </div>

        <div>
          <label className="mb-1 block text-xs font-medium text-zinc-300">
            Compute Precision
          </label>
          <select
            value={computeType}
            onChange={(e) => setComputeType(e.target.value as ComputeType)}
            className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-indigo-500 focus:outline-none"
          >
            {computeTypes.map((c) => (
              <option key={c} value={c}>
                {c === "auto" ? "auto (int8_float16 on CUDA)" : c}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="mb-1 block text-xs font-medium text-zinc-300">
            Source Language
          </label>
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-indigo-500 focus:outline-none"
          >
            {SOURCE_LANGUAGES.map((lang) => (
              <option key={lang.code || "auto"} value={lang.code}>
                {lang.label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="mb-1 block text-xs font-medium text-zinc-300">
            Auto-Dub Target (Optional)
          </label>
          <select
            value={targetLanguage}
            onChange={(e) => setTargetLanguage(e.target.value)}
            className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-indigo-500 focus:outline-none"
          >
            <option value="">Transcribe only (dub in workbench)</option>
            {Object.entries(translationLanguages).map(([code, name]) => (
              <option key={code} value={code}>
                {name} ({code})
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div>
          <label className="mb-1 block text-xs font-medium text-zinc-300">
            Primary Export Format
          </label>
          <select
            value={outputFormat}
            onChange={(e) => setOutputFormat(e.target.value as ExportFormat)}
            className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-indigo-500 focus:outline-none"
          >
            <option value="webvtt">WebVTT (.vtt)</option>
            <option value="srt">SubRip (.srt)</option>
            <option value="json">JSON (word timestamps)</option>
            <option value="csv">CSV</option>
          </select>
        </div>

        {targetLanguage && (
          <div className="space-y-2">
            <div>
              <label className="mb-1 block text-xs font-medium text-zinc-300">
                Dubbing Voice Engine
              </label>
              <select
                value={ttsEngine}
                onChange={(e) => setTtsEngine(e.target.value as TTSEngine)}
                className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-indigo-500 focus:outline-none"
              >
                <option value="auto">
                  Auto ({health?.cloning_available ? "Chatterbox Cloning if supported" : "Edge-TTS"})
                </option>
                <option value="chatterbox" disabled={!health?.cloning_available}>
                  Chatterbox Voice Cloning
                </option>
                <option value="edge_tts">Edge-TTS Neural Voices</option>
              </select>
            </div>
            <label className="flex cursor-pointer items-center gap-2 text-xs text-zinc-300">
              <input
                type="checkbox"
                checked={preserveBackground}
                onChange={(e) => setPreserveBackground(e.target.checked)}
                className="h-4 w-4 rounded border-zinc-600 accent-indigo-500"
              />
              Preserve BGM/SFX (Demucs)
            </label>
          </div>
        )}
      </div>

      <div className="mt-4 rounded-xl border border-zinc-800 bg-zinc-950/60 p-3">
        <div className="flex items-center justify-between gap-2">
          <label htmlFor="diarize" className="flex cursor-pointer items-center gap-2 text-sm text-zinc-200">
            <input
              id="diarize"
              type="checkbox"
              checked={doDiarization}
              onChange={(e) => setDoDiarization(e.target.checked)}
              className="h-4 w-4 rounded border-zinc-600 accent-indigo-500"
            />
            Speaker Diarization
          </label>
          {health && (
            <span
              className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${
                health.hf_token_configured
                  ? "bg-emerald-500/15 text-emerald-300"
                  : "bg-amber-500/15 text-amber-300"
              }`}
            >
              {health.hf_token_configured ? "HF_TOKEN ready" : "HF_TOKEN not set"}
            </span>
          )}
        </div>

        {doDiarization && (
          <div className="mt-3 grid grid-cols-2 gap-3">
            <div>
              <label className="mb-1 block text-[11px] text-zinc-400">
                Min Speakers (optional)
              </label>
              <input
                type="number"
                min={1}
                max={20}
                placeholder="e.g. 1"
                value={minSpeakers}
                onChange={(e) => setMinSpeakers(e.target.value)}
                className="w-full rounded-lg border border-zinc-700 bg-zinc-900 px-2.5 py-1.5 text-sm text-zinc-100 focus:border-indigo-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="mb-1 block text-[11px] text-zinc-400">
                Max Speakers (optional)
              </label>
              <input
                type="number"
                min={1}
                max={20}
                placeholder="e.g. 6"
                value={maxSpeakers}
                onChange={(e) => setMaxSpeakers(e.target.value)}
                className="w-full rounded-lg border border-zinc-700 bg-zinc-900 px-2.5 py-1.5 text-sm text-zinc-100 focus:border-indigo-500 focus:outline-none"
              />
            </div>
          </div>
        )}
      </div>

      <button
        type="submit"
        disabled={!file || isSubmitting}
        className="mt-4 w-full rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {isSubmitting
          ? "Uploading & Queueing..."
          : targetLanguage
            ? `Queue Transcribe + Dub (${targetLanguage.toUpperCase()})`
            : "Queue Transcription Job"}
      </button>
    </form>
  );
}
