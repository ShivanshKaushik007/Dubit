"use client";

import { formatTimestamp } from "@/lib/time";
import type { JobStatus, JobSummary } from "@/types/dubit";

interface JobQueueListProps {
  jobs: JobSummary[];
  selectedJobId: string | null;
  onSelect: (jobId: string) => void;
  onDelete: (jobId: string) => void;
}

const STATUS_STYLE: Record<JobStatus, string> = {
  queued: "bg-zinc-700/60 text-zinc-200",
  extracting: "bg-sky-500/20 text-sky-300",
  transcribing: "bg-indigo-500/20 text-indigo-300",
  aligning: "bg-violet-500/20 text-violet-300",
  diarizing: "bg-amber-500/20 text-amber-300",
  formatting: "bg-teal-500/20 text-teal-300",
  separating: "bg-fuchsia-500/20 text-fuchsia-300",
  translating: "bg-cyan-500/20 text-cyan-300",
  synthesizing: "bg-purple-500/20 text-purple-300",
  mixing: "bg-lime-500/20 text-lime-300",
  completed: "bg-emerald-500/20 text-emerald-300",
  failed: "bg-rose-500/20 text-rose-300",
};

const ACTIVE_STATUSES = new Set<JobStatus>([
  "queued",
  "extracting",
  "transcribing",
  "aligning",
  "diarizing",
  "formatting",
  "separating",
  "translating",
  "synthesizing",
  "mixing",
]);

export function JobQueueList({
  jobs,
  selectedJobId,
  onSelect,
  onDelete,
}: JobQueueListProps) {
  return (
    <section className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-5 shadow-xl">
      <div className="mb-3 flex items-center justify-between border-b border-zinc-800 pb-3">
        <h2 className="text-base font-semibold text-zinc-100">
          2. Job Queue ({jobs.length})
        </h2>
        <span className="text-xs text-zinc-500">Persisted locally</span>
      </div>

      {jobs.length === 0 ? (
        <p className="py-6 text-center text-xs text-zinc-500">
          No jobs yet. Upload an audio or video file to start.
        </p>
      ) : (
        <div className="max-h-96 space-y-2.5 overflow-y-auto pr-1">
          {jobs.map((job) => {
            const isSelected = job.job_id === selectedJobId;
            const isActive = ACTIVE_STATUSES.has(job.status);

            return (
              <div
                key={job.job_id}
                onClick={() => onSelect(job.job_id)}
                className={`cursor-pointer rounded-xl border p-3 transition ${
                  isSelected
                    ? "border-indigo-500/70 bg-indigo-500/10"
                    : "border-zinc-800 bg-zinc-950/60 hover:border-zinc-700"
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-zinc-100">
                      {job.filename}
                    </p>
                    <p className="mt-0.5 text-[11px] text-zinc-400">
                      {job.model_size}
                      {job.language ? ` · ${job.language}` : ""}
                      {job.target_language ? ` → ${job.target_language}` : ""}
                      {job.duration_seconds !== null
                        ? ` · ${formatTimestamp(job.duration_seconds)}`
                        : ""}
                      {job.segment_count > 0 ? ` · ${job.segment_count} segs` : ""}
                    </p>
                  </div>

                  <div className="flex items-center gap-1.5">
                    {(job.has_dubbed_video || job.has_dubbed_audio) && (
                      <span className="rounded-full bg-indigo-500/20 px-2 py-0.5 text-[10px] font-medium text-indigo-300">
                        DUB
                      </span>
                    )}
                    <span
                      className={`rounded-full px-2 py-0.5 text-[11px] font-medium capitalize ${STATUS_STYLE[job.status]}`}
                    >
                      {job.status}
                    </span>
                    {!isActive && (
                      <button
                        type="button"
                        title="Delete job"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDelete(job.job_id);
                        }}
                        className="rounded-md px-1.5 py-0.5 text-xs text-zinc-500 hover:bg-zinc-800 hover:text-rose-400"
                      >
                        ✕
                      </button>
                    )}
                  </div>
                </div>

                {isActive && (
                  <div className="mt-2">
                    <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
                      <div
                        className="h-full bg-indigo-500 transition-all duration-300"
                        style={{ width: `${Math.max(4, job.progress)}%` }}
                      />
                    </div>
                    <p className="mt-1 truncate text-[11px] text-zinc-400">
                      {job.message} ({Math.round(job.progress)}%)
                    </p>
                  </div>
                )}

                {job.status === "failed" && job.error && (
                  <p className="mt-1.5 line-clamp-2 text-[11px] text-rose-400">
                    {job.error}
                  </p>
                )}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
