"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { JobQueueList } from "@/components/JobQueueList";
import { TranscriptWorkbench } from "@/components/TranscriptWorkbench";
import { UploadPanel } from "@/components/UploadPanel";
import {
  createJob,
  deleteJob,
  getHealth,
  getJob,
  listJobs,
  updateJobTranscript,
} from "@/lib/api";
import type {
  CreateJobOptions,
  HealthResponse,
  JobPatchPayload,
  JobResponse,
  JobStatus,
  JobSummary,
} from "@/types/dubit";

const ACTIVE_STATUSES = new Set<JobStatus>([
  "queued",
  "extracting",
  "transcribing",
  "aligning",
  "diarizing",
  "formatting",
]);

export default function StudioPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<JobResponse | null>(null);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [bannerError, setBannerError] = useState<string | null>(null);

  const refreshAll = useCallback(async (focusJobId?: string | null) => {
    try {
      const [healthRes, jobsRes] = await Promise.all([getHealth(), listJobs()]);
      setHealth(healthRes);
      setJobs(jobsRes);
      setBannerError(null);

      const targetId = focusJobId ?? jobsRes[0]?.job_id ?? null;
      if (targetId) {
        setSelectedJobId(targetId);
        const detail = await getJob(targetId);
        setSelectedJob(detail);
      } else {
        setSelectedJobId(null);
        setSelectedJob(null);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to reach backend API.";
      setBannerError(msg);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    const init = async () => {
      try {
        const [healthRes, jobsRes] = await Promise.all([getHealth(), listJobs()]);
        if (cancelled) return;
        setHealth(healthRes);
        setJobs(jobsRes);
        if (jobsRes[0]?.job_id) {
          setSelectedJobId(jobsRes[0].job_id);
          const detail = await getJob(jobsRes[0].job_id);
          if (!cancelled) {
            setSelectedJob(detail);
          }
        }
      } catch (err: unknown) {
        if (!cancelled) {
          setBannerError(
            err instanceof Error ? err.message : "Backend API unavailable."
          );
        }
      }
    };
    void init();
    return () => {
      cancelled = true;
    };
  }, []);

  const hasActiveJobs = useMemo(
    () =>
      jobs.some((j) => ACTIVE_STATUSES.has(j.status)) ||
      (selectedJob !== null && ACTIVE_STATUSES.has(selectedJob.status)),
    [jobs, selectedJob]
  );

  useEffect(() => {
    if (!hasActiveJobs) return;
    const timer = window.setInterval(() => {
      void refreshAll(selectedJobId);
    }, 1500);
    return () => window.clearInterval(timer);
  }, [hasActiveJobs, refreshAll, selectedJobId]);

  const handleCreateJob = async (options: CreateJobOptions) => {
    setIsSubmitting(true);
    setBannerError(null);
    try {
      const created = await createJob(options);
      setSelectedJobId(created.job_id);
      setSelectedJob(created);
      await refreshAll(created.job_id);
    } catch (err: unknown) {
      setBannerError(
        err instanceof Error ? err.message : "Failed to queue job."
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleSelectJob = async (jobId: string) => {
    setSelectedJobId(jobId);
    setBannerError(null);
    try {
      const detail = await getJob(jobId);
      setSelectedJob(detail);
    } catch (err: unknown) {
      setBannerError(
        err instanceof Error ? err.message : "Failed to load job details."
      );
    }
  };

  const handleSaveEdits = async (jobId: string, payload: JobPatchPayload) => {
    setBannerError(null);
    try {
      const updated = await updateJobTranscript(jobId, payload);
      setSelectedJob(updated);
      const jobsRes = await listJobs();
      setJobs(jobsRes);
    } catch (err: unknown) {
      setBannerError(
        err instanceof Error ? err.message : "Failed to save transcript edits."
      );
    }
  };

  const handleDeleteJob = async (jobId: string) => {
    setBannerError(null);
    try {
      await deleteJob(jobId);
      const remaining = jobs.filter((j) => j.job_id !== jobId);
      setJobs(remaining);
      if (selectedJobId === jobId) {
        const nextId = remaining[0]?.job_id ?? null;
        await refreshAll(nextId);
      }
    } catch (err: unknown) {
      setBannerError(
        err instanceof Error ? err.message : "Failed to delete job."
      );
    }
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <header className="border-b border-zinc-800/80 bg-zinc-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-4 sm:px-6">
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white">
              Dubit <span className="font-normal text-indigo-400">Studio</span>
            </h1>
            <p className="text-xs text-zinc-400">
              AI Transcription, Speaker Diarization &amp; Dubbing Workbench
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs">
            {health ? (
              <>
                <span className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-emerald-300">
                  API Online
                </span>
                <span className="rounded-full border border-zinc-700 bg-zinc-900 px-3 py-1 text-zinc-300">
                  {health.gpu.cuda_available
                    ? `${health.gpu.device_name} (${health.gpu.total_vram_gb} GB)`
                    : "CPU Mode"}
                </span>
              </>
            ) : (
              <span className="rounded-full border border-zinc-700 bg-zinc-900 px-3 py-1 text-zinc-400">
                Connecting to API...
              </span>
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
        {bannerError && (
          <div className="mb-5 flex items-center justify-between rounded-xl border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
            <span>{bannerError}</span>
            <button
              type="button"
              onClick={() => setBannerError(null)}
              className="ml-4 text-xs text-rose-200 underline"
            >
              Dismiss
            </button>
          </div>
        )}

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
          <div className="space-y-6 lg:col-span-5">
            <UploadPanel
              health={health}
              isSubmitting={isSubmitting}
              onSubmit={handleCreateJob}
            />
            <JobQueueList
              jobs={jobs}
              selectedJobId={selectedJobId}
              onSelect={handleSelectJob}
              onDelete={handleDeleteJob}
            />
          </div>

          <div className="lg:col-span-7">
            {selectedJob ? (
              <TranscriptWorkbench
                key={selectedJob.job_id}
                job={selectedJob}
                onSaveEdits={handleSaveEdits}
              />
            ) : (
              <div className="flex h-full min-h-[360px] flex-col items-center justify-center rounded-2xl border border-dashed border-zinc-800 bg-zinc-900/30 p-8 text-center">
                <p className="text-sm font-medium text-zinc-300">
                  Select a job or upload a file to inspect the transcript
                </p>
                <p className="mt-1 max-w-md text-xs text-zinc-500">
                  Completed jobs open an interactive audio/video player synced to
                  editable word/segment transcripts and one-click VTT, SRT, JSON,
                  and CSV downloads.
                </p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
