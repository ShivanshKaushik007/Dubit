import type {
  CreateJobOptions,
  DubJobPayload,
  ExportFormat,
  HealthResponse,
  JobPatchPayload,
  JobResponse,
  JobSummary,
} from "@/types/dubit";

export const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/+$/, "");

async function parseErrorDetail(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: unknown; message?: unknown };
    if (typeof body.detail === "string" && body.detail.trim()) {
      return body.detail;
    }
    if (typeof body.message === "string" && body.message.trim()) {
      return body.message;
    }
  } catch {
    // Fallback to HTTP status text when response is not JSON.
  }
  return `Request failed with HTTP ${res.status} (${res.statusText || "Error"})`;
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res));
  }
  return (await res.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>(`${API_BASE_URL}/health`, {
    cache: "no-store",
  });
}

export function listJobs(): Promise<JobSummary[]> {
  return requestJson<JobSummary[]>(`${API_BASE_URL}/jobs`, {
    cache: "no-store",
  });
}

export function getJob(jobId: string): Promise<JobResponse> {
  return requestJson<JobResponse>(`${API_BASE_URL}/jobs/${encodeURIComponent(jobId)}`, {
    cache: "no-store",
  });
}

export function createJob(options: CreateJobOptions): Promise<JobResponse> {
  const formData = new FormData();
  formData.append("file", options.file);
  formData.append("output_format", options.outputFormat);
  formData.append("model_size", options.modelSize);
  formData.append("compute_type", options.computeType);
  formData.append("do_diarization", String(options.doDiarization));
  formData.append("preserve_background", String(options.preserveBackground));

  if (options.language.trim()) {
    formData.append("language", options.language.trim());
  }
  if (options.targetLanguage.trim()) {
    formData.append("target_language", options.targetLanguage.trim());
  }
  if (options.doDiarization && options.minSpeakers.trim()) {
    formData.append("min_speakers", options.minSpeakers.trim());
  }
  if (options.doDiarization && options.maxSpeakers.trim()) {
    formData.append("max_speakers", options.maxSpeakers.trim());
  }

  return requestJson<JobResponse>(`${API_BASE_URL}/jobs`, {
    method: "POST",
    body: formData,
  });
}

export function updateJobTranscript(
  jobId: string,
  payload: JobPatchPayload
): Promise<JobResponse> {
  return requestJson<JobResponse>(`${API_BASE_URL}/jobs/${encodeURIComponent(jobId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function dubJob(
  jobId: string,
  payload: DubJobPayload
): Promise<JobResponse> {
  return requestJson<JobResponse>(`${API_BASE_URL}/jobs/${encodeURIComponent(jobId)}/dub`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function deleteJob(jobId: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/jobs/${encodeURIComponent(jobId)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    throw new Error(await parseErrorDetail(res));
  }
}

export function getDownloadUrl(jobId: string, format: ExportFormat): string {
  return `${API_BASE_URL}/jobs/${encodeURIComponent(jobId)}/download/${encodeURIComponent(format)}`;
}

export function getMediaUrl(jobId: string): string {
  return `${API_BASE_URL}/jobs/${encodeURIComponent(jobId)}/media`;
}

export function getDubbedMediaUrl(jobId: string): string {
  return `${API_BASE_URL}/jobs/${encodeURIComponent(jobId)}/dubbed`;
}
