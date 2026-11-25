export type JobStatus =
  | "queued"
  | "extracting"
  | "transcribing"
  | "aligning"
  | "diarizing"
  | "formatting"
  | "completed"
  | "failed";

export type WhisperModelSize =
  | "tiny"
  | "base"
  | "small"
  | "medium"
  | "large-v3"
  | "large-v3-turbo";

export type ComputeType =
  | "auto"
  | "int8_float16"
  | "float16"
  | "int8"
  | "float32";

export type ExportFormat = "json" | "webvtt" | "srt" | "csv";

export interface WordTimestamp {
  word: string;
  start?: number;
  end?: number;
  score?: number;
  speaker?: string;
}

export interface TranscriptSegment {
  start?: number;
  end?: number;
  text: string;
  speaker?: string;
  words?: WordTimestamp[];
}

export interface TranscriptMetadata {
  model_size?: string;
  device?: string;
  compute_type?: string;
  alignment_applied?: boolean;
  diarization_applied?: boolean;
}

export interface TranscriptPayload {
  language?: string;
  segments: TranscriptSegment[];
  warnings?: string[];
  metadata?: TranscriptMetadata;
}

export interface JobSummary {
  job_id: string;
  filename: string;
  status: JobStatus;
  stage: string;
  progress: number;
  message: string;
  created_at: string;
  updated_at: string;
  duration_seconds: number | null;
  model_size: string;
  compute_type: string;
  do_diarization: boolean;
  language: string | null;
  has_video: boolean;
  segment_count: number;
  error: string | null;
}

export interface JobResponse extends JobSummary {
  output_format: ExportFormat;
  min_speakers: number | null;
  max_speakers: number | null;
  warnings: string[];
  available_formats: ExportFormat[];
  data: TranscriptPayload | null;
}

export interface HealthResponse {
  status: string;
  gpu: {
    cuda_available: boolean;
    device_name: string;
    total_vram_gb: number;
    allocated_vram_mb: number;
    reserved_vram_mb: number;
  };
  hf_token_configured: boolean;
  supported_models: WhisperModelSize[];
  supported_compute_types: ComputeType[];
  supported_formats: ExportFormat[];
  max_upload_mb: number;
  queue_depth: number;
}

export interface CreateJobOptions {
  file: File;
  outputFormat: ExportFormat;
  modelSize: WhisperModelSize;
  computeType: ComputeType;
  language: string;
  doDiarization: boolean;
  minSpeakers: string;
  maxSpeakers: string;
}

export interface JobPatchPayload {
  segments: Array<{
    index: number;
    text?: string;
    speaker?: string;
  }>;
  speaker_map: Record<string, string>;
}
