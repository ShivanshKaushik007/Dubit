"""Pydantic request and response schemas for the Dubit API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

JobStatus = Literal[
    "queued",
    "extracting",
    "transcribing",
    "aligning",
    "diarizing",
    "formatting",
    "completed",
    "failed",
]

OutputFormat = Literal["json", "webvtt", "srt", "csv"]


class TranscriptSegmentUpdate(BaseModel):
    """Inline update for a single transcript segment."""

    index: int = Field(ge=0)
    text: str | None = None
    speaker: str | None = None


class JobPatchRequest(BaseModel):
    """Payload to edit transcript text or rename speakers for a completed job."""

    segments: list[TranscriptSegmentUpdate] = Field(default_factory=list)
    speaker_map: dict[str, str] = Field(default_factory=dict)


class JobSummary(BaseModel):
    """Lightweight job status representation for queue listings and polling."""

    job_id: str
    filename: str
    status: JobStatus
    stage: str
    progress: float = Field(ge=0.0, le=100.0)
    message: str
    created_at: str
    updated_at: str
    duration_seconds: float | None = None
    model_size: str
    compute_type: str
    do_diarization: bool
    language: str | None = None
    has_video: bool = False
    segment_count: int = 0
    error: str | None = None


class JobResponse(JobSummary):
    """Full job payload including transcript segments and export metadata."""

    output_format: str = "json"
    min_speakers: int | None = None
    max_speakers: int | None = None
    warnings: list[str] = Field(default_factory=list)
    available_formats: list[str] = Field(default_factory=list)
    data: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    """System health, GPU telemetry, and capability summary."""

    status: str
    gpu: dict[str, Any]
    hf_token_configured: bool
    supported_models: list[str]
    supported_compute_types: list[str]
    supported_formats: list[str]
    max_upload_mb: int
    queue_depth: int
