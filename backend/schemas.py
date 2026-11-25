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
    "separating",
    "translating",
    "synthesizing",
    "mixing",
    "completed",
    "failed",
]

OutputFormat = Literal["json", "webvtt", "srt", "csv"]


class TranscriptSegmentUpdate(BaseModel):
    """Inline update for a single transcript or translation segment."""

    index: int = Field(ge=0)
    text: str | None = None
    translated_text: str | None = None
    speaker: str | None = None


class JobPatchRequest(BaseModel):
    """Payload to edit transcript/translated text or rename speakers for a completed job."""

    segments: list[TranscriptSegmentUpdate] = Field(default_factory=list)
    speaker_map: dict[str, str] = Field(default_factory=dict)


class DubRequest(BaseModel):
    """Configuration for dubbing an existing transcribed job into a target language."""

    target_language: str = Field(min_length=2, max_length=16)
    preserve_background: bool = True
    bgm_volume: float = Field(default=0.28, ge=0.0, le=1.0)
    max_speech_speed: float = Field(default=1.35, ge=1.0, le=2.0)
    speaker_voice_map: dict[str, str] = Field(default_factory=dict)
    skip_translation: bool = False


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
    target_language: str | None = None
    has_video: bool = False
    has_stems: bool = False
    has_dubbed_audio: bool = False
    has_dubbed_video: bool = False
    segment_count: int = 0
    error: str | None = None


class JobResponse(JobSummary):
    """Full job payload including transcript segments, dubbing voices, and export metadata."""

    output_format: str = "json"
    min_speakers: int | None = None
    max_speakers: int | None = None
    dubbed_voices: dict[str, str] = Field(default_factory=dict)
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
    translation_languages: dict[str, str]
    dubbing_voices: dict[str, list[str]]
    max_upload_mb: int
    queue_depth: int
