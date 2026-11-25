"""FastAPI server for AI Video Dubbing Studio."""

from __future__ import annotations

import logging
import mimetypes
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config import (
    SUPPORTED_COMPUTE_TYPES,
    SUPPORTED_EXPORT_FORMATS,
    SUPPORTED_WHISPER_MODELS,
    settings,
)
from schemas import HealthResponse, JobPatchRequest, JobResponse, JobSummary
from services.formatters import normalize_format_name
from services.gpu_manager import get_device_info
from services.job_manager import JobConflictError, JobManager, sanitize_filename

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dubit.api")

_CHUNK_SIZE = 1024 * 1024  # 1 MiB
_DOWNLOAD_MEDIA_TYPES = {
    "json": "application/json; charset=utf-8",
    "webvtt": "text/vtt; charset=utf-8",
    "srt": "application/x-subrip; charset=utf-8",
    "csv": "text/csv; charset=utf-8",
}


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialize directories and the single-GPU background job manager."""
    settings.ensure_directories()
    created_manager = False
    if not hasattr(application.state, "job_manager") or application.state.job_manager is None:
        application.state.job_manager = JobManager(settings)
        created_manager = True

    try:
        yield
    finally:
        if created_manager and getattr(application.state, "job_manager", None) is not None:
            application.state.job_manager.shutdown()


app = FastAPI(title="AI Dubbing API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


def _get_manager(request: Request) -> JobManager:
    manager: JobManager | None = getattr(request.app.state, "job_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="Job manager is not initialized.")
    return manager


async def _stream_upload_to_disk(file: UploadFile, destination: Path, max_bytes: int) -> None:
    written = 0
    try:
        with open(destination, "wb") as out:
            while True:
                chunk = await file.read(_CHUNK_SIZE)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=f"Uploaded file exceeds maximum allowed size of {settings.max_upload_mb} MB.",
                    )
                out.write(chunk)
    except Exception:
        if destination.exists():
            destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


def _validate_job_options(
    *,
    model_size: str,
    compute_type: str,
    output_format: str,
    min_speakers: int | None,
    max_speakers: int | None,
) -> str:
    if model_size not in SUPPORTED_WHISPER_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model_size '{model_size}'. Supported: {', '.join(SUPPORTED_WHISPER_MODELS)}",
        )
    if compute_type not in SUPPORTED_COMPUTE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported compute_type '{compute_type}'. Supported: {', '.join(SUPPORTED_COMPUTE_TYPES)}",
        )
    try:
        normalized_fmt = normalize_format_name(output_format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if min_speakers is not None and min_speakers < 1:
        raise HTTPException(status_code=400, detail="min_speakers must be at least 1.")
    if max_speakers is not None and max_speakers < 1:
        raise HTTPException(status_code=400, detail="max_speakers must be at least 1.")
    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise HTTPException(status_code=400, detail="min_speakers cannot exceed max_speakers.")

    return normalized_fmt


async def _create_job_from_upload(
    request: Request,
    *,
    file: UploadFile,
    output_format: str,
    do_diarization: bool,
    model_size: str,
    language: str | None,
    compute_type: str,
    min_speakers: int | None,
    max_speakers: int | None,
) -> JobResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    try:
        safe_name, ext = sanitize_filename(file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    normalized_fmt = _validate_job_options(
        model_size=model_size,
        compute_type=compute_type,
        output_format=output_format,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    manager = _get_manager(request)
    job_id, job_dir = manager.create_job_dir()
    input_path = job_dir / f"input{ext}"

    try:
        await _stream_upload_to_disk(file, input_path, settings.max_upload_bytes)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    return manager.enqueue_job(
        job_id=job_id,
        filename=safe_name,
        input_path=input_path,
        output_format=normalized_fmt,
        model_size=model_size,
        language=language,
        do_diarization=do_diarization,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        compute_type=compute_type,
    )


@app.get("/")
def read_root() -> dict[str, str]:
    """Basic health check root route."""
    return {"status": "Backend is running"}


@app.get("/health", response_model=HealthResponse)
def health_check(request: Request) -> HealthResponse:
    """Return GPU capabilities, configuration flags, and queue depth."""
    manager: JobManager | None = getattr(request.app.state, "job_manager", None)
    return HealthResponse(
        status="ok",
        gpu=get_device_info(),
        hf_token_configured=settings.get_hf_token() is not None,
        supported_models=list(SUPPORTED_WHISPER_MODELS),
        supported_compute_types=list(SUPPORTED_COMPUTE_TYPES),
        supported_formats=list(SUPPORTED_EXPORT_FORMATS),
        max_upload_mb=settings.max_upload_mb,
        queue_depth=manager.queue_depth() if manager is not None else 0,
    )


@app.post("/jobs", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    output_format: Annotated[str, Form()] = "json",
    do_diarization: Annotated[bool, Form()] = False,
    model_size: Annotated[str, Form()] = settings.default_model_size,
    language: Annotated[str | None, Form()] = None,
    compute_type: Annotated[str, Form()] = settings.default_compute_type,
    min_speakers: Annotated[int | None, Form()] = None,
    max_speakers: Annotated[int | None, Form()] = None,
) -> JobResponse:
    """Upload a media file and queue an asynchronous transcription/diarization job."""
    return await _create_job_from_upload(
        request,
        file=file,
        output_format=output_format,
        do_diarization=do_diarization,
        model_size=model_size,
        language=language,
        compute_type=compute_type,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )


@app.get("/jobs", response_model=list[JobSummary])
def list_jobs(request: Request) -> list[JobSummary]:
    """List persisted jobs newest first."""
    return _get_manager(request).list_jobs()


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(request: Request, job_id: str) -> JobResponse:
    """Get current status, progress, warnings, and transcript payload for a job."""
    job = _get_manager(request).get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.patch("/jobs/{job_id}", response_model=JobResponse)
def patch_job(request: Request, job_id: str, payload: JobPatchRequest) -> JobResponse:
    """Edit transcript segments or rename speaker labels and regenerate export files."""
    manager = _get_manager(request)
    try:
        updated = manager.update_transcript(job_id, payload)
    except JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if updated is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return updated


@app.get("/jobs/{job_id}/download/{fmt}")
def download_export(request: Request, job_id: str, fmt: str) -> FileResponse:
    """Download a completed job's transcript in JSON, WebVTT, SRT, or CSV format."""
    try:
        normalized = normalize_format_name(fmt)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    manager = _get_manager(request)
    try:
        resolved = manager.get_download_path(job_id, normalized)
    except JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if resolved is None:
        raise HTTPException(status_code=404, detail="Job export not found.")

    file_path, download_name = resolved
    return FileResponse(
        path=file_path,
        media_type=_DOWNLOAD_MEDIA_TYPES[normalized],
        filename=download_name,
    )


@app.get("/jobs/{job_id}/media")
def stream_media(request: Request, job_id: str) -> FileResponse:
    """Stream the uploaded audio or video file for synchronized browser playback."""
    resolved = _get_manager(request).get_media_path(job_id)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Job media not found.")

    media_path, filename = resolved
    guessed_type, _ = mimetypes.guess_type(filename)
    return FileResponse(
        path=media_path,
        media_type=guessed_type or "application/octet-stream",
    )


@app.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job(request: Request, job_id: str) -> None:
    """Delete a job and all associated files on disk."""
    try:
        deleted = _get_manager(request).delete_job(job_id)
    except JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found.")


@app.post("/upload")
async def legacy_upload_file(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    output_format: Annotated[str, Form()] = "json",
    do_diarization: Annotated[bool, Form()] = False,
    model_size: Annotated[str, Form()] = settings.default_model_size,
    language: Annotated[str | None, Form()] = None,
) -> dict[str, object]:
    """Backwards-compatible synchronous upload route backed by the async job queue."""
    job = await _create_job_from_upload(
        request,
        file=file,
        output_format=output_format,
        do_diarization=do_diarization,
        model_size=model_size,
        language=language,
        compute_type=settings.default_compute_type,
        min_speakers=None,
        max_speakers=None,
    )
    manager = _get_manager(request)
    finished = await run_in_threadpool(manager.wait_for_job, job.job_id)
    if finished is None or finished.status != "completed":
        detail = finished.error if finished and finished.error else "Transcription failed."
        raise HTTPException(status_code=500, detail=detail)

    return {
        "message": "Transcription complete",
        "job_id": finished.job_id,
        "file_path": f"/jobs/{finished.job_id}/download/{job.output_format}",
        "data": finished.data,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
