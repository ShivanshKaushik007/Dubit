"""Single-GPU sequential background job manager with persistent JSON metadata."""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from config import ALLOWED_MEDIA_EXTENSIONS, Settings
from schemas import JobPatchRequest, JobResponse, JobSummary
from services.formatters import export_all_formats, normalize_format_name
from services.media_utils import MediaError, extract_audio, probe_media
from services.transcription import transcribe_and_diarize

logger = logging.getLogger("dubit.jobs")

_ACTIVE_STATES = frozenset({"extracting", "transcribing", "aligning", "diarizing", "formatting"})
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._ -]")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_filename(raw_name: str) -> tuple[str, str]:
    """Return a safe display filename and validated lowercase extension."""
    basename = os.path.basename(raw_name.strip())
    if not basename:
        raise ValueError("Filename cannot be empty.")

    safe_name = _SAFE_FILENAME_RE.sub("_", basename).strip(" .")
    if not safe_name:
        safe_name = "upload"

    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_MEDIA_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_MEDIA_EXTENSIONS))
        raise ValueError(f"Unsupported file extension '{ext or '(none)'}'. Allowed: {allowed}")

    return safe_name, ext


class JobConflictError(RuntimeError):
    """Raised when a job operation conflicts with its current execution state."""


class JobManager:
    """Manages transcription jobs, disk persistence, and serialized GPU execution."""

    def __init__(
        self,
        settings: Settings,
        transcribe_fn: Callable[..., dict[str, Any]] = transcribe_and_diarize,
    ) -> None:
        self._settings = settings
        self._jobs_dir = Path(settings.jobs_dir)
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._transcribe_fn = transcribe_fn
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._events: dict[str, threading.Event] = {}
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._active_job_id: str | None = None
        self._stop_requested = False

        self._load_existing_jobs()
        self._worker = threading.Thread(target=self._worker_loop, name="dubit-gpu-worker", daemon=True)
        self._worker.start()

    def _meta_path(self, job_id: str) -> Path:
        return self._jobs_dir / job_id / "meta.json"

    def _persist_job(self, record: dict[str, Any]) -> None:
        job_dir = self._jobs_dir / record["job_id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        meta_path = job_dir / "meta.json"
        tmp_path = job_dir / "meta.json.tmp"
        tmp_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, meta_path)

    def _load_existing_jobs(self) -> None:
        for entry in self._jobs_dir.iterdir():
            if not entry.is_dir():
                continue
            meta_file = entry / "meta.json"
            if not meta_file.is_file():
                continue
            try:
                record = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Skipping corrupt job metadata at %s: %s", meta_file, exc)
                continue

            if record.get("status") not in ("completed", "failed"):
                record["status"] = "failed"
                record["stage"] = "failed"
                record["message"] = "Job interrupted by server restart."
                record["error"] = "Server restarted before job completed."
                record["updated_at"] = _utc_now_iso()
                self._persist_job(record)

            job_id = record["job_id"]
            self._jobs[job_id] = record
            evt = threading.Event()
            evt.set()
            self._events[job_id] = evt

    def create_job_dir(self) -> tuple[str, Path]:
        """Allocate a fresh UUID job ID and directory on disk."""
        job_id = uuid.uuid4().hex
        job_dir = self._jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_id, job_dir

    def enqueue_job(
        self,
        *,
        job_id: str,
        filename: str,
        input_path: Path,
        output_format: str = "json",
        model_size: str = "base",
        language: str | None = None,
        do_diarization: bool = False,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        compute_type: str = "auto",
    ) -> JobResponse:
        """Persist initial job metadata and enqueue it for sequential GPU execution."""
        now = _utc_now_iso()
        normalized_fmt = normalize_format_name(output_format)
        record: dict[str, Any] = {
            "job_id": job_id,
            "filename": filename,
            "input_path": str(input_path),
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "message": "Queued for processing...",
            "created_at": now,
            "updated_at": now,
            "duration_seconds": None,
            "model_size": model_size,
            "compute_type": compute_type,
            "do_diarization": do_diarization,
            "language": language.strip().lower() if language else None,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "output_format": normalized_fmt,
            "has_video": False,
            "segment_count": 0,
            "warnings": [],
            "available_formats": [],
            "error": None,
            "data": None,
        }

        with self._lock:
            self._jobs[job_id] = record
            self._events[job_id] = threading.Event()
            self._persist_job(record)

        self._queue.put(job_id)
        return self._to_response(record)

    def _update_record(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.update(updates)
            record["updated_at"] = _utc_now_iso()
            self._persist_job(record)

    def _worker_loop(self) -> None:
        while not self._stop_requested:
            job_id = self._queue.get()
            if job_id is None:
                self._queue.task_done()
                break

            try:
                with self._lock:
                    if job_id not in self._jobs:
                        continue
                    self._active_job_id = job_id

                self._process_job(job_id)
            except Exception as exc:
                logger.exception("Unhandled error in GPU worker for job %s", job_id)
                self._update_record(
                    job_id,
                    status="failed",
                    stage="failed",
                    message="Processing failed unexpectedly.",
                    error=str(exc),
                )
            finally:
                with self._lock:
                    self._active_job_id = None
                    evt = self._events.get(job_id)
                    if evt is not None:
                        evt.set()
                self._queue.task_done()

    def _process_job(self, job_id: str) -> None:
        with self._lock:
            record = dict(self._jobs[job_id])

        job_dir = self._jobs_dir / job_id
        input_path = Path(record["input_path"])
        audio_path = job_dir / "audio.wav"
        outputs_dir = job_dir / "outputs"

        try:
            self._update_record(
                job_id,
                status="extracting",
                stage="extracting",
                progress=10.0,
                message="Inspecting media and extracting 16 kHz mono audio...",
            )
            probe = probe_media(input_path)
            self._update_record(
                job_id,
                duration_seconds=probe.duration_seconds,
                has_video=probe.has_video,
            )
            extract_audio(input_path, audio_path)

            def _on_progress(stage: str, pct: float, msg: str) -> None:
                self._update_record(
                    job_id,
                    status=stage,
                    stage=stage,
                    progress=round(float(pct), 1),
                    message=msg,
                )

            result = self._transcribe_fn(
                str(audio_path),
                model_size=record["model_size"],
                do_diarization=record["do_diarization"],
                language=record["language"],
                compute_type=record["compute_type"],
                batch_size=self._settings.default_batch_size,
                min_speakers=record["min_speakers"],
                max_speakers=record["max_speakers"],
                hf_token=self._settings.get_hf_token(),
                vad_method=self._settings.default_vad_method,
                progress_callback=_on_progress,
            )

            self._update_record(
                job_id,
                status="formatting",
                stage="formatting",
                progress=95.0,
                message="Generating JSON, WebVTT, SRT, and CSV exports...",
            )
            written = export_all_formats(result, outputs_dir)

            segments = result.get("segments") or []
            warnings_list = list(result.get("warnings") or [])
            detected_lang = result.get("language") or record["language"]

            self._update_record(
                job_id,
                status="completed",
                stage="completed",
                progress=100.0,
                message="Transcription complete.",
                language=detected_lang,
                segment_count=len(segments),
                warnings=warnings_list,
                available_formats=list(written.keys()),
                data=result,
            )
        except MediaError as exc:
            logger.warning("Job %s media error: %s", job_id, exc)
            self._update_record(
                job_id,
                status="failed",
                stage="failed",
                message="Media validation or extraction failed.",
                error=str(exc),
            )
        except Exception as exc:
            logger.exception("Job %s processing failed", job_id)
            self._update_record(
                job_id,
                status="failed",
                stage="failed",
                message="Transcription pipeline failed.",
                error=str(exc),
            )
        finally:
            if audio_path.is_file():
                try:
                    audio_path.unlink()
                except OSError:
                    logger.debug("Could not delete temporary WAV %s", audio_path)

    def _to_summary(self, record: dict[str, Any]) -> JobSummary:
        return JobSummary(
            job_id=record["job_id"],
            filename=record["filename"],
            status=record["status"],
            stage=record["stage"],
            progress=record["progress"],
            message=record["message"],
            created_at=record["created_at"],
            updated_at=record["updated_at"],
            duration_seconds=record.get("duration_seconds"),
            model_size=record["model_size"],
            compute_type=record["compute_type"],
            do_diarization=record["do_diarization"],
            language=record.get("language"),
            has_video=bool(record.get("has_video", False)),
            segment_count=int(record.get("segment_count", 0)),
            error=record.get("error"),
        )

    def _to_response(self, record: dict[str, Any]) -> JobResponse:
        return JobResponse(
            **self._to_summary(record).model_dump(),
            output_format=record.get("output_format", "json"),
            min_speakers=record.get("min_speakers"),
            max_speakers=record.get("max_speakers"),
            warnings=list(record.get("warnings") or []),
            available_formats=list(record.get("available_formats") or []),
            data=record.get("data"),
        )

    def list_jobs(self) -> list[JobSummary]:
        """Return all persisted jobs ordered newest first."""
        with self._lock:
            records = sorted(
                self._jobs.values(),
                key=lambda item: item["created_at"],
                reverse=True,
            )
            return [self._to_summary(rec) for rec in records]

    def get_job(self, job_id: str) -> JobResponse | None:
        """Return full job status and transcript data if found."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            return self._to_response(record)

    def wait_for_job(self, job_id: str, timeout: float | None = None) -> JobResponse | None:
        """Block until a job finishes or timeout elapses."""
        with self._lock:
            evt = self._events.get(job_id)
        if evt is None:
            return None
        evt.wait(timeout=timeout)
        return self.get_job(job_id)

    def update_transcript(self, job_id: str, patch: JobPatchRequest) -> JobResponse | None:
        """Update segment text or speaker labels and regenerate all export files."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            if record["status"] != "completed" or not record.get("data"):
                raise JobConflictError("Transcript can only be edited once the job has completed.")

            data = json.loads(json.dumps(record["data"]))
            segments = data.get("segments") or []

            if patch.speaker_map:
                for seg in segments:
                    old_speaker = seg.get("speaker")
                    if old_speaker and old_speaker in patch.speaker_map:
                        seg["speaker"] = patch.speaker_map[old_speaker].strip()
                    for word in seg.get("words") or []:
                        word_spk = word.get("speaker")
                        if word_spk and word_spk in patch.speaker_map:
                            word["speaker"] = patch.speaker_map[word_spk].strip()

            for item in patch.segments:
                if item.index >= len(segments):
                    raise ValueError(f"Segment index {item.index} is out of range.")
                target = segments[item.index]
                if item.text is not None:
                    target["text"] = item.text.strip()
                if item.speaker is not None:
                    cleaned_speaker = item.speaker.strip()
                    if cleaned_speaker:
                        target["speaker"] = cleaned_speaker
                    else:
                        target.pop("speaker", None)

            outputs_dir = self._jobs_dir / job_id / "outputs"
            written = export_all_formats(data, outputs_dir)
            record["data"] = data
            record["segment_count"] = len(segments)
            record["available_formats"] = list(written.keys())
            record["updated_at"] = _utc_now_iso()
            self._persist_job(record)
            return self._to_response(record)

    def get_download_path(self, job_id: str, fmt: str) -> tuple[Path, str] | None:
        """Return the export file path and friendly download filename for a completed job."""
        normalized = normalize_format_name(fmt)
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            if record["status"] != "completed":
                raise JobConflictError("Exports are only available after the job completes.")
            stem = Path(record["filename"]).stem or "transcript"

        ext_map = {"json": ".json", "webvtt": ".vtt", "srt": ".srt", "csv": ".csv"}
        file_map = {"json": "transcript.json", "webvtt": "transcript.vtt", "srt": "transcript.srt", "csv": "transcript.csv"}
        path = self._jobs_dir / job_id / "outputs" / file_map[normalized]
        if not path.is_file():
            return None
        return path, f"{stem}{ext_map[normalized]}"

    def get_media_path(self, job_id: str) -> tuple[Path, str] | None:
        """Return the uploaded media file path and safe display filename."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            media_path = Path(record["input_path"])
            filename = record["filename"]
        if not media_path.is_file():
            return None
        return media_path, filename

    def delete_job(self, job_id: str) -> bool:
        """Remove a non-running job and all of its stored files from disk."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return False
            if self._active_job_id == job_id or record["status"] in _ACTIVE_STATES:
                raise JobConflictError("Cannot delete a job while it is actively executing.")
            self._jobs.pop(job_id, None)
            evt = self._events.pop(job_id, None)
            if evt is not None:
                evt.set()

        job_dir = self._jobs_dir / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
        return True

    def queue_depth(self) -> int:
        """Return approximate number of queued jobs waiting for the GPU worker."""
        return self._queue.qsize()

    def shutdown(self) -> None:
        """Stop the background worker gracefully."""
        self._stop_requested = True
        self._queue.put(None)
        if self._worker.is_alive():
            self._worker.join(timeout=5.0)
