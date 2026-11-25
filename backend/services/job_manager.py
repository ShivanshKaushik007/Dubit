"""Single-GPU sequential background job manager for transcription and AI dubbing."""

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
from schemas import DubRequest, JobPatchRequest, JobResponse, JobSummary
from services.formatters import export_all_formats, normalize_format_name
from services.media_utils import MediaError, extract_audio, merge_audio_video, probe_media
from services.mixer import mix_dubbed_audio
from services.separation import extract_speaker_reference_clips, separate_stems
from services.transcription import transcribe_and_diarize
from services.translation import normalize_lang_code, translate_segments
from services.tts import synthesize_segments

logger = logging.getLogger("dubit.jobs")

_ACTIVE_STATES = frozenset(
    {
        "extracting",
        "transcribing",
        "aligning",
        "diarizing",
        "formatting",
        "separating",
        "translating",
        "synthesizing",
        "mixing",
    }
)
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._ -]")

_EXTRA_DOWNLOAD_FILES: dict[str, tuple[str, str]] = {
    "dubbed_vtt": ("dubbed.vtt", "_dubbed.vtt"),
    "dubbed_srt": ("dubbed.srt", "_dubbed.srt"),
    "dubbed_audio": ("dubbed_audio.wav", "_dubbed.wav"),
    "dubbed_video": ("dubbed_video.mp4", "_dubbed.mp4"),
}


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
    """Manages transcription & dubbing jobs, disk persistence, and serialized GPU execution."""

    def __init__(
        self,
        settings: Settings,
        transcribe_fn: Callable[..., dict[str, Any]] = transcribe_and_diarize,
        separate_fn: Callable[..., Any] = separate_stems,
        translate_fn: Callable[..., list[dict[str, Any]]] = translate_segments,
        synth_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._settings = settings
        self._jobs_dir = Path(settings.jobs_dir)
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._transcribe_fn = transcribe_fn
        self._separate_fn = separate_fn
        self._translate_fn = translate_fn
        self._synth_fn = synth_fn
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._events: dict[str, threading.Event] = {}
        self._queue: queue.Queue[tuple[str, str, dict[str, Any] | None] | None] = queue.Queue()
        self._active_job_id: str | None = None
        self._stop_requested = False

        self._load_existing_jobs()
        self._worker = threading.Thread(target=self._worker_loop, name="dubit-gpu-worker", daemon=True)
        self._worker.start()

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
        target_language: str | None = None,
        preserve_background: bool = True,
    ) -> JobResponse:
        """Persist initial job metadata and enqueue transcription (and optional dubbing)."""
        now = _utc_now_iso()
        normalized_fmt = normalize_format_name(output_format)
        cleaned_target = normalize_lang_code(target_language) if target_language and target_language.strip() else None

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
            "target_language": cleaned_target,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "output_format": normalized_fmt,
            "has_video": False,
            "has_stems": False,
            "has_dubbed_audio": False,
            "has_dubbed_video": False,
            "dubbed_voices": {},
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

        auto_dub = (
            DubRequest(target_language=cleaned_target, preserve_background=preserve_background).model_dump()
            if cleaned_target
            else None
        )
        self._queue.put(("transcribe", job_id, auto_dub))
        return self._to_response(record)

    def enqueue_dub(self, job_id: str, dub_req: DubRequest) -> JobResponse | None:
        """Queue dubbing on an already transcribed job."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            if self._active_job_id == job_id or record["status"] in _ACTIVE_STATES or record["status"] == "queued":
                raise JobConflictError("Job is already queued or running.")
            if not record.get("data") or not (record["data"].get("segments") or []):
                raise JobConflictError("Cannot dub a job without transcribed segments.")

            record["status"] = "queued"
            record["stage"] = "queued"
            record["progress"] = 0.0
            record["message"] = f"Queued for dubbing ({dub_req.target_language})..."
            record["error"] = None
            record["target_language"] = normalize_lang_code(dub_req.target_language)
            record["updated_at"] = _utc_now_iso()
            evt = self._events.get(job_id) or threading.Event()
            evt.clear()
            self._events[job_id] = evt
            self._persist_job(record)

        self._queue.put(("dub", job_id, dub_req.model_dump()))
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
            task = self._queue.get()
            if task is None:
                self._queue.task_done()
                break

            task_kind, job_id, payload = task
            try:
                with self._lock:
                    if job_id not in self._jobs:
                        continue
                    self._active_job_id = job_id

                if task_kind == "transcribe":
                    ok = self._process_transcription(job_id)
                    if ok and payload is not None:
                        self._process_dub(job_id, payload)
                elif task_kind == "dub" and payload is not None:
                    self._process_dub(job_id, payload)
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

    def _process_transcription(self, job_id: str) -> bool:
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

            self._update_record(
                job_id,
                status="completed",
                stage="completed",
                progress=100.0,
                message="Transcription complete.",
                language=result.get("language") or record["language"],
                segment_count=len(segments),
                warnings=list(result.get("warnings") or []),
                available_formats=list(written.keys()),
                data=result,
            )
            return True
        except MediaError as exc:
            logger.warning("Job %s media error: %s", job_id, exc)
            self._update_record(
                job_id,
                status="failed",
                stage="failed",
                message="Media validation or extraction failed.",
                error=str(exc),
            )
            return False
        except Exception as exc:
            logger.exception("Job %s transcription failed", job_id)
            self._update_record(
                job_id,
                status="failed",
                stage="failed",
                message="Transcription pipeline failed.",
                error=str(exc),
            )
            return False
        finally:
            if audio_path.is_file():
                try:
                    audio_path.unlink()
                except OSError:
                    logger.debug("Could not delete temporary WAV %s", audio_path)

    def _export_dubbed_subtitles(self, data: dict[str, Any], outputs_dir: Path) -> list[str]:
        """Write dubbed WebVTT and SRT subtitle files using translated_text."""
        translated_payload = {
            "language": data.get("target_language"),
            "segments": [
                {
                    **seg,
                    "text": (seg.get("translated_text") or seg.get("text") or "").strip(),
                }
                for seg in (data.get("segments") or [])
            ],
        }
        written = export_all_formats(translated_payload, outputs_dir, stem="dubbed")
        formats = []
        if "webvtt" in written:
            formats.append("dubbed_vtt")
        if "srt" in written:
            formats.append("dubbed_srt")
        return formats

    def _process_dub(self, job_id: str, dub_payload: dict[str, Any]) -> None:
        with self._lock:
            record = dict(self._jobs[job_id])

        dub_req = DubRequest(**dub_payload)
        job_dir = self._jobs_dir / job_id
        input_path = Path(record["input_path"])
        outputs_dir = job_dir / "outputs"
        stems_dir = job_dir / "stems"
        tts_dir = job_dir / "tts"
        speakers_dir = job_dir / "speakers"

        def _on_progress(stage: str, pct: float, msg: str) -> None:
            self._update_record(
                job_id,
                status=stage,
                stage=stage,
                progress=round(float(pct), 1),
                message=msg,
            )

        try:
            data = json.loads(json.dumps(record["data"]))
            segments = data.get("segments") or []
            if not segments:
                raise RuntimeError("No transcript segments available for dubbing.")

            warnings_list = list(record.get("warnings") or [])
            background_path: Path | None = None
            vocals_path = stems_dir / "vocals.wav"
            cached_bg = stems_dir / "background.wav"

            if dub_req.preserve_background:
                if vocals_path.is_file() and cached_bg.is_file():
                    background_path = cached_bg
                else:
                    try:
                        stems = self._separate_fn(
                            input_path,
                            stems_dir,
                            progress_callback=_on_progress,
                        )
                        vocals_path = stems.vocals_path
                        background_path = stems.background_path
                        extract_speaker_reference_clips(vocals_path, segments, speakers_dir)
                    except Exception as exc:
                        msg = f"Stem separation skipped ({exc}); dubbing without background stem."
                        logger.warning(msg)
                        if msg not in warnings_list:
                            warnings_list.append(msg)
                        background_path = None

            target_lang = normalize_lang_code(dub_req.target_language)
            has_existing_translation = all("translated_text" in seg for seg in segments)
            if not (dub_req.skip_translation and has_existing_translation):
                segments = self._translate_fn(
                    segments,
                    source_language=data.get("language") or record.get("language"),
                    target_language=target_lang,
                    progress_callback=_on_progress,
                )
            data["segments"] = segments
            data["target_language"] = target_lang

            _on_progress("synthesizing", 60.0, f"Synthesizing {target_lang} dubbed speech...")
            clips, voice_map = synthesize_segments(
                segments,
                tts_dir,
                target_language=target_lang,
                speaker_voice_map=dub_req.speaker_voice_map,
                vocals_wav_path=vocals_path if vocals_path.is_file() else None,
                synth_fn=self._synth_fn,
                progress_callback=_on_progress,
            )
            if not clips:
                raise RuntimeError("No speakable segments produced audio clips.")

            _on_progress("mixing", 85.0, "Time-fitting clips and mixing dubbed audio track...")
            dubbed_audio_path = outputs_dir / "dubbed_audio.wav"
            mix_dubbed_audio(
                clips,
                dubbed_audio_path,
                total_duration_s=record.get("duration_seconds"),
                background_wav_path=background_path,
                vocals_wav_path=vocals_path if vocals_path.is_file() else None,
                bgm_volume=dub_req.bgm_volume,
                max_speech_speed=dub_req.max_speech_speed,
                work_dir=tts_dir / "fitted",
            )

            has_dubbed_video = False
            if record.get("has_video"):
                _on_progress("mixing", 93.0, "Muxing dubbed audio into video...")
                dubbed_video_path = outputs_dir / "dubbed_video.mp4"
                merge_audio_video(input_path, dubbed_audio_path, dubbed_video_path)
                has_dubbed_video = dubbed_video_path.is_file()

            written_transcript = export_all_formats(data, outputs_dir)
            dubbed_sub_formats = self._export_dubbed_subtitles(data, outputs_dir)

            available = list(written_transcript.keys()) + dubbed_sub_formats + ["dubbed_audio"]
            if has_dubbed_video:
                available.append("dubbed_video")
            if vocals_path.is_file():
                available.append("vocals")
            if background_path and background_path.is_file():
                available.append("background")

            self._update_record(
                job_id,
                status="completed",
                stage="completed",
                progress=100.0,
                message=f"Dubbing complete ({target_lang}).",
                target_language=target_lang,
                has_stems=vocals_path.is_file() and cached_bg.is_file(),
                has_dubbed_audio=dubbed_audio_path.is_file(),
                has_dubbed_video=has_dubbed_video,
                dubbed_voices=voice_map,
                warnings=warnings_list,
                available_formats=available,
                data=data,
            )
        except Exception as exc:
            logger.exception("Job %s dubbing failed", job_id)
            self._update_record(
                job_id,
                status="failed",
                stage="failed",
                message="Dubbing pipeline failed.",
                error=str(exc),
            )
        finally:
            if tts_dir.exists():
                shutil.rmtree(tts_dir, ignore_errors=True)

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
            target_language=record.get("target_language"),
            has_video=bool(record.get("has_video", False)),
            has_stems=bool(record.get("has_stems", False)),
            has_dubbed_audio=bool(record.get("has_dubbed_audio", False)),
            has_dubbed_video=bool(record.get("has_dubbed_video", False)),
            segment_count=int(record.get("segment_count", 0)),
            error=record.get("error"),
        )

    def _to_response(self, record: dict[str, Any]) -> JobResponse:
        return JobResponse(
            **self._to_summary(record).model_dump(),
            output_format=record.get("output_format", "json"),
            min_speakers=record.get("min_speakers"),
            max_speakers=record.get("max_speakers"),
            dubbed_voices=dict(record.get("dubbed_voices") or {}),
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
        """Return full job status and transcript/dubbing payload if found."""
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
        """Update segment text, translated_text, or speaker labels and regenerate exports."""
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
                        new_spk = patch.speaker_map[old_speaker].strip()
                        if new_spk:
                            seg["speaker"] = new_spk
                        else:
                            seg.pop("speaker", None)
                    for word in seg.get("words") or []:
                        word_spk = word.get("speaker")
                        if word_spk and word_spk in patch.speaker_map:
                            new_spk = patch.speaker_map[word_spk].strip()
                            if new_spk:
                                word["speaker"] = new_spk
                            else:
                                word.pop("speaker", None)

            for item in patch.segments:
                if item.index >= len(segments):
                    raise ValueError(f"Segment index {item.index} is out of range.")
                target = segments[item.index]
                if item.text is not None:
                    target["text"] = item.text.strip()
                if item.translated_text is not None:
                    target["translated_text"] = item.translated_text.strip()
                if item.speaker is not None:
                    cleaned_speaker = item.speaker.strip()
                    if cleaned_speaker:
                        target["speaker"] = cleaned_speaker
                    else:
                        target.pop("speaker", None)

            outputs_dir = self._jobs_dir / job_id / "outputs"
            written = export_all_formats(data, outputs_dir)
            available = list(written.keys())
            if data.get("target_language"):
                available.extend(self._export_dubbed_subtitles(data, outputs_dir))
            if record.get("has_dubbed_audio"):
                available.append("dubbed_audio")
            if record.get("has_dubbed_video"):
                available.append("dubbed_video")
            if record.get("has_stems"):
                available.extend(["vocals", "background"])

            record["data"] = data
            record["segment_count"] = len(segments)
            record["available_formats"] = available
            record["updated_at"] = _utc_now_iso()
            self._persist_job(record)
            return self._to_response(record)

    def get_download_path(self, job_id: str, fmt: str) -> tuple[Path, str] | None:
        """Return the export or dubbed asset file path and friendly download filename."""
        lowered = fmt.strip().lower()
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            if record["status"] != "completed":
                raise JobConflictError("Exports are only available after the job completes.")
            stem = Path(record["filename"]).stem or "transcript"

        job_dir = self._jobs_dir / job_id
        if lowered in ("vocals", "background"):
            stem_file = job_dir / "stems" / f"{lowered}.wav"
            if not stem_file.is_file():
                return None
            return stem_file, f"{stem}_{lowered}.wav"

        if lowered in _EXTRA_DOWNLOAD_FILES:
            rel_name, suffix = _EXTRA_DOWNLOAD_FILES[lowered]
            path = job_dir / "outputs" / rel_name
            if not path.is_file():
                return None
            return path, f"{stem}{suffix}"

        normalized = normalize_format_name(lowered)
        ext_map = {"json": ".json", "webvtt": ".vtt", "srt": ".srt", "csv": ".csv"}
        file_map = {
            "json": "transcript.json",
            "webvtt": "transcript.vtt",
            "srt": "transcript.srt",
            "csv": "transcript.csv",
        }
        path = job_dir / "outputs" / file_map[normalized]
        if not path.is_file():
            return None
        return path, f"{stem}{ext_map[normalized]}"

    def get_media_path(self, job_id: str, *, dubbed: bool = False) -> tuple[Path, str] | None:
        """Return original or dubbed media file path and safe display filename."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            filename = record["filename"]
            input_path = Path(record["input_path"])
            has_dubbed_video = bool(record.get("has_dubbed_video"))
            has_dubbed_audio = bool(record.get("has_dubbed_audio"))

        job_dir = self._jobs_dir / job_id
        if dubbed:
            if has_dubbed_video:
                path = job_dir / "outputs" / "dubbed_video.mp4"
                if path.is_file():
                    return path, f"{Path(filename).stem}_dubbed.mp4"
            if has_dubbed_audio:
                path = job_dir / "outputs" / "dubbed_audio.wav"
                if path.is_file():
                    return path, f"{Path(filename).stem}_dubbed.wav"
            return None

        if not input_path.is_file():
            return None
        return input_path, filename

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
