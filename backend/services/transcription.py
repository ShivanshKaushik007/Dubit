"""WhisperX transcription, forced alignment, and Pyannote speaker diarization pipeline."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

import whisperx

from .gpu_manager import clear_vram, select_device_and_compute
from .media_utils import load_audio_waveform

logger = logging.getLogger("dubit.transcription")

ProgressCallback = Callable[[str, float, str], None]


def _emit_progress(
    callback: ProgressCallback | None,
    stage: str,
    progress: float,
    message: str,
) -> None:
    if callback is not None:
        callback(stage, progress, message)


def transcribe_and_diarize(
    audio_path: str,
    model_size: str = "base",
    do_diarization: bool = False,
    language: str | None = None,
    compute_type: str = "auto",
    batch_size: int = 8,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    fill_nearest: bool = True,
    hf_token: str | None = None,
    vad_method: str = "pyannote",
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Transcribe audio with WhisperX, optionally align words, and assign speaker labels."""
    device, resolved_compute = select_device_and_compute(compute_type)
    effective_batch = batch_size if device == "cuda" else max(1, min(batch_size, 4))
    warnings_list: list[str] = []

    logger.info(
        "Starting pipeline: model=%s device=%s compute=%s batch=%s diarize=%s",
        model_size,
        device,
        resolved_compute,
        effective_batch,
        do_diarization,
    )

    _emit_progress(progress_callback, "transcribing", 20.0, f"Loading WhisperX '{model_size}' on {device}...")
    audio = load_audio_waveform(audio_path)

    model = None
    try:
        model = whisperx.load_model(
            model_size,
            device,
            compute_type=resolved_compute,
            language=language or None,
            vad_method=vad_method,
        )
        _emit_progress(progress_callback, "transcribing", 35.0, "Transcribing speech segments...")
        result: dict[str, Any] = model.transcribe(audio, batch_size=effective_batch)
    finally:
        if model is not None:
            del model
        clear_vram()

    segments = result.get("segments") or []
    detected_language = result.get("language") or language or "unknown"
    alignment_applied = False

    if not segments:
        warnings_list.append("No speech segments detected in audio; skipped alignment and diarization.")
    else:
        _emit_progress(
            progress_callback,
            "aligning",
            60.0,
            f"Aligning word timestamps ({detected_language})...",
        )
        model_a = None
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=detected_language,
                device=device,
            )
            aligned = whisperx.align(
                segments,
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=False,
            )
            result["segments"] = aligned.get("segments", segments)
            if "word_segments" in aligned:
                result["word_segments"] = aligned["word_segments"]
            alignment_applied = True
        except ValueError as exc:
            msg = f"Word alignment unavailable for language '{detected_language}': {exc}"
            logger.warning(msg)
            warnings_list.append(msg)
        except Exception as exc:
            msg = f"Word alignment failed ({exc}); using segment-level timestamps."
            logger.warning(msg)
            warnings_list.append(msg)
        finally:
            if model_a is not None:
                del model_a
            clear_vram()

    diarization_applied = False
    if do_diarization and result.get("segments"):
        token = hf_token or os.environ.get("HF_TOKEN", "").strip() or None
        if not token:
            msg = "Speaker diarization requested, but HF_TOKEN is not set; skipped speaker labeling."
            logger.warning(msg)
            warnings_list.append(msg)
        else:
            _emit_progress(progress_callback, "diarizing", 80.0, "Running speaker diarization...")
            from whisperx.diarize import DiarizationPipeline

            diarize_model = None
            try:
                diarize_model = DiarizationPipeline(token=token, device=device)

                def _diarize_hook(pct: float) -> None:
                    scaled = 80.0 + min(max(pct, 0.0), 100.0) * 0.14
                    _emit_progress(progress_callback, "diarizing", scaled, "Assigning speaker turns...")

                diarize_segments = diarize_model(
                    audio,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    progress_callback=_diarize_hook,
                )
                result = whisperx.assign_word_speakers(
                    diarize_segments,
                    result,
                    fill_nearest=fill_nearest,
                )
                diarization_applied = True
            except Exception as exc:
                msg = f"Speaker diarization failed ({exc}); returning transcript without speaker labels."
                logger.warning(msg)
                warnings_list.append(msg)
            finally:
                if diarize_model is not None:
                    del diarize_model
                clear_vram()

    result["language"] = detected_language
    result["warnings"] = warnings_list
    result["metadata"] = {
        "model_size": model_size,
        "device": device,
        "compute_type": resolved_compute,
        "alignment_applied": alignment_applied,
        "diarization_applied": diarization_applied,
    }
    return result
