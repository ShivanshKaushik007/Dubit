"""Vocal and background accompaniment separation using Demucs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
import torch

from .gpu_manager import clear_vram
from .media_utils import load_audio_waveform

logger = logging.getLogger("dubit.separation")

ProgressCallback = Callable[[str, float, str], None]


@dataclass(frozen=True)
class StemOutputs:
    """Paths to separated vocal and background accompaniment stems."""

    vocals_path: Path
    background_path: Path
    sample_rate: int


def separate_stems(
    media_path: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "htdemucs",
    segment_seconds: float = 7.0,
    progress_callback: ProgressCallback | None = None,
) -> StemOutputs:
    """Separate input audio/video into vocals and background (drums+bass+other) WAV stems."""
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocals_path = out_dir / "vocals.wav"
    background_path = out_dir / "background.wav"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if progress_callback:
        progress_callback("separating", 15.0, f"Loading Demucs '{model_name}' on {device}...")

    model = None
    try:
        model = get_model(model_name)
        model.eval()

        sample_rate = int(getattr(model, "samplerate", 44100))
        audio_channels = int(getattr(model, "audio_channels", 2))
        waveform_np = load_audio_waveform(media_path, sample_rate=sample_rate, channels=audio_channels)
        if audio_channels == 1:
            waveform_np = waveform_np[np.newaxis, :]

        wav = torch.from_numpy(waveform_np).float()
        ref = wav.mean(0)
        std = ref.std()
        if float(std) > 1e-8:
            wav = (wav - ref.mean()) / std

        if progress_callback:
            progress_callback("separating", 24.0, "Separating vocals from background music/SFX...")

        with torch.no_grad():
            sources = apply_model(
                model,
                wav.unsqueeze(0),
                shifts=1,
                split=True,
                overlap=0.25,
                segment=segment_seconds if device == "cuda" else None,
                device=device,
                progress=False,
            )[0]

        if float(std) > 1e-8:
            sources = sources * std + ref.mean()

        source_names = list(model.sources)
        vocals_idx = source_names.index("vocals")
        vocals = sources[vocals_idx].cpu().numpy()

        bg_indices = [i for i in range(len(source_names)) if i != vocals_idx]
        background = sources[bg_indices].sum(dim=0).cpu().numpy() if bg_indices else np.zeros_like(vocals)

        sf.write(str(vocals_path), np.clip(vocals.T, -1.0, 1.0), sample_rate, subtype="PCM_16")
        sf.write(str(background_path), np.clip(background.T, -1.0, 1.0), sample_rate, subtype="PCM_16")
        logger.info("Separated stems written to %s", out_dir)
        return StemOutputs(
            vocals_path=vocals_path,
            background_path=background_path,
            sample_rate=sample_rate,
        )
    finally:
        if model is not None:
            del model
        clear_vram()


def extract_speaker_reference_clips(
    vocals_path: str | Path,
    segments: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    min_clip_seconds: float = 3.0,
    max_clip_seconds: float = 12.0,
) -> dict[str, Path]:
    """Slice clean per-speaker vocal reference WAV clips for inspection or voice cloning."""
    src = Path(vocals_path)
    if not src.is_file() or not segments:
        return {}

    audio, sr = sf.read(str(src), dtype="float32")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_by_speaker: dict[str, tuple[float, float, float]] = {}
    for seg in segments:
        speaker = (seg.get("speaker") or "").strip()
        start = seg.get("start")
        end = seg.get("end")
        if not speaker or start is None or end is None:
            continue
        dur = float(end) - float(start)
        if dur <= 0.4:
            continue
        prev = best_by_speaker.get(speaker)
        if prev is None or (prev[0] < min_clip_seconds and dur > prev[0]):
            best_by_speaker[speaker] = (dur, float(start), float(end))

    saved: dict[str, Path] = {}
    for speaker, (dur, start, end) in best_by_speaker.items():
        clipped_end = min(end, start + max_clip_seconds)
        start_idx = max(0, int(start * sr))
        end_idx = min(len(audio), int(clipped_end * sr))
        if end_idx <= start_idx:
            continue
        safe_spk = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in speaker)
        dest = out_dir / f"{safe_spk}.wav"
        sf.write(str(dest), audio[start_idx:end_idx], sr, subtype="PCM_16")
        saved[speaker] = dest

    return saved
