"""Audio time-fitting, ducking, and final dubbed track mixing."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .media_utils import MediaError, get_ffmpeg_path, load_audio_waveform
from .tts import SynthesizedClip

logger = logging.getLogger("dubit.mixer")


def _build_atempo_chain(speed: float) -> str:
    """Build FFmpeg atempo filter chain supporting factors outside [0.5, 2.0]."""
    factors: list[float] = []
    remaining = speed
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={factor:.4f}" for factor in factors)


def fit_clip_to_slot(
    clip_path: Path,
    slot_duration_s: float,
    output_path: Path,
    *,
    min_speed: float = 0.88,
    max_speed: float = 1.35,
) -> tuple[Path, float]:
    """Time-stretch a TTS clip via FFmpeg atempo if it exceeds its timing slot."""
    info = sf.info(str(clip_path))
    clip_duration = float(info.duration)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    safe_slot = max(0.2, slot_duration_s)
    if clip_duration <= safe_slot + 0.05:
        shutil.copyfile(clip_path, output_path)
        return output_path, clip_duration

    speed = min(max_speed, max(min_speed, clip_duration / safe_slot))
    if abs(speed - 1.0) < 0.03:
        shutil.copyfile(clip_path, output_path)
        return output_path, clip_duration

    cmd = [
        get_ffmpeg_path(),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(clip_path),
        "-filter:a",
        _build_atempo_chain(speed),
        "-ac",
        "1",
        "-ar",
        str(info.samplerate),
        "-acodec",
        "pcm_s16le",
        str(output_path),
        "-y",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not output_path.is_file():
        detail = (proc.stderr or "").strip() or f"FFmpeg exited with status {proc.returncode}"
        raise MediaError(f"Failed to time-fit audio clip: {detail}")

    fitted_info = sf.info(str(output_path))
    return output_path, float(fitted_info.duration)


def _compute_slot_windows(clips: list[SynthesizedClip], total_duration_s: float) -> list[float]:
    """Allow speech to borrow trailing silence before the next speaker turn."""
    ordered = sorted(clips, key=lambda c: c.start)
    slots: dict[int, float] = {}
    for idx, clip in enumerate(ordered):
        next_start = ordered[idx + 1].start if idx + 1 < len(ordered) else total_duration_s
        base_dur = max(0.25, clip.end - clip.start)
        gap_cap = max(base_dur, next_start - clip.start - 0.05) if next_start > clip.start else base_dur
        slots[clip.index] = min(base_dur * 1.4, gap_cap)
    return [slots[clip.index] for clip in clips]


def _apply_subtle_room_reverb(
    stereo: np.ndarray,
    sample_rate: int,
    wet: float = 0.08,
) -> np.ndarray:
    """Add light stereo early-reflection room ambience so dry TTS sits naturally in the scene."""
    if wet <= 0.0:
        return stereo
    out = stereo.copy()
    tap1_l = int(0.017 * sample_rate)
    tap1_r = int(0.021 * sample_rate)
    tap2_l = int(0.037 * sample_rate)
    tap2_r = int(0.043 * sample_rate)

    if tap1_l < len(stereo):
        out[tap1_l:, 0] += stereo[:-tap1_l, 0] * wet
    if tap1_r < len(stereo):
        out[tap1_r:, 1] += stereo[:-tap1_r, 1] * wet
    if tap2_l < len(stereo):
        out[tap2_l:, 0] += stereo[:-tap2_l, 0] * (wet * 0.55)
    if tap2_r < len(stereo):
        out[tap2_r:, 1] += stereo[:-tap2_r, 1] * (wet * 0.55)
    return out


def mix_dubbed_audio(
    clips: list[SynthesizedClip],
    output_wav_path: str | Path,
    *,
    total_duration_s: float | None = None,
    background_wav_path: str | Path | None = None,
    vocals_wav_path: str | Path | None = None,
    bgm_volume: float = 0.28,
    duck_level: float = 0.25,
    non_speech_vocal_gain: float = 0.35,
    room_reverb: float = 0.08,
    max_speech_speed: float = 1.35,
    sample_rate: int = 44100,
    work_dir: str | Path | None = None,
) -> Path:
    """Place fitted TTS clips on the timeline, match original vocal loudness, duck BGM, and write stereo WAV."""
    out_path = Path(output_wav_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fitted_dir = Path(work_dir) if work_dir else out_path.parent / "fitted"
    fitted_dir.mkdir(parents=True, exist_ok=True)

    max_end = max((c.end for c in clips), default=0.0)
    duration_s = max(total_duration_s or 0.0, max_end, 0.5)
    slots = _compute_slot_windows(clips, duration_s)

    placed_clips: list[tuple[SynthesizedClip, np.ndarray]] = []
    for clip, slot_s in zip(clips, slots):
        fitted_path = fitted_dir / f"fitted_{clip.index:04d}.wav"
        _, fitted_dur = fit_clip_to_slot(
            clip.wav_path,
            slot_s,
            fitted_path,
            max_speed=max_speech_speed,
        )
        mono = load_audio_waveform(fitted_path, sample_rate=sample_rate, channels=1)
        placed_clips.append((clip, mono))
        duration_s = max(duration_s, clip.start + fitted_dur)

    total_samples = max(1, int(np.ceil(duration_s * sample_rate)))
    speech_mix = np.zeros((total_samples, 2), dtype=np.float32)
    duck_mask = np.zeros(total_samples, dtype=np.float32)
    ramp_samples = min(int(0.08 * sample_rate), max(1, total_samples // 10))

    orig_vocals_stereo: np.ndarray | None = None
    orig_vocals_mono: np.ndarray | None = None
    if vocals_wav_path and Path(vocals_wav_path).is_file():
        orig_vocals_stereo = load_audio_waveform(vocals_wav_path, sample_rate=sample_rate, channels=2).T
        orig_vocals_mono = orig_vocals_stereo.mean(axis=1)

    for clip, mono in placed_clips:
        start_idx = max(0, int(clip.start * sample_rate))
        end_idx = min(total_samples, start_idx + len(mono))
        seg_len = end_idx - start_idx
        if seg_len <= 0:
            continue

        segment_audio = mono[:seg_len].copy()
        clip_rms = float(np.sqrt(np.mean(np.square(segment_audio)) + 1e-12))

        if orig_vocals_mono is not None and clip_rms > 1e-4:
            orig_end_idx = min(len(orig_vocals_mono), max(start_idx + 1, int(clip.end * sample_rate)))
            if orig_end_idx > start_idx:
                orig_slice = orig_vocals_mono[start_idx:orig_end_idx]
                orig_rms = float(np.sqrt(np.mean(np.square(orig_slice)) + 1e-12))
                target_rms = max(0.03, min(0.32, orig_rms * 1.15))
                segment_audio = segment_audio * min(3.0, target_rms / clip_rms)
        else:
            peak = float(np.max(np.abs(segment_audio)))
            if peak > 0.01:
                segment_audio = segment_audio * min(2.0, 0.8 / peak)

        speech_mix[start_idx:end_idx, 0] += segment_audio
        speech_mix[start_idx:end_idx, 1] += segment_audio

        env = np.ones(seg_len, dtype=np.float32)
        if ramp_samples > 1 and seg_len > ramp_samples * 2:
            env[:ramp_samples] = np.linspace(0.0, 1.0, ramp_samples, dtype=np.float32)
            env[-ramp_samples:] = np.linspace(1.0, 0.0, ramp_samples, dtype=np.float32)
        duck_mask[start_idx:end_idx] = np.maximum(duck_mask[start_idx:end_idx], env)

    final_mix = _apply_subtle_room_reverb(speech_mix, sample_rate, wet=room_reverb)

    if orig_vocals_stereo is not None and non_speech_vocal_gain > 0.0:
        v_len = min(total_samples, len(orig_vocals_stereo))
        gap_mask = np.clip(1.0 - duck_mask[:v_len], 0.0, 1.0) * non_speech_vocal_gain
        final_mix[:v_len] += orig_vocals_stereo[:v_len] * gap_mask[:, np.newaxis]

    if background_wav_path and Path(background_wav_path).is_file():
        bg_stereo = load_audio_waveform(background_wav_path, sample_rate=sample_rate, channels=2).T
        bg_len = min(total_samples, len(bg_stereo))
        bg_gain = bgm_volume * (1.0 - duck_mask[:bg_len] * (1.0 - duck_level))
        final_mix[:bg_len] += bg_stereo[:bg_len] * bg_gain[:, np.newaxis]

    overall_peak = float(np.max(np.abs(final_mix)))
    if overall_peak > 0.95:
        final_mix = final_mix * (0.95 / overall_peak)

    sf.write(str(out_path), np.clip(final_mix, -1.0, 1.0), sample_rate, subtype="PCM_16")
    return out_path
