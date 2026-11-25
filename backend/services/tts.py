"""Multilingual neural TTS synthesis with speaker-to-voice mapping."""

from __future__ import annotations

import asyncio
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .media_utils import MediaError, get_ffmpeg_path
from .translation import normalize_lang_code

logger = logging.getLogger("dubit.tts")

ProgressCallback = Callable[[str, float, str], None]

DEFAULT_VOICES_BY_LANG: dict[str, list[str]] = {
    "en": ["en-US-JennyNeural", "en-US-GuyNeural", "en-GB-SoniaNeural", "en-GB-RyanNeural"],
    "es": ["es-ES-ElviraNeural", "es-ES-AlvaroNeural", "es-MX-DaliaNeural", "es-MX-JorgeNeural"],
    "fr": ["fr-FR-DeniseNeural", "fr-FR-HenriNeural"],
    "de": ["de-DE-KatjaNeural", "de-DE-ConradNeural"],
    "hi": ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"],
    "ja": ["ja-JP-NanamiNeural", "ja-JP-KeitaNeural"],
    "ko": ["ko-KR-SunHiNeural", "ko-KR-InJoonNeural"],
    "zh": ["zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural"],
    "pt": ["pt-BR-FranciscaNeural", "pt-BR-AntonioNeural"],
    "ar": ["ar-EG-SalmaNeural", "ar-EG-ShakirNeural"],
    "ru": ["ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"],
    "it": ["it-IT-ElsaNeural", "it-IT-DiegoNeural"],
    "nl": ["nl-NL-ColetteNeural", "nl-NL-MaartenNeural"],
    "pl": ["pl-PL-ZofiaNeural", "pl-PL-MarekNeural"],
    "tr": ["tr-TR-EmelNeural", "tr-TR-AhmetNeural"],
    "uk": ["uk-UA-PolinaNeural", "uk-UA-OstapNeural"],
    "vi": ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"],
    "id": ["id-ID-GadisNeural", "id-ID-ArdiNeural"],
    "bn": ["bn-IN-TanishaaNeural", "bn-IN-BashkarNeural"],
    "ta": ["ta-IN-PallaviNeural", "ta-IN-ValluvarNeural"],
    "te": ["te-IN-ShrutiNeural", "te-IN-MohanNeural"],
    "mr": ["mr-IN-AarohiNeural", "mr-IN-ManoharNeural"],
    "ur": ["ur-PK-UzmaNeural", "ur-PK-AsadNeural"],
}


@dataclass(frozen=True)
class SynthesizedClip:
    """Generated TTS clip aligned to a transcript segment slot."""

    index: int
    start: float
    end: float
    speaker: str
    voice: str
    wav_path: Path


def get_voices_for_language(target_language: str) -> list[str]:
    """Return available default Edge-TTS neural voices for target_language."""
    code = normalize_lang_code(target_language)
    return list(DEFAULT_VOICES_BY_LANG.get(code, DEFAULT_VOICES_BY_LANG["en"]))


def assign_speaker_voices(
    segments: list[dict[str, Any]],
    target_language: str,
    speaker_voice_map: dict[str, str] | None = None,
) -> dict[str, str]:
    """Deterministically assign voices to each speaker across available voices."""
    palette = get_voices_for_language(target_language)
    overrides = speaker_voice_map or {}
    assigned: dict[str, str] = {}
    auto_idx = 0

    for seg in segments:
        speaker = (seg.get("speaker") or "DEFAULT").strip() or "DEFAULT"
        if speaker in assigned:
            continue
        if speaker in overrides and overrides[speaker].strip():
            assigned[speaker] = overrides[speaker].strip()
        else:
            assigned[speaker] = palette[auto_idx % len(palette)]
            auto_idx += 1

    return assigned


def _convert_to_pcm_wav(src_audio: Path, dest_wav: Path, sample_rate: int = 44100) -> None:
    cmd = [
        get_ffmpeg_path(),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src_audio),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        str(dest_wav),
        "-y",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not dest_wav.is_file():
        detail = (proc.stderr or "").strip() or f"FFmpeg exited with code {proc.returncode}"
        raise MediaError(f"Failed to normalize TTS clip: {detail}")


async def _edge_tts_to_file(
    text: str,
    voice: str,
    out_mp3: Path,
    *,
    rate: str = "+0%",
    pitch: str = "+0Hz",
) -> None:
    import edge_tts

    communicator = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    await communicator.save(str(out_mp3))


def synthesize_edge_tts_clip(
    text: str,
    voice: str,
    dest_wav: Path,
    *,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    sample_rate: int = 44100,
) -> Path:
    """Synthesize a single utterance via Edge-TTS and convert it to mono PCM WAV."""
    dest_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp_mp3 = dest_wav.with_suffix(".mp3")
    try:
        asyncio.run(_edge_tts_to_file(text, voice, tmp_mp3, rate=rate, pitch=pitch))
        _convert_to_pcm_wav(tmp_mp3, dest_wav, sample_rate=sample_rate)
        return dest_wav
    finally:
        if tmp_mp3.exists():
            tmp_mp3.unlink(missing_ok=True)


ClipSynthFn = Callable[[str, str, Path], Path]


def _estimate_segment_prosody(
    segments: list[dict[str, Any]],
    vocals_wav_path: str | Path | None,
) -> dict[int, tuple[str, str]]:
    """Estimate per-segment (rate, pitch) adjustments from original vocal energy."""
    if not vocals_wav_path or not Path(vocals_wav_path).is_file():
        return {}

    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(vocals_wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    rms_by_idx: dict[int, float] = {}
    for idx, seg in enumerate(segments):
        start = seg.get("start")
        end = seg.get("end")
        if start is None or end is None:
            continue
        s_idx = max(0, int(float(start) * sr))
        e_idx = min(len(audio), int(float(end) * sr))
        if e_idx <= s_idx:
            continue
        chunk = audio[s_idx:e_idx]
        rms_by_idx[idx] = float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))

    active_rms = [v for v in rms_by_idx.values() if v > 0.005]
    if not active_rms:
        return {}

    median_rms = float(np.median(active_rms))
    adjustments: dict[int, tuple[str, str]] = {}
    for idx, rms in rms_by_idx.items():
        ratio = rms / max(median_rms, 1e-6)
        if ratio >= 1.35:
            adjustments[idx] = ("+8%", "+5Hz")
        elif ratio <= 0.70:
            adjustments[idx] = ("-6%", "-4Hz")
        else:
            adjustments[idx] = ("+0%", "+0Hz")
    return adjustments


def synthesize_segments(
    segments: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    target_language: str,
    speaker_voice_map: dict[str, str] | None = None,
    vocals_wav_path: str | Path | None = None,
    synth_fn: ClipSynthFn | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[SynthesizedClip], dict[str, str]]:
    """Synthesize translated text for all valid segments and return clips + voice map."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    voice_map = assign_speaker_voices(segments, target_language, speaker_voice_map)
    prosody_map = _estimate_segment_prosody(segments, vocals_wav_path)

    speakable = [
        (idx, seg)
        for idx, seg in enumerate(segments)
        if (seg.get("translated_text") or seg.get("text") or "").strip()
        and seg.get("start") is not None
        and seg.get("end") is not None
    ]
    total = max(1, len(speakable))
    clips: list[SynthesizedClip] = []

    for pos, (idx, seg) in enumerate(speakable, start=1):
        text = (seg.get("translated_text") or seg.get("text") or "").strip()
        speaker = (seg.get("speaker") or "DEFAULT").strip() or "DEFAULT"
        voice = voice_map.get(speaker, get_voices_for_language(target_language)[0])
        wav_path = out_dir / f"seg_{idx:04d}.wav"

        if synth_fn is not None:
            synth_fn(text, voice, wav_path)
        else:
            rate, pitch = prosody_map.get(idx, ("+0%", "+0Hz"))
            synthesize_edge_tts_clip(text, voice, wav_path, rate=rate, pitch=pitch)
        clips.append(
            SynthesizedClip(
                index=idx,
                start=float(seg["start"]),
                end=float(seg["end"]),
                speaker=speaker,
                voice=voice,
                wav_path=wav_path,
            )
        )

        if progress_callback:
            pct = 60.0 + (pos / total) * 20.0
            progress_callback(
                "synthesizing",
                round(pct, 1),
                f"Synthesized voice clip {pos}/{total}...",
            )

    return clips, voice_map
