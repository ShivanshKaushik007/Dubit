"""Multilingual TTS synthesis supporting Edge-TTS and Chatterbox zero-shot voice cloning."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from .media_utils import MediaError, get_ffmpeg_path
from .translation import normalize_lang_code

logger = logging.getLogger("dubit.tts")

ProgressCallback = Callable[[str, float, str], None]

CHATTERBOX_SUPPORTED_LANGUAGES: frozenset[str] = frozenset(
    {
        "ar",
        "da",
        "de",
        "el",
        "en",
        "es",
        "fi",
        "fr",
        "he",
        "hi",
        "it",
        "ja",
        "ko",
        "ms",
        "nl",
        "no",
        "pl",
        "pt",
        "ru",
        "sv",
        "sw",
        "tr",
        "zh",
    }
)

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


@dataclass(frozen=True)
class SegmentProsody:
    """Acoustic emotion & pacing signals derived from the original vocal segment."""

    rate: str
    pitch: str
    exaggeration: float


ClipSynthFn = Callable[[str, str, Path], Path]


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


def extract_clean_prompt_slice(
    vocals_mono: np.ndarray,
    sample_rate: int,
    start_s: float,
    end_s: float,
    *,
    max_seconds: float = 9.5,
) -> np.ndarray:
    """Extract active speech from [start_s, end_s] while stripping long internal silence gaps."""
    s_idx = max(0, int(start_s * sample_rate))
    e_idx = min(len(vocals_mono), int(end_s * sample_rate))
    if e_idx <= s_idx:
        return np.zeros(0, dtype=np.float32)

    raw = vocals_mono[s_idx:e_idx]
    peak = float(np.max(np.abs(raw)))
    if peak < 0.01:
        return np.zeros(0, dtype=np.float32)

    frame_len = max(1, int(0.04 * sample_rate))
    threshold = max(0.012, peak * 0.08)
    active_chunks: list[np.ndarray] = []
    for offset in range(0, len(raw), frame_len):
        chunk = raw[offset : offset + frame_len]
        rms = float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))
        if rms >= threshold:
            active_chunks.append(chunk)

    if not active_chunks:
        return np.zeros(0, dtype=np.float32)

    combined = np.concatenate(active_chunks)
    max_samples = int(max_seconds * sample_rate)
    return combined[:max_samples]


def prepare_segment_prompts(
    segments: list[dict[str, Any]],
    vocals_wav_path: str | Path,
    prompts_dir: Path,
    *,
    min_segment_prompt_s: float = 1.2,
) -> tuple[dict[int, Path], Path | None]:
    """Slice silence-trimmed per-segment reference prompts and speaker fallbacks from vocals.wav."""
    src = Path(vocals_wav_path)
    if not src.is_file():
        return {}, None

    audio, sr = sf.read(str(src), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    prompts_dir.mkdir(parents=True, exist_ok=True)
    segment_slices: dict[int, np.ndarray] = {}
    speaker_pools: dict[str, list[np.ndarray]] = {}
    all_slices: list[np.ndarray] = []

    for idx, seg in enumerate(segments):
        start = seg.get("start")
        end = seg.get("end")
        if start is None or end is None:
            continue
        cleaned = extract_clean_prompt_slice(audio, sr, float(start), float(end))
        if len(cleaned) == 0:
            continue
        segment_slices[idx] = cleaned
        all_slices.append(cleaned)
        spk = (seg.get("speaker") or "DEFAULT").strip() or "DEFAULT"
        speaker_pools.setdefault(spk, []).append(cleaned)

    fallback_prompt: Path | None = None
    if all_slices:
        global_audio = np.concatenate(all_slices)[: int(8.0 * sr)]
        if len(global_audio) >= int(0.5 * sr):
            fallback_prompt = prompts_dir / "fallback.wav"
            sf.write(str(fallback_prompt), np.clip(global_audio, -1.0, 1.0), sr, subtype="PCM_16")

    speaker_fallbacks: dict[str, Path] = {}
    for spk, chunks in speaker_pools.items():
        merged = np.concatenate(chunks)[: int(8.0 * sr)]
        if len(merged) >= int(0.8 * sr):
            safe_spk = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in spk)
            spk_path = prompts_dir / f"speaker_{safe_spk}.wav"
            sf.write(str(spk_path), np.clip(merged, -1.0, 1.0), sr, subtype="PCM_16")
            speaker_fallbacks[spk] = spk_path

    prompt_map: dict[int, Path] = {}
    min_samples = int(min_segment_prompt_s * sr)
    for idx, seg in enumerate(segments):
        spk = (seg.get("speaker") or "DEFAULT").strip() or "DEFAULT"
        seg_audio = segment_slices.get(idx)
        if seg_audio is not None and len(seg_audio) >= min_samples:
            seg_path = prompts_dir / f"seg_{idx:04d}.wav"
            sf.write(str(seg_path), np.clip(seg_audio, -1.0, 1.0), sr, subtype="PCM_16")
            prompt_map[idx] = seg_path
        elif spk in speaker_fallbacks:
            prompt_map[idx] = speaker_fallbacks[spk]
        elif fallback_prompt is not None:
            prompt_map[idx] = fallback_prompt

    return prompt_map, fallback_prompt


def _estimate_segment_prosody(
    segments: list[dict[str, Any]],
    vocals_wav_path: str | Path | None,
    *,
    emotion_intensity: float = 1.0,
) -> dict[int, SegmentProsody]:
    """Estimate per-segment prosody and Chatterbox exaggeration from original vocal energy."""
    if not vocals_wav_path or not Path(vocals_wav_path).is_file():
        return {}

    audio, sr = sf.read(str(vocals_wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    rms_by_idx: dict[int, float] = {}
    for idx, seg in enumerate(segments):
        start = seg.get("start")
        end = seg.get("end")
        if start is None or end is None:
            continue
        cleaned = extract_clean_prompt_slice(audio, sr, float(start), float(end))
        if len(cleaned) == 0:
            continue
        rms_by_idx[idx] = float(np.sqrt(np.mean(np.square(cleaned)) + 1e-12))

    active_rms = [v for v in rms_by_idx.values() if v > 0.005]
    if not active_rms:
        return {}

    median_rms = float(np.median(active_rms))
    profiles: dict[int, SegmentProsody] = {}
    for idx, rms in rms_by_idx.items():
        ratio = rms / max(median_rms, 1e-6)
        raw_exag = (0.55 + (ratio - 1.0) * 0.38) * emotion_intensity
        exaggeration = round(min(1.15, max(0.30, raw_exag)), 2)

        if ratio >= 1.35:
            profiles[idx] = SegmentProsody(rate="+8%", pitch="+5Hz", exaggeration=exaggeration)
        elif ratio <= 0.70:
            profiles[idx] = SegmentProsody(rate="-6%", pitch="-4Hz", exaggeration=exaggeration)
        else:
            profiles[idx] = SegmentProsody(rate="+0%", pitch="+0Hz", exaggeration=exaggeration)

    return profiles


def _run_chatterbox_batch(
    *,
    tts_python: Path,
    manifest_path: Path,
    total_items: int,
    progress_callback: ProgressCallback | None,
) -> None:
    """Invoke the isolated Chatterbox worker script and stream progress updates."""
    worker_script = Path(__file__).resolve().parent / "cloning_worker.py"
    cmd = [str(tts_python), str(worker_script), "--manifest", str(manifest_path)]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    tail_lines: list[str] = []
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.strip()
        if not line:
            continue
        tail_lines.append(line)
        if len(tail_lines) > 25:
            tail_lines.pop(0)

        if line.startswith("LOADING") and progress_callback:
            progress_callback("synthesizing", 61.0, "Loading Chatterbox Multilingual voice cloner...")
        elif line.startswith("PROGRESS ") and progress_callback:
            parts = line.split()
            if len(parts) == 3:
                try:
                    done = int(parts[1])
                    total = max(1, int(parts[2]))
                    pct = 62.0 + (done / total) * 20.0
                    progress_callback(
                        "synthesizing",
                        round(pct, 1),
                        f"Cloning voice & emotion ({done}/{total} segments)...",
                    )
                except ValueError:
                    pass

    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Chatterbox worker exited with code {return_code}: {' | '.join(tail_lines[-5:])}"
        )


def resolve_tts_engine(
    requested_engine: str,
    *,
    target_language: str,
    tts_python: Path | None,
    vocals_wav_path: str | Path | None,
    warnings_out: list[str] | None = None,
) -> str:
    """Determine whether to run 'chatterbox' or 'edge_tts' based on availability and language."""
    norm_lang = normalize_lang_code(target_language)
    can_clone = (
        tts_python is not None
        and tts_python.is_file()
        and vocals_wav_path is not None
        and Path(vocals_wav_path).is_file()
        and norm_lang in CHATTERBOX_SUPPORTED_LANGUAGES
    )

    if requested_engine == "edge_tts":
        return "edge_tts"

    if can_clone:
        return "chatterbox"

    if requested_engine == "chatterbox" and warnings_out is not None:
        if not tts_python or not tts_python.is_file():
            warnings_out.append("Chatterbox TTS environment not found; fell back to Edge-TTS.")
        elif not vocals_wav_path or not Path(vocals_wav_path).is_file():
            warnings_out.append("Chatterbox cloning requires separated vocals stem; fell back to Edge-TTS.")
        elif norm_lang not in CHATTERBOX_SUPPORTED_LANGUAGES:
            warnings_out.append(
                f"Chatterbox Multilingual does not support '{norm_lang}'; fell back to Edge-TTS."
            )

    return "edge_tts"


def synthesize_segments(
    segments: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    target_language: str,
    source_language: str | None = None,
    tts_engine: str = "auto",
    tts_python: Path | None = None,
    speaker_voice_map: dict[str, str] | None = None,
    vocals_wav_path: str | Path | None = None,
    emotion_intensity: float = 1.0,
    accent_bleed_control: float = 0.25,
    synth_fn: ClipSynthFn | None = None,
    progress_callback: ProgressCallback | None = None,
    warnings_out: list[str] | None = None,
    engine_used_out: list[str] | None = None,
) -> tuple[list[SynthesizedClip], dict[str, str]]:
    """Synthesize translated text using Chatterbox voice cloning or Edge-TTS."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    voice_map = assign_speaker_voices(segments, target_language, speaker_voice_map)
    prosody_map = _estimate_segment_prosody(
        segments,
        vocals_wav_path,
        emotion_intensity=emotion_intensity,
    )

    speakable = [
        (idx, seg)
        for idx, seg in enumerate(segments)
        if (seg.get("translated_text") or seg.get("text") or "").strip()
        and seg.get("start") is not None
        and seg.get("end") is not None
    ]
    total = max(1, len(speakable))

    chosen_engine = (
        "custom"
        if synth_fn is not None
        else resolve_tts_engine(
            tts_engine,
            target_language=target_language,
            tts_python=tts_python,
            vocals_wav_path=vocals_wav_path,
            warnings_out=warnings_out,
        )
    )

    if chosen_engine == "chatterbox" and tts_python is not None and vocals_wav_path is not None:
        prompts_dir = out_dir / "prompts"
        prompt_map, fallback_prompt = prepare_segment_prompts(
            segments,
            vocals_wav_path,
            prompts_dir,
        )
        norm_tgt = normalize_lang_code(target_language)
        norm_src = normalize_lang_code(source_language) if source_language else ""
        cfg_weight = 0.5 if norm_src == norm_tgt else float(accent_bleed_control)

        manifest_items: list[dict[str, Any]] = []
        clips: list[SynthesizedClip] = []
        cloned_voices: dict[str, str] = {}

        for idx, seg in speakable:
            text = (seg.get("translated_text") or seg.get("text") or "").strip()
            speaker = (seg.get("speaker") or "DEFAULT").strip() or "DEFAULT"
            wav_path = out_dir / f"seg_{idx:04d}.wav"
            prompt_path = prompt_map.get(idx) or fallback_prompt
            prosody = prosody_map.get(idx)
            exaggeration = prosody.exaggeration if prosody else round(min(1.15, max(0.3, 0.55 * emotion_intensity)), 2)

            manifest_items.append(
                {
                    "index": idx,
                    "text": text,
                    "language_id": norm_tgt,
                    "audio_prompt_path": str(prompt_path) if prompt_path else None,
                    "output_wav": str(wav_path),
                    "exaggeration": exaggeration,
                    "cfg_weight": cfg_weight,
                }
            )
            cloned_voices[speaker] = "chatterbox-cloned"
            clips.append(
                SynthesizedClip(
                    index=idx,
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    speaker=speaker,
                    voice="chatterbox-cloned",
                    wav_path=wav_path,
                )
            )

        manifest_path = out_dir / "chatterbox_manifest.json"
        manifest_path.write_text(json.dumps({"items": manifest_items}, indent=2), encoding="utf-8")

        try:
            _run_chatterbox_batch(
                tts_python=tts_python,
                manifest_path=manifest_path,
                total_items=len(manifest_items),
                progress_callback=progress_callback,
            )
            if all(clip.wav_path.is_file() for clip in clips):
                if engine_used_out is not None:
                    engine_used_out.append("chatterbox")
                return clips, cloned_voices
            raise RuntimeError("Chatterbox completed without generating all expected audio clips.")
        except Exception as exc:
            logger.warning("Chatterbox synthesis failed (%s); falling back to Edge-TTS.", exc)
            if warnings_out is not None:
                warnings_out.append(f"Chatterbox voice cloning failed ({exc}); fell back to Edge-TTS.")

    clips = []
    for pos, (idx, seg) in enumerate(speakable, start=1):
        text = (seg.get("translated_text") or seg.get("text") or "").strip()
        speaker = (seg.get("speaker") or "DEFAULT").strip() or "DEFAULT"
        voice = voice_map.get(speaker, get_voices_for_language(target_language)[0])
        wav_path = out_dir / f"seg_{idx:04d}.wav"

        if synth_fn is not None:
            synth_fn(text, voice, wav_path)
        else:
            prosody = prosody_map.get(idx)
            rate = prosody.rate if prosody else "+0%"
            pitch = prosody.pitch if prosody else "+0Hz"
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

    if engine_used_out is not None:
        engine_used_out.append("edge_tts" if synth_fn is None else "custom")
    return clips, voice_map
