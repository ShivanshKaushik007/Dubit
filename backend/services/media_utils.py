"""FFmpeg media probing, audio extraction, waveform loading, and A/V muxing utilities."""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import numpy as np

logger = logging.getLogger("dubit.media")

WHISPER_SAMPLE_RATE = 16000
_DURATION_PATTERN = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_AUDIO_STREAM_PATTERN = re.compile(r"Stream #\d+:\d+.*Audio:")
_VIDEO_STREAM_PATTERN = re.compile(r"Stream #\d+:\d+.*Video:")


class MediaError(RuntimeError):
    """Raised when input media cannot be probed, decoded, or muxed."""


@dataclass(frozen=True)
class MediaProbe:
    """Metadata parsed from FFmpeg stream inspection."""

    has_audio: bool
    has_video: bool
    duration_seconds: float | None


def get_ffmpeg_path() -> str:
    """Return the bundled FFmpeg executable path from imageio-ffmpeg."""
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_media(media_path: str | Path) -> MediaProbe:
    """Inspect streams and duration of a media file using the bundled FFmpeg binary."""
    path = Path(media_path)
    if not path.is_file():
        raise MediaError(f"Media file does not exist: {path}")

    cmd = [get_ffmpeg_path(), "-hide_banner", "-i", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    stderr = proc.stderr or ""

    has_audio = bool(_AUDIO_STREAM_PATTERN.search(stderr))
    has_video = bool(_VIDEO_STREAM_PATTERN.search(stderr))

    duration_seconds: float | None = None
    match = _DURATION_PATTERN.search(stderr)
    if match:
        hours, minutes, seconds = match.groups()
        duration_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    if not has_audio and not has_video:
        tail = stderr.strip().splitlines()[-1] if stderr.strip() else "Unrecognized media format"
        raise MediaError(f"Invalid or unreadable media file: {tail}")

    return MediaProbe(has_audio=has_audio, has_video=has_video, duration_seconds=duration_seconds)


def extract_audio(
    media_path: str | Path,
    output_audio_path: str | Path,
    sample_rate: int = WHISPER_SAMPLE_RATE,
) -> str:
    """Extract a 16 kHz mono 16-bit PCM WAV track from an audio or video file."""
    src = Path(media_path)
    dst = Path(output_audio_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    probe = probe_media(src)
    if not probe.has_audio:
        raise MediaError("Uploaded media file does not contain an audio stream.")

    if dst.exists():
        dst.unlink()

    cmd = [
        get_ffmpeg_path(),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        str(dst),
        "-y",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not dst.is_file():
        detail = (proc.stderr or "").strip() or f"FFmpeg exited with status {proc.returncode}"
        raise MediaError(f"Failed to extract audio: {detail}")

    return str(dst)


def load_audio_waveform(audio_path: str | Path, sample_rate: int = WHISPER_SAMPLE_RATE) -> np.ndarray:
    """Decode audio via imageio-ffmpeg into a mono float32 waveform without requiring PATH ffmpeg."""
    src = Path(audio_path)
    if not src.is_file():
        raise MediaError(f"Audio file not found: {src}")

    cmd = [
        get_ffmpeg_path(),
        "-nostdin",
        "-threads",
        "0",
        "-i",
        str(src),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise MediaError(f"Failed to decode audio waveform: {err}")

    if not proc.stdout:
        raise MediaError("Decoded audio stream is empty.")

    return np.frombuffer(proc.stdout, dtype=np.int16).flatten().astype(np.float32) / 32768.0


def merge_audio_video(
    video_path: str | Path,
    new_audio_path: str | Path,
    output_video_path: str | Path,
) -> str:
    """Replace the video's audio track with newly synthesized audio while keeping video stream intact."""
    video_src = Path(video_path)
    audio_src = Path(new_audio_path)
    dst = Path(output_video_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    probe = probe_media(video_src)
    if not probe.has_video:
        raise MediaError("Cannot mux audio+video because the source file has no video stream.")

    cmd = [
        get_ffmpeg_path(),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_src),
        "-i",
        str(audio_src),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        str(dst),
        "-y",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not dst.is_file():
        detail = (proc.stderr or "").strip() or f"FFmpeg exited with status {proc.returncode}"
        raise MediaError(f"Failed to mux audio and video: {detail}")

    return str(dst)
