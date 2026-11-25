"""Integration tests for FFmpeg media probing, extraction, waveform decoding, and muxing."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.media_utils import (
    MediaError,
    extract_audio,
    get_ffmpeg_path,
    load_audio_waveform,
    merge_audio_video,
    probe_media,
)


def _generate_synthetic_wav(dest: Path, duration: float = 1.0) -> None:
    cmd = [
        get_ffmpeg_path(),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={duration}",
        "-ac",
        "2",
        "-ar",
        "44100",
        str(dest),
        "-y",
    ]
    subprocess.run(cmd, check=True)


def _generate_silent_mp4(dest: Path, duration: float = 1.0) -> None:
    cmd = [
        get_ffmpeg_path(),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s=160x120:d={duration}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(dest),
        "-y",
    ]
    subprocess.run(cmd, check=True)


class MediaUtilsTests(unittest.TestCase):
    def test_extract_and_load_waveform_without_path_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            src_wav = tmp_dir / "stereo.wav"
            extracted = tmp_dir / "mono16k.wav"
            _generate_synthetic_wav(src_wav, duration=1.0)

            probe = probe_media(src_wav)
            self.assertTrue(probe.has_audio)
            self.assertFalse(probe.has_video)
            self.assertIsNotNone(probe.duration_seconds)

            out_path = extract_audio(src_wav, extracted)
            self.assertTrue(Path(out_path).is_file())

            waveform = load_audio_waveform(out_path)
            self.assertEqual(waveform.dtype, np.float32)
            self.assertGreater(waveform.shape[0], 14000)
            self.assertLessEqual(float(np.max(np.abs(waveform))), 1.0)

    def test_silent_video_raises_media_error_and_merge_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            silent_mp4 = tmp_dir / "silent.mp4"
            audio_wav = tmp_dir / "tone.wav"
            muxed_mp4 = tmp_dir / "muxed.mp4"
            _generate_silent_mp4(silent_mp4, duration=1.0)
            _generate_synthetic_wav(audio_wav, duration=1.0)

            with self.assertRaises(MediaError):
                extract_audio(silent_mp4, tmp_dir / "should_fail.wav")

            out_video = merge_audio_video(silent_mp4, audio_wav, muxed_mp4)
            self.assertTrue(Path(out_video).is_file())
            muxed_probe = probe_media(out_video)
            self.assertTrue(muxed_probe.has_video)
            self.assertTrue(muxed_probe.has_audio)


if __name__ == "__main__":
    unittest.main()
