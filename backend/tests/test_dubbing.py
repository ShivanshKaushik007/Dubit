"""Tests for translation, TTS voice assignment, FFmpeg time-fitting/mixing, and dubbing API."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main
from config import Settings
from services.job_manager import JobManager
from services.media_utils import get_ffmpeg_path
from services.mixer import fit_clip_to_slot, mix_dubbed_audio
from services.separation import StemOutputs
from services.translation import estimate_max_new_tokens, translate_segments
from services.tts import SynthesizedClip, assign_speaker_voices


def _make_tone_wav(dest: Path, *, duration: float = 0.8, freq: int = 440, channels: int = 1, sr: int = 44100) -> Path:
    subprocess.run(
        [
            get_ffmpeg_path(),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={freq}:duration={duration}",
            "-ac",
            str(channels),
            "-ar",
            str(sr),
            str(dest),
            "-y",
        ],
        check=True,
    )
    return dest


class DubbingServicesTests(unittest.TestCase):
    def test_translation_same_language_and_token_budget(self) -> None:
        segs = [{"start": 0.0, "end": 2.0, "text": "Hello studio"}]
        out = translate_segments(segs, source_language="en", target_language="en")
        self.assertEqual(out[0]["translated_text"], "Hello studio")
        self.assertEqual(out[0]["original_text"], "Hello studio")
        self.assertGreaterEqual(estimate_max_new_tokens(2.0, "Hello studio"), 24)

    def test_assign_speaker_voices_and_override(self) -> None:
        segs = [
            {"speaker": "SPEAKER_00", "text": "One"},
            {"speaker": "SPEAKER_01", "text": "Two"},
        ]
        assigned = assign_speaker_voices(
            segs,
            "es",
            speaker_voice_map={"SPEAKER_01": "es-MX-JorgeNeural"},
        )
        self.assertEqual(assigned["SPEAKER_00"], "es-ES-ElviraNeural")
        self.assertEqual(assigned["SPEAKER_01"], "es-MX-JorgeNeural")

    def test_mixer_time_fit_and_ducking_mix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            long_clip = _make_tone_wav(tmp_dir / "long.wav", duration=1.6, freq=520)
            bg_wav = _make_tone_wav(tmp_dir / "bg.wav", duration=2.0, freq=220, channels=2)
            fitted_path = tmp_dir / "fitted.wav"

            _, fitted_dur = fit_clip_to_slot(long_clip, slot_duration_s=1.0, output_path=fitted_path, max_speed=1.6)
            self.assertLessEqual(fitted_dur, 1.15)

            mixed_out = mix_dubbed_audio(
                [
                    SynthesizedClip(
                        index=0,
                        start=0.2,
                        end=1.2,
                        speaker="SPEAKER_00",
                        voice="en-US-JennyNeural",
                        wav_path=long_clip,
                    )
                ],
                tmp_dir / "dubbed.wav",
                total_duration_s=2.0,
                background_wav_path=bg_wav,
                work_dir=tmp_dir / "work",
            )
            self.assertTrue(mixed_out.is_file())
            info = sf.info(str(mixed_out))
            self.assertEqual(info.channels, 2)
            self.assertEqual(info.samplerate, 44100)
            data, _ = sf.read(str(mixed_out), dtype="float32")
            self.assertLessEqual(float(np.max(np.abs(data))), 1.0)

    def test_loudness_matching_and_non_speech_gap_preservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            sr = 44100
            orig_vocals = np.zeros((sr * 3, 2), dtype=np.float32)
            # Quiet segment [0.0, 0.8], non-speech breath/laugh in [1.0, 1.5], loud segment [1.8, 2.6]
            t1 = np.linspace(0, 0.8, int(sr * 0.8), endpoint=False, dtype=np.float32)
            orig_vocals[: len(t1), :] = (0.04 * np.sin(2 * np.pi * 440 * t1))[:, None]
            t_gap = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
            orig_vocals[sr : sr + len(t_gap), :] = (0.15 * np.sin(2 * np.pi * 330 * t_gap))[:, None]
            t2 = np.linspace(0, 0.8, int(sr * 0.8), endpoint=False, dtype=np.float32)
            orig_vocals[int(1.8 * sr) : int(1.8 * sr) + len(t2), :] = (0.30 * np.sin(2 * np.pi * 440 * t2))[:, None]

            vocals_path = tmp_dir / "vocals.wav"
            sf.write(str(vocals_path), orig_vocals, sr, subtype="PCM_16")

            uniform_clip_1 = _make_tone_wav(tmp_dir / "c1.wav", duration=0.6, freq=500)
            uniform_clip_2 = _make_tone_wav(tmp_dir / "c2.wav", duration=0.6, freq=500)

            mixed_out = mix_dubbed_audio(
                [
                    SynthesizedClip(0, 0.0, 0.8, "S0", "v", uniform_clip_1),
                    SynthesizedClip(1, 1.8, 2.6, "S0", "v", uniform_clip_2),
                ],
                tmp_dir / "matched.wav",
                total_duration_s=3.0,
                vocals_wav_path=vocals_path,
                room_reverb=0.0,
                work_dir=tmp_dir / "work",
            )
            mixed, _ = sf.read(str(mixed_out), dtype="float32")
            quiet_rms = float(np.sqrt(np.mean(np.square(mixed[int(0.1 * sr) : int(0.5 * sr)]))))
            loud_rms = float(np.sqrt(np.mean(np.square(mixed[int(1.9 * sr) : int(2.3 * sr)]))))
            gap_rms = float(np.sqrt(np.mean(np.square(mixed[int(1.1 * sr) : int(1.4 * sr)]))))

            self.assertGreater(loud_rms, quiet_rms * 2.0)
            self.assertGreater(gap_rms, 0.01)


class DubbingApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self._settings = Settings(
            base_dir=root,
            jobs_dir=root / "jobs",
            uploads_dir=root / "uploads",
            workdir_dir=root / "workdir",
            output_dir=root / "output",
        )

        def _fake_transcribe(audio_path: str, **kwargs):
            return {
                "language": "en",
                "segments": [
                    {"start": 0.0, "end": 0.8, "text": "Hello world", "speaker": "SPEAKER_00"}
                ],
                "warnings": [],
                "metadata": {"model_size": "tiny", "device": "cpu", "compute_type": "int8"},
            }

        def _fake_separate(media_path, output_dir, **kwargs):
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            v = _make_tone_wav(out / "vocals.wav", duration=0.8, freq=440, channels=2)
            b = _make_tone_wav(out / "background.wav", duration=0.8, freq=220, channels=2)
            return StemOutputs(vocals_path=v, background_path=b, sample_rate=44100)

        def _fake_translate(segments, *, source_language, target_language, **kwargs):
            return [
                {**seg, "original_text": seg["text"], "translated_text": "Hola mundo"}
                for seg in segments
            ]

        def _fake_synth(text: str, voice: str, dest_wav: Path) -> Path:
            dest_wav.parent.mkdir(parents=True, exist_ok=True)
            return _make_tone_wav(dest_wav, duration=0.6, freq=500, channels=1)

        self._manager = JobManager(
            self._settings,
            transcribe_fn=_fake_transcribe,
            separate_fn=_fake_separate,
            translate_fn=_fake_translate,
            synth_fn=_fake_synth,
        )
        main.app.state.job_manager = self._manager
        self._client_ctx = TestClient(main.app)
        self.client = self._client_ctx.__enter__()
        self.wav_bytes = _make_tone_wav(root / "input.wav", duration=0.8, sr=16000).read_bytes()

    def tearDown(self) -> None:
        self._client_ctx.__exit__(None, None, None)
        self._manager.shutdown()
        main.app.state.job_manager = None
        self._tmp.cleanup()

    def test_transcribe_then_dub_edit_and_resynthesize(self) -> None:
        create_res = self.client.post(
            "/jobs",
            files={"file": ("clip.wav", self.wav_bytes, "audio/wav")},
            data={"do_diarization": "true", "model_size": "tiny"},
        )
        self.assertEqual(create_res.status_code, 202)
        job_id = create_res.json()["job_id"]
        finished = self._manager.wait_for_job(job_id, timeout=10.0)
        self.assertEqual(finished.status, "completed")

        dub_res = self.client.post(
            f"/jobs/{job_id}/dub",
            json={"target_language": "es", "preserve_background": True},
        )
        self.assertEqual(dub_res.status_code, 202)
        dubbed = self._manager.wait_for_job(job_id, timeout=10.0)
        self.assertEqual(dubbed.status, "completed")
        self.assertTrue(dubbed.has_dubbed_audio)
        self.assertTrue(dubbed.has_stems)
        self.assertEqual(dubbed.target_language, "es")
        self.assertEqual(dubbed.data["segments"][0]["translated_text"], "Hola mundo")

        stream_res = self.client.get(f"/jobs/{job_id}/dubbed")
        self.assertEqual(stream_res.status_code, 200)

        vtt_res = self.client.get(f"/jobs/{job_id}/download/dubbed_vtt")
        self.assertEqual(vtt_res.status_code, 200)
        self.assertIn("Hola mundo", vtt_res.text)

        patch_res = self.client.patch(
            f"/jobs/{job_id}",
            json={"segments": [{"index": 0, "translated_text": "Hola amigos"}]},
        )
        self.assertEqual(patch_res.status_code, 200)

        redub_res = self.client.post(
            f"/jobs/{job_id}/dub",
            json={"target_language": "es", "skip_translation": True},
        )
        self.assertEqual(redub_res.status_code, 202)
        redubbed = self._manager.wait_for_job(job_id, timeout=10.0)
        self.assertEqual(redubbed.data["segments"][0]["translated_text"], "Hola amigos")


if __name__ == "__main__":
    unittest.main()
