"""Unit tests for transcription, alignment fallbacks, diarization wiring, and VRAM cleanup."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services import transcription


class TranscriptionTests(unittest.TestCase):
    def test_skips_diarization_when_disabled_and_clears_vram(self) -> None:
        fake_audio = np.zeros(1600, dtype=np.float32)
        fake_model = MagicMock()
        fake_model.transcribe.return_value = {
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
        }
        fake_align_model = MagicMock()

        with (
            patch.object(transcription, "load_audio_waveform", return_value=fake_audio),
            patch.object(transcription, "select_device_and_compute", return_value=("cuda", "int8_float16")),
            patch.object(transcription, "clear_vram") as mock_clear,
            patch.object(transcription.whisperx, "load_model", return_value=fake_model),
            patch.object(transcription.whisperx, "load_align_model", return_value=(fake_align_model, {"lang": "en"})),
            patch.object(
                transcription.whisperx,
                "align",
                return_value={"segments": [{"start": 0.0, "end": 1.0, "text": "Hello", "words": []}]},
            ),
        ):
            res = transcription.transcribe_and_diarize(
                "dummy.wav",
                model_size="tiny",
                do_diarization=False,
                hf_token="hf_dummy",
            )

        self.assertEqual(res["language"], "en")
        self.assertTrue(res["metadata"]["alignment_applied"])
        self.assertFalse(res["metadata"]["diarization_applied"])
        self.assertEqual(mock_clear.call_count, 2)

    def test_diarization_uses_whisperx_diarize_and_token_kwarg(self) -> None:
        fake_audio = np.zeros(1600, dtype=np.float32)
        fake_model = MagicMock()
        fake_model.transcribe.return_value = {
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
        }
        fake_align_model = MagicMock()
        fake_diarize_instance = MagicMock(return_value="DIARIZE_DF")
        fake_diarize_cls = MagicMock(return_value=fake_diarize_instance)

        fake_diarize_module = types.ModuleType("whisperx.diarize")
        fake_diarize_module.DiarizationPipeline = fake_diarize_cls

        with (
            patch.dict(sys.modules, {"whisperx.diarize": fake_diarize_module}),
            patch.object(transcription, "load_audio_waveform", return_value=fake_audio),
            patch.object(transcription, "select_device_and_compute", return_value=("cpu", "int8")),
            patch.object(transcription, "clear_vram") as mock_clear,
            patch.object(transcription.whisperx, "load_model", return_value=fake_model),
            patch.object(transcription.whisperx, "load_align_model", return_value=(fake_align_model, {"lang": "en"})),
            patch.object(
                transcription.whisperx,
                "align",
                return_value={"segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}]},
            ),
            patch.object(
                transcription.whisperx,
                "assign_word_speakers",
                return_value={"segments": [{"start": 0.0, "end": 1.0, "text": "Hello", "speaker": "SPEAKER_00"}]},
            ) as mock_assign,
        ):
            res = transcription.transcribe_and_diarize(
                "dummy.wav",
                model_size="tiny",
                do_diarization=True,
                min_speakers=1,
                max_speakers=3,
                hf_token="hf_secret",
            )

        fake_diarize_cls.assert_called_once_with(token="hf_secret", device="cpu")
        mock_assign.assert_called_once()
        self.assertEqual(mock_assign.call_args.kwargs.get("fill_nearest"), True)
        self.assertTrue(res["metadata"]["diarization_applied"])
        self.assertEqual(res["segments"][0]["speaker"], "SPEAKER_00")
        self.assertEqual(mock_clear.call_count, 3)

    def test_unsupported_alignment_language_falls_back_gracefully(self) -> None:
        fake_audio = np.zeros(1600, dtype=np.float32)
        fake_model = MagicMock()
        fake_model.transcribe.return_value = {
            "language": "or",
            "segments": [{"start": 0.0, "end": 1.2, "text": "Sample text"}],
        }

        with (
            patch.object(transcription, "load_audio_waveform", return_value=fake_audio),
            patch.object(transcription, "select_device_and_compute", return_value=("cpu", "int8")),
            patch.object(transcription, "clear_vram"),
            patch.object(transcription.whisperx, "load_model", return_value=fake_model),
            patch.object(
                transcription.whisperx,
                "load_align_model",
                side_effect=ValueError("No default align-model for language: or"),
            ),
        ):
            res = transcription.transcribe_and_diarize(
                "dummy.wav",
                model_size="tiny",
                do_diarization=True,
                hf_token=None,
            )

        self.assertFalse(res["metadata"]["alignment_applied"])
        self.assertFalse(res["metadata"]["diarization_applied"])
        self.assertEqual(len(res["warnings"]), 2)
        self.assertEqual(res["segments"][0]["text"], "Sample text")

    def test_vram_cleared_even_when_transcription_raises(self) -> None:
        fake_audio = np.zeros(1600, dtype=np.float32)
        fake_model = MagicMock()
        fake_model.transcribe.side_effect = RuntimeError("Simulated CUDA error")

        with (
            patch.object(transcription, "load_audio_waveform", return_value=fake_audio),
            patch.object(transcription, "select_device_and_compute", return_value=("cuda", "int8_float16")),
            patch.object(transcription, "clear_vram") as mock_clear,
            patch.object(transcription.whisperx, "load_model", return_value=fake_model),
        ):
            with self.assertRaises(RuntimeError):
                transcription.transcribe_and_diarize("dummy.wav", model_size="tiny")

        mock_clear.assert_called_once()


if __name__ == "__main__":
    unittest.main()
