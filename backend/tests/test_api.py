"""API and JobManager integration tests using FastAPI TestClient and synthetic FFmpeg media."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main
from config import Settings
from services.job_manager import JobManager
from services.media_utils import get_ffmpeg_path


def _create_wav_bytes(tmp_dir: Path) -> bytes:
    wav_path = tmp_dir / "fixture.wav"
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
            "sine=frequency=440:duration=0.6",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav_path),
            "-y",
        ],
        check=True,
    )
    return wav_path.read_bytes()


def _fake_transcribe(
    audio_path: str,
    *,
    progress_callback=None,
    do_diarization: bool = False,
    **kwargs,
):
    if progress_callback:
        progress_callback("transcribing", 40.0, "Mock transcribing...")
        progress_callback("aligning", 70.0, "Mock aligning...")
    return {
        "language": "en",
        "segments": [
            {
                "start": 0.0,
                "end": 0.6,
                "text": "Test segment one",
                **({"speaker": "SPEAKER_00"} if do_diarization else {}),
                "words": [{"word": "Test", "start": 0.0, "end": 0.2}],
            }
        ],
        "warnings": [],
        "metadata": {
            "model_size": kwargs.get("model_size", "tiny"),
            "device": "cpu",
            "compute_type": "int8",
            "alignment_applied": True,
            "diarization_applied": do_diarization,
        },
    }


class ApiTests(unittest.TestCase):
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
        self._manager = JobManager(self._settings, transcribe_fn=_fake_transcribe)
        main.app.state.job_manager = self._manager
        self._client_ctx = TestClient(main.app)
        self.client = self._client_ctx.__enter__()
        self.wav_bytes = _create_wav_bytes(root)

    def tearDown(self) -> None:
        self._client_ctx.__exit__(None, None, None)
        self._manager.shutdown()
        main.app.state.job_manager = None
        self._tmp.cleanup()

    def test_health_endpoint(self) -> None:
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("tiny", body["supported_models"])
        self.assertIn("srt", body["supported_formats"])

    def test_full_job_lifecycle_edit_download_media_and_delete(self) -> None:
        create_res = self.client.post(
            "/jobs",
            files={"file": ("../../evil/sample clip.wav", self.wav_bytes, "audio/wav")},
            data={
                "output_format": "webvtt",
                "do_diarization": "true",
                "model_size": "tiny",
            },
        )
        self.assertEqual(create_res.status_code, 202)
        created = create_res.json()
        job_id = created["job_id"]
        self.assertEqual(created["filename"], "sample clip.wav")

        finished = self._manager.wait_for_job(job_id, timeout=10.0)
        self.assertIsNotNone(finished)
        self.assertEqual(finished.status, "completed")

        get_res = self.client.get(f"/jobs/{job_id}")
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.json()["data"]["segments"][0]["speaker"], "SPEAKER_00")

        patch_res = self.client.patch(
            f"/jobs/{job_id}",
            json={
                "segments": [{"index": 0, "text": "Edited segment"}],
                "speaker_map": {"SPEAKER_00": "Host"},
            },
        )
        self.assertEqual(patch_res.status_code, 200)
        patched_seg = patch_res.json()["data"]["segments"][0]
        self.assertEqual(patched_seg["text"], "Edited segment")
        self.assertEqual(patched_seg["speaker"], "Host")

        vtt_res = self.client.get(f"/jobs/{job_id}/download/webvtt")
        self.assertEqual(vtt_res.status_code, 200)
        self.assertIn("[Host]: Edited segment", vtt_res.text)

        srt_res = self.client.get(f"/jobs/{job_id}/download/srt")
        self.assertEqual(srt_res.status_code, 200)
        self.assertIn("[Host]: Edited segment", srt_res.text)

        media_res = self.client.get(f"/jobs/{job_id}/media")
        self.assertEqual(media_res.status_code, 200)
        self.assertEqual(media_res.content, self.wav_bytes)

        del_res = self.client.delete(f"/jobs/{job_id}")
        self.assertEqual(del_res.status_code, 204)
        self.assertEqual(self.client.get(f"/jobs/{job_id}").status_code, 404)

    def test_validation_rejects_bad_extension_and_oversized_upload(self) -> None:
        bad_ext = self.client.post(
            "/jobs",
            files={"file": ("malware.exe", b"MZ", "application/octet-stream")},
        )
        self.assertEqual(bad_ext.status_code, 400)

        with patch.object(
            type(main.settings),
            "max_upload_bytes",
            new_callable=PropertyMock,
            return_value=100,
        ):
            too_big = self.client.post(
                "/jobs",
                files={"file": ("big.wav", self.wav_bytes, "audio/wav")},
            )
            self.assertEqual(too_big.status_code, 413)

    def test_legacy_upload_route_still_works(self) -> None:
        res = self.client.post(
            "/upload",
            files={"file": ("legacy.wav", self.wav_bytes, "audio/wav")},
            data={"output_format": "JSON", "do_diarization": "false"},
        )
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["message"], "Transcription complete")
        self.assertIn("job_id", payload)
        self.assertEqual(payload["data"]["segments"][0]["text"], "Test segment one")


if __name__ == "__main__":
    unittest.main()
