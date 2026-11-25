"""Unit tests for transcript export formatters."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.formatters import (
    export_all_formats,
    format_srt_time,
    format_vtt_time,
    normalize_format_name,
    result_to_csv,
    result_to_srt,
    result_to_vtt,
)


class FormatterTests(unittest.TestCase):
    def test_timestamp_rounding_and_carry(self) -> None:
        self.assertEqual(format_vtt_time(0.0), "00:00:00.000")
        self.assertEqual(format_vtt_time(-2.5), "00:00:00.000")
        self.assertEqual(format_vtt_time(59.9999), "00:01:00.000")
        self.assertEqual(format_srt_time(3661.234), "01:01:01,234")

    def test_vtt_and_srt_omit_unknown_when_not_diarized(self) -> None:
        payload = {
            "language": "en",
            "segments": [
                {"start": 0.5, "end": 2.25, "text": " Hello world "},
                {"start": 2.5, "end": 4.0, "speaker": "SPEAKER_00", "text": "Second line"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            vtt_path = Path(tmp) / "out.vtt"
            srt_path = Path(tmp) / "out.srt"
            result_to_vtt(payload, vtt_path)
            result_to_srt(payload, srt_path)

            vtt_text = vtt_path.read_text(encoding="utf-8")
            srt_text = srt_path.read_text(encoding="utf-8")

        self.assertTrue(vtt_text.startswith("WEBVTT\n\n"))
        self.assertIn("00:00:00.500 --> 00:00:02.250\nHello world\n", vtt_text)
        self.assertNotIn("[UNKNOWN]", vtt_text)
        self.assertIn("00:00:02.500 --> 00:00:04.000\n[SPEAKER_00]: Second line\n", vtt_text)

        self.assertIn("1\n00:00:00,500 --> 00:00:02,250\nHello world\n", srt_text)
        self.assertIn("2\n00:00:02,500 --> 00:00:04,000\n[SPEAKER_00]: Second line\n", srt_text)

    def test_csv_sanitizes_formula_injection(self) -> None:
        payload = {
            "segments": [
                {"start": 0.1234, "end": 1.5, "speaker": "=CMD", "text": "+SUM(1,2)"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "out.csv"
            result_to_csv(payload, csv_path)
            with open(csv_path, encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))

        self.assertEqual(rows[0], ["start_time", "end_time", "speaker", "text"])
        self.assertEqual(rows[1], ["0.123", "1.5", "'=CMD", "'+SUM(1,2)"])

    def test_export_all_formats_and_alias(self) -> None:
        self.assertEqual(normalize_format_name("vtt"), "webvtt")
        self.assertEqual(normalize_format_name("SRT"), "srt")
        with self.assertRaises(ValueError):
            normalize_format_name("docx")

        payload = {"language": "en", "segments": [{"start": 0.0, "end": 1.0, "text": "Hi"}]}
        with tempfile.TemporaryDirectory() as tmp:
            written = export_all_formats(payload, tmp)
            self.assertEqual(set(written.keys()), {"json", "webvtt", "srt", "csv"})
            for path in written.values():
                self.assertTrue(path.is_file())
            parsed = json.loads(written["json"].read_text(encoding="utf-8"))
            self.assertEqual(parsed["segments"][0]["text"], "Hi")


if __name__ == "__main__":
    unittest.main()
