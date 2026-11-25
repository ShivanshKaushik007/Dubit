"""Export formatters for JSON, WebVTT, SubRip (SRT), and CSV transcripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _split_timestamp(seconds: float) -> tuple[int, int, int, int]:
    """Convert non-negative seconds into (hours, minutes, seconds, milliseconds) with rounding."""
    total_ms = round(max(0.0, float(seconds)) * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1_000)
    return hours, minutes, secs, millis


def format_vtt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm for WebVTT cues."""
    hours, minutes, secs, millis = _split_timestamp(seconds)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_srt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm for SubRip (SRT) cues."""
    hours, minutes, secs, millis = _split_timestamp(seconds)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _sanitize_csv_cell(value: str) -> str:
    """Neutralize spreadsheet formula injection when exporting untrusted text to CSV."""
    if value and value[0] in _CSV_FORMULA_PREFIXES:
        return f"'{value}"
    return value


def _format_subtitle_cues(
    result: dict[str, Any],
    *,
    decimal_marker: str,
    include_index: bool,
) -> list[str]:
    """Build subtitle cue blocks shared by WebVTT and SRT writers."""
    time_formatter = format_vtt_time if decimal_marker == "." else format_srt_time
    lines: list[str] = []
    if not include_index:
        lines.append("WEBVTT")
        lines.append("")

    for idx, segment in enumerate(result.get("segments", []), start=1):
        start_val = segment.get("start")
        end_val = segment.get("end")
        if start_val is None or end_val is None:
            continue

        start = time_formatter(float(start_val))
        end = time_formatter(float(end_val))
        speaker = (segment.get("speaker") or "").strip()
        text = (segment.get("text") or "").strip()
        cue_text = f"[{speaker}]: {text}" if speaker else text

        if include_index:
            lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(cue_text)
        lines.append("")

    return lines


def result_to_json(result: dict[str, Any], output_path: str | Path) -> None:
    """Write transcription payload to formatted UTF-8 JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def result_to_vtt(result: dict[str, Any], output_path: str | Path) -> None:
    """Write transcription segments as WebVTT (.vtt)."""
    lines = _format_subtitle_cues(result, decimal_marker=".", include_index=False)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def result_to_srt(result: dict[str, Any], output_path: str | Path) -> None:
    """Write transcription segments as SubRip (.srt)."""
    lines = _format_subtitle_cues(result, decimal_marker=",", include_index=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def result_to_csv(result: dict[str, Any], output_path: str | Path) -> None:
    """Write transcription segments to CSV with formula-injection protection."""
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_time", "end_time", "speaker", "text"])
        for segment in result.get("segments", []):
            start_val = segment.get("start")
            end_val = segment.get("end")
            speaker = (segment.get("speaker") or "").strip()
            text = (segment.get("text") or "").strip()
            writer.writerow(
                [
                    round(float(start_val), 3) if start_val is not None else "",
                    round(float(end_val), 3) if end_val is not None else "",
                    _sanitize_csv_cell(speaker),
                    _sanitize_csv_cell(text),
                ]
            )


FORMAT_WRITERS = {
    "json": ("transcript.json", result_to_json),
    "webvtt": ("transcript.vtt", result_to_vtt),
    "srt": ("transcript.srt", result_to_srt),
    "csv": ("transcript.csv", result_to_csv),
}


def normalize_format_name(fmt: str) -> str:
    """Normalize format aliases such as 'vtt' -> 'webvtt'."""
    lowered = fmt.strip().lower()
    if lowered == "vtt":
        return "webvtt"
    if lowered not in FORMAT_WRITERS:
        raise ValueError(f"Unsupported export format: {fmt}")
    return lowered


def export_all_formats(
    result: dict[str, Any],
    output_dir: str | Path,
    stem: str | None = None,
) -> dict[str, Path]:
    """Generate JSON, WebVTT, SRT, and CSV outputs in output_dir and return a format->path map."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for fmt, (default_name, writer) in FORMAT_WRITERS.items():
        ext = Path(default_name).suffix
        filename = f"{stem}{ext}" if stem else default_name
        out_path = target_dir / filename
        writer(result, out_path)
        written[fmt] = out_path
    return written
