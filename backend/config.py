"""Application configuration and lightweight .env loader for the Dubit backend."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def load_dotenv_file(env_path: Path) -> None:
    """Load key=value pairs from a .env file without overriding existing environment vars."""
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        cleaned = value.strip()
        if (
            len(cleaned) >= 2
            and cleaned[0] == cleaned[-1]
            and cleaned[0] in ("'", '"')
        ):
            cleaned = cleaned[1:-1]
        os.environ.setdefault(key, cleaned)


load_dotenv_file(BASE_DIR / ".env")


def _parse_origins(raw: str | None) -> list[str]:
    if not raw:
        return ["http://localhost:3000", "http://127.0.0.1:3000"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


SUPPORTED_WHISPER_MODELS: tuple[str, ...] = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "large-v3-turbo",
)

SUPPORTED_COMPUTE_TYPES: tuple[str, ...] = (
    "auto",
    "int8_float16",
    "float16",
    "int8",
    "float32",
)

SUPPORTED_EXPORT_FORMATS: tuple[str, ...] = ("json", "webvtt", "srt", "csv")

ALLOWED_MEDIA_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp4",
        ".mkv",
        ".mov",
        ".webm",
        ".avi",
        ".m4v",
        ".mp3",
        ".wav",
        ".m4a",
        ".aac",
        ".flac",
        ".ogg",
        ".opus",
        ".mpeg",
        ".mpg",
    }
)


@dataclass(frozen=True)
class Settings:
    """Runtime settings derived from environment variables."""

    base_dir: Path = BASE_DIR
    jobs_dir: Path = field(default_factory=lambda: Path(os.environ.get("DUBIT_JOBS_DIR", BASE_DIR / "jobs")))
    uploads_dir: Path = field(default_factory=lambda: BASE_DIR / "uploads")
    workdir_dir: Path = field(default_factory=lambda: BASE_DIR / "workdir")
    output_dir: Path = field(default_factory=lambda: BASE_DIR / "output")
    allowed_origins: list[str] = field(default_factory=lambda: _parse_origins(os.environ.get("ALLOWED_ORIGINS")))
    max_upload_mb: int = field(default_factory=lambda: max(1, int(os.environ.get("MAX_UPLOAD_MB", "500"))))
    default_model_size: str = field(default_factory=lambda: os.environ.get("WHISPER_MODEL", "base"))
    default_compute_type: str = field(default_factory=lambda: os.environ.get("COMPUTE_TYPE", "auto"))
    default_batch_size: int = field(default_factory=lambda: max(1, int(os.environ.get("WHISPER_BATCH_SIZE", "8"))))
    default_vad_method: str = field(default_factory=lambda: os.environ.get("WHISPER_VAD_METHOD", "pyannote"))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def get_hf_token(self) -> str | None:
        token = os.environ.get("HF_TOKEN", "").strip()
        return token or None

    def ensure_directories(self) -> None:
        for directory in (self.jobs_dir, self.uploads_dir, self.workdir_dir, self.output_dir):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
