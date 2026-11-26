"""Isolated subprocess worker for Chatterbox Multilingual zero-shot voice cloning.

Intended to be executed with `backend/tts_venv/Scripts/python.exe` so Chatterbox's
dependency pins stay isolated from WhisperX/FastAPI in `backend/venv`.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path


def _clear_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    gc.collect()


def run_manifest(manifest_path: Path) -> None:
    """Load ChatterboxMultilingualTTS once and synthesize all clips in manifest_path."""
    import numpy as np
    import soundfile as sf
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    if not os.environ.get("HF_TOKEN", "").strip():
        os.environ.pop("HF_TOKEN", None)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("items") or []
    if not items:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"LOADING device={device}", flush=True)

    model = None
    try:
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        total = len(items)
        for pos, item in enumerate(items, start=1):
            text = str(item["text"]).strip()
            language_id = str(item["language_id"]).strip().lower()
            prompt_path = item.get("audio_prompt_path")
            out_path = Path(item["output_wav"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            exaggeration = float(item.get("exaggeration", 0.55))
            cfg_weight = float(item.get("cfg_weight", 0.25))

            if not text:
                continue

            wav = model.generate(
                text=text,
                language_id=language_id,
                audio_prompt_path=prompt_path if prompt_path and Path(prompt_path).is_file() else None,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
            audio_np = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
            sf.write(str(out_path), np.clip(audio_np, -1.0, 1.0), int(model.sr), subtype="PCM_16")
            print(f"PROGRESS {pos} {total}", flush=True)

            if pos % 8 == 0:
                _clear_cuda()
    finally:
        if model is not None:
            del model
        _clear_cuda()


def main() -> int:
    parser = argparse.ArgumentParser(description="Chatterbox Multilingual Dubbing Worker")
    parser.add_argument("--manifest", required=True, help="Path to JSON synthesis manifest")
    args = parser.parse_args()

    run_manifest(Path(args.manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
