"""Local multilingual translation service powered by M2M100 with duration awareness."""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable

import torch

from .gpu_manager import clear_vram

logger = logging.getLogger("dubit.translation")

ProgressCallback = Callable[[str, float, str], None]

SUPPORTED_DUB_LANGUAGES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "hi": "Hindi",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "pt": "Portuguese",
    "ar": "Arabic",
    "ru": "Russian",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "ur": "Urdu",
}

_LANGUAGE_ALIASES = {
    "zh-cn": "zh",
    "zh-tw": "zh",
    "nb": "no",
    "nn": "no",
    "jw": "jv",
}


def normalize_lang_code(lang: str | None) -> str:
    """Normalize Whisper/ISO language tags to M2M100 language codes."""
    if not lang:
        return "en"
    cleaned = lang.strip().lower()
    return _LANGUAGE_ALIASES.get(cleaned, cleaned.split("-")[0])


def estimate_max_new_tokens(duration_seconds: float, source_text: str) -> int:
    """Estimate a generous token cap proportional to utterance duration and source length."""
    duration_budget = max(16, int(duration_seconds * 9.5) + 16)
    source_budget = max(24, int(len(source_text.split()) * 2.5) + 16)
    return min(256, max(duration_budget, source_budget))


def translate_segments(
    segments: list[dict[str, Any]],
    *,
    source_language: str | None,
    target_language: str,
    model_name: str = "facebook/m2m100_418M",
    batch_size: int = 8,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Translate transcript segments into target_language while preserving timing metadata."""
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    src_code = normalize_lang_code(source_language)
    tgt_code = normalize_lang_code(target_language)
    translated_segments = copy.deepcopy(segments)

    if not translated_segments:
        return translated_segments

    if src_code == tgt_code:
        for seg in translated_segments:
            text = (seg.get("text") or "").strip()
            seg.setdefault("original_text", text)
            seg["translated_text"] = text
        return translated_segments

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    if progress_callback:
        progress_callback("translating", 40.0, f"Loading translation model ({src_code} -> {tgt_code})...")

    tokenizer = None
    model = None
    try:
        try:
            tokenizer = M2M100Tokenizer.from_pretrained(model_name, local_files_only=True)
            model = M2M100ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                local_files_only=True,
            )
        except Exception:
            logger.info("Local cache miss for %s; fetching from HuggingFace Hub.", model_name)
            tokenizer = M2M100Tokenizer.from_pretrained(model_name)
            model = M2M100ForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)

        if src_code not in tokenizer.lang_code_to_id:
            raise ValueError(f"Unsupported source translation language: '{src_code}'")
        if tgt_code not in tokenizer.lang_code_to_id:
            raise ValueError(f"Unsupported target translation language: '{tgt_code}'")

        model.to(device)
        model.eval()
        tokenizer.src_lang = src_code
        forced_bos_id = tokenizer.get_lang_id(tgt_code)

        non_empty_indices = [
            idx for idx, seg in enumerate(translated_segments) if (seg.get("text") or "").strip()
        ]
        total = max(1, len(non_empty_indices))

        for offset in range(0, len(non_empty_indices), batch_size):
            batch_indices = non_empty_indices[offset : offset + batch_size]
            batch_texts = [(translated_segments[i].get("text") or "").strip() for i in batch_indices]
            max_dur = max(
                float(translated_segments[i].get("end") or 0.0) - float(translated_segments[i].get("start") or 0.0)
                for i in batch_indices
            )
            max_tokens = max(estimate_max_new_tokens(max_dur, txt) for txt in batch_texts)

            encoded = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    forced_bos_token_id=forced_bos_id,
                    max_new_tokens=max_tokens,
                    num_beams=3,
                )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

            for idx, orig, trans in zip(batch_indices, batch_texts, decoded):
                seg = translated_segments[idx]
                seg["original_text"] = orig
                seg["translated_text"] = trans.strip()

            if progress_callback:
                pct = 42.0 + ((offset + len(batch_indices)) / total) * 16.0
                progress_callback(
                    "translating",
                    round(pct, 1),
                    f"Translated {min(offset + len(batch_indices), total)}/{total} segments...",
                )

        for seg in translated_segments:
            if "original_text" not in seg:
                seg["original_text"] = (seg.get("text") or "").strip()
            if "translated_text" not in seg:
                seg["translated_text"] = ""

        return translated_segments
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        clear_vram()
