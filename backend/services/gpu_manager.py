"""GPU diagnostics, device selection, and VRAM cleanup utilities."""

from __future__ import annotations

import gc
import logging

logger = logging.getLogger("dubit.gpu")


def get_device_info() -> dict[str, object]:
    """Return CUDA availability, GPU model name, and current VRAM usage."""
    info: dict[str, object] = {
        "cuda_available": False,
        "device_name": "cpu",
        "total_vram_gb": 0.0,
        "allocated_vram_mb": 0.0,
        "reserved_vram_mb": 0.0,
    }
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update(
                {
                    "cuda_available": True,
                    "device_name": torch.cuda.get_device_name(0),
                    "total_vram_gb": round(props.total_memory / (1024**3), 2),
                    "allocated_vram_mb": round(torch.cuda.memory_allocated(0) / (1024**2), 2),
                    "reserved_vram_mb": round(torch.cuda.memory_reserved(0) / (1024**2), 2),
                }
            )
    except ImportError:
        logger.debug("PyTorch not installed; reporting CPU only.")
    return info


def select_device_and_compute(requested_compute: str = "auto") -> tuple[str, str]:
    """Choose execution device and CTranslate2 compute type suitable for available hardware."""
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if has_cuda:
        if requested_compute == "auto":
            # int8_float16 balances accuracy and memory headroom on 6 GB GPUs.
            return "cuda", "int8_float16"
        return "cuda", requested_compute

    if requested_compute in ("auto", "float16", "int8_float16"):
        return "cpu", "int8"
    return "cpu", requested_compute


def clear_vram() -> None:
    """Force garbage collection and release cached CUDA allocations between pipeline stages."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass

    gc.collect()
    logger.debug("Cleared Python GC and GPU VRAM cache.")
