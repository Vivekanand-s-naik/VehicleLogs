"""
Dynamic Device & Hardware Acceleration Management
================================================
Handles dynamic selection and fallback between CUDA, MPS, and CPU.
Provides hardware diagnostics and memory telemetry.
"""

import os
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger("anpr.device")


def resolve_device(preferred: str = "auto") -> Tuple[str, Dict[str, Any]]:
    """
    Dynamically resolve the execution device.
    
    Args:
        preferred: "auto", "cuda", "cpu", "mps", or explicit index like "cuda:0".
        
    Returns:
        (resolved_device_str, diagnostics_dict)
    """
    ort_providers = []
    try:
        import onnxruntime as ort
        ort_providers = list(ort.get_available_providers())
    except Exception:
        pass

    normalized_pref = (preferred or "auto").strip().lower()

    try:
        import torch
    except ImportError:
        if normalized_pref != "cpu" and "DmlExecutionProvider" in ort_providers:
            logger.info("Using ONNX Runtime DirectML GPU acceleration (PyTorch-free runtime).")
            return "dml", {
                "device": "dml",
                "name": "Windows DirectML GPU (ONNX Runtime)",
                "cuda_available": False,
                "onnx_providers": ort_providers,
                "cpu_count": os.cpu_count(),
            }
        logger.info("Using ONNX Runtime CPU execution (PyTorch-free runtime).")
        return "cpu", {
            "device": "cpu",
            "name": "CPU (ONNX Runtime)" if ort_providers else "CPU",
            "cuda_available": False,
            "onnx_providers": ort_providers,
            "cpu_count": os.cpu_count(),
        }

    cuda_available = torch.cuda.is_available()
    device_str = "cpu"
    diagnostics: Dict[str, Any] = {
        "preferred": preferred,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "onnx_providers": ort_providers,
    }

    if normalized_pref == "auto":
        if cuda_available and torch.cuda.device_count() > 0:
            try:
                device_str = "cuda:0"
                gpu_name = torch.cuda.get_device_name(0)
                diagnostics.update({
                    "device": device_str,
                    "name": gpu_name,
                    "total_memory_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 2),
                    "allocated_memory_mb": round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2),
                })
                logger.info(f"Dynamic device selection: Auto-selected CUDA GPU -> {gpu_name} ({device_str})")
            except Exception as e:
                logger.warning(f"Error accessing CUDA:0 ({e}). Falling back to CPU.")
                device_str = "cpu"
                diagnostics.update({"device": "cpu", "name": "CPU", "cpu_count": os.cpu_count()})
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
            diagnostics.update({"device": device_str, "name": "Apple Silicon MPS"})
            logger.info("Dynamic device selection: Auto-selected Apple Silicon MPS")
        else:
            device_str = "cpu"
            diagnostics.update({"device": "cpu", "name": "CPU", "cpu_count": os.cpu_count()})
            logger.info(f"Dynamic device selection: Auto-selected CPU ({os.cpu_count()} cores)")

    elif normalized_pref.startswith("cuda"):
        if cuda_available and torch.cuda.device_count() > 0:
            dev_idx = 0
            if ":" in normalized_pref:
                try:
                    dev_idx = int(normalized_pref.split(":")[1])
                except ValueError:
                    dev_idx = 0

            # Guard against invalid device index
            if dev_idx < 0 or dev_idx >= torch.cuda.device_count():
                logger.warning(
                    f"Requested CUDA device index {dev_idx} is out of range (count={torch.cuda.device_count()}). "
                    "Falling back to default 'cuda:0'."
                )
                dev_idx = 0

            try:
                device_str = f"cuda:{dev_idx}"
                gpu_name = torch.cuda.get_device_name(dev_idx)
                diagnostics.update({
                    "device": device_str,
                    "name": gpu_name,
                    "total_memory_mb": round(torch.cuda.get_device_properties(dev_idx).total_memory / (1024 * 1024), 2),
                    "allocated_memory_mb": round(torch.cuda.memory_allocated(dev_idx) / (1024 * 1024), 2),
                })
                logger.info(f"Dynamic device selection: Using requested CUDA -> {gpu_name} ({device_str})")
            except Exception as e:
                logger.warning(f"Error querying CUDA properties: {e}. Falling back to CPU.")
                device_str = "cpu"
                diagnostics.update({
                    "device": "cpu",
                    "name": "CPU (CUDA fallback)",
                    "cpu_count": os.cpu_count(),
                    "fallback_occurred": True,
                })
        else:
            logger.warning(
                f"Requested device '{preferred}' but CUDA is NOT available. Gracefully falling back to CPU."
            )
            device_str = "cpu"
            diagnostics.update({
                "device": "cpu",
                "name": "CPU (CUDA fallback)",
                "cpu_count": os.cpu_count(),
                "fallback_occurred": True,
            })

    elif normalized_pref == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
            diagnostics.update({"device": device_str, "name": "Apple Silicon MPS"})
        else:
            logger.warning("Requested MPS but MPS is not available. Falling back to CPU.")
            device_str = "cpu"
            diagnostics.update({"device": "cpu", "name": "CPU (MPS fallback)", "fallback_occurred": True})

    else:
        # Default or explicit 'cpu'
        device_str = "cpu"
        diagnostics.update({"device": "cpu", "name": "CPU", "cpu_count": os.cpu_count()})
        logger.info(f"Dynamic device selection: Using CPU ({os.cpu_count()} cores)")

    return device_str, diagnostics


def easyocr_gpu_flag(device_str: str) -> bool:
    """Check if EasyOCR should use GPU flag."""
    return str(device_str).startswith("cuda")


def get_device_diagnostics() -> Dict[str, Any]:
    """Return live device diagnostics including ONNX Runtime providers and GPU VRAM if available."""
    ort_providers = []
    try:
        import onnxruntime as ort
        ort_providers = list(ort.get_available_providers())
    except Exception:
        pass

    if "DmlExecutionProvider" in ort_providers:
        return {
            "active_device": "dml",
            "device_name": "Windows DirectML GPU (ONNX Runtime)",
            "onnx_providers": ort_providers,
            "cpu_count": os.cpu_count(),
        }

    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            dev_idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(dev_idx)
            total = props.total_memory / (1024 * 1024)
            allocated = torch.cuda.memory_allocated(dev_idx) / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(dev_idx) / (1024 * 1024)
            return {
                "active_device": f"cuda:{dev_idx}",
                "device_name": props.name,
                "total_vram_mb": round(total, 2),
                "allocated_vram_mb": round(allocated, 2),
                "free_vram_mb": round(total - reserved, 2),
                "utilization_pct": round((allocated / total) * 100, 1) if total > 0 else 0,
                "onnx_providers": ort_providers,
            }
    except Exception:
        pass
    return {
        "active_device": "cpu",
        "device_name": "CPU (ONNX Runtime)" if ort_providers else "CPU",
        "onnx_providers": ort_providers,
        "cpu_count": os.cpu_count(),
    }
