#!/usr/bin/env python3
"""Device detection utilities for cross-platform support."""

import platform
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import torch


class DeviceType(Enum):
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


@dataclass
class DeviceInfo:
    device_type: DeviceType
    device_name: str
    dtype: torch.dtype
    attn_implementation: str
    whisper_backend: Literal["faster-whisper", "mlx-whisper"]

    @property
    def torch_device(self) -> str:
        if self.device_type == DeviceType.CUDA:
            return "cuda:0"
        elif self.device_type == DeviceType.MPS:
            return "mps"
        return "cpu"

    @property
    def is_gpu(self) -> bool:
        return self.device_type in (DeviceType.CUDA, DeviceType.MPS)


def detect_device() -> DeviceInfo:
    """Detect the best available device for inference."""

    # Check CUDA first (Linux/Windows with NVIDIA GPU)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_properties(0).name
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        # Check for FlashAttention support
        try:
            import flash_attn  # noqa: F401

            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"

        return DeviceInfo(
            device_type=DeviceType.CUDA,
            device_name=f"{gpu_name} ({gpu_memory:.1f}GB)",
            dtype=torch.float16,
            attn_implementation=attn_impl,
            whisper_backend="faster-whisper",
        )

    # Check MPS (Apple Silicon Mac)
    if torch.backends.mps.is_available():
        # Get Mac model info
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            )
            cpu_name = result.stdout.strip()
        except Exception:
            cpu_name = "Apple Silicon"

        return DeviceInfo(
            device_type=DeviceType.MPS,
            device_name=cpu_name,
            dtype=torch.float32,  # MPS requires float32 for Qwen3-TTS
            attn_implementation="sdpa",  # FlashAttention not available on Mac
            whisper_backend="mlx-whisper",
        )

    # Fallback to CPU
    return DeviceInfo(
        device_type=DeviceType.CPU,
        device_name=platform.processor() or "CPU",
        dtype=torch.float32,
        attn_implementation="sdpa",
        whisper_backend="mlx-whisper",
    )


def print_device_info(device_info: DeviceInfo) -> None:
    """Print device information in a formatted way."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Device Configuration")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Device Type", device_info.device_type.value.upper())
    table.add_row("Device Name", device_info.device_name)
    table.add_row("PyTorch Device", device_info.torch_device)
    table.add_row("Data Type", str(device_info.dtype))
    table.add_row("Attention Implementation", device_info.attn_implementation)
    table.add_row("Whisper Backend", device_info.whisper_backend)

    console.print(table)


if __name__ == "__main__":
    device = detect_device()
    print_device_info(device)
