#!/usr/bin/env python3
"""Audio post-processing pipeline for TTS output enhancement and source separation."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

# Lazy imports to avoid loading heavy libraries until needed
_noisereduce = None
_pedalboard = None
_pyloudnorm = None


# ============== Source Separation (Demucs) ==============


def separate_vocals(
    input_path: Path,
    output_dir: Path | None = None,
    model: str = "htdemucs",
    device: str = "auto",
    quiet: bool = True,
) -> Path:
    """
    Separate vocals from background music using Demucs.

    Args:
        input_path: Path to input audio file
        output_dir: Directory for output (default: same as input)
        model: Demucs model to use (htdemucs, htdemucs_ft, mdx_extra)
        device: Device to use (auto, cuda, cpu, mps)
        quiet: Suppress demucs progress output

    Returns:
        Path to extracted vocals file
    """
    input_path = Path(input_path)
    if output_dir is None:
        output_dir = input_path.parent

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if device == "auto":
        import torch

        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Run demucs
    cmd = [
        "python",
        "-m",
        "demucs",
        "--two-stems",
        "vocals",
        "-n",
        model,
        "-d",
        device,
        "-o",
        str(output_dir),
        str(input_path),
    ]

    if quiet:
        # Suppress tqdm and other output
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print(f"Running Demucs ({model}) on {device}...")
        subprocess.run(cmd, check=True)

    # Find output vocals file
    stem_name = input_path.stem
    vocals_path = output_dir / model / stem_name / "vocals.wav"

    if not vocals_path.exists():
        raise FileNotFoundError(f"Vocals file not found: {vocals_path}")

    return vocals_path


def separate_vocals_to_file(
    input_path: Path,
    output_path: Path,
    model: str = "htdemucs",
    device: str = "auto",
    cleanup: bool = True,
    quiet: bool = True,
) -> Path:
    """
    Separate vocals and save to a specific output path.

    Args:
        input_path: Path to input audio file
        output_path: Path for output vocals file
        model: Demucs model to use
        device: Device to use
        cleanup: Remove intermediate files after extraction
        quiet: Suppress demucs progress output

    Returns:
        Path to extracted vocals file
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Use temp directory for demucs output
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Run separation
        vocals_path = separate_vocals(input_path, temp_dir, model, device, quiet=quiet)

        # Copy to final destination
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(vocals_path, output_path)

    return output_path


def check_demucs_available() -> bool:
    """Check if Demucs is installed and available."""
    try:
        result = subprocess.run(
            ["python", "-m", "demucs", "--help"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def _get_noisereduce():
    global _noisereduce
    if _noisereduce is None:
        import noisereduce as nr

        _noisereduce = nr
    return _noisereduce


def _get_pedalboard():
    global _pedalboard
    if _pedalboard is None:
        import pedalboard

        _pedalboard = pedalboard
    return _pedalboard


def _get_pyloudnorm():
    global _pyloudnorm
    if _pyloudnorm is None:
        import pyloudnorm as pyln

        _pyloudnorm = pyln
    return _pyloudnorm


def reduce_noise(
    audio: np.ndarray,
    sample_rate: int,
    stationary: bool = False,
    prop_decrease: float = 0.8,
) -> np.ndarray:
    """
    Remove background noise using spectral gating.

    Args:
        audio: Audio signal as numpy array
        sample_rate: Sample rate in Hz
        stationary: If True, use stationary noise reduction (faster, for constant noise)
                   If False, use non-stationary (better for varying noise)
        prop_decrease: Proportion to reduce noise by (0.0-1.0)

    Returns:
        Denoised audio signal
    """
    nr = _get_noisereduce()

    # Ensure float32 for noisereduce
    audio_float = audio.astype(np.float32)

    reduced = nr.reduce_noise(
        y=audio_float,
        sr=sample_rate,
        stationary=stationary,
        prop_decrease=prop_decrease,
        n_fft=512,  # Optimized for speech
    )

    return reduced


def apply_voice_eq(
    audio: np.ndarray,
    sample_rate: int,
    highpass_freq: float = 80.0,
    lowpass_freq: float = 12000.0,
) -> np.ndarray:
    """
    Apply EQ optimized for voice clarity.

    Args:
        audio: Audio signal as numpy array
        sample_rate: Sample rate in Hz
        highpass_freq: High-pass filter cutoff (removes low rumble)
        lowpass_freq: Low-pass filter cutoff (removes high-freq noise)

    Returns:
        EQ'd audio signal
    """
    pb = _get_pedalboard()

    # Ensure correct shape for pedalboard (samples, channels) or (samples,)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)

    board = pb.Pedalboard(
        [
            pb.HighpassFilter(cutoff_frequency_hz=highpass_freq),
            pb.LowpassFilter(cutoff_frequency_hz=lowpass_freq),
        ]
    )

    processed = board(audio.T, sample_rate).T
    return processed.squeeze()


def apply_dynamics(
    audio: np.ndarray,
    sample_rate: int,
    compressor_threshold_db: float = -20.0,
    compressor_ratio: float = 3.0,
    limiter_threshold_db: float = -1.0,
    makeup_gain_db: float = 0.0,
) -> np.ndarray:
    """
    Apply dynamics processing (compression + limiting).

    Args:
        audio: Audio signal as numpy array
        sample_rate: Sample rate in Hz
        compressor_threshold_db: Compressor threshold in dB
        compressor_ratio: Compression ratio (e.g., 3.0 = 3:1)
        limiter_threshold_db: Limiter threshold in dB (prevents clipping)
        makeup_gain_db: Additional gain after compression

    Returns:
        Dynamics-processed audio signal
    """
    pb = _get_pedalboard()

    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)

    board = pb.Pedalboard(
        [
            pb.Compressor(
                threshold_db=compressor_threshold_db,
                ratio=compressor_ratio,
                attack_ms=5.0,
                release_ms=100.0,
            ),
            pb.Gain(gain_db=makeup_gain_db),
            pb.Limiter(threshold_db=limiter_threshold_db),
        ]
    )

    processed = board(audio.T, sample_rate).T
    return processed.squeeze()


def normalize_loudness(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float = -14.0,
) -> np.ndarray:
    """
    Normalize audio to target loudness (LUFS).

    Args:
        audio: Audio signal as numpy array
        sample_rate: Sample rate in Hz
        target_lufs: Target loudness in LUFS (-14 is streaming standard)

    Returns:
        Loudness-normalized audio signal
    """
    import warnings

    pyln = _get_pyloudnorm()

    # Ensure float64 for pyloudnorm
    audio_float = audio.astype(np.float64)

    # Measure current loudness
    meter = pyln.Meter(sample_rate)
    current_loudness = meter.integrated_loudness(audio_float)

    # Skip if audio is silent or too quiet to measure
    if current_loudness == float("-inf") or np.isnan(current_loudness):
        return audio

    # Calculate gain needed
    gain_db = target_lufs - current_loudness
    gain_linear = 10 ** (gain_db / 20.0)

    # Check if normalization would cause clipping
    peak_after = np.max(np.abs(audio_float)) * gain_linear
    if peak_after > 1.0:
        # Reduce gain to prevent clipping (leave 1dB headroom)
        gain_linear = 0.89 / np.max(np.abs(audio_float))

    # Apply gain manually to avoid pyloudnorm's clipping warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        normalized = audio_float * gain_linear

    # Final safety check
    max_val = np.max(np.abs(normalized))
    if max_val > 1.0:
        normalized = normalized / max_val * 0.99

    return normalized.astype(np.float32)


def post_process_audio(
    audio: np.ndarray,
    sample_rate: int,
    denoise: bool = True,
    eq: bool = True,
    dynamics: bool = True,
    loudness_normalize: bool = True,
    target_lufs: float = -14.0,
    denoise_strength: float = 0.8,
) -> np.ndarray:
    """
    Full post-processing pipeline for TTS output.

    Pipeline: Denoise → EQ → Dynamics → Loudness Normalization

    Args:
        audio: Audio signal as numpy array
        sample_rate: Sample rate in Hz
        denoise: Apply noise reduction
        eq: Apply voice EQ (highpass + lowpass)
        dynamics: Apply compression and limiting
        loudness_normalize: Normalize to target LUFS
        target_lufs: Target loudness level
        denoise_strength: Noise reduction strength (0.0-1.0)

    Returns:
        Processed audio signal
    """
    processed = audio.copy()

    # 1. Noise reduction
    if denoise:
        processed = reduce_noise(
            processed,
            sample_rate,
            stationary=False,
            prop_decrease=denoise_strength,
        )

    # 2. Voice EQ
    if eq:
        processed = apply_voice_eq(processed, sample_rate)

    # 3. Dynamics (compression + limiting)
    if dynamics:
        processed = apply_dynamics(processed, sample_rate)

    # 4. Loudness normalization
    if loudness_normalize:
        processed = normalize_loudness(processed, sample_rate, target_lufs)

    return processed


def post_process_file(
    input_path: Path,
    output_path: Path | None = None,
    **kwargs,
) -> Path:
    """
    Post-process an audio file.

    Args:
        input_path: Path to input audio file
        output_path: Path for output (default: overwrites input)
        **kwargs: Arguments passed to post_process_audio()

    Returns:
        Path to processed file
    """
    if output_path is None:
        output_path = input_path

    # Load audio
    audio, sample_rate = sf.read(str(input_path))

    # Process
    processed = post_process_audio(audio, sample_rate, **kwargs)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), processed, sample_rate)

    return output_path


def post_process_directory(
    input_dir: Path,
    output_dir: Path | None = None,
    pattern: str = "*.wav",
    **kwargs,
) -> list[Path]:
    """
    Post-process all audio files in a directory.

    Args:
        input_dir: Directory containing audio files
        output_dir: Output directory (default: overwrites in place)
        pattern: Glob pattern for audio files
        **kwargs: Arguments passed to post_process_audio()

    Returns:
        List of processed file paths
    """
    from tqdm import tqdm

    if output_dir is None:
        output_dir = input_dir

    input_files = list(input_dir.rglob(pattern))
    processed_files = []

    for input_file in tqdm(input_files, desc="Post-processing"):
        # Maintain directory structure
        relative_path = input_file.relative_to(input_dir)
        output_file = output_dir / relative_path

        post_process_file(input_file, output_file, **kwargs)
        processed_files.append(output_file)

    return processed_files


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python post_process.py <input_path> [output_path]")
        print("       python post_process.py <input_dir> [output_dir] --dir")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "--dir" else None
    is_dir = "--dir" in sys.argv

    if is_dir or input_path.is_dir():
        print(f"Post-processing directory: {input_path}")
        processed = post_process_directory(input_path, output_path)
        print(f"Processed {len(processed)} files")
    else:
        print(f"Post-processing file: {input_path}")
        result = post_process_file(input_path, output_path)
        print(f"Saved to: {result}")
