#!/usr/bin/env python3
"""Setup and test Qwen3-TTS 1.7B model for GPU (A100)."""

import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def check_gpu():
    """Check GPU availability."""
    print("Checking GPU support...")

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return False

    gpu_count = torch.cuda.device_count()
    print(f"CUDA available: True")
    print(f"GPU count: {gpu_count}")

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

    return True


def download_model():
    """Download Qwen3-TTS 1.7B model for voice cloning."""
    from huggingface_hub import snapshot_download

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    local_dir = MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-Base"

    if local_dir.exists():
        print(f"Model already downloaded: {local_dir}")
        return local_dir

    print(f"Downloading {model_name}...")
    snapshot_download(repo_id=model_name, local_dir=str(local_dir))

    # Also download tokenizer
    tokenizer_name = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    tokenizer_dir = MODELS_DIR / "Qwen3-TTS-Tokenizer-12Hz"

    if not tokenizer_dir.exists():
        print(f"Downloading {tokenizer_name}...")
        snapshot_download(repo_id=tokenizer_name, local_dir=str(tokenizer_dir))

    print("Model download complete!")
    return local_dir


def test_model():
    """Test the TTS model with voice cloning on GPU."""
    from qwen_tts import Qwen3TTSModel
    import soundfile as sf
    import json

    model_path = MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-Base"
    ref_audio = PROJECT_ROOT / "assets" / "clean" / "karina_clean.wav"
    transcript_file = PROJECT_ROOT / "assets" / "transcripts" / "karina_clean_transcript.json"

    # Check if reference files exist
    if not ref_audio.exists():
        print(f"Reference audio not found: {ref_audio}")
        print("Skipping voice clone test - run transcription first")
        return False

    if not transcript_file.exists():
        print(f"Transcript not found: {transcript_file}")
        print("Skipping voice clone test - run transcription first")
        return False

    # Load transcript
    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)
    ref_text = transcript_data["text"]

    print("Loading Qwen3-TTS 1.7B model on GPU...")
    model = Qwen3TTSModel.from_pretrained(
        str(model_path),
        dtype=torch.float16,  # float16 for GPU
        attn_implementation="flash_attention_2",  # FlashAttention for A100
        device_map="cuda:0",
    )

    # Test voice cloning
    print("Testing voice cloning...")
    test_text = "안녕하세요, 테스트입니다."

    print(f"Reference audio: {ref_audio}")
    print(f"Reference text: {ref_text[:50]}...")

    wavs, sr = model.generate_voice_clone(
        text=test_text,
        ref_audio=str(ref_audio),
        ref_text=ref_text,
        language="korean",
        non_streaming_mode=True,
    )

    output_path = PROJECT_ROOT / "output" / "test_output.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), wavs[0], sr)

    print(f"Test audio saved to: {output_path}")
    return True


if __name__ == "__main__":
    if not check_gpu():
        exit(1)
    download_model()
    test_model()
