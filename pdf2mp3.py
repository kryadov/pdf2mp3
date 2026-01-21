#!/usr/bin/env python3
"""
PDF to MP3 Converter

This script converts PDF documents to MP3 audiobooks using LLM for text processing
and Nvidia GPU for text-to-speech conversion.
"""

import os
import argparse
import warnings
import torch
import numpy as np
from tqdm import tqdm
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pydub import AudioSegment

# Silence a noisy (but harmless) PyTorch performance advisory that can be triggered by
# upstream dependencies constructing `TransformerEncoderLayer` with `batch_first=False`.
# Example warning:
#   enable_nested_tensor is True, but self.use_nested_tensor is False because
#   encoder_layer.self_attn.batch_first was not True
warnings.filterwarnings(
    "ignore",
    message=r"enable_nested_tensor is True, but self.use_nested_tensor is False.*batch_first.*",
    category=UserWarning,
)

# PyTorch 2.6+ uses a restricted unpickler by default.
# IMPORTANT: `torch.serialization.add_safe_globals()` expects *objects* (types/functions),
# not strings. Passing strings will crash with: `'str' object has no attribute '__module__'`.
try:
    from torch.serialization import add_safe_globals

    _safe = {}
    try:
        # These come from the `dp` (DeepPhonemizer) package, which some TTS checkpoints may rely on.
        from dp.preprocessing.text import Preprocessor  # type: ignore
        from dp.phonemizer import Phonemizer  # type: ignore
        from dp.model.model import (  # type: ignore
            G2p,
            Dictionary,
            Tokenizer,
            Normalizer,
            load_checkpoint,
        )

        _safe.update(
            {
                "dp.preprocessing.text.Preprocessor": Preprocessor,
                "dp.phonemizer.Phonemizer": Phonemizer,
                "dp.model.model.G2p": G2p,
                "dp.model.model.Dictionary": Dictionary,
                "dp.model.model.Tokenizer": Tokenizer,
                "dp.model.model.Normalizer": Normalizer,
                "dp.model.model.load_checkpoint": load_checkpoint,
            }
        )
    except Exception:
        # `dp` is optional; if it's not installed we simply don't register these entries.
        pass

    if _safe:
        add_safe_globals(_safe)
        print("Added safe globals for PyTorch 2.6+ compatibility")
except ImportError:
    print("Using PyTorch version that doesn't require safe globals configuration")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert PDF to MP3 audiobook using LLM and GPU")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", "-o", default=None, help="Output MP3 file path (default: same as PDF with .mp3 extension)")
    parser.add_argument("--voice", "-v", default="en_US", help="Voice to use for TTS (default: en_US)")
    parser.add_argument(
        "--tts-backend",
        choices=["speecht5", "torchaudio"],
        default="speecht5",
        help="TTS backend to use (default: speecht5). 'torchaudio' uses Tacotron2+WaveRNN and can be very slow.",
    )
    parser.add_argument("--model", "-m", default="facebook/bart-large-cnn", help="LLM model to use for text processing")
    parser.add_argument("--chunk-size", "-c", type=int, default=1000, help="Text chunk size for processing (default: 1000)")
    parser.add_argument("--batch-size", "-b", type=int, default=16, help="Batch size for LLM processing (default: 16)")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM text processing (fastest; uses raw extracted text)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--volume", type=int, default=0, help="Volume adjustment in dB (default: 0, negative values decrease volume, positive values increase volume)")
    return parser.parse_args()

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    print(f"Extracting text from {pdf_path}...")
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(tqdm(reader.pages, desc="Reading pages")):
        text += page.extract_text() + "\n"
    return text

def _chunk_text(text, chunk_size):
    """Split text into roughly `chunk_size` character chunks, preferring whitespace boundaries."""
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + chunk_size, n)
        if end < n:
            # Try not to split in the middle of a word/sentence.
            split_at = text.rfind(" ", i, end)
            if split_at == -1:
                split_at = text.rfind("\n", i, end)
            if split_at != -1 and split_at > i + chunk_size * 0.5:
                end = split_at
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        i = max(end, i + 1)
    return chunks


def process_text_with_llm(text, model_name, chunk_size, batch_size, use_gpu):
    """Process text with a language model to improve TTS quality."""
    print(f"Processing text with {model_name}...")

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    # Set device
    device = 0 if cuda_available and use_gpu else -1
    if device == 0:
        print("Using GPU for text processing")
        torch.cuda.set_device(0)
        print(f"Active CUDA device: {torch.cuda.current_device()}")
    else:
        print("Using CPU for text processing")
        print("To use GPU, make sure CUDA is available and --no-gpu flag is not set")

    # Initialize the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()

    # Move model to the appropriate device
    if device == 0:
        model = model.cuda()
        print("Model moved to CUDA device")

    # Split text into chunks
    chunks = _chunk_text(text, chunk_size)
    if not chunks:
        return ""

    processed_chunks = []
    # Use fixed generation limits to enable true batching (per-item limits aren't supported in batched calls).
    # These defaults favor throughput over aggressive summarization.
    max_new_tokens = 128
    min_length = 16
    # Explicit input length cap to avoid HF warning: model/tokenizer may not define max length.
    max_input_tokens = 1024

    use_amp = device == 0

    for i in tqdm(range(0, len(chunks), batch_size), desc="Processing text chunks"):
        batch = [c for c in chunks[i : i + batch_size] if c.strip()]
        if not batch:
            continue

        with torch.inference_mode():
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_tokens,
            )
            if device == 0:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            if use_amp:
                # `torch.cuda.amp.autocast` is deprecated; use the unified API.
                with torch.amp.autocast(device_type="cuda"):
                    summary_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        min_length=min_length,
                        do_sample=False,
                    )
            else:
                summary_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_length=min_length,
                    do_sample=False,
                )

            summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

        processed_chunks.extend([s for s in summaries if s.strip()])

    # Join the processed chunks
    processed_text = " ".join(processed_chunks)
    return processed_text

def text_to_speech(text, voice, use_gpu, tts_backend="speecht5"):
    """Convert text to speech.

    Note: torchaudio's Tacotron2+WaveRNN backend is frequently CPU-bound and can appear
    "stuck" (tqdm at 0%) until the first sentence finishes.
    """
    print(f"Converting text to speech using {voice} voice... (backend: {tts_backend})")

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if not cuda_available and use_gpu:
        print("Warning: CUDA is not available but GPU usage was requested")
        print("Falling back to CPU for text-to-speech conversion")

    # Set device
    device = "cuda" if cuda_available and use_gpu else "cpu"
    if device == "cuda":
        print("Using GPU for text-to-speech conversion")
        torch.cuda.set_device(0)
        print(f"Active CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU for text-to-speech conversion")
        if use_gpu:
            print("To use GPU, make sure CUDA is available and properly set up")

    def _split_tts_units(text_to_split: str, max_chars: int = 240):
        import re

        normalized = re.sub(r"\s+", " ", text_to_split.replace("\n", " ")).strip()
        if not normalized:
            return []

        # First pass: sentence-ish splitting.
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", normalized) if p.strip()]
        # Second pass: enforce a max length to avoid very long first item saying at 0%.
        units: list[str] = []
        for p in parts:
            if len(p) <= max_chars:
                units.append(p)
                continue
            units.extend(_chunk_text(p, max_chars))
        return units

    if tts_backend == "torchaudio":
        # Use torchaudio for TTS
        try:
            import torchaudio
            from torchaudio.pipelines import TACOTRON2_WAVERNN_PHONE_LJSPEECH

            if device == "cuda":
                print(
                    "Note: torchaudio Tacotron2+WaveRNN is often slow and may not show meaningful GPU usage. "
                    "If it appears stuck at 0%, try --tts-backend speecht5."
                )

            bundle = TACOTRON2_WAVERNN_PHONE_LJSPEECH
            try:
                processor = bundle.get_text_processor()
                tacotron2 = bundle.get_tacotron2().to(device).eval()
                vocoder = bundle.get_vocoder().to(device).eval()
            except AttributeError as e:
                print(f"Error loading TTS model components: {e}")
                print("This is likely due to compatibility issues with PyTorch 2.6+")
                raise Exception("TTS model loading failed")

            units = _split_tts_units(text)
            audio_segments = []
            pbar = tqdm(units, desc="Converting sentences to speech")
            for unit in pbar:
                if not unit.strip():
                    continue
                pbar.set_postfix(chars=len(unit))
                # Ensure terminal punctuation for better prosody.
                if unit[-1] not in ".!?":
                    unit += "."

                with torch.inference_mode():
                    processed, lengths = processor(unit)
                    processed = processed.to(device)
                    lengths = lengths.to(device)

                    spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
                    waveforms, _ = vocoder(spec, spec_lengths)
                    audio_segments.append(waveforms[0].detach().cpu().numpy())

            audio = np.concatenate(audio_segments) if audio_segments else np.zeros((0,), dtype=np.float32)
            return audio, bundle.sample_rate

        except Exception as e:
            print(f"Error in torchaudio TTS: {e}")
            print("Falling back to SpeechT5...")
            tts_backend = "speecht5"

    # SpeechT5 backend (default)
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    import datasets

    print("Loading SpeechT5 model and speaker embeddings...")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device).eval()
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device).eval()

    # Load speaker embeddings from the CMU Arctic dataset
    try:
        embeddings_dataset = datasets.load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embeddings = torch.tensor(embeddings_dataset[7]["xvector"]).unsqueeze(0).to(device)
        print("Speaker embeddings loaded successfully")
    except Exception as e:
        print(f"Error loading speaker embeddings: {e}")
        print("Using default speaker embeddings")
        speaker_embeddings = torch.randn(1, 512).to(device)

    units = _split_tts_units(text, max_chars=420)
    audio_segments = []
    pbar = tqdm(units, desc="Converting text chunks to speech")
    for unit in pbar:
        if not unit.strip():
            continue
        pbar.set_postfix(chars=len(unit))

        inputs = processor(text=unit, return_tensors="pt").to(device)
        with torch.inference_mode():
            speech_hidden_states = model.generate_speech(inputs["input_ids"], speaker_embeddings=speaker_embeddings)
            speech = vocoder(speech_hidden_states)

        audio_segments.append(speech.detach().cpu().numpy())

    audio = np.concatenate(audio_segments) if audio_segments else np.zeros((0,), dtype=np.float32)
    return audio, 16000  # 16kHz sample rate

def save_audio_to_mp3(audio, sample_rate, output_path, volume_db=0):
    """Save audio data to an MP3 file with optional volume adjustment.

    Args:
        audio: Audio data as numpy array
        sample_rate: Sample rate of the audio
        output_path: Path to save the MP3 file
        volume_db: Volume adjustment in dB (default: 0, negative values decrease volume, positive values increase volume)
    """
    print(f"Saving audio to {output_path}...")

    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)

    # Create a temporary WAV file
    temp_wav = output_path.replace('.mp3', '_temp.wav')
    import wave
    with wave.open(temp_wav, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    # Convert WAV to MP3
    audio_segment = AudioSegment.from_wav(temp_wav)

    # Adjust volume if needed
    if volume_db != 0:
        print(f"Adjusting volume by {volume_db} dB")
        audio_segment = audio_segment.apply_gain(volume_db)

    # Export to MP3
    audio_segment.export(output_path, format="mp3")

    # Remove temporary WAV file
    os.remove(temp_wav)

    print(f"MP3 file saved to {output_path}")

def check_gpu_availability():
    """Check if GPU is available and print information about it."""
    print("\n=== GPU Availability Check ===")
    if torch.cuda.is_available():
        print("✓ CUDA is available")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"✓ GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"✓ Current device: {torch.cuda.current_device()}")
        print("Device set to use GPU")
    else:
        print("✗ CUDA is not available")
        print("✗ PyTorch will use CPU instead")
        print("To use GPU acceleration, please:")
        print("1. Make sure you have an NVIDIA GPU")
        print("2. Install the appropriate NVIDIA drivers")
        print("3. Install CUDA toolkit and cuDNN")
        print("4. Reinstall PyTorch with CUDA support")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("Device set to use CPU")
    print("==============================\n")

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()

    # Check GPU availability
    check_gpu_availability()

    # Set output path if not specified
    if args.output is None:
        args.output = os.path.splitext(args.pdf_path)[0] + ".mp3"

    # Extract text from PDF
    text = extract_text_from_pdf(args.pdf_path)

    # Process text with LLM
    if args.no_llm:
        processed_text = text
    else:
        processed_text = process_text_with_llm(text, args.model, args.chunk_size, args.batch_size, not args.no_gpu)

    # Convert text to speech
    audio, sample_rate = text_to_speech(
        processed_text,
        args.voice,
        not args.no_gpu,
        tts_backend=args.tts_backend,
    )

    # Save audio to MP3
    save_audio_to_mp3(audio, sample_rate, args.output, args.volume)

if __name__ == "__main__":
    main()
