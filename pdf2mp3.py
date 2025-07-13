#!/usr/bin/env python3
"""
PDF to MP3 Converter

This script converts PDF documents to MP3 audiobooks using LLM for text processing
and Nvidia GPU for text-to-speech conversion.
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from pydub import AudioSegment

# Add safe globals for PyTorch 2.6+ compatibility
try:
    from torch.serialization import add_safe_globals
    add_safe_globals([
        "dp.preprocessing.text.Preprocessor",
        "dp.model.model.Phonemizer",
        "dp.phonemizer.Phonemizer",
        "dp.model.model.load_checkpoint",
        "dp.model.model.G2p",
        # Add additional safe globals that might be needed
        "dp.model.model.Dictionary",
        "dp.model.model.Tokenizer",
        "dp.model.model.Normalizer"
    ])
    # Also add functions that might be used during unpickling
    import torch.nn
    import torch.nn.functional
    import torch.nn.modules
    import torch.nn.parameter
    print("Added safe globals for PyTorch 2.6+ compatibility")
except ImportError:
    print("Using PyTorch version that doesn't require safe globals configuration")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert PDF to MP3 audiobook using LLM and GPU")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", "-o", default=None, help="Output MP3 file path (default: same as PDF with .mp3 extension)")
    parser.add_argument("--voice", "-v", default="en_US", help="Voice to use for TTS (default: en_US)")
    parser.add_argument("--model", "-m", default="facebook/bart-large-cnn", help="LLM model to use for text processing")
    parser.add_argument("--chunk-size", "-c", type=int, default=1000, help="Text chunk size for processing (default: 1000)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    return parser.parse_args()

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    print(f"Extracting text from {pdf_path}...")
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(tqdm(reader.pages, desc="Reading pages")):
        text += page.extract_text() + "\n"
    return text

def process_text_with_llm(text, model_name, chunk_size, use_gpu):
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

    # Move model to the appropriate device
    if device == 0:
        model = model.cuda()
        print("Model moved to CUDA device")

    # Create a summarization pipeline
    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    # Split text into chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    # Process each chunk
    processed_chunks = []
    for chunk in tqdm(chunks, desc="Processing text chunks"):
        # Skip empty chunks
        if not chunk.strip():
            continue

        # Process the chunk
        processed_chunk = summarizer(
            chunk, 
            max_new_tokens=min(len(chunk.split()) // 2, chunk_size // 2),
            min_length=min(len(chunk.split()) // 4, chunk_size // 4),
            do_sample=False
        )[0]['summary_text']

        processed_chunks.append(processed_chunk)

    # Join the processed chunks
    processed_text = " ".join(processed_chunks)
    return processed_text

def text_to_speech(text, voice, use_gpu):
    """Convert text to speech using GPU acceleration."""
    print(f"Converting text to speech using {voice} voice...")

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

    # Use torchaudio for TTS
    try:
        import torchaudio
        from torchaudio.pipelines import TACOTRON2_WAVERNN_PHONE_LJSPEECH

        # Load the TTS model
        bundle = TACOTRON2_WAVERNN_PHONE_LJSPEECH
        try:
            processor = bundle.get_text_processor()
            tacotron2 = bundle.get_tacotron2().to(device)
            vocoder = bundle.get_vocoder().to(device)
        except AttributeError as e:
            print(f"Error loading TTS model components: {e}")
            print("This is likely due to compatibility issues with PyTorch 2.6+")
            print("Falling back to alternative TTS method...")
            raise Exception("TTS model loading failed") # This will trigger the fallback method

        # Split text into sentences to avoid memory issues
        sentences = text.replace('\n', ' ').split('. ')

        # Process each sentence
        audio_segments = []
        for sentence in tqdm(sentences, desc="Converting sentences to speech"):
            # Skip empty sentences
            if not sentence.strip():
                continue

            # Add period if missing
            if not sentence.endswith('.'):
                sentence += '.'

            # Process the sentence
            with torch.no_grad():
                processed, lengths = processor(sentence)
                processed = processed.to(device)
                lengths = lengths.to(device)

                spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
                waveforms, _ = vocoder(spec, spec_lengths)

                # Convert to numpy array
                audio_np = waveforms[0].detach().cpu().numpy()
                audio_segments.append(audio_np)

        # Concatenate audio segments
        audio = np.concatenate(audio_segments)

        return audio, bundle.sample_rate

    except Exception as e:
        print(f"Error in TTS: {e}")
        print("Falling back to simpler TTS method...")

        # Fallback to a simpler TTS method
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        import datasets

        print("Loading SpeechT5 model and speaker embeddings...")
        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

        # Load speaker embeddings from the CMU Arctic dataset
        try:
            # Load speaker embeddings
            embeddings_dataset = datasets.load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            speaker_embeddings = torch.tensor(embeddings_dataset[7]["xvector"]).unsqueeze(0).to(device)
            print("Speaker embeddings loaded successfully")
        except Exception as e:
            print(f"Error loading speaker embeddings: {e}")
            print("Using default speaker embeddings")
            # Create a default speaker embedding if loading fails
            speaker_embeddings = torch.randn(1, 512).to(device)

        # Split text into chunks to avoid memory issues
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]

        # Process each chunk
        audio_segments = []
        for chunk in tqdm(chunks, desc="Converting text chunks to speech"):
            # Skip empty chunks
            if not chunk.strip():
                continue

            # Process the chunk
            inputs = processor(text=chunk, return_tensors="pt").to(device)

            # Generate speech with speaker embeddings
            try:
                # Try the standard way first with speaker embeddings
                speech_hidden_states = model.generate_speech(
                    inputs["input_ids"], 
                    speaker_embeddings=speaker_embeddings
                )
                # Use vocoder separately
                speech = vocoder(speech_hidden_states)
            except Exception as e:
                print(f"Error in speech generation: {e}")
                print("Trying alternative method...")
                try:
                    # Alternative approach
                    speech_hidden_states = model.generate_speech(
                        inputs["input_ids"], 
                        speaker_embeddings=speaker_embeddings,
                        vocoder=None
                    )
                    speech = vocoder(speech_hidden_states)
                except Exception as e2:
                    print(f"Alternative method also failed: {e2}")
                    raise

            # Convert to numpy array
            audio_np = speech.detach().cpu().numpy()
            audio_segments.append(audio_np)

        # Concatenate audio segments
        audio = np.concatenate(audio_segments)

        return audio, 16000  # 16kHz sample rate

def save_audio_to_mp3(audio, sample_rate, output_path):
    """Save audio data to an MP3 file."""
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
    processed_text = process_text_with_llm(text, args.model, args.chunk_size, not args.no_gpu)

    # Convert text to speech
    audio, sample_rate = text_to_speech(processed_text, args.voice, not args.no_gpu)

    # Save audio to MP3
    save_audio_to_mp3(audio, sample_rate, args.output)

if __name__ == "__main__":
    main()
