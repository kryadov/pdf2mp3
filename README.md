# PDF to MP3 Converter

A Python application that converts PDF documents to MP3 audiobooks using LLM (Large Language Model) for text processing and Nvidia GPU for text-to-speech conversion.

## Features

- Extract text from PDF documents
- Process and enhance text using LLM
- Convert text to speech using GPU acceleration
- Save output as MP3 audiobook

## Requirements

- Python 3.13
- NVIDIA GPU with CUDA support (for optimal performance)
- FFmpeg (required for audio processing)
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/pdf2mp3.git
   cd pdf2mp3
   ```

2. Install FFmpeg:
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg` or equivalent for your distribution

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
   If you get "Using CPU for text processing" in the log then run
   ```
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. Make sure you have the latest NVIDIA drivers and CUDA toolkit installed if you want to use GPU acceleration.

## Usage

Basic usage:

```
python pdf2mp3.py path/to/your/document.pdf
```

This will convert the PDF to an MP3 file with the same name in the same directory.

### Command Line Options

```
python pdf2mp3.py [-h] [--output OUTPUT] [--voice VOICE] [--model MODEL] [--chunk-size CHUNK_SIZE] [--no-gpu] pdf_path
```

Arguments:

- `pdf_path`: Path to the PDF file (required)
- `--output`, `-o`: Output MP3 file path (default: same as PDF with .mp3 extension)
- `--voice`, `-v`: Voice to use for TTS (default: en_US)
- `--model`, `-m`: LLM model to use for text processing (default: facebook/bart-large-cnn)
- `--chunk-size`, `-c`: Text chunk size for processing (default: 1000)
- `--no-gpu`: Disable GPU acceleration

### Examples

Convert a PDF to MP3 with default settings:
```
python pdf2mp3.py document.pdf
```

Specify an output file:
```
python pdf2mp3.py document.pdf --output audiobook.mp3
```

Use a different LLM model:
```
python pdf2mp3.py document.pdf --model t5-base
```

Process without GPU acceleration:
```
python pdf2mp3.py document.pdf --no-gpu
```

## How It Works

1. **Text Extraction**: The application extracts text from the PDF document using PyPDF2.
2. **Text Processing**: The extracted text is processed using a language model to improve readability and TTS quality.
3. **Text-to-Speech**: The processed text is converted to speech using either torchaudio's TACOTRON2 model or Microsoft's SpeechT5 model.
4. **Audio Export**: The generated audio is saved as an MP3 file.

## GPU Acceleration

This application is designed to leverage NVIDIA GPUs for faster text processing and speech synthesis. When you run the application, it will automatically check for GPU availability and provide detailed information.

### Verifying GPU Usage

When you run the application, it will display GPU information at startup:

```
=== GPU Availability Check ===
✓ CUDA is available
✓ CUDA version: 11.8
✓ GPU count: 1
✓ GPU 0: NVIDIA GeForce RTX 3080
✓ Current device: 0
Device set to use GPU
==============================
```

During processing, you should see messages like:
- "Using GPU for text processing"
- "Using GPU for text-to-speech conversion"

If you see "Using CPU" instead, check the troubleshooting section below.

### Setting Up GPU Support

To ensure GPU acceleration works properly:

1. **Hardware Requirements**:
   - NVIDIA GPU (GeForce GTX 1060 or better recommended)
   - At least 6GB of VRAM for processing large documents

2. **Software Requirements**:
   - NVIDIA GPU drivers (latest version recommended)
   - CUDA Toolkit (version 11.7 or newer)
   - cuDNN library (compatible with your CUDA version)

3. **PyTorch with CUDA**:
   Make sure PyTorch is installed with CUDA support:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   Replace `cu118` with your CUDA version (e.g., `cu117` for CUDA 11.7)

## Troubleshooting

- **Memory Issues**: If you encounter memory issues with large PDFs, try increasing the chunk size with `--chunk-size`.
- **Model Download**: The first time you run the application, it will download the required models, which may take some time depending on your internet connection.

### GPU-Related Issues

- **GPU Not Detected**: If the application reports "CUDA is not available":
  1. Verify your NVIDIA drivers are installed and up-to-date
  2. Check that CUDA toolkit is properly installed
  3. Ensure PyTorch was installed with CUDA support
  4. Run `nvidia-smi` in terminal to verify GPU is recognized by the system

- **Out of Memory Errors**: If you encounter CUDA out of memory errors:
  1. Reduce chunk size with `--chunk-size` option
  2. Close other GPU-intensive applications
  3. Try a smaller model with `--model` option

- **Slow Performance Despite GPU**: If processing is slow even with GPU:
  1. Check GPU utilization with `nvidia-smi` during processing
  2. Ensure you're not running in a virtual environment with limited GPU access
  3. Update GPU drivers to the latest version
- **Tokenizers Installation**: If you encounter errors related to tokenizers compilation during installation, you may need to install Rust:
  1. Download and install Rust from [https://rustup.rs/](https://rustup.rs/)
  2. Make sure Rust is in your PATH
  3. Run `pip install -r requirements.txt` again
- **PyAudio Installation Issues**: If you encounter errors related to PyAudio installation:
  - Windows: You may need to install Visual C++ Build Tools first, or use a pre-compiled wheel:
    ```
    pip install pipwin
    pipwin install pyaudio
    ```
  - macOS: `brew install portaudio` before installing PyAudio
  - Linux: `sudo apt-get install python3-pyaudio` or `sudo apt-get install portaudio19-dev` before installing PyAudio
- **Audio Processing Errors**: If you encounter errors related to audio processing:
  1. Make sure FFmpeg is installed and in your PATH
  2. Try reinstalling pydub: `pip install --upgrade pydub`
- **Text-to-Speech Errors**: If you encounter errors related to text-to-speech conversion:
  1. If you see "DeepPhonemizer is not installed" error, run: `pip install deep-phonemizer>=0.0.17`
  2. If you see "SpeechT5Tokenizer requires the SentencePiece library" error, run: `pip install sentencepiece`
  3. For other TTS errors, the application will automatically fall back to a simpler TTS method
  4. If both TTS methods fail, try updating torchaudio: `pip install --upgrade torchaudio`
- **Text Processing Errors**: If you encounter errors related to text processing:
  1. Try using a different LLM model with `--model` option
  2. Adjust the chunk size with `--chunk-size` option (smaller chunks may process better)
  3. For large documents, consider processing in smaller sections

If you get issues with sentencepiece:
```
pip install https://github.com/Somi-Project/SP313/releases/download/V1.0.0/sentencepiece-0.2.0-cp313-cp313-win_amd64.whl
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
