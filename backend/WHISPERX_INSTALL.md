# WhisperX installation & tips (CPU/GPU)

Recommended environment: Linux with Python 3.10/3.11.

1) Create venv and activate
   python -m venv .venv
   source .venv/bin/activate

2) Install PyTorch:
- With CUDA (example for CUDA 11.8):
  pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118
- CPU-only:
  pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu

3) Install whisperx and helpers:
  pip install soundfile
  pip install git+https://github.com/m-bain/whisperX.git
  pip install g2p_en

Notes:
- whisperx may require ffmpeg installed on system: apt-get install ffmpeg
- For large models (small/medium) use GPU; for CPU prefer tiny/base
- If installation problems occur, check whisperx README: https://github.com/m-bain/whisperX

4) Quick test script example:
  python -c "from backend.asr_whisperx import transcribe_file; print(transcribe_file('examples/sample.wav', device='cpu'))"
