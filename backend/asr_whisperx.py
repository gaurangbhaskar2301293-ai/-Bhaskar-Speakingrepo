"""
WhisperX integration for transcription + phoneme alignment.

Usage:
- transcribe_file(path, device="cuda" or "cpu") -> returns dict with:
  { "transcript": str,
    "segments": [...],       # whisperx aligned segments
    "phonemes": [...],       # aligned phoneme sequence (list of phoneme tokens)
    "asr_confidence": float, # heuristic in [0,1]
    "audio_features": { "phoneme_error_rate": float, "speech_rate_z": float }
  }

Requirements (install separately):
  pip install torch soundfile whisperx g2p_en
  # whisperx may require extra deps; see INSTALL.md below.
"""
import os
import tempfile
import json
from typing import Tuple, Dict, Any, List

def _safe_import_whisperx():
    try:
        import whisperx
        return whisperx
    except Exception as e:
        raise RuntimeError("whisperx not available. Install whisperx and its deps. Error: " + str(e))

def _save_bytes_to_tempfile(b: bytes, suffix=".wav") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(b)
    return path

def transcribe_file(path: str, device: str = "cpu"):
    whisperx = _safe_import_whisperx()
    # load model (choose model size per your GPU/CPU)
    model = whisperx.load_model("small", device=device)  # change to base/tiny if CPU only
    # transcribe
    result = model.transcribe(path)
    # load alignment model
    # whisperx provides load_align_model(language_code, device)
    align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
    # run alignment on the segments
    result_aligned = whisperx.align(result["segments"], align_model, metadata, path, device)
    # get phonemes (list per segment) and flatten
    phonemes = whisperx.get_phonemes(result_aligned, "en")
    flat_phonemes = []
    for seg in phonemes:
        # seg is dict with 'phonemes' key (list of tokens)
        p = seg.get("phonemes", [])
        flat_phonemes.extend([ph["phoneme"] for ph in p if "phoneme" in ph])
    # asr confidence heuristic: use word avg_logprob if available
    avg_logprob = None
    try:
        probs = [s.get("avg_logprob") for s in result.get("segments", []) if s.get("avg_logprob") is not None]
        if probs:
            avg_logprob = sum(probs) / len(probs)
    except Exception:
        avg_logprob = None
    # map avg_logprob -> [0,1], approximate
    if avg_logprob is None:
        asr_confidence = 0.85
    else:
        # avg_logprob is negative; higher (closer to 0) better
        asr_confidence = max(0.0, min(1.0, (1.0 + avg_logprob)))  # rough heuristic

    # speech_rate_z placeholder (compute later from duration & token count)
    try:
        total_duration = sum([seg.get("end", 0) - seg.get("start", 0) for seg in result.get("segments", [])])
        total_words = sum([len(seg.get("text","")) for seg in result.get("segments", [])]) or 1
        words_per_sec = total_words / max(0.001, total_duration)
        # approximate z-score vs typical 3.5 wps
        speech_rate_z = (words_per_sec - 3.5) / 1.0
    except Exception:
        speech_rate_z = 0.0

    return {
        "transcript": result.get("text", "").strip(),
        "segments": result_aligned,
        "phonemes": flat_phonemes,
        "asr_confidence": asr_confidence,
        "audio_features": {"phoneme_error_rate": None, "speech_rate_z": speech_rate_z}
    }

def transcribe_bytes(audio_bytes: bytes, device: str = "cpu"):
    path = _save_bytes_to_tempfile(audio_bytes, suffix=".wav")
    try:
        out = transcribe_file(path, device=device)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
    return out
