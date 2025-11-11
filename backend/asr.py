"""
Wrapper ASR module: tries whisperx implementation then falls back.
"""
import os
from dotenv import load_dotenv
load_dotenv()
USE_LOCAL_ASR = os.getenv("USE_LOCAL_ASR", "true").lower() in ("1","true","yes")
USE_WHISPERX = os.getenv("USE_WHISPERX", "true").lower() in ("1","true","yes")

def transcribe_bytes(audio_bytes: bytes):
    if USE_LOCAL_ASR and USE_WHISPERX:
        try:
            from .asr_whisperx import transcribe_bytes as tx
            out = tx(audio_bytes, device=os.getenv("WHISPERX_DEVICE","cpu"))
            # compute phoneme error rate if possible
            try:
                from .phoneme_scoring import compute_phoneme_error_rate
                per = compute_phoneme_error_rate(out["transcript"], out.get("phonemes", []))
                out["audio_features"]["phoneme_error_rate"] = per
            except Exception:
                out["audio_features"]["phoneme_error_rate"] = None
            return out["transcript"], {"asr_confidence": out.get("asr_confidence",0.8), "audio_features": out.get("audio_features", {})}
        except Exception as e:
            print("whisperx failed:", e)
    # fallback placeholder (existing behavior)
    transcript = "Placeholder transcript (fallback)."
    metadata = {"asr_confidence": 0.85, "audio_features": {"phoneme_error_rate":0.12, "speech_rate_z": -0.2}}
    return transcript, metadata
