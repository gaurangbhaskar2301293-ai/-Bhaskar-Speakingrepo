from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

from .adaptive_engine import AdaptiveEngine
from .asr import transcribe_bytes
from .llm import generate_feedback

app = FastAPI(title="Bhaskar Speaking Coach API (prototype)")

# Demo in-memory store
USER_STORE = {
    1: {"skill_scores": {"pronunciation":60, "grammar":50, "vocab":70, "fluency":55, "listening":65}, "learning_history":[]}
}

class AssessRequest(BaseModel):
    user_id: int
    transcript: str
    audio_features: Dict[str, Any] = {}

@app.post("/api/assess")
def assess(req: AssessRequest):
    profile = USER_STORE.get(req.user_id)
    if not profile:
        return {"error":"user not found"}
    engine = AdaptiveEngine(profile)
    updated_scores = engine.assess_utterance(req.transcript, req.audio_features)
    exercises = engine.next_exercises(n=3)
    # use LLM to generate feedback text for the transcript
    feedback = generate_feedback(req.transcript, profile)
    return {"scores": updated_scores, "exercises": exercises, "feedback": feedback}

@app.post("/api/upload_audio")
async def upload_audio(user_id: int = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    # Transcribe with ASR (local or cloud depending on config)
    transcript, asr_meta = transcribe_bytes(contents)
    audio_features = asr_meta.get("audio_features", {"phoneme_error_rate":0.12, "speech_rate_z": -0.2, "asr_confidence": 0.85})
    profile = USER_STORE.get(user_id)
    engine = AdaptiveEngine(profile)
    scores = engine.assess_utterance(transcript, audio_features)
    feedback = generate_feedback(transcript, profile)
    return {"transcript": transcript, "scores": scores, "feedback": feedback}

@app.get("/api/profile/{user_id}")
def get_profile(user_id:int):
    return USER_STORE.get(user_id, {})

# For local dev
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
