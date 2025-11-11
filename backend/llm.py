"""
LLM connector for feedback/exercise generation.

Example shows OpenAI ChatCompletion usage. You can replace with local LLM calls
(Llama2 via text-generation inference, or other providers).
"""
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def generate_feedback(transcript: str, profile: dict) -> dict:
    """
    Returns a dict with feedback_text and suggested corrections.
    This is a lightweight wrapper; replace or expand prompts as needed.
    """
    # Build a prompt based on profile
    prompt = f"""
You are an English pronunciation and grammar tutor.
User transcript: "{transcript}"
User profile skill scores: {profile.get('skill_scores')}
Provide:
1) Short assessment (1-2 lines)
2) Specific corrections (if grammar issues, corrected sentence)
3) Pronunciation tips (if phoneme errors)
4) One practice prompt for the user.
Return JSON with keys: assessment, corrections, pronunciation_tips, practice_prompt.
"""
    # Minimal local fallback without OpenAI
    if not OPENAI_API_KEY:
        return {
            "assessment": "Quick assessment (local fallback): needs practice on grammar and consonant clusters.",
            "corrections": ["Example: 'He went to school yesterday.'"],
            "pronunciation_tips": ["Pay attention to final consonant sounds, practice with minimal pairs."],
            "practice_prompt": "Repeat: 'She sells seashells by the seashore.'"
        }
    # If OpenAI key exists, call ChatCompletion
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"You are an expert English tutor."},
                      {"role":"user","content":prompt}],
            max_tokens=400,
            temperature=0.3
        )
        text = resp["choices"][0]["message"]["content"]
        # Expect JSON from model; try parse
        import json
        try:
            parsed = json.loads(text)
            return parsed
        except Exception:
            return {"assessment": text}
    except Exception as e:
        return {"error": str(e)}
