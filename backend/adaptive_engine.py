import random
from typing import Dict, Any

class AdaptiveEngine:
    """
    Core personalization logic (simplified).
    - assess_utterance: updates profile skill scores based on audio_features
    - next_exercises: returns exercises for weakest skills
    """

    def __init__(self, user_profile: Dict[str, Any]):
        self.profile = user_profile

    def assess_utterance(self, transcript: str, audio_features: Dict[str, Any]) -> Dict[str, float]:
        scores = self.profile.get("skill_scores", {})
        pronunciation_delta = -audio_features.get("phoneme_error_rate", 0) * 100
        fluency_delta = (audio_features.get("speech_rate_z", 0)) * 5
        asr_conf = float(audio_features.get("asr_confidence", 0.8))
        grammar_delta = -max(0, 1 - asr_conf) * 10

        scores["pronunciation"] = max(0, min(100, scores.get("pronunciation",50) + pronunciation_delta))
        scores["fluency"] = max(0, min(100, scores.get("fluency",50) + fluency_delta))
        scores["grammar"] = max(0, min(100, scores.get("grammar",50) + grammar_delta))

        history = self.profile.get("learning_history", [])
        history.append({"transcript": transcript, "features": audio_features, "scores": scores.copy()})
        self.profile["skill_scores"] = scores
        self.profile["learning_history"] = history[-200:]
        return scores

    def next_exercises(self, n=3):
        scores = self.profile.get("skill_scores", {})
        sorted_skills = sorted(scores.items(), key=lambda x: x[1])
        exercises = []
        for skill, _ in sorted_skills[:n]:
            exercises.append(self._make_exercise_for_skill(skill))
        return exercises

    def _make_exercise_for_skill(self, skill):
        if skill == "pronunciation":
            return {"type":"pronunciation", "prompt":"Repeat: 'The quick brown fox jumps over the lazy dog.'", "target_skill":skill, "difficulty":"medium"}
        if skill == "grammar":
            return {"type":"grammar", "prompt":"Correct: 'He go to school yesterday.'", "target_skill":skill, "difficulty":"easy"}
        if skill == "vocab":
            return {"type":"vocab", "prompt":"Use the word 'meticulous' in a sentence about your daily routine.", "target_skill":skill, "difficulty":"medium"}
        if skill == "fluency":
            return {"type":"fluency", "prompt":"Talk for 60 seconds about your favorite hobby.", "target_skill":skill, "difficulty":"hard"}
        if skill == "listening":
            return {"type":"listening", "prompt":"Listen and answer: summarize the short audio in one sentence.", "target_skill":skill, "difficulty":"medium"}
        return {"type":"general", "prompt":"Practice speaking about your last vacation.", "target_skill":skill, "difficulty":"easy"}
