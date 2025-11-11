"""
Training pipeline placeholder.
- prepare_finetune_data: convert sessions to instruction-response format
- run_finetune: placeholder to produce adapter weights (LoRA)
"""
import json
from typing import List

def prepare_finetune_data(sessions: List[dict], out_path: str):
    data = []
    for s in sessions:
        data.append({
            "instruction": "Assess and correct the user's English sentence.",
            "input": s.get("transcript",""),
            "output": s.get("feedback","Good job. Try to reduce consonant cluster errors.")
        })
    with open(out_path, "w") as f:
        for item in data:
            f.write(json.dumps(item)+"\n")
    return out_path

def run_finetune(prepared_data_path: str, user_id: int):
    # TODO: hook into HF transformers + PEFT to produce LoRA adapter
    out_weights = f"/artifacts/user_{user_id}_adapter.bin"
    open(out_weights, "wb").close()
    return out_weights
