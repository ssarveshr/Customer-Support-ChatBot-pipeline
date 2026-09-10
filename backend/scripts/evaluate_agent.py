import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.routes import predict
from app.models.schemas import PredictRequest
from evaluation.golden_set import load_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the real support-agent pipeline on labeled golden records.")
    parser.add_argument("--golden-set", type=Path, default=Path("evaluation/golden_set_delta_final.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/agent_records_delta_final.jsonl"))
    args = parser.parse_args()
    records = load_records(args.golden_set)
    if any(not record.get("intent") or record["intent"] == "LABEL_ME" for record in records):
        raise ValueError("Label every golden-set intent before running agent evaluation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for record in records:
            response = predict(PredictRequest(message=record["message"]))
            generated = {
                **record,
                "predicted_intent": response.intent,
                "intent_confidence": response.intent_confidence,
                "reply": response.reply,
                "decision": response.decision,
                "reason": response.reason,
                "evidence": [evidence.model_dump() for evidence in response.evidence],
            }
            output.write(json.dumps(generated) + "\n")
    print(f"Generated agent records for {len(records)} examples at {args.output}")