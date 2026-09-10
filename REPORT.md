# Customer Support Agent Report

## Executive summary

This project implements a brand-specific support-agent pipeline over a configurable sample of the Customer Support on Twitter dataset. The system classifies intents, retrieves historical resolutions, drafts an Ollama response, and deterministically decides whether to auto-handle or escalate.

The selected account is Delta. Baseline metrics are recorded in `backend/evaluation/results_delta_final.json`. Judge metrics are written to `backend/evaluation/judge_results_delta_final.json` after the final agent and judge commands run.

## Dataset and sampling

The source is Kaggle's `thoughtvector/customer-support-on-twitter` dataset. Inbound tweets are paired with outbound responses through `response_tweet_id`. Development uses `MAX_EXAMPLES` after normalization and a fixed random seed. The golden set is a separate 150-250 example sample and must be labeled without using the model prediction as the label.

## Evaluation protocol

The harness reports intent accuracy and macro-F1 for the majority and rule baselines, plus escalation precision and recall. Reply quality and evidence grounding are scored on a 1-5 rubric by the Ollama judge. A human score can be added to each scored record; the harness reports exact and within-one agreement between human and judge scores.

## Results

The final evaluation used 200 labeled examples from the Delta support-account sample.

### Intent and escalation

| Metric | Majority baseline | Rule baseline |
| --- | ---: | ---: |
| Intent accuracy | 13.0% | 3.0% |
| Intent macro-F1 | 0.0050 | 0.0100 |

The rule-based escalation policy achieved 20.42% precision and 74.36% recall.

### Reply quality and grounding

The Ollama judge scored all 200 generated replies on a 1-5 scale:

- Reply quality: **3.75 / 5**
- Evidence grounding correctness: **3.88 / 5**

Human calibration was performed on 40 records:

- Exact judge-human agreement: **70.0%**
- Agreement within one point: **100.0%**

The source files are `backend/evaluation/results_delta_final.json` and `backend/evaluation/judge_results_delta_final.json`.

## Limitations and next steps

- The first intent model is a transparent keyword baseline; a supervised model should follow once labels are available.
- The response judge was calibrated against human scores on 40 records; broader calibration would reduce uncertainty.
- Brand names in `twcs.csv` are represented by account IDs, so the selected account ID must be recorded in `.env` and in the final submission.