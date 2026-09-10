# Delta Customer Support Agent

An evaluated AI customer-support agent. The system uses historical Customer Support on Twitter conversations to classify intent, retrieve similar resolutions, draft a grounded response with Ollama, and decide whether to auto-handle or escalate.

## What It Does

```text
Customer message
			|
			v
Intent classification --> Similarity retrieval --> Ollama response
			|                         |                       |
			+-------------------------+-----------------------+
																v
										AUTO_HANDLE or ESCALATE
```

- Selects one support account/brand from the dataset. The checked-in evaluation uses Delta.
- Normalizes Kaggle's `twcs.csv` reply-linked tweet format.
- Uses a transparent keyword intent baseline with replaceable classifier interfaces.
- Embeds historical examples with Sentence-Transformers and stores them in ChromaDB.
- Generates concise, evidence-grounded drafts through a configurable Ollama model.
- Escalates low-confidence, weak-evidence, sensitive, or ungrounded requests.
- Includes baseline evaluation, an LLM judge rubric, and human judge calibration.

## Stack

Python, FastAPI, Pydantic, Pandas, Sentence-Transformers, ChromaDB, Ollama, and pytest.

## Quick Start

Run from the repository root in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
Copy-Item .env.example .env
```

Install Ollama from [ollama.com](https://ollama.com), then download the configured model:

```powershell
ollama pull llama3.2:3b
```

## Dataset Setup

Install and configure the Kaggle CLI with an API token. Place `kaggle.json` at `$HOME\.kaggle\kaggle.json`, then run:

```powershell
python -m pip install kaggle
kaggle datasets download -d thoughtvector/customer-support-on-twitter -p backend\data\raw --unzip
```

The loader searches recursively, ignores placeholder files, and supports the Kaggle `twcs.csv` schema. It pairs inbound customer tweets with outbound brand replies through `response_tweet_id`.

The local `.env` selects Delta:

```dotenv
SELECTED_BRAND=Delta
MAX_EXAMPLES=10000
OLLAMA_MODEL=llama3.2:3b
```

`twcs.csv` identifies accounts by ID. For a different brand, set `SELECTED_BRAND` to the responding account ID and regenerate the normalized data and vector index.

## Prepare And Run

Start Ollama in one terminal:

```powershell
ollama serve
```

In a second terminal:

```powershell
cd backend
python scripts\preprocess.py
python scripts\index_examples.py
python -m uvicorn app.main:app --reload
```

The first embedding run downloads the Sentence-Transformers model. The pipeline is intentionally limited by `MAX_EXAMPLES` so local development does not process the full dataset.

## API

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Prediction:

```powershell
Invoke-RestMethod `
	-Method Post `
	http://127.0.0.1:8000/predict `
	-ContentType "application/json" `
	-Body '{"message":"My flight was delayed and I need help"}'
```

Response fields include `intent`, `intent_confidence`, `reply`, `decision`, `reason`, and retrieved `evidence`.

## Evaluation

Run tests:

```powershell
cd backend
python -m pytest tests -q
```

The final evaluation uses 200 manually labeled Delta examples:

```powershell
python scripts\evaluate.py `
	--golden-set evaluation\golden_set_delta_final.jsonl `
	--output evaluation\results_delta_final.json
python scripts\evaluate_agent.py
python scripts\judge_records.py
python scripts\summarize_judged.py
```

Final measured results:

| Metric | Result |
| --- | ---: |
| Majority intent accuracy | 13.0% |
| Majority intent macro-F1 | 0.0050 |
| Rule intent accuracy | 3.0% |
| Rule intent macro-F1 | 0.0100 |
| Escalation precision | 20.42% |
| Escalation recall | 74.36% |
| Reply quality, Ollama judge | 3.75 / 5 |
| Evidence grounding, Ollama judge | 3.88 / 5 |
| Exact human-judge agreement, 40 records | 70.0% |
| Human-judge agreement within one point | 100.0% |

Results are stored in [results_delta_final.json](backend/evaluation/results_delta_final.json) and [judge_results_delta_final.json](backend/evaluation/judge_results_delta_final.json). The 200-example golden set is [golden_set_delta_final.jsonl](backend/evaluation/golden_set_delta_final.jsonl).

## Repository Structure

```text
backend/
	app/              FastAPI app, schemas, and replaceable services
	data/             local raw data, normalized data, and Chroma index
	evaluation/       golden set, metrics, judge, and final results
	scripts/           preprocessing, indexing, evaluation, and reporting
	tests/             unit tests
DECISION_LOG.md     non-obvious engineering decisions
REPORT.md           assignment report and limitations
```

## Limitations

- The current intent model is a keyword baseline, not a trained supervised classifier.
- The selected dataset contains noisy, multilingual, and context-dependent messages.
- Retrieval evidence is not always sufficient for a reliable answer; escalation is preferred in those cases.
- The judge is calibrated on 40 human-reviewed records, so agreement estimates have uncertainty.
- Raw Kaggle data, generated Chroma files, and local environment files are excluded from Git.

## Documentation

- [Assignment report](REPORT.md)
- [Decision log](DECISION_LOG.md)
- [Backend README](backend/README.md)
