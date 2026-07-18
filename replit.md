# Multi-Lingual Disaster Intelligence Dashboard

A Flask (Python) web app for real-time multilingual disaster intelligence. The Python backend runs all NLP model inference; the frontend is a JavaScript single-page dashboard that calls Python API endpoints.

## How to run

The **Run Flask app** workflow starts the server:

```
python app.py
```

Dashboard is served at port 5000.

## Architecture

- **`app.py`** — Flask backend (686 lines). All API routes, news/social ingestion, and NLP pipeline calls live here.
- **`templates/index.html`** — JS frontend dashboard (~1954 lines). Tailwind CSS + vanilla JS, talks to the Python API via `fetch()`.
- **`nlp/pipeline.py`** — NLP pipeline. Loads mT5-small, IndicBART, mBART-50 models and runs 5 tasks: disaster classification, location extraction, translation, sentiment analysis, summarization.
- **`nlp/train_all.py`** — Optional fine-tuning script for all 3 models × 5 tasks.
- **`nlp/evaluate_all.py`** — Writes metrics to `evaluation_results/metrics.json`.

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/fetch-news` | GET | Fetch & analyze disaster news (GNews, NewsAPI, GDELT) |
| `/api/fetch-social` | GET | Fetch & analyze social posts (Mastodon) |
| `/api/analyze` | POST | Run NLP on arbitrary text with one model |
| `/api/analyze-compare` | POST | Run all 3 models × all 5 tasks for comparison |
| `/api/analyze-document` | POST | Upload TXT/PDF/DOCX for analysis |
| `/api/evaluation-metrics` | GET | Read stored eval metrics |
| `/api/run-evaluation` | POST | Trigger background evaluation |
| `/api/run-training` | POST | Trigger background fine-tuning |
| `/api/training-logs` | GET | Stream training log output |
| `/api/clear-caches` | POST | Clear news/social feed caches |

## Environment variables

Set in `.env`:
- `GNEWS_KEY` — GNews API key (optional, has a default)
- `NEWS_API_KEY` / `NEWSAPI_KEY` — NewsAPI key (optional)
- `MASTODON_API_KEY` — Mastodon bearer token (optional; public Mastodon works without it)
- `TWITTER_API_KEY` — Twitter/X bearer token (optional)

GDELT news ingestion works with no API key at all.

## Notes

- `server.ts` and `package.json` are from an earlier abandoned Express version — do not run them.
- NLP models are downloaded from HuggingFace Hub on first use (~1–3 GB); subsequent starts use the local cache.
- Fine-tuned checkpoints (from `nlp/train_all.py`) are used automatically when present; otherwise base models run via task-prefix prompting.

## User preferences

- Python as backend for all model inference and analysis.
- Frontend is a vanilla JS page (no React/Vue framework).
