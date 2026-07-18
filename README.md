# Multi-Lingual Disaster Intelligence Dashboard

Flask app (`app.py`) is the single, canonical backend. Do not use `server.ts`
or `npm run dev` — those belonged to an earlier, unfixed Express/Gemini
version and have been removed from this project to prevent accidentally
running the broken path.

## Run Locally

**Prerequisites:** Python 3.9+

1. Install dependencies:
   ```
   pip install -r requirements_nlp.txt
   ```
2. (Optional) Add real API keys to `.env.example` for GNews/NewsAPI live
   news ingestion — GDELT works with no key. No Gemini/LLM API key is
   required or used anywhere in this project.
3. Run the app:
   ```
   python app.py
   ```
   Dashboard is served at http://localhost:5000

## Real model fine-tuning (optional but required for trained-checkpoint results)

```
python -m nlp.train_all          # fine-tunes all 3 models x all 5 tasks
python -m nlp.evaluate_all       # computes real metrics -> evaluation_results/metrics.json
```

Without running these, the dashboard still works — it uses the raw
pretrained base models (mT5-small, IndicBART, mBART-50) via task-prefix
prompting, and every API response includes `is_fine_tuned: false` so it's
always clear which results come from a fine-tuned checkpoint vs. the base
model.

## Models

- mT5-small (`google/mt5-small`)
- IndicBART (`ai4bharat/IndicBART`)
- mBART-50 (`facebook/mbart-large-50-many-to-many-mmt`)

All three independently perform all 5 tasks: disaster type classification,
location extraction, translation, sentiment analysis, summarization.
