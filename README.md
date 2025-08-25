## Reproducible Outputs

This service extracts ICD-10/CPT codes using the OpenAI API. To keep outputs consistent across runs, the server enforces deterministic settings and supports an optional seed.

### Requirements

- Python 3.9+
- `OPENAI_API_KEY` environment variable

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the API

```bash
export OPENAI_API_KEY="sk-..."
# Optional: set a seed to increase reproducibility
export OPENAI_SEED=42

uvicorn app:app --host 0.0.0.0 --port 8000
```

### Determinism Details

- The backend requests are made with `temperature=0`, `top_p=1`, `n=1`, and no penalties.
- If `OPENAI_SEED` is set, the request includes a `seed` parameter (if supported by your OpenAI account/model). If the API rejects the seed, the server retries without it.
- The model response is normalized into a JSON array when possible and deterministically sorted by `type`, `code_type`, `code`, `description`, and `reasoning` so the order is stable across runs.

### Health and Landing

- Landing page: `GET /` serves `landing.html` if present.
- Health check: `GET /health`
- Extraction endpoint: `POST /api/extract` with body `{ "note": "..." }`

### Notes

- Even with deterministic settings, upstream model changes or availability can affect results. Using a fixed `OPENAI_SEED` and a pinned model can improve reproducibility.
