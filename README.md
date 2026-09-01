# Sentiment ML Service

An end-to-end, recruiter-friendly example of serving a sentiment model through a tested FastAPI application.

## What it demonstrates

- a clean boundary between inference and HTTP code;
- single and batch predictions;
- model loading once at application startup;
- deterministic local demo requiring no cloud account or private model;
- container health checks and automated tests;
- an upgrade path from the demo baseline to a trained transformer.

```text
client → FastAPI → SentimentService → model adapter → label + score
```

## Run locally

```bash
python -m venv .venv
pip install -e .[dev]
uvicorn sentiment_service.api:app --reload
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"texts":["I loved the clear documentation"]}'
```

Open `http://localhost:8000/docs` for the interactive API.

## Reproducible demo

The default `LexiconModel` is intentionally small and transparent. It is not presented as a production-quality ML model: it makes the API, tests and container fully reproducible. A production adapter can implement the same `SentimentModel` protocol and load a versioned artifact at startup.

## API

- `GET /health` — service readiness;
- `POST /v1/predict` — one or more texts, limited to 100 per request.

## Development

```bash
pytest
ruff check .
docker build -t sentiment-ml-service .
```

## Provenance and safety

This public-ready reconstruction is inspired by the concepts demonstrated in `Projet7` and `p7_api`. It contains no trained model, private dataset, cloud identifier or copied Git history. Sentiment140 can be added later through a documented download-and-training pipeline after dataset-license validation.

## License

Released under the [MIT License](LICENSE).
