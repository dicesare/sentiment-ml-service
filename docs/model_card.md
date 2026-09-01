# Model card — demo baseline

## Purpose

The bundled `LexiconModel` exists only to make the API contract, tests and container reproducible without publishing a private trained artifact. It is not suitable for production moderation or reputation monitoring.

## Historical experimentation represented by the project

The source project compared a simple custom model, a recurrent/deep-learning approach and BERT-style transfer learning, with experiment tracking through MLflow and deployment behind an inference API. Those concepts inform this clean service architecture; historical models, logs and cloud configuration are not redistributed.

## Intended evaluation for a trained replacement

- macro F1 and per-class precision/recall;
- confusion matrix on a held-out set;
- inference latency and memory footprint;
- error slices for negation, sarcasm, URLs, mentions and slang;
- drift monitoring and human escalation policy.

## Limitations and ethics

Sentiment is not equivalent to harmful content or reputational risk. Predictions can encode dataset and language bias. A production system requires documented data provenance, human review, monitoring and a clear appeal/escalation process.
