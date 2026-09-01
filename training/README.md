# Training pipeline contract

A future training implementation should remain separate from the serving application and produce a versioned artifact plus metrics and provenance metadata.

Expected stages:

1. download a clearly licensed dataset without committing it;
2. validate labels, language and train/validation/test separation;
3. train a transparent baseline before deep-learning candidates;
4. log parameters, metrics and artifact checksums with MLflow;
5. export the selected model behind the `SentimentModel` protocol;
6. run API contract and latency tests before deployment.

No historical model, cloud credential, TensorBoard log or absolute local path belongs in this public repository.
