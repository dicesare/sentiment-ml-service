import re
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True, slots=True)
class Prediction:
    label: str
    score: float


class SentimentModel(Protocol):
    name: str

    def predict(self, texts: Sequence[str]) -> list[Prediction]: ...


class LexiconModel:
    """Transparent offline baseline used only for the reproducible demo."""

    name = "lexicon-demo-v1"
    positive = {"amazing", "clear", "excellent", "good", "great", "love", "loved", "useful"}
    negative = {"awful", "bad", "broken", "confusing", "hate", "hated", "poor", "terrible"}

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        predictions = []
        for text in texts:
            tokens = set(re.findall(r"[a-z']+", text.lower()))
            margin = len(tokens & self.positive) - len(tokens & self.negative)
            label = "positive" if margin >= 0 else "negative"
            score = min(0.99, 0.5 + abs(margin) * 0.12)
            predictions.append(Prediction(label, round(score, 3)))
        return predictions


class SentimentService:
    def __init__(self, model: SentimentModel):
        self.model = model

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        if not texts:
            raise ValueError("At least one text is required")
        if len(texts) > 100:
            raise ValueError("A request can contain at most 100 texts")
        if any(not text.strip() for text in texts):
            raise ValueError("Texts cannot be blank")
        return self.model.predict(texts)
