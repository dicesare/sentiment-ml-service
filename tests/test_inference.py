import pytest

from sentiment_service import LexiconModel, SentimentService


def test_predicts_positive_and_negative():
    service = SentimentService(LexiconModel())
    predictions = service.predict(["Great and useful", "A terrible broken experience"])
    assert [item.label for item in predictions] == ["positive", "negative"]


def test_rejects_blank_text():
    with pytest.raises(ValueError, match="blank"):
        SentimentService(LexiconModel()).predict([" "])
