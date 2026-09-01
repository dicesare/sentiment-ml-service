from fastapi.testclient import TestClient

from sentiment_service.api import app


def test_health_and_prediction():
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        response = client.post("/v1/predict", json={"texts": ["Excellent documentation"]})
        assert response.status_code == 200
        assert response.json()[0]["label"] == "positive"
