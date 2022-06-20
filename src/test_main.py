import json

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"msg": "Up"}


def test_predict_greater_than_50k():
    request_body = {
        "age": 25,
        "workclass": "Federal-gov",
        "education": "Some-college",
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Unmarried",
        "race": "Black",
        "sex": "Male",
        "hoursPerWeek": 40,
        "nativeCountry": "Italy",
    }
    response = client.post("/predict", data=json.dumps(request_body))
    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}


def test_predict_less_than_50k():
    request_body = {
        "age": 0,
        "workclass": "State-gov",
        "education": "Bachelors",
        "maritalStatus": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "hoursPerWeek": 0,
        "nativeCountry": "United-States",
    }
    response = client.post("/predict", data=json.dumps(request_body))
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}
